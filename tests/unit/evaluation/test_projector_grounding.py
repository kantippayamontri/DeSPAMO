"""CPU contracts for the dev-only, exact-length visual grounding diagnostic."""

import importlib
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.provenance import digest, file_hash, read_json
from despamo.data.batch import PhoenixSample, collate_phoenix


@pytest.fixture
def grounding():
    return importlib.import_module("scripts.diagnose_projector_grounding")


@pytest.fixture
def samples():
    return {
        clip: PhoenixSample(
            clip, "Signer03", f"secret reference {clip}.", "secret gloss",
            "secret english", "secret spanish", "secret french",
            torch.full((length, 2048), float(index)),
            torch.full((2, 1024), float(index + 10)),
        )
        for index, (clip, length) in enumerate((("a", 3), ("b", 3), ("c", 5), ("d", 5)))
    }


def _records(samples):
    return [{"fileid": s.clip_id, "signer": s.signer, "text": s.text}
            for s in samples.values()]


def test_selection_deterministic_sorted_exact_and_distinct(grounding, samples):
    records = _records(samples)
    spatial = {key: len(s.spatial) for key, s in samples.items()}
    motion = {key: len(s.motion) for key, s in samples.items()}
    first = grounding.select_donor_pairs(records, spatial, motion, count=4)
    second = grounding.select_donor_pairs(records[::-1], spatial, motion, count=4)
    assert first == second
    assert first["selected_ids"] == ["a", "b", "c", "d"]
    assert first["donor_map"] == {"a": "b", "b": "a", "c": "d", "d": "c"}
    assert first["eligibility"]["eligible_groups"] == 2
    assert first["donor_map_hash"] == digest(first["donor_map"])


def test_singletons_and_normalized_same_text_groups_excluded(grounding):
    rows = [{"fileid": clip, "signer": "Signer03", "text": text}
            for clip, text in (("a", "different"), ("b", "other"),
                               ("c", "SAME ."), ("d", " same . "), ("e", "alone"))]
    lengths = {"a": 1, "b": 1, "c": 2, "d": 2, "e": 3}
    result = grounding.select_donor_pairs(rows, lengths, lengths, count=2)
    assert result["selected_ids"] == ["a", "b"]
    assert result["eligibility"] == {
        "total_clips": 5, "total_groups": 3, "eligible_clips": 2, "eligible_groups": 1,
        "excluded_clips": 3, "excluded_groups": 2,
        "singleton_groups": 1, "same_text_groups": 1,
    }
    with pytest.raises(ValueError, match="eligible"):
        grounding.select_donor_pairs(rows, lengths, lengths, count=4)
    with pytest.raises(ValueError, match="even"):
        grounding.select_donor_pairs(rows, lengths, lengths, count=3)
    rows[0]["signer"] = "Signer07"
    with pytest.raises(ValueError, match="Signer03"):
        grounding.select_donor_pairs(rows, lengths, lengths, count=2)


@pytest.mark.parametrize("condition", ["normal", "shuffle_spatial", "shuffle_motion",
                                       "shuffle_both"])
def test_conditions_preserve_metadata_masks_and_originals(grounding, samples, condition):
    ids = ["a", "b", "c", "d"]
    donors = {"a": "b", "b": "a", "c": "d", "d": "c"}
    originals = {key: (s.spatial.clone(), s.motion.clone()) for key, s in samples.items()}
    result = grounding.build_condition_samples(samples, ids, donors, condition)
    normal = collate_phoenix(list(samples.values()))
    batch = collate_phoenix(result)
    assert torch.equal(batch.spatial_mask, normal.spatial_mask)
    assert torch.equal(batch.motion_mask, normal.motion_mask)
    for item in result:
        source = samples[item.clip_id]
        assert replace(item, spatial=source.spatial, motion=source.motion) == source
        donor = samples[donors[item.clip_id]]
        expected_s = donor if condition in {"shuffle_spatial", "shuffle_both"} else source
        expected_m = donor if condition in {"shuffle_motion", "shuffle_both"} else source
        assert torch.equal(item.spatial, expected_s.spatial)
        assert torch.equal(item.motion, expected_m.motion)
        item.spatial.add_(100)
        item.motion.add_(100)
    for key, sample in samples.items():
        assert torch.equal(sample.spatial, originals[key][0])
        assert torch.equal(sample.motion, originals[key][1])


def test_condition_rejects_shape_and_incoherent_donor_maps(grounding, samples):
    for donors in ({"a": "a", "b": "b"}, {"a": "b", "b": "c"},
                   {"a": "c", "c": "a"}):
        with pytest.raises(ValueError):
            grounding.build_condition_samples(samples, sorted(donors), donors, "shuffle_both")


class FakeModel(torch.nn.Module):
    prompt_template = "Translate to {}"

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.zeros(1))
        self.language_model = SimpleNamespace(generate_text=self.generate)

    def visual_adapter(self, spatial, spatial_mask, motion, motion_mask):
        assert not torch.is_grad_enabled()
        return spatial[:, 0, :1] + motion[:, 0, :1], spatial_mask[:, :1]

    def generate(self, visual, valid, prompts, mode, beams, length):
        assert prompts == ["Translate to German"] * len(prompts)
        assert (mode, beams, length) == ("deterministic", 5, 64)
        assert not self.training
        return [f"visual {int(value)}" for value in visual[:, 0]]


def test_fresh_generation_only_visuals_and_language_prompt(grounding, samples):
    ids = list(samples)
    donors = {"a": "b", "b": "a", "c": "d", "d": "c"}
    normal = grounding.score_condition(FakeModel(), samples, ids, donors, "normal", "split")
    both = grounding.score_condition(FakeModel(), samples, ids, donors, "shuffle_both", "split")
    assert [i["prediction"] for i in normal["items"]] == [
        "visual 10", "visual 12", "visual 14", "visual 16"]
    assert [i["prediction"] for i in both["items"]] == [
        "visual 12", "visual 10", "visual 16", "visual 14"]
    assert [i["reference"] for i in both["items"]] == [s.text for s in samples.values()]
    assert [i["donor_id"] for i in both["items"]] == ["b", "a", "d", "c"]


def test_summary_tracks_donor_matches_and_informative_pairs(grounding):
    donors = {"a": "b", "b": "a", "c": "d", "d": "c"}
    def report(predictions):
        return {"items": [{"clip_id": clip, "prediction": pred, "reference": clip,
                           "donor_id": donors[clip]}
                          for clip, pred in zip(donors, predictions, strict=True)],
                "metrics": {"bleu4": 2.0, "rougeL_f1": 0.2}}
    normal = report(["one one", "two", "generic", "generic"])
    both = report(["two", "one one", "generic", "generic"])
    both["metrics"]["bleu4"] = 1.0
    result = grounding.summarize_grounding(normal, {"normal": normal, "shuffle_both": both})
    assert result["conditions"]["shuffle_both"]["metric_deltas"]["bleu4"] == -1.0
    assert result["conditions"]["shuffle_both"]["prediction_change_rate"] == 0.5
    donor = result["conditions"]["shuffle_both"]["donor_consistency"]
    assert donor == {"matches": 4, "rate": 1.0, "informative_recipients": 2,
                     "informative_pairs": 1, "informative_matches": 2,
                     "informative_rate": 1.0}
    repetition = result["conditions"]["normal"]["repetition"]
    assert repetition["unique_predictions"] == 3
    assert repetition["mean_whitespace_length"] == 1.25
    assert repetition["repeated_token_fraction"] == 0.2


def test_condition_publication_reuse_and_source_drift(grounding, tmp_path, samples):
    ids = list(samples)
    donors = {"a": "b", "b": "a", "c": "d", "d": "c"}
    report = grounding.score_condition(FakeModel(), samples, ids, donors, "normal", "split")
    binding = {"source_hash": "original", "donor_map_hash": digest(donors)}
    path = tmp_path / "normal.json"
    grounding.publish_condition(path, report, binding)
    original = path.read_bytes()
    assert grounding.read_condition(path, binding, ids, donors) == read_json(path)
    grounding.publish_condition(path, report, binding)
    assert path.read_bytes() == original
    with pytest.raises(ValueError, match="provenance"):
        grounding.read_condition(path, {**binding, "source_hash": "changed"}, ids, donors)
    with pytest.raises(ValueError):
        grounding.publish_condition(path, {**report, "metrics": {}}, binding)
    value = read_json(path)
    value["items"][0]["prediction"] = "tampered"
    from despamo.appearance.provenance import atomic_json
    atomic_json(path, value)
    with pytest.raises(ValueError, match="hash"):
        grounding.read_condition(path, binding, ids, donors)


@pytest.fixture
def runtime(grounding, samples, tmp_path, monkeypatch):
    from despamo.appearance.provenance import atomic_json

    args = SimpleNamespace(mode="run", output_dir=tmp_path / "diagnostic",
                           gpu_cap_seconds=3600, count=4)
    for name in grounding.INPUTS:
        setattr(args, name.replace("-", "_"), tmp_path / "sources" / name)
    lengths = {c: len(s.spatial) for c, s in samples.items()}
    motions = {c: len(s.motion) for c, s in samples.items()}
    selection = grounding.select_donor_pairs(_records(samples), lengths, motions, count=4)
    binding = {"models": {str(seed): {"checkpoint_hash": f"checkpoint{seed}"}
                          for seed in (0, 1)},
               "decoder_hash": digest(grounding.DECODER),
               "donor_map_hash": selection["donor_map_hash"],
               "split_hash": "split", "gpu_cap_seconds": 3600}
    models = {0: {"seed": 0}, 1: {"seed": 1}}
    monkeypatch.setattr(grounding, "MODELS", {
        0: tmp_path / "models/seed0", 1: tmp_path / "models/seed1"})
    monkeypatch.setattr(grounding, "_prepare", lambda _: (binding, selection, samples, models))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1234)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    loaded = []

    class CPUModel(FakeModel):
        def to(self, *args, **kwargs):
            return self

    def load(selected):
        loaded.append(selected["seed"])
        return CPUModel()

    monkeypatch.setattr(grounding, "_load_projector_model", load)
    return SimpleNamespace(args=args, binding=binding, selection=selection,
                           loaded=loaded, atomic_json=atomic_json)


def test_readonly_preflight_and_sequential_run_reuse(grounding, runtime):
    runtime.args.mode = "preflight"
    ready = grounding.diagnose(runtime.args)
    assert ready["selection"]["selected_ids"] == ["a", "b", "c", "d"]
    assert not runtime.args.output_dir.exists()
    assert not runtime.args.output_dir.with_suffix(".lock").exists()
    runtime.args.mode = "run"
    complete = grounding.diagnose(runtime.args)
    assert complete["status"] == "complete"
    assert runtime.loaded == [0, 1]
    assert complete["peak_gpu_bytes"] == 1234
    assert complete["seeds"]["0"]["grounding"]["conditions"]["shuffle_both"][
        "donor_consistency"]["informative_rate"] == 1.0
    before = {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()}
    assert grounding.diagnose(runtime.args) == complete
    assert runtime.loaded == [0, 1]
    after = {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()}
    assert after == before
    runtime.args.mode = "preflight"
    assert grounding.diagnose(runtime.args)["status"] == "ready"
    assert {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()} == before


def test_resume_skips_completed_conditions_and_preserves_reports(grounding, runtime, monkeypatch):
    real_score = grounding.score_condition
    calls = []

    def interrupted(*args, **kwargs):
        condition = args[4]
        calls.append(condition)
        if condition == "shuffle_motion":
            raise RuntimeError("interrupted")
        return real_score(*args, **kwargs)

    monkeypatch.setattr(grounding, "score_condition", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        grounding.diagnose(runtime.args)
    assert calls == ["normal", "shuffle_spatial", "shuffle_motion"]
    normal = runtime.args.output_dir / "seed0-normal.json"
    saved = normal.read_bytes()
    calls.clear()
    monkeypatch.setattr(grounding, "score_condition", real_score)
    result = grounding.diagnose(runtime.args)
    assert result["status"] == "complete"
    assert normal.read_bytes() == saved
    assert runtime.loaded == [0, 0, 1]


def test_complete_reuse_refuses_drift_without_changing_artifacts(grounding, runtime):
    grounding.diagnose(runtime.args)
    before = {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()}
    runtime.binding["source_hash"] = "changed"
    with pytest.raises(ValueError, match="provenance"):
        grounding.diagnose(runtime.args)
    assert {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()} == before


def test_budget_exhaustion_refuses_loading_model(grounding, runtime):
    runtime.args.gpu_cap_seconds = 10
    with pytest.raises(TimeoutError, match="budget"):
        grounding.diagnose(runtime.args)
    assert not runtime.loaded
    state = read_json(runtime.args.output_dir / "state.json")
    assert state["status"] == "partial"


def test_output_rejects_source_and_symlink_paths(grounding, runtime, tmp_path):
    runtime.args.output_dir = runtime.args.adapted_root / "diagnostic"
    with pytest.raises(ValueError, match="separate"):
        grounding.diagnose(runtime.args)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    runtime.args.output_dir = link / "diagnostic"
    with pytest.raises(ValueError, match="separate"):
        grounding.diagnose(runtime.args)


def test_output_cannot_create_state_in_model_cache(grounding, runtime, tmp_path):
    cache = tmp_path / "frozen-model-cache"
    runtime.atomic_json(grounding.MODELS[0] / "budget.json", {
        "identity": {"config": {"model": {"cache_dir": str(cache)}}}})
    runtime.args.output_dir = cache / "diagnostic"
    with pytest.raises(ValueError, match="separate"):
        grounding.diagnose(runtime.args)
    assert not cache.exists()


def test_cumulative_resume_budget_does_not_reset(grounding, runtime):
    runtime.atomic_json(runtime.args.output_dir / "state.json", {
        "run_policy": grounding.RUN_POLICY, "status": "partial", "binding": None,
        "elapsed_seconds": 3590.0, "peak_gpu_bytes": 0})
    with pytest.raises(TimeoutError):
        grounding.diagnose(runtime.args)
    assert not runtime.loaded
    assert read_json(runtime.args.output_dir / "state.json")["elapsed_seconds"] >= 3590


def test_stable_lock_excludes_second_runner(grounding, runtime):
    import fcntl

    lock_path = runtime.args.output_dir.with_name(runtime.args.output_dir.name + ".lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        with pytest.raises(BlockingIOError):
            grounding.diagnose(runtime.args)
    assert not runtime.args.output_dir.exists()
    assert not runtime.loaded


def test_resume_all_reports_complete_needs_no_gpu(grounding, runtime, monkeypatch):
    grounding.diagnose(runtime.args)
    (runtime.args.output_dir / "summary.json").unlink()
    path = runtime.args.output_dir / "state.json"
    state = read_json(path)
    runtime.atomic_json(path, {**state, "status": "partial"})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = grounding.diagnose(runtime.args)
    assert result["status"] == "complete"
    assert runtime.loaded == [0, 1]


@pytest.mark.parametrize("artifact", ["provenance.json", "selection.json"])
def test_completed_reuse_requires_saved_provenance(grounding, runtime, artifact):
    grounding.diagnose(runtime.args)
    (runtime.args.output_dir / artifact).unlink()
    runtime.args.mode = "preflight"
    with pytest.raises(ValueError, match="missing"):
        grounding.diagnose(runtime.args)


@pytest.fixture
def admission(grounding, samples, runtime, monkeypatch):
    """Replace expensive external source/model reads, retain all admission logic."""
    args = runtime.args
    records = _records(samples) + [
        {"fileid": f"x{i:03}", "signer": "Signer03", "text": f"reference {i}"}
        for i in range(578)
    ]
    ids = [r["fileid"] for r in records]
    adapted = {"checkpoint_hash": "adaptation"}
    protocol = {"protocol_hash": grounding.PROTOCOL_HASH, "decoder": grounding.DECODER,
                "supervision_policy": grounding.POLICY, "human_review_status": "not_assessed",
                "split_hash": "split", "split": {"groups": {
                    "dev": {"clip_ids": ids, "signers": ["Signer03"]}}}}

    class View:
        def __init__(self):
            self.records = records

        def __getitem__(self, index):
            return samples[ids[index]]

    class UnusedView:
        def __getitem__(self, index):
            raise AssertionError("non-dev sample accessed")

    class Manifest:
        def __init__(self, motion=False):
            self.motion = motion

        def require(self, split, clip):
            assert split == "train"  # Logical dev originates in physical train.
            length = (len(samples[clip].motion if self.motion else samples[clip].spatial)
                      if clip in samples else int(clip[1:]) + 20)
            return SimpleNamespace(length=length, path=f"train/{clip}.npy")

    lineage = {"adaptation_checkpoint_hash": "adaptation", "adapted_encoder_key": digest(adapted),
               "adapted_manifest_hash": "bytes", "adapted_frame_rows_hash": "bytes"}
    selections = {}
    checkpoints = {}
    for seed in (0, 1):
        data = {"batch_size": 4, "num_workers": 0,
                "spatial_root": str(args.adapted_root),
                "spatial_manifest": str(args.adapted_root / "complete/manifest.json"),
                "motion_root": str(args.motion_root), "motion_manifest": str(args.motion_manifest),
                "annotation_root": str(args.annotation.parent)}
        identity = {"projector": "linear", "seed": seed, **lineage,
                    "config": {"data": data, "model": {"use_in_context": False,
                                                       "warm_up_steps": 2000},
                               "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1}}}
        selections[seed] = {"identity": identity, "checkpoint": f"checkpoint{seed}",
                            "checkpoint_hash": "bytes", "step": 16000}
        checkpoints[f"checkpoint{seed}"] = {"run_metadata": identity, "global_step": 16000,
                                             "state_dict": {"weight": torch.ones(1)}}

    def validate(run, given_protocol, given_ids):
        assert given_ids == tuple(ids)
        return selections[next(seed for seed, path in grounding.MODELS.items() if path == run)]

    # Restore real admission, overridden by the runtime fixture.
    monkeypatch.undo()
    monkeypatch.setattr(grounding, "read_json", lambda path: adapted)
    monkeypatch.setattr(grounding, "file_hash", lambda path: "bytes")
    monkeypatch.setattr(grounding, "load_e3_views", lambda *a: (
        protocol, {"dev": View(), "test": UnusedView(), "train": UnusedView()}))
    monkeypatch.setattr(grounding.FeatureManifest, "load",
                        lambda path: Manifest(path == args.motion_manifest))
    monkeypatch.setattr(grounding, "validate_projector_selection", validate)
    monkeypatch.setattr(torch, "load", lambda path, **kwargs: checkpoints[path])
    return SimpleNamespace(args=args, protocol=protocol, records=records,
                           selections=selections, checkpoints=checkpoints)


def test_admission_dev_only_full_metadata_and_cache_lineage(grounding, admission):
    binding, selected, samples, models = grounding._prepare(admission.args)
    assert binding["dev_clips"] == 582
    assert selected["selected_ids"] == ["a", "b", "c", "d"]
    assert list(samples) == ["a", "b", "c", "d"]
    assert list(models) == [0, 1]
    assert binding["features"]["a"] == {"spatial": "bytes", "motion": "bytes"}


@pytest.mark.parametrize("drift", ["metadata", "lineage", "decoder", "signer", "step"])
def test_admission_rejects_frozen_source_or_checkpoint_drift(grounding, admission, drift):
    if drift == "metadata":
        admission.checkpoints["checkpoint1"]["run_metadata"] = {"wrong": "metadata"}
    elif drift == "lineage":
        admission.selections[0]["identity"]["adapted_manifest_hash"] = "changed"
    elif drift == "decoder":
        admission.protocol["decoder"] = {**grounding.DECODER, "beam_size": 1}
    elif drift == "signer":
        admission.records[0]["signer"] = "Signer07"
    else:
        admission.checkpoints["checkpoint0"]["global_step"] = 5500
    with pytest.raises(ValueError):
        grounding._prepare(admission.args)


def test_inference_manifest_covers_local_decoder_adapter_and_dependencies(grounding):
    helper = getattr(grounding, "inference_source_files", None)
    assert callable(helper), "missing complete inference source manifest"
    files = grounding.inference_source_files(grounding.ROOT)
    package_files = {p.relative_to(grounding.ROOT).as_posix()
                     for p in (grounding.ROOT / "src/despamo").rglob("*.py")}
    required = {
        "src/despamo/models/flan_t5.py", "src/despamo/models/visual_adapter.py",
        "src/despamo/models/temporal.py", "src/despamo/factory.py",
        "scripts/diagnose_projector_grounding.py", "scripts/evaluate_projector_pilot.py",
        "scripts/evaluate_e3_pilot.py", "scripts/train_projector_pilot.py",
        "scripts/train_signer_pilot.py", "scripts/train_e3_pilot.py",
        "scripts/run_projector_overnight.py", "tools/dinov3/storage.py",
        "tools/dinov3/identity.py", "pyproject.toml", "uv.lock",
    }
    assert package_files | required == set(files)
    assert list(files) == sorted(files)
    assert all(path.is_file() and path == grounding.ROOT / name for name, path in files.items())


@pytest.fixture
def synthetic_inference_sources(grounding, tmp_path):
    helper = getattr(grounding, "inference_source_files", None)
    assert callable(helper), "missing complete inference source manifest"
    project = tmp_path / "synthetic-project"
    for name, source in grounding.inference_source_files(grounding.ROOT).items():
        destination = project / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return project


@pytest.mark.parametrize("module", ["flan_t5.py", "visual_adapter.py"])
def test_prepare_binds_real_inference_source_bytes(
    grounding, admission, synthetic_inference_sources, monkeypatch, module,
):
    project = synthetic_inference_sources
    monkeypatch.setattr(grounding, "ROOT", project)
    monkeypatch.setattr(grounding, "file_hash", lambda path: (
        file_hash(path) if path.is_relative_to(project) else "bytes"))
    before = grounding._prepare(admission.args)[0]
    path = project / "src/despamo/models" / module
    path.write_bytes(path.read_bytes() + b"\n# synthetic changed inference implementation\n")
    after = grounding._prepare(admission.args)[0]
    key = path.relative_to(project).as_posix()
    assert before["source_hashes"][key] != after["source_hashes"][key]
    assert after["source_hashes"][key] == file_hash(path)
    assert digest(before) != digest(after)


@pytest.mark.parametrize("status", ["partial", "complete"])
@pytest.mark.parametrize("module", ["flan_t5.py", "visual_adapter.py"])
def test_inference_byte_drift_rejected_before_generation_or_artifact_writes(
    grounding, runtime, synthetic_inference_sources, monkeypatch, status, module,
):
    project = synthetic_inference_sources
    prepared = grounding._prepare(runtime.args)

    def prepare(args):
        files = grounding.inference_source_files(project)
        binding = {**prepared[0], "source_hashes": {
            name: file_hash(path) for name, path in files.items()
        }}
        return binding, *prepared[1:]

    monkeypatch.setattr(grounding, "_prepare", prepare)
    real_score = grounding.score_condition
    if status == "partial":
        def interrupt(*args, **kwargs):
            if args[4] == "shuffle_spatial":
                raise RuntimeError("interrupt after fresh normal")
            return real_score(*args, **kwargs)
        monkeypatch.setattr(grounding, "score_condition", interrupt)
        with pytest.raises(RuntimeError, match="interrupt"):
            grounding.diagnose(runtime.args)
    else:
        grounding.diagnose(runtime.args)
    before = {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()}
    path = project / "src/despamo/models" / module
    path.write_bytes(path.read_bytes() + b"\n# changed after normal generation\n")
    monkeypatch.setattr(grounding, "score_condition", real_score)
    loaded_before = list(runtime.loaded)
    with pytest.raises(ValueError, match="provenance"):
        grounding.diagnose(runtime.args)
    assert runtime.loaded == loaded_before
    assert {p.name: p.read_bytes() for p in runtime.args.output_dir.iterdir()} == before

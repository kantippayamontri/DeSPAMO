"""CPU-only held-out admission, reconstruction, and crash-resume contracts."""

import copy
import fcntl
import importlib
import signal
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import PhoenixSample
from despamo.models.residual_projector import enable_residual_projectors
from despamo.models.visual_adapter import SpaMoVisualAdapter

DECODER = {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}
PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
POLICY = "qwen-schema98-unreviewed-v1"


@pytest.fixture
def cli():
    script = Path(__file__).resolve().parents[3] / "scripts/evaluate_projector_pilot.py"
    assert script.is_file(), "projector held-out evaluator is not implemented"
    return importlib.import_module("scripts.evaluate_projector_pilot")


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatial_projector = nn.Linear(2, 2)
        self.motion_projector = nn.Linear(2, 2)

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        return self.spatial_projector(spatial) + self.motion_projector(motion), spatial_mask


class TinyModel(nn.Module):
    def __init__(self, calls, fail_batch=None):
        super().__init__()
        self.visual_adapter = TinyAdapter()
        self.prompt_template = "Translate into {}."
        self.language_model = SimpleNamespace(generate_text=self.generate_text)
        self.calls = calls
        self.fail_batch = fail_batch

    def generate_text(self, visual, mask, prompts, mode, beam_size, max_length):
        assert not self.training and torch.is_inference_mode_enabled()
        assert prompts == ["Translate into German."] * len(prompts)
        assert (mode, beam_size, max_length) == ("deterministic", 5, 64)
        if self.fail_batch is not None and len(self.calls) == self.fail_batch:
            raise TimeoutError("simulated generation cap")
        self.calls.append(len(prompts))
        return ["am montag regen im norden"] * len(prompts)


class TinyView:
    def __init__(self, ids):
        self.records = [{"fileid": clip, "signer": "Signer07"} for clip in ids]
        self.reads = []
        self.reference = "SECRET HELDOUT REFERENCE"

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self.reads.append(index)
        return PhoenixSample(
            clip_id=self.records[index]["fileid"], signer="Signer07", text=self.reference,
            gloss="SECRET GLOSS", en_text="SECRET EN", es_text="SECRET ES", fr_text="SECRET FR",
            spatial=torch.ones(1, 2), motion=torch.ones(1, 2),
        )


@pytest.fixture
def study(tmp_path, monkeypatch, cli):
    args = SimpleNamespace(
        mode="preflight", protocol=tmp_path / "protocol.json", dataset=tmp_path / "dataset",
        text_manifest=tmp_path / "text.json", dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion", motion_manifest=tmp_path / "motion.json",
        annotation=tmp_path / "annotations/train_info_ml.npy", adapted_root=tmp_path / "adapted",
        projector_run=tmp_path / "run", output=tmp_path / "results/report.json",
        registry=tmp_path / "claims/registry.json", gpu_cap_seconds=1800,
    )
    ids = tuple(f"test-{i:04d}" for i in range(768))
    dev_ids = ("dev-0", "dev-1")
    protocol = {
        "protocol_hash": PROTOCOL_HASH, "split_hash": "split", "decoder": DECODER,
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
        "source_hash": "source", "frame_rows_hash": "original-rows",
        "split": {"groups": {"test": {"clip_ids": list(ids), "signers": ["Signer07"]},
                             "dev": {"clip_ids": list(dev_ids)}}},
    }
    adapted = {"checkpoint_hash": "adapt-checkpoint", "protocol_hash": PROTOCOL_HASH,
               "source_hash": "source", "frame_rows_hash": "original-rows"}
    for path, value in (
        (args.protocol, protocol), (args.text_manifest, {"text": "frozen"}),
        (args.motion_manifest, {"motion": "frozen"}), (args.annotation, {"annotation": "frozen"}),
        (args.dataset / "source.json", {"source": "frozen"}),
        (args.dataset / "version.json", {"version": "frozen"}),
        (args.dino_root / "complete/manifest.json", {"dino": "frozen"}),
        (args.dino_root / "complete/frame_rows.json", {"rows": "frozen"}),
        (args.adapted_root / "complete/identity.json", adapted),
        (args.adapted_root / "complete/manifest.json", {"adapted": "frozen"}),
        (args.adapted_root / "complete/frame_rows.json", {"rows": "adapted"}),
    ):
        atomic_json(path, value)
    config = {
        "seed": 0,
        "data": {"num_workers": 0, "batch_size": 4,
                 "spatial_root": str(args.adapted_root),
                 "spatial_manifest": str(args.adapted_root / "complete/manifest.json"),
                 "motion_root": str(args.motion_root), "motion_manifest": str(args.motion_manifest),
                 "annotation_root": str(args.annotation.parent)},
        "model": {"use_in_context": False, "num_in_context": 0, "warm_up_steps": 2000,
                  "cache_dir": str(tmp_path / "model-cache")},
        "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1},
    }
    identity = {
        "run_policy": "signer-pilot-projector-ablation-v1", "variant": "E3_projector_linear",
        "projector": "linear", "steps": 16000, "seed": 0, "residual_hidden_dim": None,
        "residual_seed": None, "dev_checkpoints": [5500, 10400, 16000],
        "warmup_steps": 2000, "config": config,
        "protocol_hash": PROTOCOL_HASH, "split_hash": "split",
        "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
        "adapted_encoder_key": digest(adapted),
        "adapted_manifest_hash": file_hash(args.adapted_root / "complete/manifest.json"),
        "adapted_frame_rows_hash": file_hash(args.adapted_root / "complete/frame_rows.json"),
        "supervision_policy": POLICY, "human_review_status": "not_assessed", "run_key": "run",
    }
    calls = []
    builder_calls = []
    failure = {"batch": None}

    def builder(config, steps, seed):
        builder_calls.append((steps, seed))
        return TinyModel(calls, failure["batch"])

    monkeypatch.setitem(sys.modules, "scripts.train_projector_pilot", SimpleNamespace(
        build_seeded_projector_model=builder,
    ))
    view = TinyView(ids)
    monkeypatch.setattr(cli, "load_e3_views", lambda *args: (protocol, {"test": view}))
    monkeypatch.setattr(cli, "_runtime_device", lambda: torch.device("cpu"))
    state = SimpleNamespace(args=args, ids=ids, dev_ids=dev_ids, protocol=protocol,
                            identity=identity, calls=calls, view=view, failure=failure,
                            builder_calls=builder_calls)
    _write_run(state)
    return state


def _write_run(study):
    run, identity = study.args.projector_run, study.identity
    atomic_json(run / "budget.json", {"status": "closed", "identity": identity})
    atomic_json(run / "run-status.json", {"status": "complete", "global_step": 16000,
                                        "run_key": "run"})
    model = TinyModel([])
    if identity["projector"] == "residual":
        enable_residual_projectors(cast(SpaMoVisualAdapter, model.visual_adapter), 1024,
                                   identity.get("residual_seed", 1))
    for step, bleu in ((5500, 1.0), (10400, 2.0), (16000, 3.0)):
        checkpoint = run / f"step-{step}.ckpt"
        torch.save({"state_dict": model.state_dict(), "run_metadata": copy.deepcopy(identity),
                    "global_step": step}, checkpoint)
        report = {
            "checkpoint_step": step, "checkpoint_hash": file_hash(checkpoint),
            "split": "dev", "split_hash": "split", "decoding": DECODER,
            "model_variant": identity["variant"], "metrics": {"bleu4": bleu},
            "items": [{"clip_id": clip, "prediction": "hyp", "reference": "ref"}
                      for clip in study.dev_ids],
        }
        atomic_json(run / f"dev-{step}.json", report)
    best = read_json(run / "dev-16000.json")
    atomic_json(run / "selected-checkpoint.json", {
        "step": 16000, "checkpoint_hash": best["checkpoint_hash"], "metrics": best["metrics"],
        "model_variant": identity["variant"], "protocol_hash": PROTOCOL_HASH,
        "adaptation_checkpoint_hash": identity["adaptation_checkpoint_hash"],
        "supervision_policy": POLICY, "human_review_status": "not_assessed",
    })


def test_complete_selection_returns_bound_checkpoint(cli, study):
    selected = cli.validate_projector_selection(
        study.args.projector_run, study.protocol, study.dev_ids,
    )
    assert selected["step"] == 16000
    assert selected["checkpoint"] == study.args.projector_run / "step-16000.ckpt"
    assert selected["checkpoint_hash"] == file_hash(selected["checkpoint"])
    assert selected["identity"] == study.identity


@pytest.mark.parametrize("change", [
    "partial", "open_budget", "policy", "horizon", "missing_report", "dev_order",
    "dev_split", "decoder", "variant", "schedule", "not_best", "selected_hash", "bytes",
])
def test_selection_refuses_incomplete_or_unbound_runs(cli, study, change):
    run = study.args.projector_run
    if change == "missing_report":
        (run / "dev-10400.json").unlink()
    elif change == "bytes":
        (run / "step-16000.ckpt").write_bytes(b"changed")
    else:
        path = run / "dev-10400.json"
        if change == "partial":
            path = run / "run-status.json"
        elif change in {"open_budget", "policy", "horizon", "schedule"}:
            path = run / "budget.json"
        elif change in {"not_best", "selected_hash"}:
            path = run / "selected-checkpoint.json"
        data = read_json(path)
        if change == "partial":
            data["status"] = "partial"
        elif change == "open_budget":
            data["status"] = "open"
        elif change in {"policy", "horizon", "schedule"}:
            key, value = {"policy": ("run_policy", "other"), "horizon": ("steps", 8000),
                          "schedule": ("dev_checkpoints", [5500, 10400])}[change]
            data["identity"][key] = value
        elif change == "dev_order":
            data["items"].reverse()
        elif change == "dev_split":
            data["split"] = "test"
        elif change == "decoder":
            data["decoding"]["beam_size"] = 1
        elif change == "variant":
            data["model_variant"] = "E3_projector_residual"
        elif change == "not_best":
            data["step"] = 5500
        elif change == "selected_hash":
            data["checkpoint_hash"] = "wrong"
        atomic_json(path, data)
    with pytest.raises(ValueError):
        cli.validate_projector_selection(run, study.protocol, study.dev_ids)


@pytest.mark.parametrize("arm,seed,residual_seed", [("linear", 0, None), ("residual", 0, 1),
                                                    ("residual", 1, 2), ("linear", 1, None)])
def test_model_reconstruction_strictly_matches_variant_and_seed(cli, study, arm, seed,
                                                              residual_seed):
    study.identity.update(projector=arm, variant=f"E3_projector_{arm}", seed=seed,
                          residual_hidden_dim=1024 if arm == "residual" else None,
                          residual_seed=residual_seed)
    study.identity["config"]["seed"] = seed
    _write_run(study)
    selected = cli.validate_projector_selection(study.args.projector_run, study.protocol,
                                                study.dev_ids)
    model = cli._load_projector_model(selected)
    assert study.builder_calls == [(16000, seed)]
    assert isinstance(model.visual_adapter.spatial_projector, nn.Linear) == (arm == "linear")
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    for name, value in model.state_dict().items():
        assert torch.equal(value, checkpoint["state_dict"][name])


@pytest.mark.parametrize("change", ["metadata", "step", "state_dict"])
def test_checkpoint_full_metadata_and_strict_state_are_required(cli, study, change):
    selected = cli.validate_projector_selection(study.args.projector_run, study.protocol,
                                                study.dev_ids)
    checkpoint = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    if change == "metadata":
        checkpoint["run_metadata"]["seed"] = 4
    elif change == "step":
        checkpoint["global_step"] = 10400
    else:
        checkpoint["state_dict"]["unexpected.weight"] = torch.ones(1)
    torch.save(checkpoint, selected["checkpoint"])
    selected["checkpoint_hash"] = file_hash(selected["checkpoint"])
    with pytest.raises((ValueError, RuntimeError)):
        cli._load_projector_model(selected)


def test_preflight_is_read_only_and_verifies_current_cache(cli, study):
    result = cli.evaluate(study.args)
    assert result["status"] == "ready" and result["test_clips"] == 768
    assert not study.args.registry.parent.exists() and not study.args.output.parent.exists()
    assert not study.builder_calls and not study.view.reads
    atomic_json(study.args.adapted_root / "complete/frame_rows.json", {"changed": True})
    with pytest.raises(ValueError, match="lineage|cache"):
        cli.evaluate(study.args)
    assert not study.args.registry.parent.exists()


@pytest.mark.parametrize("change", ["count", "order", "signer", "decoder", "output_source",
                                   "registry_source", "cap", "output_symlink"])
def test_invalid_inputs_never_claim_registry(cli, study, change):
    args = study.args
    if change == "count":
        study.protocol["split"]["groups"]["test"]["clip_ids"].pop()
    elif change == "order":
        study.view.records.reverse()
    elif change == "signer":
        study.view.records[0]["signer"] = "Signer03"
    elif change == "decoder":
        study.protocol["decoder"] = {**DECODER, "beam_size": 1}
    elif change == "output_source":
        args.output = args.motion_root / "heldout.json"
    elif change == "registry_source":
        args.registry = args.projector_run / "registry.json"
    elif change == "cap":
        args.gpu_cap_seconds = 1801
    elif change == "output_symlink":
        args.output.parent.mkdir()
        args.output.symlink_to(args.protocol)
    args.mode = "run"
    with pytest.raises(ValueError):
        cli.evaluate(args)
    assert not args.registry.exists()
    assert not study.calls


def test_completed_checkpoint_reuses_report_without_generation_or_metric_recompute(cli, study,
                                                                                 monkeypatch):
    study.args.mode = "run"
    result = cli.evaluate(study.args)
    assert len(study.calls) == 192 and set(study.calls) == {4}
    assert len(study.view.reads) == 768
    assert result["human_review_status"] == "not_assessed"
    assert result["supervision_policy"] == POLICY
    assert result["seed"] == 0
    assert [item["clip_id"] for item in result["items"]] == list(study.ids)
    assert result["items"][0]["reference"] == "secret heldout reference"
    before = study.args.output.read_bytes()
    monkeypatch.setattr(cli, "_load_projector_model", lambda *args: pytest.fail("regenerated"))
    monkeypatch.setattr(cli, "evaluate_translations", lambda *args: pytest.fail("rescored"))
    assert cli.evaluate(study.args) == result
    assert study.args.output.read_bytes() == before
    claim = read_json(study.args.registry)["evaluations"][result["checkpoint_hash"]]
    assert claim["status"] == "complete"
    assert claim["report_hash"] == file_hash(study.args.output)


def test_partial_journal_resumes_only_uncommitted_batches(cli, study):
    study.args.mode = "run"
    study.failure["batch"] = 2
    with pytest.raises(TimeoutError):
        cli.evaluate(study.args)
    assert study.view.reads[:8] == list(range(8))
    claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
    assert claim["status"] == "partial" and claim["stage"] == "generation"
    journal = read_json(Path(claim["journal_path"]))
    assert [item["clip_id"] for item in journal["items"]] == list(study.ids[:8])
    assert not study.args.output.exists()
    study.view.reads.clear()
    study.failure["batch"] = None
    result = cli.evaluate(study.args)
    assert study.view.reads == list(range(8, 768))
    assert len(study.calls) == 192
    assert result["status"] == "complete"


@pytest.mark.parametrize("complete", [False, True])
@pytest.mark.parametrize("drift", ["split", "decoder", "output", "source", "config", "journal"])
def test_claim_and_journal_refuse_provenance_or_output_drift(cli, study, complete, drift):
    study.args.mode = "run"
    if not complete:
        study.failure["batch"] = 1
        with pytest.raises(TimeoutError):
            cli.evaluate(study.args)
    else:
        cli.evaluate(study.args)
    count = len(study.calls)
    if drift == "split":
        study.protocol["split_hash"] = "changed"
    elif drift == "decoder":
        study.protocol["decoder"] = {**DECODER, "beam_size": 1}
    elif drift == "output":
        study.args.output = study.args.output.with_name("another.json")
    elif drift == "source":
        atomic_json(study.args.text_manifest, {"changed": True})
    elif drift == "config":
        data = read_json(study.args.projector_run / "budget.json")
        data["identity"]["config"]["model"]["cache_dir"] += "-changed"
        atomic_json(study.args.projector_run / "budget.json", data)
    else:
        claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
        path = Path(claim["journal_path"])
        journal = read_json(path)
        journal["binding"]["split_hash"] = "changed"
        atomic_json(path, journal)
    with pytest.raises(ValueError):
        cli.evaluate(study.args)
    assert len(study.calls) == count


def test_completed_report_tampering_refused_without_regeneration(cli, study):
    study.args.mode = "run"
    cli.evaluate(study.args)
    report = read_json(study.args.output)
    report["items"][0]["prediction"] = "tampered"
    atomic_json(study.args.output, report)
    with pytest.raises(ValueError, match="report|hash"):
        cli.evaluate(study.args)
    assert len(study.calls) == 192


def test_final_report_publication_never_clobbers_existing_file(cli, tmp_path):
    output = tmp_path / "result.json"
    output.write_text("existing")
    with pytest.raises((ValueError, FileExistsError)):
        cli._publish_report(output, {"new": True})
    assert output.read_text() == "existing"


def test_legacy_seed_zero_defaults_do_not_mutate_checkpoint_identity(cli, study):
    study.identity.update(projector="residual", variant="E3_projector_residual")
    for key in ("seed", "residual_seed", "residual_hidden_dim"):
        del study.identity[key]
    del study.identity["config"]["seed"]
    _write_run(study)
    before = copy.deepcopy(study.identity)
    selected = cli.validate_projector_selection(study.args.projector_run, study.protocol,
                                                study.dev_ids)
    cli._load_projector_model(selected)
    assert selected["identity"] == before
    study.args.mode = "run"
    report = cli.evaluate(study.args)
    assert report["seed"] == 0 and report["residual_seed"] == 1
    assert report["residual_hidden_dim"] == 1024


@pytest.mark.parametrize("change", ["decoder_bool", "seed_bool"])
def test_bool_integer_aliases_cannot_change_decoder_or_seed_identity(cli, study, change):
    if change == "decoder_bool":
        report = read_json(study.args.projector_run / "dev-5500.json")
        report["decoding"]["in_context"] = 0
        atomic_json(study.args.projector_run / "dev-5500.json", report)
    else:
        study.identity["seed"] = False
        study.identity["config"]["seed"] = False
        _write_run(study)
    with pytest.raises(ValueError):
        cli.validate_projector_selection(study.args.projector_run, study.protocol, study.dev_ids)


@pytest.mark.parametrize("stage", ["before_link", "after_link"])
def test_publication_failure_reuses_durable_metrics_and_predictions(cli, study, monkeypatch, stage):
    study.args.mode = "run"
    publish = cli._publish_report

    def interrupted(path, value):
        if stage == "after_link":
            publish(path, value)
        raise OSError("publication interrupted")

    monkeypatch.setattr(cli, "_publish_report", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        cli.evaluate(study.args)
    assert len(study.calls) == 192
    claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
    assert claim["status"] == "partial" and claim["stage"] == "publication"
    monkeypatch.setattr(cli, "_publish_report", publish)
    monkeypatch.setattr(cli, "_load_projector_model", lambda *args: pytest.fail("regenerated"))
    monkeypatch.setattr(cli, "evaluate_translations", lambda *args: pytest.fail("rescored"))
    assert cli.evaluate(study.args)["status"] == "complete"
    completed = next(iter(read_json(study.args.registry)["evaluations"].values()))
    assert completed["status"] == "complete"


def test_signal_cap_publishes_distinct_stage_and_cannot_reset_budget(cli, study, monkeypatch):
    study.args.mode = "run"
    study.args.gpu_cap_seconds = 1
    monkeypatch.setattr(cli, "_load_projector_model", lambda *args: time.sleep(3))
    with pytest.raises(cli.StageCapTimeout, match="model_loading"):
        cli.evaluate(study.args)
    claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
    assert claim["status"] == "partial" and claim["timeout_kind"] == "stage_cap"
    assert claim["gpu_seconds"] >= 1
    budget = read_json(Path(claim["journal_path"] + ".budget.json"))
    assert budget["status"] == "closed" and budget["gpu_cap_seconds"] == 1
    assert not study.calls and not study.args.output.exists()
    with pytest.raises(cli.StageCapTimeout):
        cli.evaluate(study.args)
    study.args.gpu_cap_seconds = 1800
    with pytest.raises(ValueError, match="drift"):
        cli.evaluate(study.args)


def test_unclaimed_output_never_adopted_or_overwritten(cli, study):
    atomic_json(study.args.output, {"status": "complete"})
    before = study.args.output.read_bytes()
    study.args.mode = "run"
    with pytest.raises(ValueError, match="without matching claim"):
        cli.evaluate(study.args)
    assert study.args.output.read_bytes() == before
    assert not study.args.registry.exists() and not study.calls


def test_partial_journal_corruption_refused_even_when_item_digest_updated(cli, study):
    study.args.mode = "run"
    study.failure["batch"] = 1
    with pytest.raises(TimeoutError):
        cli.evaluate(study.args)
    claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
    path = Path(claim["journal_path"])
    journal = read_json(path)
    journal["items"].reverse()
    journal["items_hash"] = digest(journal["items"])
    atomic_json(path, journal)
    with pytest.raises(ValueError, match="reordered"):
        cli.evaluate(study.args)
    assert len(study.calls) == 1


def test_registry_lock_remains_exclusive_through_generation_and_publication(cli, study,
                                                                          monkeypatch):
    study.args.mode = "run"
    generate, publish = TinyModel.generate_text, cli._publish_report
    checked = set()

    def locked(stage):
        with Path(str(study.args.registry) + ".lock").open("a+") as competing:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competing.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        checked.add(stage)

    def generate_under_lock(self, *args):
        if not self.calls:
            locked("generation")
        return generate(self, *args)

    def publish_under_lock(*args):
        locked("publication")
        return publish(*args)

    monkeypatch.setattr(TinyModel, "generate_text", generate_under_lock)
    monkeypatch.setattr(cli, "_publish_report", publish_under_lock)
    assert cli.evaluate(study.args)["status"] == "complete"
    assert checked == {"generation", "publication"}


def test_immediate_timer_expiry_restores_previous_signal_handler(cli, monkeypatch):
    original = signal.getsignal(signal.SIGALRM)
    clock = cli._Clock({"status": "partial", "gpu_seconds": 0, "stage": "claimed"}, 1800)

    def expire_at_install(which, seconds, interval=0):
        if seconds > 0:
            raise cli.StageCapTimeout("timer install expired")

    monkeypatch.setattr(cli.signal, "setitimer", expire_at_install)
    try:
        with pytest.raises(cli.StageCapTimeout, match="install"):
            with cli._stage_timer(clock):
                pytest.fail("expired timer entered scoring")
        assert signal.getsignal(signal.SIGALRM) == original
    finally:
        signal.signal(signal.SIGALRM, original)

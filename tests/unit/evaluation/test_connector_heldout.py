"""CPU-only contracts for connector admission and the isolated once-only engine."""

import argparse
import copy
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import PhoenixSample
from despamo.models.connector_adapter import MaskedQueryConnector


class TinyModel(nn.Module):
    def __init__(self, arm, calls, failure):
        super().__init__()
        self.visual_adapter = TinyAdapter(arm)
        self.prompt_template = "Translate into {}."
        self.language_model = SimpleNamespace(generate_text=self.generate_text)
        self.calls, self.failure = calls, failure

    def generate_text(self, visual, valid, prompts, mode, beam_size, max_length):
        assert not self.training and torch.is_inference_mode_enabled()
        assert prompts == ["Translate into German."] * 4
        assert (mode, beam_size, max_length) == ("deterministic", 5, 64)
        if self.failure[0] == len(self.calls):
            raise TimeoutError("interrupted batch")
        self.calls.append(4)
        return ["am montag regen"] * 4


class TinyAdapter(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.spatial_projector = (nn.Sequential(nn.Linear(2, 3), nn.GELU(), nn.Linear(3, 2))
                                  if arm == "plain_mlp" else nn.Linear(2, 2))
        self.motion_projector = nn.Linear(2, 2)
        self.multimodal_projector = (MaskedQueryConnector(2, 2, 2, 1, 1, 4, 0.0)
                                     if arm == "qformer" else nn.Linear(2, 2))

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        memory = self.spatial_projector(spatial) + self.motion_projector(motion)
        if isinstance(self.multimodal_projector, MaskedQueryConnector):
            return self.multimodal_projector(memory, spatial_mask)
        return self.multimodal_projector(memory), spatial_mask


class TinyView:
    def __init__(self, ids):
        self.records = [{"fileid": clip, "signer": "Signer07"} for clip in ids]
        self.reads = []

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self.reads.append(index)
        return PhoenixSample(
            clip_id=self.records[index]["fileid"], signer="Signer07", text="SECRET TEXT",
            gloss="SECRET GLOSS", en_text="SECRET EN", es_text="SECRET ES", fr_text="SECRET FR",
            spatial=torch.ones(1, 2), motion=torch.ones(1, 2),
        )


@pytest.fixture
def study(tmp_path, monkeypatch):
    path = Path(__file__).resolve().parents[3] / "scripts/evaluate_connector_study.py"
    assert path.is_file(), "connector held-out adapter missing"
    cli = importlib.import_module("scripts.evaluate_connector_study")
    training = importlib.import_module("scripts.train_connector_study")
    engine = cli._new_engine()
    args = argparse.Namespace(
        mode="preflight", protocol=tmp_path / "protocol.json", dataset=tmp_path / "dataset",
        text_manifest=tmp_path / "text.json", dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion", motion_manifest=tmp_path / "motion.json",
        annotation=tmp_path / "annotation/train.npy", adapted_root=tmp_path / "adapted",
        connector_run=tmp_path / "runs/run", output=tmp_path / "reports/test.json",
        registry=tmp_path / "registry/claims.json", gpu_cap_seconds=1800,
    )
    ids, dev_ids = tuple(f"t{i}" for i in range(768)), ("d0", "d1")
    protocol = {
        "protocol_hash": engine.PROTOCOL_HASH, "split_hash": "split", "decoder": engine.DECODER,
        "supervision_policy": engine.POLICY, "human_review_status": "not_assessed",
        "split": {"groups": {"dev": {"clip_ids": list(dev_ids)},
                              "test": {"clip_ids": list(ids), "signers": ["Signer07"]}}},
    }
    adapted = {"checkpoint_hash": "adapted"}
    files = {
        args.protocol: protocol, args.text_manifest: {}, args.motion_manifest: {},
        args.annotation: {}, args.dataset / "source.json": {}, args.dataset / "version.json": {},
        args.dino_root / "complete/manifest.json": {},
        args.dino_root / "complete/frame_rows.json": {},
        args.adapted_root / "complete/identity.json": adapted,
        args.adapted_root / "complete/manifest.json": {},
        args.adapted_root / "complete/frame_rows.json": {},
    }
    for target, value in files.items():
        atomic_json(target, value)
    identity = {
        "run_policy": training.RUN_POLICY, "variant": "E3_connector_reference",
        "arm": "reference", "seed": 0, "module_seed": 100, "steps": 16000,
        "architecture": training.arm_configs()["reference"],
        "recipe": copy.deepcopy(training.RECIPE),
        "dev_checkpoints": list(training.DEV_STEPS), "gpu_cap_seconds": 10800,
        "schema_hash": file_hash(cli.ROOT / "src/despamo/appearance/schema.py"),
        "architecture_schema_hash": digest(training.arm_configs()),
        "code_hashes": training.connector_code_hashes(cli.ROOT),
        "initial_base_hash": "a" * 64, "initialization_profile_hash": "b" * 64,
        "protocol_hash": engine.PROTOCOL_HASH, "split_hash": "split",
        "adaptation_checkpoint_hash": "adapted", "adapted_encoder_key": digest(adapted),
        "adapted_manifest_hash": file_hash(args.adapted_root / "complete/manifest.json"),
        "adapted_frame_rows_hash": file_hash(args.adapted_root / "complete/frame_rows.json"),
        "supervision_policy": engine.POLICY, "human_review_status": "not_assessed",
        "config": {
            "seed": 0, "data": {"num_workers": 0, "batch_size": 4,
                                  "spatial_root": str(args.adapted_root),
                                  "spatial_manifest": str(
                                      args.adapted_root / "complete/manifest.json"),
                                  "motion_root": str(args.motion_root),
                                  "motion_manifest": str(args.motion_manifest),
                                  "annotation_root": str(args.annotation.parent)},
            "model": {"warm_up_steps": 2000, "use_in_context": False, "num_in_context": 0},
            "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1},
        },
    }
    # Completed training predates this newly added evaluator; historical inventory
    # must remain admissible without pretending it originally included the adapter.
    identity["code_hashes"].pop("scripts/evaluate_connector_study.py")
    calls, built, failure = [], [], [None]

    def builder(config, arm, seed, initial_base_hash):
        built.append((arm, seed, config.seed, initial_base_hash))
        return TinyModel(arm, calls, failure)

    monkeypatch.setattr(training, "build_connector_model", builder)
    view = TinyView(ids)
    monkeypatch.setattr(engine, "load_e3_views", lambda *args: (protocol, {"test": view}))
    monkeypatch.setattr(engine, "_runtime_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(cli, "_new_engine", lambda: engine)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    state = SimpleNamespace(cli=cli, training=training, engine=engine, args=args,
                            identity=identity, protocol=protocol, dev_ids=dev_ids, ids=ids,
                            calls=calls, built=built, failure=failure, view=view)
    write_run(state)
    return state


def write_run(study, arm="reference", seed=0):
    identity, training = study.identity, study.training
    identity.update(arm=arm, seed=seed, module_seed=100 + seed,
                    variant=f"E3_connector_{arm}", architecture=training.arm_configs()[arm])
    identity["config"]["seed"] = seed
    identity["run_key"] = training._run_key(identity)
    run = study.args.connector_run.parent / identity["run_key"]
    study.args.connector_run = run
    atomic_json(run / "budget.json", {"identity": identity, "status": "closed",
                                      "optimizer_step": 16000, "gpu_seconds": 10,
                                      "ceiling_seconds": 10800})
    atomic_json(run / "run-status.json", {"status": "complete", "global_step": 16000,
                                          "run_key": run.name})
    best_step = 10400 if (arm, seed) == ("qformer", 1) else 16000
    for step in training.DEV_STEPS:
        path = run / f"step-{step}.ckpt"
        torch.save({"run_metadata": copy.deepcopy(identity),
                    "pilot_identity": copy.deepcopy(identity), "global_step": step,
                    "state_dict": TinyModel(arm, [], [None]).state_dict()}, path)
        report = {"checkpoint_step": step, "checkpoint_hash": file_hash(path), "split": "dev",
                  "split_hash": "split", "decoding": training.DECODER,
                  "metrics": {"bleu4": 3.0 if step == best_step else 1.0},
                  "items": [{"clip_id": clip, "prediction": "hyp", "reference": "ref"}
                            for clip in study.dev_ids], **training.report_binding(identity)}
        report["content_hash"] = digest(report)
        atomic_json(run / f"dev-{step}.json", report)
    best = read_json(run / f"dev-{best_step}.json")
    atomic_json(run / "selected-checkpoint.json", {
        "step": best_step, "checkpoint_hash": best["checkpoint_hash"], "metrics": best["metrics"],
        "dev_report_hashes": training._dev_report_hashes(run), **training.report_binding(identity),
    })


@pytest.mark.parametrize("arm", ["reference", "plain_mlp", "qformer"])
@pytest.mark.parametrize("seed", [0, 1])
def test_exact_model_and_dev_selection(study, arm, seed):
    write_run(study, arm, seed)
    selected = study.cli.validate_connector_selection(
        study.args.connector_run, study.protocol, study.dev_ids,
    )
    assert selected["step"] == (10400 if (arm, seed) == ("qformer", 1) else 16000)
    model = study.cli._load_connector_model(selected)
    assert study.built == [(arm, seed, seed, "a" * 64)]
    saved = torch.load(selected["checkpoint"], map_location="cpu", weights_only=False)
    assert set(model.state_dict()) == set(saved["state_dict"])
    for key, value in model.state_dict().items():
        assert torch.equal(value, saved["state_dict"][key])
    result = study.cli.evaluate(study.args)
    assert result["training_horizon"] == 16000 and result["checkpoint_step"] == selected["step"]
    assert not study.view.reads and not study.args.registry.exists()


@pytest.mark.parametrize("change", ["policy", "schema", "source", "architecture", "module_seed",
                                    "seed", "recipe", "dev_bytes", "checkpoint_metadata"])
def test_admission_refuses_drift_before_claim(study, change):
    run = study.args.connector_run
    if change == "dev_bytes":
        with (run / "dev-5500.json").open("a") as output:
            output.write(" ")
    elif change == "checkpoint_metadata":
        path = run / "step-5500.ckpt"
        checkpoint = torch.load(path, weights_only=False)
        checkpoint["pilot_identity"]["seed"] = 7
        torch.save(checkpoint, path)
    else:
        changes = {"policy": ("run_policy", "other"), "schema": ("schema_hash", "wrong"),
                   "architecture": ("architecture", {}), "module_seed": ("module_seed", 1),
                   "seed": ("seed", False), "recipe": ("recipe", {})}
        if change == "source":
            study.identity["code_hashes"]["src/despamo/models/visual_adapter.py"] = "changed"
        else:
            key, value = changes[change]
            study.identity[key] = value
        atomic_json(run / "budget.json", {"identity": study.identity})
    study.args.mode = "run"
    with pytest.raises(ValueError):
        study.cli.evaluate(study.args)
    assert not study.args.registry.exists() and not study.built and not study.calls


def test_isolated_globals_and_old_claims_completed_reuse(study, monkeypatch):
    original = importlib.import_module("scripts.evaluate_projector_pilot")
    before = dict(vars(original))
    old = {f"old-{i}": {"status": "complete", "old_binding": i} for i in range(4)}
    atomic_json(study.args.registry, {"evaluation_policy": original.EVALUATION_POLICY,
                                     "evaluations": old})
    study.args.mode = "run"
    report = study.cli.evaluate(study.args)
    assert report["connector_policy"] == "three-arm-connector-heldout-v1"
    assert isinstance(original.__file__, str)
    assert report["evaluator_hash"] == file_hash(Path(original.__file__))
    assert report["connector_adapter_hash"] == file_hash(Path(study.cli.__file__))
    assert report["connector_run"] == str(study.args.connector_run.resolve())
    assert report["items"][0]["reference"] == "secret text"
    assert len(study.calls) == 192
    claims = read_json(study.args.registry)["evaluations"]
    assert len(claims) == 5 and all(claims[key] == value for key, value in old.items())
    raw = study.args.output.read_bytes()
    monkeypatch.setattr(study.engine, "_load_projector_model", lambda *a: pytest.fail("load"))
    monkeypatch.setattr(study.engine, "evaluate_translations", lambda *a: pytest.fail("score"))
    assert study.cli.evaluate(study.args) == report
    assert study.args.output.read_bytes() == raw
    study.args.output = study.args.output.with_name("other.json")
    with pytest.raises(ValueError, match="drift"):
        study.cli.evaluate(study.args)
    assert all(vars(original)[key] is value for key, value in before.items())


@pytest.mark.parametrize("drift", ["adapter", "policy"])
def test_partial_resume_and_adapter_dependency_drift(study, monkeypatch, drift):
    study.args.mode = "run"
    study.failure[0] = 2
    with pytest.raises(TimeoutError):
        study.cli.evaluate(study.args)
    claim = next(iter(read_json(study.args.registry)["evaluations"].values()))
    assert claim["status"] == "partial" and claim["committed_items"] == 8
    original = study.cli.inference_code_hashes
    with monkeypatch.context() as patch:
        if drift == "adapter":
            patch.setattr(study.cli, "inference_code_hashes",
                          lambda: {**original(), "scripts/evaluate_connector_study.py": "changed"})
        else:
            patch.setattr(study.cli, "CONNECTOR_POLICY", "changed")
        with pytest.raises(ValueError, match="drift"):
            study.cli.evaluate(study.args)
    assert len(study.calls) == 2
    study.failure[0] = None
    study.view.reads.clear()
    assert study.cli.evaluate(study.args)["status"] == "complete"
    assert study.view.reads == list(range(8, 768)) and len(study.calls) == 192


def test_inference_closure_includes_new_adapter_and_all_execution_dependencies(study):
    hashes = study.cli.inference_code_hashes()
    expected = {
        "scripts/evaluate_connector_study.py", "scripts/evaluate_projector_pilot.py",
        "scripts/evaluate_e3_pilot.py", "scripts/train_connector_study.py",
        "scripts/train_projector_pilot.py", "scripts/run_connector_study.py", "uv.lock",
        "src/despamo/models/connector_adapter.py", "src/despamo/models/flan_t5.py",
        "src/despamo/training/pilot_factory.py", "src/despamo/evaluation/signer_pilot.py",
        "src/despamo/evaluation/metrics.py",
    }
    assert expected.issubset(hashes)
    for path in expected:
        assert hashes[path] == file_hash(study.cli.ROOT / path)
    assert "scripts/evaluate_connector_study.py" not in study.identity["code_hashes"]
    assert study.cli._recipe(study.identity)["inference_dependencies_digest"] == digest(hashes)


@pytest.mark.parametrize("change", ["bytes", "metadata", "pilot_identity", "step", "state_dict"])
def test_loader_checks_bytes_full_metadata_and_strict_keys(study, change):
    selected = study.cli.validate_connector_selection(
        study.args.connector_run, study.protocol, study.dev_ids,
    )
    path = selected["checkpoint"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if change in {"metadata", "pilot_identity"}:
        checkpoint["run_metadata" if change == "metadata" else change]["seed"] = 8
    elif change == "step":
        checkpoint["global_step"] = 10400
    else:
        checkpoint["state_dict"]["unexpected"] = torch.ones(1)
    torch.save(checkpoint, path)
    if change != "bytes":
        selected["checkpoint_hash"] = file_hash(path)
    with pytest.raises((ValueError, RuntimeError)):
        study.cli._load_connector_model(selected)


def test_fresh_engine_instances_do_not_share_adapters():
    cli = importlib.import_module("scripts.evaluate_connector_study")
    legacy = importlib.import_module("scripts.evaluate_projector_pilot")
    before = dict(vars(legacy))
    first, second = cli._new_engine(), cli._new_engine()
    assert first is not second and first is not legacy
    assert first._generate is not second._generate
    assert first._generate.__globals__ is vars(first)
    assert first._recipe is cli._recipe
    first._recipe = lambda identity: {}
    assert second._recipe is cli._recipe
    assert all(vars(legacy)[key] is value for key, value in before.items())


@pytest.mark.parametrize("dependency", ["src/despamo/appearance/schema.py",
                                        "src/despamo/models/connector_adapter.py",
                                        "scripts/train_projector_pilot.py", "uv.lock"])
def test_current_bound_source_change_refused_without_identity_changes(study, monkeypatch,
                                                                     dependency):
    original = study.cli.inference_code_hashes
    monkeypatch.setattr(study.cli, "inference_code_hashes",
                        lambda: {**original(), dependency: "changed"})
    study.args.mode = "run"
    with pytest.raises(ValueError, match="source/schema"):
        study.cli.evaluate(study.args)
    assert not study.args.registry.exists() and not study.built


@pytest.mark.parametrize("change", ["cap", "icl", "signer", "order", "decoder", "cache"])
def test_frozen_inference_inputs_refused(study, change):
    if change == "cap":
        study.args.gpu_cap_seconds = 1801
    elif change == "icl":
        study.identity["config"]["model"]["use_in_context"] = True
        write_run(study)
    elif change == "signer":
        study.view.records[0]["signer"] = "Signer03"
    elif change == "order":
        study.view.records.reverse()
    elif change == "decoder":
        study.protocol["decoder"] = {**study.protocol["decoder"], "beam_size": 1}
    else:
        atomic_json(study.args.adapted_root / "complete/frame_rows.json", {"changed": True})
    study.args.mode = "run"
    with pytest.raises(ValueError):
        study.cli.evaluate(study.args)
    assert not study.args.registry.exists() and not study.built

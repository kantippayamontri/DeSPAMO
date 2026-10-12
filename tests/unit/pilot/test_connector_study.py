"""CPU-only contracts for the independent connector runtime."""

import copy
import importlib
import json
import random
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.training.pilot_replay import capture_rng


@pytest.fixture
def runtime():
    return importlib.import_module("scripts.train_connector_study")


@pytest.fixture
def source_hasher(runtime):
    return runtime.connector_code_hashes


@pytest.fixture
def inputs(tmp_path, monkeypatch, runtime):
    args = SimpleNamespace(
        mode="initialize", arm="reference", seed=0, steps=16000, gpu_cap_seconds=10800,
        initialization_profile=tmp_path / "profiles/common.json", resume_checkpoint=None,
        authorize_connector_run=True, output_base=tmp_path / "runs",
    )
    args.output_base.mkdir()
    config = {
        "seed": 0, "model": {"warm_up_steps": 2000, "use_in_context": False},
        "data": {"batch_size": 4, "num_workers": 0},
        "trainer": {"max_steps": 16000, "accumulate_grad_batches": 1,
                    "precision": "bf16", "gradient_clip_val": 1.0},
        "optimizer": {"learning_rate": 6e-4, "weight_decay": 0.01},
    }
    protocol = {"protocol_hash": "protocol", "split_hash": "split", "decoder": runtime.DECODER,
                "split": {"groups": {"dev": {"clip_ids": ["d1", "d2"]}}}}
    adapted = {"clips": 7096, "checkpoint_hash": "adaptation", "encoder_key": "cache",
               "manifest_hash": "manifest", "frame_rows_hash": "rows"}
    code = {"src/despamo/appearance/schema.py": "a" * 64,
            "scripts/train_connector_study.py": "b" * 64}
    architectures = {name: {"name": name} for name in runtime.ARMS}
    monkeypatch.setattr(runtime, "arm_configs", lambda: copy.deepcopy(architectures))
    monkeypatch.setattr(runtime, "connector_code_hashes", lambda project: dict(code))

    def prepare(namespace):
        resolved = copy.deepcopy(config)
        resolved["seed"] = namespace.seed
        return protocol, {"train": list(range(8)), "dev": [1, 2], "test": [99]}, \
            OmegaConf.create(resolved), adapted, "baseline", dict(code)

    monkeypatch.setattr(runtime.legacy, "_prepare_projector_inputs", prepare)
    monkeypatch.setattr(runtime, "build_seeded_projector_model",
                        lambda config, steps, seed: TinyModel(seed))
    monkeypatch.setattr(runtime, "E1_SHARED_HASH", runtime.shared_tensor_hash(TinyModel(0)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return args, protocol, adapted, code


class TinyAdapter(torch.nn.Module):
    def __init__(self, query):
        super().__init__()
        self.spatial_projector = torch.nn.Linear(4, 4)
        self.motion_projector = torch.nn.Linear(4, 4)
        self.multimodal_projector = TinyQuery() if query else torch.nn.Linear(4, 4)


class TinyModel(torch.nn.Module):
    counter: torch.Tensor

    def __init__(self, seed=0, query=False, broken=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.visual_adapter = TinyAdapter(query)
            self.language_model = torch.nn.Linear(4, 2)
        self.register_buffer("counter", torch.zeros(1))
        self.rng = random.Random(seed)
        self.vt_weight = 1.0
        self.broken = broken
        self.weight_decay = 0.01
        self._trainer = None

    def configure_optimizers(self):
        from despamo.training.pilot_factory import _pilot_optimizer

        return _pilot_optimizer(self)  # pyright: ignore[reportArgumentType]

    def compute_losses(self, batch):
        self.counter.add_(1)
        random.random()
        np.random.rand()
        self.rng.random()
        spatial = self.visual_adapter.spatial_projector(batch)
        motion = self.visual_adapter.motion_projector(batch)
        connector = self.visual_adapter.multimodal_projector(spatial + motion)
        if self.broken:
            connector = connector.detach()
        loss = self.language_model(torch.nn.functional.dropout(connector, 0.1)).square().mean()
        return {"loss": loss, "contra_loss": loss * 0.25}


class TinyQuery(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.queries = torch.nn.Parameter(torch.randn(2, 4))
        self.attention = torch.nn.MultiheadAttention(4, 2, batch_first=True)

    def forward(self, memory):
        memory = torch.stack((memory, memory * 2, memory * 3), dim=1)
        valid = torch.tensor([True, True, False]).expand(memory.shape[0], -1)
        queries = self.queries[None].expand(memory.shape[0], -1, -1)
        output, _ = self.attention(queries, memory, memory, key_padding_mask=~valid)
        return output.mean(1)


def test_common_profile_binds_both_seeds_and_all_arms(inputs, runtime):
    args, _, _, _ = inputs
    result = runtime.initialize_connector(args)
    profile = read_json(args.initialization_profile)
    assert result["status"] == "initialized"
    assert profile["profile_policy"] == "connector-common-initialization-v1"
    assert set(profile["seeds"]) == {"0", "1"}
    assert profile["seeds"]["0"]["initial_base_hash"] == runtime.E1_SHARED_HASH
    assert profile["seeds"]["1"]["initial_base_hash"] != runtime.E1_SHARED_HASH
    before = args.initialization_profile.read_bytes()
    identities = []
    for seed in (0, 1):
        args.seed = seed
        paired = []
        for arm in runtime.ARMS:
            args.arm = arm
            identity, output, _, _, config = runtime.prepare_connector(args)
            assert output.name.startswith(f"connector-{arm}-seed{seed}-16000-")
            assert identity["run_policy"] == "signer-pilot-three-arm-connectors-v1"
            assert identity["seed"] == config.seed == seed
            assert config.optimizer.learning_rate == 6e-4
            assert identity["config"]["optimizer"]["learning_rate"] == 6e-4
            assert identity["recipe"]["adapter_lr"] == 3e-4
            assert identity["recipe"]["flan_lr"] == 5e-5
            assert identity["module_seed"] == 100 + seed
            assert "residual_hidden_dim" not in identity
            assert identity["dev_checkpoints"] == [5500, 10400, 16000]
            paired.append(identity)
        for key in ("initial_base_hash", "initialization_profile_hash", "input_identity",
                    "config", "code_hashes", "gpu_cap_seconds", "recipe"):
            assert paired[0][key] == paired[1][key] == paired[2][key]
        identities.extend(paired)
    assert len({identity["run_key"] for identity in identities}) == 6
    assert args.initialization_profile.read_bytes() == before
    assert list(args.output_base.iterdir()) == []


@pytest.mark.parametrize("change", ["code", "cap", "config", "input", "schema", "hash"])
def test_preflight_refuses_profile_drift_without_writes(inputs, runtime, monkeypatch, change):
    args, protocol, _, code = inputs
    runtime.initialize_connector(args)
    before = args.initialization_profile.read_bytes()
    if change == "code":
        code["src/other.py"] = "c" * 64
    elif change == "cap":
        args.gpu_cap_seconds = 10000
    elif change == "config":
        prepare = runtime.legacy._prepare_projector_inputs

        def changed(namespace):
            values = prepare(namespace)
            values[2].optimizer.weight_decay = 0.02
            return values

        monkeypatch.setattr(runtime.legacy, "_prepare_projector_inputs", changed)
    elif change == "input":
        protocol["extra"] = "changed"
    else:
        profile = read_json(args.initialization_profile)
        if change == "schema":
            profile["seeds"]["1"]["schema_hash"] = "wrong"
        else:
            profile["seeds"]["0"]["initial_base_hash"] = "0" * 64
        profile["profile_hash"] = digest({k: v for k, v in profile.items() if k != "profile_hash"})
        atomic_json(args.initialization_profile, profile)
        before = args.initialization_profile.read_bytes()
    with pytest.raises(ValueError):
        runtime.prepare_connector(args)
    assert args.initialization_profile.read_bytes() == before
    assert list(args.output_base.iterdir()) == []


def test_initialize_reference_only_and_seed0_actual_hash(inputs, runtime, monkeypatch):
    args, _, _, _ = inputs
    args.arm = "qformer"
    with pytest.raises(ValueError, match="reference"):
        runtime.initialize_connector(args)
    args.arm = "reference"
    monkeypatch.setattr(runtime, "E1_SHARED_HASH", "0" * 64)
    with pytest.raises(ValueError, match="seed0"):
        runtime.initialize_connector(args)
    assert not args.initialization_profile.exists()


def test_full_source_closure_detects_added_source(tmp_path, runtime):
    source = tmp_path / "src/despamo"
    source.mkdir(parents=True)
    (source / "model.py").write_text("original")
    for name in runtime.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    first = runtime.connector_code_hashes(tmp_path)
    (source / "new_dependency.py").write_text("new")
    assert runtime.connector_code_hashes(tmp_path) != first
    assert "scripts/train_projector_pilot.py" in first
    assert "tools/dinov3/storage.py" in first
    assert "uv.lock" in first


@pytest.mark.parametrize("query", [False, True])
def test_joint_smoke_restores_rng_buffers_modes_and_clears_grads(runtime, query):
    model = TinyModel(query=query)
    model.eval()
    model.visual_adapter.motion_projector.train()
    batch = torch.randn(4, 4)
    rng = capture_rng()
    prompt = model.rng.getstate()
    state = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    result = runtime.connector_gradient_smoke(model, batch, "qformer" if query else "reference")
    assert result["status"] == "passed" and result["optimizer_updates"] == 0
    assert all(value > 0 for value in result["gradient_norms"].values())
    if query:
        assert result["gradient_norms"]["learned_queries"] > 0
        assert result["gradient_norms"]["attention"] > 0
    assert [module.training for module in model.modules()] == modes
    assert model.rng.getstate() == prompt
    assert all(parameter.grad is None for parameter in model.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)
    after = capture_rng()
    assert after["python"] == rng["python"]
    assert np.array_equal(after["numpy"][1], rng["numpy"][1])
    assert torch.equal(after["torch"], rng["torch"])


def test_failed_smoke_also_restores_state(runtime):
    model = TinyModel(query=True, broken=True)
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(RuntimeError, match="gradient"):
        runtime.connector_gradient_smoke(model, torch.randn(4, 4), "qformer")
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], rtol=0, atol=0)


def test_joint_smoke_restores_all_cuda_rng_without_cuda_work(runtime, monkeypatch):
    model = TinyModel(query=True)
    states = [torch.tensor([1, 2], dtype=torch.uint8), torch.tensor([3, 4], dtype=torch.uint8)]
    original = [state.clone() for state in states]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all",
                        lambda: [state.clone() for state in states])

    def restore_cuda(saved):
        states[:] = saved

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", restore_cuda)
    compute = model.compute_losses

    def consume(batch):
        states[0].add_(10)
        states[1].add_(20)
        return compute(batch)

    monkeypatch.setattr(model, "compute_losses", consume)
    runtime.connector_gradient_smoke(model, torch.randn(4, 4), "qformer")
    assert all(torch.equal(actual, saved) for actual, saved in zip(states, original, strict=True))


def test_owned_retention_prunes_only_after_durable_save(tmp_path, runtime):
    identity = {"run_policy": runtime.RUN_POLICY, "protocol_hash": "p", "gpu_cap_seconds": 10800}
    ledger = runtime.ConnectorLedger(tmp_path, identity, 0, 0, lambda: 0, lambda: 0)
    callback = runtime.ConnectorResumeSnapshots(tmp_path, ledger)
    for step in (500, 1000):
        (tmp_path / f"resume-{step}.ckpt").write_bytes(str(step).encode())
    (tmp_path / "step-5500.ckpt").write_bytes(b"dev")

    def fail(path):
        Path(path).write_bytes(b"partial")
        raise OSError("disk failed")

    trainer = SimpleNamespace(global_step=1500, save_checkpoint=fail)
    with pytest.raises(OSError):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert sorted(path.name for path in tmp_path.glob("resume-*.ckpt")) == [
        "resume-1000.ckpt", "resume-500.ckpt"]
    trainer.save_checkpoint = lambda path: Path(path).write_bytes(b"durable")
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert sorted(path.name for path in tmp_path.glob("resume-*.ckpt")) == [
        "resume-1000.ckpt", "resume-1500.ckpt"]
    assert (tmp_path / "step-5500.ckpt").read_bytes() == b"dev"
    identity["run_policy"] = "signer-pilot-projector-ablation-v1"
    with pytest.raises(ValueError, match="owned"):
        runtime.ConnectorResumeSnapshots(tmp_path, ledger)


def test_ledger_enforces_per_run_cap(tmp_path, runtime):
    clock = [0.0]
    ledger = runtime.ConnectorLedger(
        tmp_path, {"run_policy": runtime.RUN_POLICY, "gpu_cap_seconds": 10800},
        0, 0, lambda: clock[0], lambda: 0,
    )
    clock[0] = 10700
    with pytest.raises(RuntimeError, match="reserve"):
        ledger.require_remaining(300, step=0, phase="setup")
    clock[0] = 10801
    with pytest.raises(RuntimeError, match="cap"):
        ledger.charge(step=1, phase="train")
    assert read_json(tmp_path / "budget.json")["gpu_seconds"] == 10801


def test_completed_resume_refusal_does_not_write(inputs, runtime):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    atomic_json(output / "selected-checkpoint.json", {"complete": True})
    atomic_json(output / "budget.json", {"sentinel": True})
    args.resume_checkpoint = output / "resume-16000.ckpt"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="completed"):
        with runtime.owned_connector_ledger(args, identity, output):
            pytest.fail("completed run accepted")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_complete_status_alone_refuses_resume_without_writes(inputs, runtime):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    atomic_json(output / "run-status.json", {"status": "complete", "global_step": 16000})
    atomic_json(output / "budget.json", {"sentinel": True})
    args.resume_checkpoint = output / "resume-16000.ckpt"
    before = {path.name: path.read_bytes() for path in output.iterdir()}
    with pytest.raises(ValueError, match="completed"):
        with runtime.owned_connector_ledger(args, identity, output):
            pytest.fail("completed run accepted")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def write_completed(runtime, args, protocol):
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    for step, bleu in zip(runtime.DEV_STEPS, (1, 3, 2), strict=True):
        checkpoint = output / f"step-{step}.ckpt"
        torch.save({"run_metadata": identity, "pilot_identity": identity,
                    "global_step": step, "state_dict": {}}, checkpoint)
        report = {"checkpoint_step": step, "checkpoint_hash": file_hash(checkpoint),
                  "split": "dev", "split_hash": protocol["split_hash"],
                  "items": [{"clip_id": name, "prediction": "x", "reference": "x"}
                            for name in ("d1", "d2")],
                  "metrics": {"bleu4": bleu}, "decoding": runtime.DECODER,
                  **runtime.report_binding(identity)}
        report["content_hash"] = digest(report)
        atomic_json(output / f"dev-{step}.json", report)
    atomic_json(output / "budget.json", {"identity": identity, "status": "closed",
                "optimizer_step": 16000, "gpu_seconds": 100, "ceiling_seconds": 10800})
    atomic_json(output / "run-status.json", {"status": "complete", "global_step": 16000,
                "run_key": identity["run_key"]})
    report = read_json(output / "dev-10400.json")
    atomic_json(output / "selected-checkpoint.json", {
        "step": 10400, "checkpoint_hash": report["checkpoint_hash"],
        "metrics": report["metrics"], **runtime.report_binding(identity),
        "dev_report_hashes": {f"dev-{step}.json": file_hash(output / f"dev-{step}.json")
                              for step in runtime.DEV_STEPS},
    })
    return identity, output


@pytest.mark.parametrize("field", ["prediction", "reference"])
@pytest.mark.parametrize("step", [5500, 10400, 16000])
def test_selection_binds_all_dev_prediction_reference_bytes(inputs, runtime, field, step):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output = write_completed(runtime, args, protocol)
    path = output / f"dev-{step}.json"
    report = read_json(path)
    report["items"][0][field] = "changed bytes, same IDs and metrics"
    # Honest new self-binding must still conflict with the published selection's exact bytes.
    report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
    atomic_json(path, report)
    with pytest.raises(ValueError, match="report.*hash|hash.*report"):
        runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)


def test_resume_dev_callback_refuses_report_content_mutation(inputs, runtime):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output = write_completed(runtime, args, protocol)
    path = output / "dev-5500.json"
    report = read_json(path)
    report["items"][0]["prediction"] = "mutated"
    atomic_json(path, report)
    ledger = runtime.ConnectorLedger(output, identity, 0, 16000, lambda: 0, lambda: 0)
    callback = runtime.ConnectorDevCheckpoints([1, 2], ("d1", "d2"), protocol["split_hash"],
                output, 16000, 2000, 4, identity["variant"], ledger=ledger,
                checkpoint_schedule=runtime.DEV_STEPS)
    with pytest.raises(ValueError, match="content.*hash"):
        callback.on_train_start(SimpleNamespace(global_step=16000), TinyModel())


def test_selection_binds_exact_report_bytes_not_only_canonical_content(inputs, runtime):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output = write_completed(runtime, args, protocol)
    path = output / "dev-16000.json"
    before = read_json(path)
    path.write_text(json.dumps(before, separators=(",", ":")))
    assert read_json(path) == before
    with pytest.raises(ValueError, match="byte hashes"):
        runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)


@pytest.mark.parametrize("controller", ["run_connector_study.py", "run_projector_overnight.py"])
def test_controller_source_inventory_drift_invalidates_profile_before_model(
    source_hasher, inputs, runtime, tmp_path, monkeypatch, controller,
):
    args, _, _, _ = inputs
    project = tmp_path / "source-inventory"
    schema = project / "src/despamo/appearance/schema.py"
    schema.parent.mkdir(parents=True)
    schema.write_text("schema")
    required = set(runtime.SOURCE_FILES) | {
        "scripts/run_connector_study.py", "scripts/run_projector_overnight.py"}
    for name in required:
        path = project / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    hash_sources = source_hasher
    before = hash_sources(project)
    assert {"scripts/run_connector_study.py", "scripts/run_projector_overnight.py"} <= set(before)
    monkeypatch.setattr(runtime, "connector_code_hashes", lambda unused: hash_sources(project))
    runtime.initialize_connector(args)
    (project / "scripts" / controller).write_text("controller changed")
    assert hash_sources(project) != before

    def forbidden(*args):
        pytest.fail("source-drift preflight constructed a model")

    monkeypatch.setattr(runtime, "build_seeded_projector_model", forbidden)
    with pytest.raises(ValueError, match="source"):
        runtime.prepare_connector(args)


@pytest.mark.parametrize("damage", [None, "test", "order", "decoder", "hash", "step", "cache"])
def test_completion_strict_dev_only_selection(inputs, runtime, damage):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output = write_completed(runtime, args, protocol)
    if damage:
        path = output / "dev-5500.json"
        report = read_json(path)
        if damage == "test":
            report["split"] = "test"
        elif damage == "order":
            report["items"].reverse()
        elif damage == "decoder":
            report["decoding"]["beam_size"] = 1
        elif damage == "cache":
            report["adapted_encoder_key"] = "wrong"
        elif damage == "hash":
            (output / "step-5500.ckpt").write_bytes(b"corrupt")
        else:
            checkpoint = output / "step-5500.ckpt"
            torch.save({"run_metadata": identity, "pilot_identity": identity,
                        "global_step": 5501}, checkpoint)
            report["checkpoint_hash"] = file_hash(checkpoint)
        atomic_json(path, report)
        with pytest.raises(ValueError):
            runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)
    else:
        result = runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)
        assert result["step"] == 10400 and result["metrics"]["bleu4"] == 3
        assert result["checkpoint"] == output / "step-10400.ckpt"


def test_runtime_import_does_not_require_new_model_module(runtime):
    # Model attachment is deferred until a model or architecture identity is requested.
    assert callable(runtime.prepare_connector)
    assert "scripts.train_connector_study" in sys.modules


def test_lock_refusal_preserves_budget_and_status(inputs, runtime, monkeypatch):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    atomic_json(output / "budget.json", {"identity": identity, "sentinel": True})
    atomic_json(output / "run-status.json", {"status": "partial", "sentinel": True})
    args.resume_checkpoint = output / "resume-500.ckpt"
    before = {path.name: path.read_bytes() for path in output.iterdir()}

    def refused(root):
        raise RuntimeError("lock refused")

    monkeypatch.setattr(runtime, "writer", refused)
    with pytest.raises(RuntimeError, match="lock refused"):
        with runtime.owned_connector_ledger(args, identity, output):
            pytest.fail("lock refusal accepted")
    assert {path.name: path.read_bytes() for path in output.iterdir()} == before


def test_accepted_failure_records_only_own_run_under_lock(inputs, runtime, monkeypatch):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    original = args.output_base / "old-projector-run"
    original.mkdir()
    (original / "run-status.json").write_bytes(b"untouched")
    real_atomic = runtime.atomic_json
    locked = [False]
    real_writer = runtime.writer
    @contextmanager
    def tracked_writer(root):
        with real_writer(root):
            locked[0] = True
            try:
                yield
            finally:
                locked[0] = False

    def checked_atomic(path, record):
        assert locked[0] and path.parent == output
        real_atomic(path, record)

    monkeypatch.setattr(runtime, "writer", tracked_writer)
    monkeypatch.setattr(runtime, "atomic_json", checked_atomic)
    with pytest.raises(RuntimeError, match="model failed"):
        with runtime.owned_connector_ledger(args, identity, output):
            raise RuntimeError("model failed")
    assert read_json(output / "budget.json")["status"] == "closed"
    assert read_json(output / "run-status.json")["status"] == "partial"
    assert (original / "run-status.json").read_bytes() == b"untouched"


def test_retention_directory_fsync_failure_preserves_old_snapshots(tmp_path, runtime, monkeypatch):
    identity = {"run_policy": runtime.RUN_POLICY, "gpu_cap_seconds": 10800}
    ledger = runtime.ConnectorLedger(tmp_path, identity, 0, 0, lambda: 0, lambda: 0)
    for step in (500, 1000):
        (tmp_path / f"resume-{step}.ckpt").write_bytes(b"old")
    callback = runtime.ConnectorResumeSnapshots(tmp_path, ledger)
    trainer = SimpleNamespace(global_step=1500,
                              save_checkpoint=lambda path: Path(path).write_bytes(b"new"))

    def fail(path):
        raise OSError("directory sync failed")

    monkeypatch.setattr(runtime, "_fsync_directory", fail)
    with pytest.raises(OSError, match="directory sync"):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert (tmp_path / "resume-500.ckpt").read_bytes() == b"old"
    assert (tmp_path / "resume-1000.ckpt").read_bytes() == b"old"


@pytest.mark.parametrize("kind", ["missing", "symlink", "seed1"])
def test_preflight_requires_complete_regular_common_profile(inputs, runtime, kind):
    args, _, _, _ = inputs
    if kind != "missing":
        runtime.initialize_connector(args)
        if kind == "symlink":
            original = args.initialization_profile
            args.initialization_profile = original.with_name("linked.json")
            args.initialization_profile.symlink_to(original)
        else:
            profile = read_json(args.initialization_profile)
            del profile["seeds"]["1"]
            atomic_json(args.initialization_profile, profile)
    with pytest.raises(ValueError, match="profile"):
        runtime.prepare_connector(args)


@pytest.mark.parametrize("budget_damage", ["identity", "step", "cap", "nan", "seconds"])
def test_resume_refuses_invalid_budget_without_writes(inputs, runtime, budget_damage):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    budget = {"identity": identity, "status": "closed", "gpu_seconds": 100,
              "optimizer_step": 500, "ceiling_seconds": 10800}
    args.resume_checkpoint = output / "resume-500.ckpt"
    torch.save({"pilot_identity": identity, "run_metadata": identity, "global_step": 500,
                "gpu_seconds_cumulative": 80, "optimizer_states": [1], "lr_schedulers": [1],
                "pilot_sampler": {"batches_yielded": 500}, "pilot_rng": {"present": True}},
               args.resume_checkpoint)
    if budget_damage == "identity":
        budget["identity"] = {"other": True}
    elif budget_damage == "step":
        budget["optimizer_step"] = 400
    elif budget_damage == "cap":
        budget["ceiling_seconds"] = 21600
    elif budget_damage == "nan":
        budget["gpu_seconds"] = float("nan")
    else:
        budget["gpu_seconds"] = 10500
    # Write NaN using torch-independent JSON to exercise malformed external records.
    (output / "budget.json").write_text(json.dumps(budget))
    before = (output / "budget.json").read_bytes()
    with pytest.raises(ValueError):
        with runtime.owned_connector_ledger(args, identity, output):
            pytest.fail("invalid budget accepted")
    assert (output / "budget.json").read_bytes() == before
    assert not (output / "run-status.json").exists()


def test_preflight_json_contains_full_identity(inputs, runtime, monkeypatch, capsys):
    args, _, _, _ = inputs
    runtime.initialize_connector(args)
    args.mode = "preflight"
    monkeypatch.setattr(runtime.argparse.ArgumentParser, "parse_args", lambda self: args)
    runtime.main()
    result = json.loads(capsys.readouterr().out)
    assert result["identity"]["run_policy"] == runtime.RUN_POLICY
    assert result["output"] == str(args.output_base / result["identity"]["run_key"])


def test_run_requires_authorization_before_any_data_reads(inputs, runtime, monkeypatch):
    args, _, _, _ = inputs
    args.authorize_connector_run = False

    def unexpected(args):
        pytest.fail("unauthorized run read inputs")

    monkeypatch.setattr(runtime, "prepare_connector", unexpected)
    with pytest.raises(ValueError, match="authorize"):
        runtime.run_connector(args)


@pytest.mark.parametrize("seed", [0, 1])
def test_public_attachment_uses_training_seed_not_double_offset(runtime, monkeypatch, seed):
    from despamo.models.visual_adapter import SpaMoVisualAdapter

    def build(config, steps, seed):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = torch.nn.Module()
            model.add_module("visual_adapter", SpaMoVisualAdapter(4, 3, 8, 6))
            model.add_module("language_model", torch.nn.Linear(6, 6))
            return model

    monkeypatch.setattr(runtime, "build_seeded_projector_model", build)
    base = build(None, 16000, seed)
    expected_hash = runtime.shared_tensor_hash(base)
    rng = torch.get_rng_state().clone()
    reference = runtime.build_connector_model(None, "reference", seed, expected_hash)
    mlp = runtime.build_connector_model(None, "plain_mlp", seed, expected_hash)
    assert runtime.shared_tensor_hash(reference) == expected_hash
    assert torch.equal(torch.get_rng_state(), rng)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(100 + seed)
        expected = torch.nn.Linear(4, 1024)
    torch.testing.assert_close(mlp.visual_adapter.spatial_projector[0].weight, expected.weight,
                               rtol=0, atol=0)
    torch.testing.assert_close(mlp.visual_adapter.motion_projector.weight,
                               reference.visual_adapter.motion_projector.weight, rtol=0, atol=0)
    with pytest.raises(ValueError, match="initialization"):
        runtime.build_connector_model(None, "reference", seed, "0" * 64)


def tiny_pilot_module():
    from despamo.losses.vt_align import VTAlignLoss
    from despamo.models.visual_adapter import SpaMoVisualAdapter
    from despamo.training.baseline_module import SpaMoBaselineModule
    from despamo.training.pilot_factory import PilotBaselineModule

    base = SpaMoBaselineModule(
        SpaMoVisualAdapter(4, 3, 8, 6), torch.nn.Linear(6, 6),  # pyright: ignore[reportArgumentType]
        VTAlignLoss(),
        "Translate {}", False, 0, "mean", 1.0, 2000, 6e-4, 0.01, seed=0,
    )
    model = PilotBaselineModule(base, 16000)
    model.visual_adapter.multimodal_projector = torch.nn.Sequential(TinyQuery())
    return model


def test_pilot_optimizer_includes_every_registered_connector_parameter(runtime):
    model = tiny_pilot_module()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped})
    assert {id(parameter) for parameter in grouped} == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    assert [group["lr"] for group in optimizer.param_groups] == [3e-4, 5e-5]
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.01


def test_effective_optimizer_validates_actual_pilot_groups_with_generic_lr(runtime):
    model = tiny_pilot_module()
    assert model.learning_rate == 6e-4
    initial_hash = runtime.shared_tensor_hash(model)
    result = runtime.validate_effective_optimizer(model)
    assert result["group_learning_rates"] == {"adapter": 3e-4, "flan": 5e-5}
    assert runtime.shared_tensor_hash(model) == initial_hash
    assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("damage", ["adapter_lr", "flan_lr", "missing", "wrong_group"])
def test_effective_optimizer_refuses_lr_or_parameter_group_drift(runtime, monkeypatch, damage):
    model = tiny_pilot_module()
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    if damage == "adapter_lr":
        optimizer.param_groups[0]["lr"] = 6e-4
    elif damage == "flan_lr":
        optimizer.param_groups[1]["lr"] = 3e-4
    elif damage == "missing":
        optimizer.param_groups[0]["params"].pop()
    else:
        first = optimizer.param_groups[0]["params"]
        second = optimizer.param_groups[1]["params"]
        first[0], second[0] = second[0], first[0]
    monkeypatch.setattr(model, "configure_optimizers", lambda: optimizer)
    with pytest.raises(ValueError, match="optimizer"):
        runtime.validate_effective_optimizer(model)


def test_dev_callback_publishes_bound_report_atomically(inputs, runtime, monkeypatch):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output, *_ = runtime.prepare_connector(args)
    output.mkdir()
    ledger = runtime.ConnectorLedger(output, identity, 0, 0, lambda: 0, lambda: 0)
    dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
    callback = runtime.ConnectorDevCheckpoints([1, 2], dev_ids, protocol["split_hash"],
                output, 16000, 2000, 4, identity["variant"], ledger=ledger,
                checkpoint_schedule=runtime.DEV_STEPS)
    model = TinyModel()
    published = []
    real_atomic = runtime.atomic_json

    def tracked(path, report):
        if path.name.startswith("dev-"):
            published.append(report)
        real_atomic(path, report)

    def score(model, loader, expected, split, split_hash, **kwargs):
        assert split == "dev" and expected == dev_ids
        kwargs["on_batch_end"](2)
        return {"split": split, "split_hash": split_hash, "decoding": runtime.DECODER,
                "metrics": {"bleu4": 2.0}, "items": [{"clip_id": name} for name in expected]}

    monkeypatch.setattr(runtime, "score_pilot_batches", score)
    monkeypatch.setattr(runtime, "atomic_json", tracked)
    trainer = SimpleNamespace(global_step=5500,
                              save_checkpoint=lambda path: Path(path).write_bytes(b"dev-model"))
    callback.on_train_batch_end(trainer, model, None, None, 0)
    assert len(published) == 1
    assert published[0]["adapted_encoder_key"] == identity["adapted_encoder_key"]
    assert published[0]["run_identity_hash"] == digest(identity)
    assert callback.completed == {5500}


@pytest.mark.parametrize("damage", [None, "source", "parameters", "late_source"])
def test_complete_runtime_with_fake_training_never_reads_heldout(
    inputs, runtime, monkeypatch, damage,
):
    args, protocol, _, code = inputs
    runtime.initialize_connector(args)
    args.mode = "run"
    identity, output, *_ = runtime.prepare_connector(args)
    prepare = runtime.legacy._prepare_projector_inputs

    class Heldout:
        def __getitem__(self, key):
            pytest.fail("runtime accessed held-out examples")

        def __iter__(self):
            pytest.fail("runtime scored held-out examples")

    def with_heldout(namespace):
        values = prepare(namespace)
        values[1]["test"] = Heldout()
        return values

    monkeypatch.setattr(runtime.legacy, "_prepare_projector_inputs", with_heldout)
    monkeypatch.setattr(runtime, "build_connector_model", lambda *args: TinyModel())
    monkeypatch.setattr(TinyModel, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(runtime, "collate_phoenix",
                        lambda examples: SimpleNamespace(clip_ids=["t1", "t2", "t3", "t4"]))
    def smoke(model, batch, arm):
        if damage == "source":
            code["src/changed.py"] = "c" * 64
        elif damage == "parameters":
            with torch.no_grad():
                model.visual_adapter.motion_projector.weight.add_(1)
        return {"status": "passed", "fit_16gib": True, "optimizer_updates": 0,
                "cuda_peak_bytes": 1024}

    monkeypatch.setattr(runtime, "connector_gradient_smoke", smoke)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(runtime.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=60 * 1024**3))
    alarms = []
    monkeypatch.setattr(runtime.signal, "setitimer", lambda which, seconds: alarms.append(seconds))

    def score(model, loader, expected, split, split_hash, **kwargs):
        assert split == "dev" and expected == ("d1", "d2")
        kwargs["on_batch_end"](2)
        return {"split": "dev", "split_hash": split_hash, "decoding": runtime.DECODER,
                "metrics": {"bleu4": 3}, "items": [{"clip_id": name} for name in expected]}

    monkeypatch.setattr(runtime, "score_pilot_batches", score)

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            assert kwargs["max_steps"] == 16000 and kwargs["precision"] == "bf16"
            self.global_step = 0

        def save_checkpoint(self, path):
            torch.save({"global_step": self.global_step, "run_metadata": identity,
                        "pilot_identity": identity, "state_dict": {}}, path)

        def fit(self, model, datamodule, ckpt_path):
            assert ckpt_path is None
            for callback in self.callbacks:
                if hasattr(callback, "on_train_start"):
                    callback.on_train_start(self, model)
            for step in runtime.DEV_STEPS:
                self.global_step = step
                for callback in self.callbacks:
                    callback.on_train_batch_end(self, model, None, None, 0)
            if damage == "late_source":
                code["src/changed.py"] = "d" * 64

    monkeypatch.setattr(runtime.pl, "Trainer", FakeTrainer)
    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            runtime.run_connector(args)
        assert read_json(output / "run-status.json")["status"] == "partial"
        assert not (output / "selected-checkpoint.json").exists()
        assert alarms[-1] == 0
        return
    result = runtime.run_connector(args)
    assert result["status"] == "complete"
    assert result["selected_step"] == 5500  # Tie rule chooses earliest dev checkpoint.
    assert 0 < alarms[0] <= 10800 and alarms[-1] == 0
    assert read_json(output / "gradient-smoke.json")["fit_16gib"] is True
    selected = runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)
    assert selected["step"] == 5500


@pytest.mark.parametrize("snapshot_kind", ["step", "resume"])
@pytest.mark.parametrize("damage", [None, "prior_report", "state", "prior_content"])
def test_terminal_resume_repairs_final_dev_without_training_or_optimizer(
    inputs, runtime, monkeypatch, snapshot_kind, damage,
):
    args, protocol, _, _ = inputs
    runtime.initialize_connector(args)
    identity, output = write_completed(runtime, args, protocol)
    fresh = TinyModel()
    variant_hash = runtime.shared_tensor_hash(fresh)
    optimizer_recipe = runtime.validate_effective_optimizer(fresh)
    atomic_json(output / "initialization.json", {
        "initial_base_hash": identity["initial_base_hash"], "initial_variant_hash": variant_hash,
        "initialization_profile_hash": identity["initialization_profile_hash"],
        "effective_optimizer": optimizer_recipe,
        "trainable_parameters": sum(p.numel() for p in fresh.parameters() if p.requires_grad),
        "adapter_trainable_parameters": sum(p.numel() for p in fresh.visual_adapter.parameters()
                                            if p.requires_grad),
        "flan_trainable_parameters": sum(p.numel() for p in fresh.language_model.parameters()
                                         if p.requires_grad),
    })
    atomic_json(output / "gradient-smoke.json", {"status": "passed", "original": True})
    atomic_json(output / "run-status.json", {"status": "partial", "global_step": 16000})
    (output / "selected-checkpoint.json").unlink()
    (output / "dev-16000.json").unlink()
    data = runtime.build_seeded_projector_data(list(range(8)), 16000, 0)
    sampler = data.state_dict()
    sampler["batches_yielded"] = 16000
    trained = TinyModel()
    with torch.no_grad():
        for parameter in trained.parameters():
            parameter.add_(0.25)
        trained.counter.fill_(7)
    trained.rng.random()
    expected_state = copy.deepcopy(trained.state_dict())
    rng = capture_rng()
    rng["cuda"] = []  # Synthetic CPU checkpoint; no real CUDA work in this test.
    optimizer = trained.configure_optimizers()
    assert isinstance(optimizer, torch.optim.Optimizer)
    saved = {
        "run_metadata": identity, "pilot_identity": identity, "global_step": 16000,
        "state_dict": expected_state, "gpu_seconds_cumulative": 80,
        "pilot_sampler": sampler, "pilot_rng": rng, "prompt_rng_state": trained.rng.getstate(),
        "optimizer_states": [optimizer.state_dict()], "lr_schedulers": [{"last_epoch": 16000}],
    }
    if damage == "state":
        saved["state_dict"] = {k: v for k, v in expected_state.items()
                               if k != "language_model.bias"}
    args.resume_checkpoint = output / f"{snapshot_kind}-16000.ckpt"
    if snapshot_kind == "resume":
        (output / "step-16000.ckpt").unlink()
    torch.save(saved, args.resume_checkpoint)
    if damage == "prior_report":
        (output / "dev-10400.json").unlink()
    elif damage == "prior_content":
        prior = read_json(output / "dev-10400.json")
        prior["items"][0]["reference"] = "mutated"
        atomic_json(output / "dev-10400.json", prior)
    prior_bytes = {path.name: path.read_bytes() for path in output.iterdir()
                   if path.name in {"step-5500.ckpt", "step-10400.ckpt", "dev-5500.json",
                                    "dev-10400.json", "initialization.json", "gradient-smoke.json"}}
    calls = {"optimizer": 0, "fit": 0, "smoke": 0, "score": 0}
    configure = TinyModel.configure_optimizers

    def counted_optimizer(model):
        calls["optimizer"] += 1
        return configure(model)

    monkeypatch.setattr(TinyModel, "configure_optimizers", counted_optimizer)
    monkeypatch.setattr(runtime, "build_connector_model", lambda *args: TinyModel())
    monkeypatch.setattr(TinyModel, "to", lambda self, *args, **kwargs: self)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3))
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: None)
    monkeypatch.setattr(runtime.shutil, "disk_usage",
                        lambda path: SimpleNamespace(free=60 * 1024**3))
    monkeypatch.setattr(runtime, "collate_phoenix",
                        lambda examples: SimpleNamespace(clip_ids=["t1", "t2", "t3", "t4"]))

    def smoke(*args):
        calls["smoke"] += 1
        return {"status": "passed", "fit_16gib": True}

    monkeypatch.setattr(runtime, "connector_gradient_smoke", smoke)

    class TerminalLightning:
        def __init__(self, **kwargs):
            self.global_step = 16000

        def fit(self, *args, **kwargs):
            calls["fit"] += 1
            # Lightning 1.9.5 skips the fit loop before on_train_start at max_steps.

    monkeypatch.setattr(runtime.pl, "Trainer", TerminalLightning)

    def score(model, loader, expected, split, split_hash, **kwargs):
        calls["score"] += 1
        assert split == "dev" and expected == ("d1", "d2")
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, expected_state[name], rtol=0, atol=0)
        assert model.rng.getstate() == saved["prompt_rng_state"]
        assert torch.equal(torch.get_rng_state(), rng["torch"])
        kwargs["on_batch_end"](2)
        return {"split": "dev", "split_hash": split_hash, "decoding": runtime.DECODER,
                "metrics": {"bleu4": 4}, "items": [{"clip_id": name, "prediction": "x",
                "reference": "x"} for name in expected]}

    monkeypatch.setattr(runtime, "score_pilot_batches", score)
    if damage:
        with pytest.raises((ValueError, RuntimeError)):
            runtime.run_connector(args)
        assert not (output / "selected-checkpoint.json").exists()
    else:
        result = runtime.run_connector(args)
        assert result["status"] == "complete" and result["selected_step"] == 16000
        selected = runtime.validate_connector_selection(output, protocol, ("d1", "d2"), identity)
        assert selected["step"] == 16000 and calls["score"] == 1
    assert calls["fit"] == calls["optimizer"] == calls["smoke"] == 0
    assert {name: (output / name).read_bytes() for name in prior_bytes} == prior_bytes

"""Synthetic software contracts; no claim about historical extraction chronology."""

import copy
import importlib
import importlib.util
import random
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.motion_timing import CONVENTION, build_motion_timing
from despamo.models.aligned_fusion import enable_aligned_fusion
from despamo.models.connector_adapter import enable_connector_arm
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.pilot_factory import PilotBaselineModule, _pilot_optimizer, shared_tensor_hash
from despamo.training.pilot_replay import capture_rng


@pytest.fixture
def runtime():
    assert importlib.util.find_spec("scripts.train_fusion_study") is not None, "runner missing"
    return importlib.import_module("scripts.train_fusion_study")


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


class TinyPilot(torch.nn.Module):
    """Actual four-input fusion forward, lightweight downstream joint objective."""

    counter: torch.Tensor
    ephemeral: torch.Tensor

    def __init__(self, seed=0, arm="reference", detached=False):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            base = SpaMoVisualAdapter(2048, 1024, 768, 2048)
            self.visual_adapter = enable_aligned_fusion(
                enable_connector_arm(base, "plain_mlp", seed), arm, seed,
            )
            self.language_model = torch.nn.Linear(2048, 2)
            self.vt_align = torch.nn.Linear(2048, 2)
        self.register_buffer("counter", torch.zeros(1))
        self.rng = random.Random(seed)
        self.vt_weight = 1.0
        self.weight_decay = 0.01
        self._trainer = None
        self.detached = detached

    def configure_optimizers(self):
        return _pilot_optimizer(cast(PilotBaselineModule, self))

    def compute_losses(self, batch):
        self.counter.add_(1)
        self.rng.random()
        random.random()
        np.random.rand()
        tokens, mask = self.visual_adapter(*batch)
        tokens = (tokens * mask[..., None]).sum(1) / mask.sum(1)[:, None]
        if self.detached:
            tokens = tokens.detach()
        return {"loss": self.language_model(tokens).square().mean(),
                "contra_loss": self.vt_align(tokens).square().mean()}


def batch():
    generator = torch.Generator().manual_seed(73)
    return (torch.randn(2, 24, 2048, generator=generator), torch.ones(2, 24, dtype=torch.bool),
            torch.randn(2, 2, 1024, generator=generator), torch.ones(2, 2, dtype=torch.bool))


@pytest.mark.parametrize("arm", ["reference", "aligned_fixed", "aligned_learned"])
def test_smoke_two_backward_restores_parameters_grads_buffers_modes_rng(runtime, arm):
    model = TinyPilot(arm=arm)
    model.eval()
    model.visual_adapter.motion_projector.train()
    parameter = next(model.parameters())
    parameter.grad = torch.ones_like(parameter)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    modes = [m.training for m in model.modules()]
    rng = capture_rng()
    prompt = model.rng.getstate()
    before = shared_tensor_hash(model)
    result = runtime.fusion_gradient_smoke(model, batch(), arm)
    assert result["status"] == "passed"
    assert result["optimizer_updates"] == 0
    assert result["disposable_optimizer_updates"] == (0 if arm == "reference" else 1)
    assert result["initial_model_hash"] == result["restored_model_hash"] == before
    if arm != "reference":
        assert result["first_gradient_norms"]["corrector_last"] > 0
        assert result["first_gradient_norms"]["corrector_early"] == 0
        assert result["second_gradient_norms"]["corrector_early"] > 0
        assert result["disposable_learning_rate"] == 3e-4
    if arm == "aligned_learned":
        assert result["first_gradient_norms"]["gate"] == 0
        assert result["second_gradient_norms"]["gate"] > 0
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert torch.equal(parameter.grad, torch.ones_like(parameter))
    assert all(p.grad is None for p in list(model.parameters())[1:])
    assert [m.training for m in model.modules()] == modes
    assert model.rng.getstate() == prompt
    after = capture_rng()
    assert after["python"] == rng["python"]
    assert np.array_equal(after["numpy"][1], rng["numpy"][1])
    assert torch.equal(after["torch"], rng["torch"])


def test_smoke_detached_fusion_fails_and_restores(runtime):
    model = TinyPilot(arm="aligned_learned", detached=True)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    with pytest.raises(RuntimeError, match="gradient"):
        runtime.fusion_gradient_smoke(model, batch(), "aligned_learned")
    assert all(torch.equal(v, before[k]) for k, v in model.state_dict().items())


def test_optimizer_accounts_for_new_branches_once_at_adapter_rate(runtime):
    model = TinyPilot(arm="aligned_learned")
    assert runtime.validate_effective_optimizer(model)["group_learning_rates"] == {
        "adapter": 3e-4, "flan": 5e-5,
    }
    original = model.configure_optimizers

    def missing():
        optimizer = original()
        assert isinstance(optimizer, torch.optim.AdamW)
        gate = cast(torch.nn.Linear, model.visual_adapter.gate)
        optimizer.param_groups[0]["params"] = [
            p for p in optimizer.param_groups[0]["params"]
            if p is not gate.weight
        ]
        return optimizer

    model.configure_optimizers = missing
    with pytest.raises(ValueError, match="missing"):
        runtime.validate_effective_optimizer(model)


def test_common_hash_excludes_only_extra_fusion_branches(runtime):
    models = [TinyPilot(seed=1, arm=arm) for arm in runtime.FUSION_ARMS]
    assert len({runtime.common_plain_hash(m) for m in models}) == 1
    assert len({shared_tensor_hash(m) for m in models}) == 3
    with torch.no_grad():
        models[1].visual_adapter.motion_projector.weight.add_(1)
    assert runtime.common_plain_hash(models[1]) != runtime.common_plain_hash(models[0])


def synthetic_timing(tmp_path):
    evidence = tmp_path / "synthetic-evidence.json"
    atomic_json(evidence, {"fixture": "not historical evidence"})
    frames = {"encoder_key": "synthetic", "clips": {"clip": {
        "frame_count": 25, "source_indices": list(range(25)),
        "paths": [f"/synthetic/{i:04}.png" for i in range(25)],
        "source_hash": "a" * 64, "feature_hash": "b" * 64,
    }}}
    record = {"clip_id": "clip", "split": "train", "path": "train/clip.npy",
              "length": 2, "width": 1024, "dtype": "float32"}
    manifest = {"schema_version": 1, "encoder": "MCG-NJU/videomae-large",
                "expected_dim": 1024, "records": [record]}
    provenance = {"schema_version": 1, "status": "original_receipt", **CONVENTION,
                  "physical_split": "train", "spatial_crop_mode": "full",
                  "input_sha256": {"frame_rows": digest(frames),
                                   "motion_manifest": digest(manifest)},
                  "file_sha256": {k: "c" * 64 for k in
                                  ("frame_rows", "motion_manifest", "annotation",
                                   "extractor", "helpers")},
                  "evidence": [{"path": str(evidence), "sha256": file_hash(evidence),
                                "claims": ["historical_window_parameters", "motion_row_order",
                                           "source_frame_order"]}],
                  "clips": {"clip": {"frame_count": 25, "source_hash": "a" * 64,
                                     "motion_record_sha256": digest(record)}}}
    return frames, manifest, build_motion_timing(frames, manifest, provenance)


def test_timing_join_rejects_renumbered_or_reordered_receipts(runtime, tmp_path):
    frames, manifest, timing = synthetic_timing(tmp_path)
    receipt = copy.deepcopy(frames["clips"]["clip"])
    runtime.validate_timing_join(timing, frames, manifest, {"clip": receipt})
    for field in ("source_indices", "paths", "source_hash", "feature_hash"):
        bad = copy.deepcopy(receipt)
        bad[field] = list(reversed(bad[field])) if isinstance(bad[field], list) else "0" * 64
        with pytest.raises(ValueError, match="receipt"):
            runtime.validate_timing_join(timing, frames, manifest, {"clip": bad})
    cropped = copy.deepcopy(frames)
    cropped["clips"]["clip"]["source_indices"] = [0, 1]
    with pytest.raises(ValueError):
        runtime.validate_timing_join(timing, cropped, manifest, {"clip": receipt})


@pytest.fixture
def prepared(runtime, tmp_path, monkeypatch):
    args = SimpleNamespace(arm="reference", seed=0, steps=16000, gpu_cap_seconds=10800,
                           initialization_profile=tmp_path / "profile.json",
                           resume_checkpoint=None, output_base=tmp_path / "runs",
                           authorize_fusion_run=True)
    args.output_base.mkdir()
    config = OmegaConf.create({"seed": 0, "model": {"warm_up_steps": 2000,
                              "use_in_context": False}, "data": {"batch_size": 4,
                              "num_workers": 0}, "trainer": {"max_steps": 16000,
                              "precision": "bf16", "gradient_clip_val": 1.0,
                              "accumulate_grad_batches": 1}, "optimizer": {"weight_decay": 0.01}})
    ids = tuple(f"dev{i}" for i in range(582))
    protocol = {"protocol_hash": "protocol", "split_hash": "split", "decoder": runtime.DECODER,
                "split": {"groups": {"dev": {"clip_ids": list(ids)}}}}
    binding = {"hardware": {"gpu_name": "synthetic", "driver": "fixture"},
               "code_hashes": {"schema": "a" * 64}, "timing_hash": "b" * 64,
               "recipe": copy.deepcopy(runtime.RECIPE), "protocol_hash": "protocol",
               "split_hash": "split", "baseline_checkpoint_hash": "c" * 64,
               "dev_frame_coverage": {"real_frame_count": 582 * 25,
                                      "supported_frame_count": 582 * 24, "clip_count": 582}}

    def prepare(namespace):
        cfg = copy.deepcopy(config)
        cfg.seed = namespace.seed
        return protocol, {"train": [], "dev": [], "test": []}, cfg, copy.deepcopy(binding)

    monkeypatch.setattr(runtime, "_prepare_inputs", prepare)
    monkeypatch.setattr(runtime, "build_seeded_projector_model", lambda c, s, seed: TinyPilot(seed))
    monkeypatch.setattr(runtime, "_attach_plain", lambda m, seed: m)
    monkeypatch.setattr(runtime, "E1_SHARED_HASH", shared_tensor_hash(TinyPilot(0)))
    monkeypatch.setattr(runtime, "_safe_destination", lambda path, namespace: None)
    monkeypatch.setattr(runtime, "_check_live", lambda identity: None)
    return args, protocol, binding, ids


def test_profile_both_seed_plain_hashes_readonly_preflight(runtime, prepared, monkeypatch):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    profile = read_json(args.initialization_profile)
    assert set(profile["seeds"]) == {"0", "1"}
    assert all(len(e["common_plain_hash"]) == 64 for e in profile["seeds"].values())
    monkeypatch.setattr(runtime, "build_seeded_projector_model",
                        lambda *args: pytest.fail("preflight built model"))
    identities = []
    for seed in (0, 1):
        args.seed = seed
        for arm in runtime.FUSION_ARMS:
            args.arm = arm
            identity, output, *_ = runtime.prepare_fusion(args)
            identities.append(identity)
            assert output.name.startswith(f"fusion-{arm}-seed{seed}-16000-")
    assert len({i["run_key"] for i in identities}) == 6
    assert list(args.output_base.iterdir()) == []


@pytest.mark.parametrize("field", ["hardware", "code_hashes", "timing_hash", "recipe"])
def test_profile_drift_fails_without_output(runtime, prepared, field):
    args, _, binding, _ = prepared
    runtime.initialize_fusion(args)
    binding[field] = {"changed": True}
    with pytest.raises(ValueError, match="profile"):
        runtime.prepare_fusion(args)
    assert list(args.output_base.iterdir()) == []


@pytest.mark.parametrize("field,value", [("seed", True), ("seed", 2), ("steps", 8000),
                                       ("gpu_cap_seconds", 10801), ("arm", "plain_mlp")])
def test_frozen_recipe_arguments_rejected(runtime, field, value):
    args = SimpleNamespace(arm="reference", seed=0, steps=16000, gpu_cap_seconds=10800)
    setattr(args, field, value)
    with pytest.raises(ValueError):
        runtime._require_args(args)


def completed_run(runtime, prepared):
    args, protocol, _, ids = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    model = TinyPilot()
    runtime.ensure_fusion_initialization(output, model, identity, existing=False)
    initialization_hash = file_hash(output / "initialization.json")
    telemetry = runtime.capture_fusion_peak(model, "training")
    telemetry = runtime.capture_fusion_peak(model, "dev", telemetry)
    stats = {"supported_frame_count": 582 * 24, "gate_count": 0, "gate_sum": None,
             "gate_sumsq": None, "gate_min": None, "gate_max": None,
             "gate_below_0_05": None, "gate_above_0_95": None, "correction_norm": None,
             "real_frame_count": 582 * 25, "clip_count": 582}
    reports = []
    for step, score in ((5500, 3.0), (10400, 4.0), (16000, 4.0)):
        checkpoint = output / f"step-{step}.ckpt"
        torch.save({"run_metadata": identity, "pilot_identity": identity,
                    "global_step": step, "initialization_hash": initialization_hash,
                    "fusion_telemetry": telemetry}, checkpoint)
        report = {"checkpoint_step": step, "checkpoint_hash": file_hash(checkpoint),
                  "split": "dev", "split_hash": "split", "decoding": runtime.DECODER,
                  "items": [{"clip_id": i} for i in ids], "metrics": {"bleu4": score},
                  "fusion_stats": stats, **runtime.report_binding(identity),
                  "initialization_hash": initialization_hash,
                  "checkpoint_size_bytes": checkpoint.stat().st_size,
                  "visual_token_lengths": [{"clip_id": i, "valid_tokens": 3,
                                             "padded_tokens": 3} for i in ids],
                  "visual_token_length_scope": "eval_diagnostic_forward_actual_output_mask",
                  "cuda_memory": telemetry["dev"]}
        report["content_hash"] = digest(report)
        atomic_json(output / f"dev-{step}.json", report)
        reports.append(report)
    atomic_json(output / "selected-checkpoint.json", {
        "step": 10400, "checkpoint_hash": reports[1]["checkpoint_hash"],
        "metrics": reports[1]["metrics"], **runtime.report_binding(identity),
        "dev_report_hashes": runtime._dev_report_hashes(output),
        "initialization_hash": initialization_hash, "fusion_telemetry": telemetry,
        "fusion_telemetry_hash": digest(telemetry),
        "checkpoint_sizes_bytes": runtime.fusion_checkpoint_sizes(output),
    })
    atomic_json(output / "budget.json", {"identity": identity, "gpu_seconds": 100.0,
                "optimizer_step": 16000, "ceiling_seconds": 10800, "status": "closed",
                "initialization_hash": initialization_hash, "fusion_telemetry": telemetry,
                "checkpoint_sizes_bytes": runtime.fusion_checkpoint_sizes(output)})
    atomic_json(output / "run-status.json", {"status": "complete", "global_step": 16000,
                                           "run_key": output.name,
                                           "initialization_hash": initialization_hash,
                                           "fusion_telemetry_hash": digest(telemetry)})
    return output, identity, protocol, ids


def test_selection_ties_earlier_all_three_report_bytes(runtime, prepared):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    assert runtime.validate_fusion_selection(output, protocol, ids, identity)["step"] == 10400
    path = output / "dev-5500.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="byte"):
        runtime.validate_fusion_selection(output, protocol, ids, identity)


def test_selection_rejects_reordered_ids_and_checkpoint_identity(runtime, prepared):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    with pytest.raises(ValueError):
        runtime.validate_fusion_selection(output, protocol, tuple(reversed(ids)), identity)
    identity = copy.deepcopy(identity)
    identity["timing_hash"] = "0" * 64
    with pytest.raises(ValueError):
        runtime.validate_fusion_selection(output, protocol, ids, identity)


def test_wrong_lock_owner_never_writes_status(runtime, prepared):
    from tools.dinov3.storage import writer

    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    args.resume_checkpoint = output / "resume-500.ckpt"
    with writer(output), pytest.raises(RuntimeError, match="writer"):
        with runtime.owned_fusion_ledger(args, identity, output):
            pytest.fail("wrong owner admitted")
    assert not (output / "run-status.json").exists()
    assert not (output / "budget.json").exists()


def test_terminal_recovery_strict_reload_without_optimizer_or_fit(runtime, tmp_path):
    model = TinyPilot()
    identity = {"run_policy": runtime.RUN_POLICY, "timing_hash": "timing",
                "protocol_hash": "synthetic-protocol", "run_key": tmp_path.name,
                "arm": "reference", "initial_base_hash": "a" * 64,
                "common_plain_hash": runtime.common_plain_hash(model)}
    runtime.ensure_fusion_initialization(tmp_path, model, identity, existing=False)
    target = tmp_path / "resume-16000.ckpt"
    checkpoint = {"global_step": 16000, "prompt_rng_state": model.rng.getstate(),
                  "state_dict": model.state_dict(), "pilot_sampler": {"batches_yielded": 16000},
                  "pilot_rng": capture_rng(), "pilot_identity": identity, "run_metadata": identity,
                  "optimizer_states": [{"saved": True}], "lr_schedulers": [{"saved": True}],
                  "gpu_seconds_cumulative": 1.0,
                  "initialization_hash": file_hash(tmp_path / "initialization.json"),
                  "fusion_telemetry": runtime.capture_fusion_peak(model, "training")}
    torch.save(checkpoint, target)
    data = SimpleNamespace(load_state_dict=lambda state: None)
    ledger = SimpleNamespace(last=10.0, charge=lambda **kw: None,
                             require_remaining=lambda *a, **kw: None)
    scored = []
    dev = SimpleNamespace(on_train_start=lambda trainer, m: scored.append(trainer.global_step))
    model.configure_optimizers = lambda: pytest.fail("terminal optimizer created")
    runtime._recover_terminal_dev(model, data, dev, target, tmp_path, identity, ledger)
    assert scored == [16000]
    assert (tmp_path / "step-16000.ckpt").read_bytes() == target.read_bytes()
    checkpoint["state_dict"]["unexpected"] = torch.ones(1)
    torch.save(checkpoint, tmp_path / "step-16000.ckpt")
    with pytest.raises(RuntimeError, match="Unexpected"):
        runtime._recover_terminal_dev(model, data, dev, target, tmp_path, identity, ledger)


def test_source_closure_requires_controller_and_lock(runtime, tmp_path):
    for name in runtime.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("synthetic source fixture")
    first = runtime.fusion_code_hashes(tmp_path)
    assert "scripts/run_fusion_study.py" in first
    assert "uv.lock" in first
    added = tmp_path / "src/despamo/new_dependency.py"
    added.parent.mkdir(parents=True, exist_ok=True)
    added.write_text("synthetic addition")
    assert runtime.fusion_code_hashes(tmp_path) != first


def test_nonpersistent_buffers_restored_on_smoke_failure(runtime):
    model = TinyPilot(arm="aligned_learned", detached=True)
    model.register_buffer("ephemeral", torch.zeros(1), persistent=False)
    compute = model.compute_losses

    def loss(batch):
        model.ephemeral.add_(1)
        return compute(batch)

    model.compute_losses = loss
    with pytest.raises(RuntimeError, match="gradient"):
        runtime.fusion_gradient_smoke(model, batch(), "aligned_learned")
    assert model.ephemeral.item() == 0


def test_run_terminal_branch_finishes_selection_without_trainer(runtime, prepared, monkeypatch):
    args, protocol, _, ids = prepared
    output, identity, _, _ = completed_run(runtime, prepared)
    # Synthetic crash before selection publication. Earlier reports/models remain.
    (output / "selected-checkpoint.json").unlink()
    atomic_json(output / "run-status.json", {"status": "partial", "global_step": 16000,
                                           "run_key": output.name})
    model = TinyPilot()
    source = output / "step-16000.ckpt"
    torch.save({"global_step": 16000, "prompt_rng_state": model.rng.getstate(),
                "state_dict": model.state_dict(), "pilot_sampler": {"batches_yielded": 16000},
                "pilot_rng": capture_rng(), "pilot_identity": identity, "run_metadata": identity,
                "optimizer_states": [{"saved": True}], "lr_schedulers": [{"saved": True}],
                "gpu_seconds_cumulative": 1.0}, source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint.update(initialization_hash=file_hash(output / "initialization.json"),
                      fusion_telemetry=read_json(output / "budget.json")["fusion_telemetry"])
    torch.save(checkpoint, source)
    report = read_json(output / "dev-16000.json")
    report["checkpoint_hash"] = file_hash(source)
    report["checkpoint_size_bytes"] = source.stat().st_size
    report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
    atomic_json(output / "dev-16000.json", report)
    args.resume_checkpoint = source
    monkeypatch.setattr(runtime, "prepare_fusion", lambda a:
                        (identity, output, protocol, {"train": [], "dev": []}, {}))
    monkeypatch.setattr(runtime, "require_storage", lambda p: {})
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "build_fusion_model", lambda *a: model)
    monkeypatch.setattr(model, "to", lambda *a: model)
    monkeypatch.setattr(runtime, "validate_effective_optimizer",
                        lambda *a: pytest.fail("terminal optimizer initialized"))
    monkeypatch.setattr(runtime.pl, "Trainer", lambda *a, **kw: pytest.fail("terminal fit"))
    monkeypatch.setattr(runtime, "_smoke_batches", lambda *a: pytest.fail("terminal backward"))
    monkeypatch.setattr(runtime, "build_seeded_projector_data",
                        lambda *a: SimpleNamespace(load_state_dict=lambda state: None))
    monkeypatch.setattr(runtime, "restore_rng", lambda rng: None)  # CPU-only saved fixture.
    monkeypatch.setattr(runtime, "FusionDevCheckpoints",
                        lambda *a, **kw: SimpleNamespace(on_train_start=lambda *a: None))
    identity["common_plain_hash"] = runtime.common_plain_hash(model)
    # Identity must remain immutable: use the originally attested TinyPilot(seed0).
    assert identity["common_plain_hash"] == read_json(args.initialization_profile)[
        "seeds"]["0"]["common_plain_hash"]
    result = runtime.run_fusion(args)
    assert result["status"] == "complete"
    assert result["selected_step"] == 10400
    assert runtime.validate_fusion_selection(output, protocol, ids, identity)["step"] == 10400


def test_snapshots_prune_after_directory_fsync(runtime, tmp_path, monkeypatch):
    identity = {"run_policy": runtime.RUN_POLICY, "protocol_hash": "synthetic"}
    ledger = SimpleNamespace(root=tmp_path, identity=identity,
                             require_remaining=lambda *a, **k: None,
                             charge=lambda **kw: None)
    events = []

    def sync(directory):
        events.append(sorted(p.name for p in directory.glob("*.ckpt")))

    monkeypatch.setattr(runtime, "_fsync_directory", sync)
    callback = runtime.FusionResumeSnapshots(tmp_path, ledger)
    for step in (500, 1000, 1500):
        trainer = SimpleNamespace(global_step=step, save_checkpoint=lambda path, step=step:
                                  torch.save({"global_step": step, "pilot_identity": identity,
                                              "run_metadata": identity}, path))
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert events[-2] == ["resume-1000.ckpt", "resume-1500.ckpt", "resume-500.ckpt"]
    assert events[-1] == ["resume-1000.ckpt", "resume-1500.ckpt"]
    torch.save({"retained": True}, tmp_path / "step-5500.ckpt")
    assert (tmp_path / "step-5500.ckpt").is_file()


def test_fsync_failure_never_prunes_old_snapshots(runtime, tmp_path, monkeypatch):
    identity = {"run_policy": runtime.RUN_POLICY, "protocol_hash": "synthetic"}
    for step in (500, 1000):
        torch.save({"global_step": step, "pilot_identity": identity, "run_metadata": identity},
                   tmp_path / f"resume-{step}.ckpt")
    ledger = SimpleNamespace(root=tmp_path, identity=identity,
                             require_remaining=lambda *a, **k: None,
                             charge=lambda **kw: None)
    monkeypatch.setattr(runtime, "_fsync_directory",
                        lambda p: (_ for _ in ()).throw(OSError("fsync")))
    trainer = SimpleNamespace(global_step=1500, save_checkpoint=lambda p: torch.save({}, p))
    with pytest.raises(OSError, match="fsync"):
        runtime.FusionResumeSnapshots(tmp_path, ledger).on_train_batch_end(
            trainer, None, None, None, 0,
        )
    assert (tmp_path / "resume-500.ckpt").exists()
    assert (tmp_path / "resume-1000.ckpt").exists()


def test_frozen_config_mutations_rejected(runtime):
    config = OmegaConf.create({"model": {"warm_up_steps": 2000, "use_in_context": False},
                              "data": {"batch_size": 4, "num_workers": 0},
                              "trainer": {"max_steps": 16000, "precision": "bf16",
                                          "gradient_clip_val": 1.0, "accumulate_grad_batches": 1},
                              "optimizer": {"weight_decay": 0.01}})
    runtime._require_config(config)
    for group, key, value in (("trainer", "precision", "32"), ("model", "warm_up_steps", 1000),
                              ("data", "batch_size", 2), ("trainer", "accumulate_grad_batches", 2),
                              ("optimizer", "weight_decay", 0.02)):
        changed = copy.deepcopy(config)
        changed[group][key] = value
        with pytest.raises(ValueError, match="recipe"):
            runtime._require_config(changed)


def test_dev_stats_recomputation_matches_applied_correction(runtime):
    model = TinyPilot(arm="aligned_learned").eval()
    physical = batch()
    b = SimpleNamespace(spatial=physical[0], spatial_mask=physical[1],
                        motion=physical[2], motion_mask=physical[3])
    before = shared_tensor_hash(model)
    stats = runtime._batch_frame_stats(model, b, "aligned_learned")
    assert stats["clip_count"] == 2
    assert stats["real_frame_count"] == stats["supported_frame_count"] == 48
    assert stats["gate_count"] == 48
    assert stats["gate_sum"] == 24
    assert stats["gate_sumsq"] == 12
    assert stats["gate_min"] == stats["gate_max"] == 0.5
    assert stats["correction_norm"] == 0
    assert shared_tensor_hash(model) == before


def test_smoke_ledger_adoption_preserves_spend(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    runtime.ensure_fusion_initialization(output, TinyPilot(), identity, existing=False)
    atomic_json(output / "budget.json", {"identity": identity, "gpu_seconds": 123.0,
                "optimizer_step": 0, "ceiling_seconds": 10800, "status": "closed",
                "initialization_hash": file_hash(output / "initialization.json"),
                "fusion_telemetry": runtime._empty_fusion_telemetry()})
    atomic_json(output / "run-status.json", {"status": "smoke_passed", "global_step": 0,
                                           "run_key": output.name})
    with runtime.owned_fusion_ledger(args, identity, output) as ledger:
        assert ledger.last >= 123
        assert ledger.step == 0
    assert read_json(output / "budget.json")["gpu_seconds"] >= 123


def test_partial_selection_can_resume_without_reset_or_early_status_write(runtime, prepared):
    args, _, _, _ = prepared
    output, identity, _, _ = completed_run(runtime, prepared)
    atomic_json(output / "run-status.json", {"status": "partial", "global_step": 16000,
                                           "run_key": output.name})
    # Selection published, status missing: allow lock admission, then reject malformed resume.
    args.resume_checkpoint = output / "step-16000.ckpt"
    before = (output / "run-status.json").read_bytes()
    with pytest.raises(ValueError, match="optimizer"):
        with runtime.owned_fusion_ledger(args, identity, output):
            pytest.fail("incomplete optimizer state admitted")
    assert (output / "run-status.json").read_bytes() == before


@pytest.mark.parametrize("changed", ["source", "hardware", "input", "profile"])
def test_live_identity_rejects_drift(runtime, tmp_path, monkeypatch, changed):
    source = {"source": "a" * 64}
    hardware = {"gpu": "synthetic"}
    file = tmp_path / "input.json"
    profile = tmp_path / "profile.json"
    atomic_json(file, {"synthetic": True})
    atomic_json(profile, {"synthetic": True})
    identity = {"code_hashes": dict(source), "hardware": dict(hardware),
                "bound_files": {str(file): file_hash(file)},
                "initialization_profile_path": str(profile),
                "initialization_profile_file_hash": file_hash(profile)}
    monkeypatch.setattr(runtime, "fusion_code_hashes", lambda p: source)
    monkeypatch.setattr(runtime, "hardware_identity", lambda: hardware)
    if changed == "source":
        source["new"] = "b" * 64
    elif changed == "hardware":
        hardware["driver"] = "changed"
    elif changed == "input":
        atomic_json(file, {"synthetic": False})
    else:
        atomic_json(profile, {"synthetic": False})
    # prepared fixture does not override this real admission function.
    with pytest.raises(ValueError, match="changed"):
        runtime._check_live(identity)


def test_inherited_pilot_loss_and_neutral_scorer_reach_fusion(runtime):
    from despamo.losses.vt_align import VTAlignLoss
    from despamo.models.flan_t5 import FlanT5Backbone
    from despamo.training.baseline_module import SpaMoBaselineModule

    class Language(torch.nn.Linear):
        def __init__(self):
            super().__init__(2048, 2)

        def target_embeddings(self, texts):
            return self.weight.T[None].expand(len(texts), -1, -1), None

        def translation_loss(self, visual, mask, prompts, texts):
            return self(visual).square().mean()

        def generate_text(self, visual, mask, prompts, *decoder):
            # Prediction depends on fusion values, not a stubbed scorer.
            value = self(visual).mean().item()
            return [f"synthetic sentence {value:.6f}."] * len(prompts)

    class Align(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(2048, 2)

        def forward(self, visual, mask, text, text_mask, pooling):
            return (self.project(visual).mean(1) - text.mean(1)).square().mean()

    tiny = TinyPilot(arm="aligned_learned")
    base = SpaMoBaselineModule(tiny.visual_adapter, cast(FlanT5Backbone, Language()),
                              cast(VTAlignLoss, Align()), "Translate {}", False,
                              0, "mean", 1.0, 2000, 3e-4, 0.01, 0)
    model = PilotBaselineModule(base, 16000)
    physical = batch()
    b = SimpleNamespace(spatial=physical[0], spatial_mask=physical[1], motion=physical[2],
                        motion_mask=physical[3], clip_ids=["one", "two"],
                        texts=["positive sentence.", "negative sentence."])
    result = runtime.fusion_gradient_smoke(model, b, "aligned_learned")
    assert result["second_gradient_norms"]["gate"] > 0
    first = runtime.score_pilot_batches(model, [b], ("one", "two"), "dev", "synthetic")
    with torch.no_grad():
        corrector = cast(torch.nn.Sequential, model.visual_adapter.corrector)
        cast(torch.nn.Linear, corrector[-1]).bias.fill_(20)
    second = runtime.score_pilot_batches(model, [b], ("one", "two"), "dev", "synthetic")
    assert first["items"] != second["items"]
    # Verify real scorer forwards through fusion by capturing downstream numeric change.
    assert first["decoding"] == second["decoding"] == {
        "mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False,
    }


def test_dev_statistics_must_match_admitted_timing_coverage(runtime, prepared):
    output, identity, _, ids = completed_run(runtime, prepared)
    report = read_json(output / "dev-5500.json")
    report["fusion_stats"]["supported_frame_count"] -= 1
    report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
    with pytest.raises(ValueError, match="coverage"):
        runtime._validate_report(output, report, 5500, identity, ids)


def test_extractor_helper_bytes_need_actual_source_identity(runtime, tmp_path):
    _, _, timing = synthetic_timing(tmp_path)
    with pytest.raises(ValueError, match="extractor"):
        runtime._validate_extraction_sources(timing, {"one_source": "1" * 64})
    runtime._validate_extraction_sources(timing, {"synthetic_source": "c" * 64})


def test_hardware_profile_measures_driver_vram_and_capability(runtime, monkeypatch):
    monkeypatch.setattr(runtime.subprocess, "check_output", lambda *a, **kw:
                        "Synthetic GPU, 8.9, 24564, 610.62, GPU-fixture\n")
    result = runtime.hardware_identity()
    assert result["gpu_name"] == "Synthetic GPU"
    assert result["vram_mib"] == 24564
    assert result["capability"] == "8.9"
    assert result["driver"] == "610.62"
    assert result["gpu_uuid"] == "GPU-fixture"


def test_rehashed_profile_cannot_change_verified_seed0_base(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    profile = read_json(args.initialization_profile)
    profile["seeds"]["0"]["initial_base_hash"] = "0" * 64
    profile["profile_hash"] = digest({k: v for k, v in profile.items() if k != "profile_hash"})
    atomic_json(args.initialization_profile, profile)
    with pytest.raises(ValueError, match="seed0"):
        runtime.prepare_fusion(args)


def test_factory_verifies_original_hash_before_any_replacement(runtime, monkeypatch):
    def factory(config, steps, seed):
        model = TinyPilot(seed)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model.visual_adapter = SpaMoVisualAdapter(2048, 1024, 768, 2048)
        return model

    monkeypatch.setattr(runtime, "build_seeded_projector_model", factory)
    initial = shared_tensor_hash(factory(None, 16000, 1))
    models = [runtime.build_fusion_model(None, arm, 1, initial) for arm in runtime.FUSION_ARMS]
    assert len({runtime.common_plain_hash(m) for m in models}) == 1
    monkeypatch.setattr(runtime, "_attach_plain",
                        lambda *a: pytest.fail("replaced before hash check"))
    with pytest.raises(ValueError, match="original base"):
        runtime.build_fusion_model(None, "aligned_learned", 1, "0" * 64)


def test_terminal_recovery_publishes_missing_final_dev(runtime, prepared, monkeypatch):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    old_report = read_json(output / "dev-16000.json")
    (output / "dev-16000.json").unlink()
    model = TinyPilot()
    source = output / "step-16000.ckpt"
    torch.save({"global_step": 16000, "prompt_rng_state": model.rng.getstate(),
                "state_dict": model.state_dict(), "pilot_sampler": {"batches_yielded": 16000},
                "pilot_rng": capture_rng(), "pilot_identity": identity, "run_metadata": identity,
                "optimizer_states": [{"saved": True}], "lr_schedulers": [{"saved": True}],
                "gpu_seconds_cumulative": 1.0}, source)
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    checkpoint.update(initialization_hash=file_hash(output / "initialization.json"),
                      fusion_telemetry=read_json(output / "budget.json")["fusion_telemetry"])
    torch.save(checkpoint, source)
    ledger = SimpleNamespace(identity=identity, last=10.0, charge=lambda **kw: None,
                             require_remaining=lambda *a, **kw: None)
    dev = runtime.FusionDevCheckpoints([], ids, protocol["split_hash"], output, 16000, 2000, 4,
                                       identity["variant"], ledger=ledger,
                                       checkpoint_schedule=(5500, 10400, 16000))

    class Loader:
        generator = torch.Generator().manual_seed(0)

        def __iter__(self):
            yield "synthetic physical batch"

    dev.loader = Loader()

    def bounded_generation(m, batches, expected, split, split_hash, on_batch_end):
        # Replace external generation cost only; retain callback aggregation/publication.
        assert list(batches) == ["synthetic physical batch"]
        on_batch_end(582)
        return {k: old_report[k] for k in ("items", "metrics", "split", "split_hash", "decoding")}

    monkeypatch.setattr(runtime, "score_pilot_batches", bounded_generation)
    monkeypatch.setattr(runtime, "_batch_frame_stats", lambda *a: old_report["fusion_stats"])
    monkeypatch.setattr(runtime, "fusion_visual_token_lengths",
                        lambda *a: old_report["visual_token_lengths"])
    model.configure_optimizers = lambda: pytest.fail("terminal optimizer")
    runtime._recover_terminal_dev(model, SimpleNamespace(load_state_dict=lambda s: None), dev,
                                  source, output, identity, ledger)
    report = read_json(output / "dev-16000.json")
    assert report["checkpoint_hash"] == file_hash(source)
    assert tuple(item["clip_id"] for item in report["items"]) == ids
    assert report["fusion_stats"]["clip_count"] == 582
    assert dev.completed == {5500, 10400, 16000}


def test_unknown_status_cannot_be_adopted_or_rewritten(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    atomic_json(output / "run-status.json", {"status": "other_policy", "global_step": 0,
                                           "run_key": output.name})
    atomic_json(output / "budget.json", {"identity": identity, "gpu_seconds": 100.0,
                "optimizer_step": 500, "ceiling_seconds": 10800, "status": "closed"})
    args.resume_checkpoint = output / "resume-500.ckpt"
    before = (output / "run-status.json").read_bytes()
    with pytest.raises(ValueError, match="unknown fusion status"):
        with runtime.owned_fusion_ledger(args, identity, output):
            pytest.fail("unknown status accepted")
    assert (output / "run-status.json").read_bytes() == before


def test_smoke_mode_cannot_relabel_production_resume(runtime, prepared):
    args, _, _, _ = prepared
    args.resume_checkpoint = args.output_base / "resume-500.ckpt"
    with pytest.raises(ValueError, match="smoke.*resume"):
        runtime._execute(args, smoke_only=True)


@pytest.mark.parametrize("arm", ["reference", "aligned_fixed", "aligned_learned"])
def test_initialization_receipt_covers_every_trainable_parameter(runtime, prepared, arm):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    args.arm = arm
    identity, output, *_ = runtime.prepare_fusion(args)
    model = TinyPilot(arm=arm)
    output.mkdir()
    receipt = runtime.ensure_fusion_initialization(output, model, identity, existing=False)
    counts = receipt["trainable_parameters"]
    assert counts["total"] == sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert counts["adapter"] == sum(p.numel() for p in model.visual_adapter.parameters())
    assert counts["vt"] == 4098
    assert counts["flan"] == 4098
    assert counts["adapter_vt"] == counts["adapter"] + 4098
    assert counts["total"] == counts["adapter_vt"] + counts["flan"]
    assert receipt["initial_variant_hash"] == shared_tensor_hash(model)
    assert receipt["common_plain_hash"] == identity["common_plain_hash"]
    assert receipt["run_identity_hash"] == digest(identity)
    before = (output / "initialization.json").read_bytes()
    runtime.ensure_fusion_initialization(output, model, identity, existing=True)
    assert (output / "initialization.json").read_bytes() == before
    with torch.no_grad():
        model.visual_adapter.motion_projector.weight.add_(1)
    with pytest.raises(ValueError, match="initialization"):
        runtime.ensure_fusion_initialization(output, model, identity, existing=True)


def test_initialization_inventory_rejects_unowned_trainables(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    model = TinyPilot()
    model.extra = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="unowned"):
        runtime.fusion_initialization(model, identity)
    assert not (output / "initialization.json").exists()


def test_initialization_adoption_requires_immutable_receipt(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    with pytest.raises(ValueError, match="initialization"):
        runtime.ensure_fusion_initialization(output, TinyPilot(), identity, existing=True)
    receipt = runtime.ensure_fusion_initialization(output, TinyPilot(), identity, existing=False)
    receipt["trainable_parameters"]["total"] -= 1
    receipt["content_hash"] = digest({k: v for k, v in receipt.items() if k != "content_hash"})
    atomic_json(output / "initialization.json", receipt)
    with pytest.raises(ValueError, match="initialization"):
        runtime.ensure_fusion_initialization(output, TinyPilot(), identity, existing=True)


def test_runtime_lengths_are_measured_from_actual_output_mask(runtime):
    class Adapter(torch.nn.Module):
        def forward(self, spatial, smask, motion, mmask):
            mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
            return spatial.new_zeros(2, 4, 3), mask

    model = TinyPilot()
    model.visual_adapter = cast(SpaMoVisualAdapter, Adapter())
    physical = batch()
    b = SimpleNamespace(spatial=physical[0], spatial_mask=physical[1], motion=physical[2],
                        motion_mask=physical[3], clip_ids=["one", "two"])
    assert runtime.fusion_visual_token_lengths(model, b) == [
        {"clip_id": "one", "valid_tokens": 2, "padded_tokens": 4},
        {"clip_id": "two", "valid_tokens": 3, "padded_tokens": 4},
    ]


def test_peak_maxima_persist_cpu_and_simulated_cuda_without_initializing_gpu(runtime, monkeypatch):
    class DeviceModel:
        def parameters(self):
            yield SimpleNamespace(device=torch.device("cuda:0"))

    values = [100, 160]
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda device: values[0])
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda device: values[1])
    state = runtime.capture_fusion_peak(DeviceModel(), "training")
    assert state["training"]["peak_allocated_bytes"] == 100
    assert state["training"]["peak_reserved_bytes"] == 160
    assert state["dev"]["peak_allocated_bytes"] is None
    values[:] = [30, 80]
    state = runtime.capture_fusion_peak(DeviceModel(), "training", state)
    assert state["training"]["peak_allocated_bytes"] == 100
    assert state["training"]["peak_reserved_bytes"] == 160
    assert state["training"]["sample_count"] == 2
    cpu = runtime.capture_fusion_peak(TinyPilot(), "dev")
    assert cpu["dev"]["device_type"] == "cpu"
    assert cpu["dev"]["peak_allocated_bytes"] == cpu["dev"]["peak_reserved_bytes"] == 0
    assert cpu["training"]["device_type"] is None


def test_peak_recovery_does_not_drop_durable_budget_maxima(runtime):
    state = runtime.capture_fusion_peak(TinyPilot(), "training")
    saved = copy.deepcopy(state)
    state["training"].update(device_type="cuda", peak_allocated_bytes=300,
                             peak_reserved_bytes=400,
                             evidence_scope="process_allocator_high_water_sampled_during_phase")
    saved["training"].update(device_type="cuda", peak_allocated_bytes=100,
                             peak_reserved_bytes=200,
                             evidence_scope="process_allocator_high_water_sampled_during_phase")
    restored = runtime.merge_fusion_telemetry(state, saved)
    assert restored["training"]["peak_allocated_bytes"] == 300
    assert restored["training"]["peak_reserved_bytes"] == 400
    bad = copy.deepcopy(saved)
    bad["training"]["peak_reserved_bytes"] = 99
    with pytest.raises(ValueError, match="telemetry"):
        runtime.merge_fusion_telemetry(state, bad)


def test_training_guard_embeds_and_recovers_peak_checkpoint_metadata(runtime, tmp_path):
    identity = {"run_policy": runtime.RUN_POLICY, "protocol_hash": "synthetic",
                "gpu_cap_seconds": 10800}
    ledger = runtime.FusionLedger(tmp_path, identity, 0, 0, lambda: 0.0, lambda: 0.0)
    ledger.initialization_hash = "a" * 64
    guard = runtime.FusionRuntimeGuard(ledger)
    model = TinyPilot()
    trainer = SimpleNamespace(global_step=500, datamodule=SimpleNamespace(state_dict=lambda: {}))
    checkpoint = {}
    guard.on_save_checkpoint(trainer, model, checkpoint)
    assert checkpoint["fusion_telemetry"]["training"]["device_type"] == "cpu"
    assert checkpoint["initialization_hash"] == "a" * 64
    assert read_json(tmp_path / "budget.json")["fusion_telemetry"] == checkpoint["fusion_telemetry"]
    assert guard.ledger.telemetry["training"]["sample_count"] > 0


def test_selection_binds_initialization_and_peak_claims(runtime, prepared):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    # Old completion records must not silently qualify as measured telemetry.
    assert runtime.validate_fusion_selection(output, protocol, ids, identity)["initialization"][
        "trainable_parameters"]["total"] > 0
    path = output / "initialization.json"
    atomic_json(path, {"tampered": True})
    with pytest.raises(ValueError, match="initialization"):
        runtime.validate_fusion_selection(output, protocol, ids, identity)


def test_selection_rejects_rehashed_budget_peak_changes(runtime, prepared):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    budget = read_json(output / "budget.json")
    budget["fusion_telemetry"]["training"]["sample_count"] += 1
    atomic_json(output / "budget.json", budget)
    with pytest.raises(ValueError, match="telemetry"):
        runtime.validate_fusion_selection(output, protocol, ids, identity)


def test_saved_checkpoint_sizes_use_real_stat_and_no_smoke_guess(runtime, tmp_path):
    assert runtime.fusion_checkpoint_sizes(tmp_path) == {}
    path = tmp_path / "resume-500.ckpt"
    torch.save({"synthetic": torch.ones(17)}, path)
    assert runtime.fusion_checkpoint_sizes(tmp_path) == {path.name: path.stat().st_size}


def test_measured_length_report_rejects_reordered_ids_after_rehash(runtime, prepared):
    output, identity, _, ids = completed_run(runtime, prepared)
    report = read_json(output / "dev-5500.json")
    report["visual_token_lengths"].reverse()
    report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
    with pytest.raises(ValueError, match="token lengths"):
        runtime._validate_report(output, report, 5500, identity, ids)


def test_smoke_adoption_rejects_receipt_byte_tampering_before_budget_write(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    runtime.ensure_fusion_initialization(output, TinyPilot(), identity, existing=False)
    atomic_json(output / "budget.json", {"identity": identity, "gpu_seconds": 100.0,
                "optimizer_step": 0, "ceiling_seconds": 10800, "status": "closed",
                "initialization_hash": file_hash(output / "initialization.json"),
                "fusion_telemetry": runtime._empty_fusion_telemetry()})
    atomic_json(output / "run-status.json", {"status": "smoke_passed", "global_step": 0,
                                           "run_key": output.name})
    before = (output / "budget.json").read_bytes()
    path = output / "initialization.json"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="initialization byte binding"):
        with runtime.owned_fusion_ledger(args, identity, output):
            pytest.fail("tampered receipt admitted")
    assert (output / "budget.json").read_bytes() == before


def test_durable_peak_maxima_survive_real_ledger_resume_and_checkpoint_load(runtime, prepared):
    args, _, _, _ = prepared
    runtime.initialize_fusion(args)
    identity, output, *_ = runtime.prepare_fusion(args)
    output.mkdir()
    model = TinyPilot()
    runtime.ensure_fusion_initialization(output, model, identity, existing=False)
    init_hash = file_hash(output / "initialization.json")
    state = runtime.capture_fusion_peak(model, "training")
    state["training"].update(device_type="cuda", peak_allocated_bytes=300,
                             peak_reserved_bytes=400,
                             evidence_scope="process_allocator_high_water_sampled_during_phase")
    atomic_json(output / "budget.json", {"identity": identity, "gpu_seconds": 10.0,
                "optimizer_step": 500, "ceiling_seconds": 10800, "status": "closed",
                "initialization_hash": init_hash, "fusion_telemetry": state})
    ledger = runtime.FusionLedger.resume(output, identity, 1.0, checkpoint_step=500)
    saved = copy.deepcopy(state)
    saved["training"].update(peak_allocated_bytes=100, peak_reserved_bytes=200)
    checkpoint = {"pilot_identity": identity, "gpu_seconds_cumulative": 1.0,
                  "pilot_sampler": {}, "pilot_rng": capture_rng(), "initialization_hash": init_hash,
                  "fusion_telemetry": saved}
    trainer = SimpleNamespace(datamodule=SimpleNamespace(load_state_dict=lambda s: None))
    runtime.FusionRuntimeGuard(ledger).on_load_checkpoint(trainer, model, checkpoint)
    ledger.close()
    receipt = read_json(output / "budget.json")
    assert receipt["fusion_telemetry"]["training"]["peak_allocated_bytes"] == 300
    assert receipt["fusion_telemetry"]["training"]["peak_reserved_bytes"] == 400
    assert receipt["initialization_hash"] == init_hash


def test_completed_run_cannot_claim_unobserved_training_peak(runtime, prepared):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    budget = read_json(output / "budget.json")
    budget["fusion_telemetry"]["training"] = runtime._empty_fusion_telemetry()["training"]
    atomic_json(output / "budget.json", budget)
    # Keep local hashes consistent; missing measurement itself must reject completion.
    selected = read_json(output / "selected-checkpoint.json")
    selected["fusion_telemetry"] = budget["fusion_telemetry"]
    selected["fusion_telemetry_hash"] = digest(budget["fusion_telemetry"])
    atomic_json(output / "selected-checkpoint.json", selected)
    status = read_json(output / "run-status.json")
    status["fusion_telemetry_hash"] = selected["fusion_telemetry_hash"]
    atomic_json(output / "run-status.json", status)
    for step in (5500, 10400, 16000):
        path = output / f"step-{step}.ckpt"
        saved = torch.load(path, map_location="cpu", weights_only=False)
        saved["fusion_telemetry"] = budget["fusion_telemetry"]
        torch.save(saved, path)
        report_path = output / f"dev-{step}.json"
        report = read_json(report_path)
        report["checkpoint_hash"] = file_hash(path)
        report["checkpoint_size_bytes"] = path.stat().st_size
        report["content_hash"] = digest({k: v for k, v in report.items() if k != "content_hash"})
        atomic_json(report_path, report)
    selected["checkpoint_hash"] = file_hash(output / "step-10400.ckpt")
    selected["dev_report_hashes"] = runtime._dev_report_hashes(output)
    selected["checkpoint_sizes_bytes"] = runtime.fusion_checkpoint_sizes(output)
    atomic_json(output / "selected-checkpoint.json", selected)
    budget["checkpoint_sizes_bytes"] = selected["checkpoint_sizes_bytes"]
    atomic_json(output / "budget.json", budget)
    with pytest.raises(ValueError, match="unobserved training"):
        runtime.validate_fusion_selection(output, protocol, ids, identity)


def test_dev_telemetry_cap_failure_still_restores_modes_and_rng(runtime, prepared, monkeypatch):
    output, identity, protocol, ids = completed_run(runtime, prepared)
    template = read_json(output / "dev-5500.json")
    model = TinyPilot()
    model.language_model.eval()
    modes = [m.training for m in model.modules()]
    prompt = model.rng.getstate()
    rng = capture_rng()

    def charge(**kwargs):
        if kwargs["phase"] == "dev_telemetry_finalized":
            raise RuntimeError("synthetic cap exhausted")

    ledger = SimpleNamespace(identity=identity, charge=charge,
                             require_remaining=lambda *a, **kw: None)
    dev = runtime.FusionDevCheckpoints([], ids, protocol["split_hash"], output, 16000, 2000, 4,
                                       identity["variant"], ledger=ledger,
                                       checkpoint_schedule=(5500, 10400, 16000))

    class Loader:
        generator = torch.Generator().manual_seed(0)

        def __iter__(self):
            yield "synthetic"

    dev.loader = Loader()

    def score(m, batches, *args, **kwargs):
        list(batches)
        model.rng.random()
        random.random()
        np.random.rand()
        torch.rand(1)
        return {k: template[k] for k in ("items", "metrics", "split", "split_hash", "decoding")}

    monkeypatch.setattr(runtime, "score_pilot_batches", score)
    monkeypatch.setattr(runtime, "_batch_frame_stats", lambda *a: template["fusion_stats"])
    monkeypatch.setattr(runtime, "fusion_visual_token_lengths",
                        lambda *a: template["visual_token_lengths"])
    with pytest.raises(RuntimeError, match="synthetic cap"):
        dev._score_saved(model, 5500, output / "step-5500.ckpt")
    assert [m.training for m in model.modules()] == modes
    assert model.rng.getstate() == prompt
    after = capture_rng()
    assert after["python"] == rng["python"]
    assert np.array_equal(after["numpy"][1], rng["numpy"][1])
    assert torch.equal(after["torch"], rng["torch"])

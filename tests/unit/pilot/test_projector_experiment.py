import copy
import os
import random
import stat
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.provenance import read_json
from despamo.models.residual_projector import enable_residual_projectors
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.e1_budget import E1Ledger
from scripts import train_projector_pilot as experiment
from scripts import train_signer_pilot
from scripts.train_projector_pilot import (
    BoundedResumeSnapshots,
    build_projector_identity,
    projector_checkpoint_steps,
    projector_gradient_smoke,
)


def test_projector_schedule_supports_declared_long_horizon_without_relaxing_frozen_tiers():
    assert projector_checkpoint_steps(8000) == (3500, 5600, 8000)
    assert projector_checkpoint_steps(16000) == (5500, 10400, 16000)
    with pytest.raises(ValueError):
        train_signer_pilot.checkpoint_steps(16000, 2000)
    with pytest.raises(ValueError):
        projector_checkpoint_steps(12000)


def test_explicit_long_dev_schedule_scores_its_actual_checkpoint(tmp_path, monkeypatch):
    callback = train_signer_pilot.FullDevCheckpoints(
        ["placeholder"], ("dev-id",), "split", tmp_path, 16000, 2000, 4,
        "E3_projector_linear", checkpoint_schedule=(5500, 10400, 16000),
    )
    monkeypatch.setattr(train_signer_pilot, "score_pilot_batches", lambda *a, **k: {
        "items": [{"clip_id": "dev-id"}], "metrics": {"bleu4": 1.0},
    })
    trainer = SimpleNamespace(global_step=5500,
                              save_checkpoint=lambda path: path.write_bytes(b"checkpoint"))
    callback.on_train_batch_end(trainer, SimpleNamespace(train=lambda: None), None, None, 0)
    assert read_json(tmp_path / "dev-5500.json")["checkpoint_step"] == 5500


def test_retention_keeps_latest_two_resumes_and_all_dev_checkpoints(tmp_path):
    root = tmp_path / "run"
    ledger = E1Ledger.create(root, {"protocol_hash": "protocol",
                                  "run_policy": "signer-pilot-projector-ablation-v1"}, 0)
    callback = BoundedResumeSnapshots(root, ledger, keep=2)
    (root / "step-3500.ckpt").write_bytes(b"selected-dev")
    trainer = SimpleNamespace(global_step=0,
                              save_checkpoint=lambda path: path.write_bytes(b"full-state"))
    for step in (500, 1000, 1500, 2000):
        trainer.global_step = step
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert sorted(path.name for path in root.glob("resume-*.ckpt")) == [
        "resume-1500.ckpt", "resume-2000.ckpt",
    ]
    assert (root / "step-3500.ckpt").read_bytes() == b"selected-dev"
    assert (root / "resume-2000.ckpt").read_bytes() == b"full-state"


def test_failed_snapshot_write_does_not_prune_recovery_points(tmp_path):
    root = tmp_path / "run"
    ledger = E1Ledger.create(root, {"protocol_hash": "protocol",
                                  "run_policy": "signer-pilot-projector-ablation-v1"}, 0)
    callback = BoundedResumeSnapshots(root, ledger, keep=2)
    for step in (500, 1000):
        (root / f"resume-{step}.ckpt").write_bytes(b"saved")

    def fail(path):
        raise OSError("disk write failed")

    trainer = SimpleNamespace(global_step=1500, save_checkpoint=fail)
    with pytest.raises(OSError, match="disk write"):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert sorted(path.name for path in root.glob("resume-*.ckpt")) == [
        "resume-1000.ckpt", "resume-500.ckpt",
    ]


def test_retention_refuses_historical_run_policy(tmp_path):
    root = tmp_path / "old-run"
    ledger = E1Ledger.create(root, {"protocol_hash": "protocol", "run_policy": "old-policy"}, 0)
    (root / "resume-500.ckpt").write_bytes(b"historical")
    with pytest.raises(ValueError, match="owned projector"):
        BoundedResumeSnapshots(root, ledger)
    assert (root / "resume-500.ckpt").read_bytes() == b"historical"


def test_projector_identity_separates_architecture_horizon_and_retention():
    config = {"seed": 0, "model": {"warm_up_steps": 2000},
              "data": {"batch_size": 4, "num_workers": 0},
              "trainer": {"max_steps": 8000, "accumulate_grad_batches": 1}}
    protocol = {"protocol_hash": "protocol", "split_hash": "split"}
    adapted = {"checkpoint_hash": "adaptation", "encoder_key": "encoder",
               "manifest_hash": "manifest", "frame_rows_hash": "rows", "clips": 7096}
    base = build_projector_identity(protocol, adapted, config, "residual", 8000, 10800,
                                    "baseline", {"runner": "code"})
    control_config = copy.deepcopy(config)
    control_config["trainer"]["max_steps"] = 16000
    control = build_projector_identity(protocol, adapted, control_config, "linear", 16000, 14400,
                                       "baseline", {"runner": "code"})
    assert base["run_key"] != control["run_key"]
    assert base["resume_retention"] == 2 and base["residual_hidden_dim"] == 1024
    assert base["supervision_policy"] == "qwen-schema98-unreviewed-v1"
    assert base["human_review_status"] == "not_assessed"
    changed = build_projector_identity(protocol, adapted, config, "residual", 8000, 10800,
                                       "baseline", {"runner": "changed"})
    assert changed["run_key"] != base["run_key"]
    with pytest.raises(ValueError):
        build_projector_identity(protocol, adapted, config, "linear", 8000, 10800,
                                 "baseline", {"runner": "code"})


@pytest.mark.parametrize("fail", [False, True])
def test_joint_gradient_smoke_restores_weights_buffers_rng_and_clears_gradients(fail):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.visual_adapter = SpaMoVisualAdapter(4, 3, 8, 6)
            enable_residual_projectors(self.visual_adapter, hidden_dim=12)
            self.language_model = torch.nn.Linear(6, 6)
            self.rng = random.Random(17)
            self.vt_weight = 1.0

        def compute_losses(self, batch):
            self.rng.random()
            tokens, mask = self.visual_adapter(*batch)
            vt = tokens[mask].square().mean() * torch.rand(())
            loss = self.language_model(tokens[mask]).square().mean()
            return {"loss": loss * float("nan") if fail else loss, "contra_loss": vt}

    torch.manual_seed(22)
    model = Model().eval()
    batch = (torch.randn(2, 20, 4), torch.ones(2, 20, dtype=torch.bool),
             torch.randn(2, 8, 3), torch.ones(2, 8, dtype=torch.bool))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng, prompt_rng = torch.get_rng_state().clone(), model.rng.getstate()
    if fail:
        with pytest.raises(FloatingPointError):
            projector_gradient_smoke(model, batch)
    else:
        report = projector_gradient_smoke(model, batch)
        assert report["status"] == "passed"
        assert report["gradient_norms"]["spatial"] > 0
        assert report["gradient_norms"]["motion"] > 0
        assert report["gradient_norms"]["language"] > 0
    assert torch.equal(torch.get_rng_state(), rng)
    assert model.rng.getstate() == prompt_rng
    assert model.training is False
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    assert all(param.grad is None for param in model.parameters())


def test_late_checkpoint_flush_failure_preserves_previous_recovery_points(tmp_path, monkeypatch):
    root = tmp_path / "run"
    ledger = E1Ledger.create(root, {"protocol_hash": "protocol",
                                  "run_policy": "signer-pilot-projector-ablation-v1"}, 0)
    callback = BoundedResumeSnapshots(root, ledger)
    for step in (500, 1000):
        (root / f"resume-{step}.ckpt").write_bytes(b"durable")
    original_fsync = os.fsync

    def fail_checkpoint_flush(fd):
        path = os.readlink(f"/proc/self/fd/{fd}")
        if stat.S_ISREG(os.fstat(fd).st_mode) and "resume-1500.ckpt" in path:
            raise OSError("late checkpoint writeback failed")
        original_fsync(fd)

    monkeypatch.setattr(experiment.os, "fsync", fail_checkpoint_flush)
    trainer = SimpleNamespace(global_step=1500,
                              save_checkpoint=lambda path: path.write_bytes(b"new-checkpoint"))
    with pytest.raises(OSError, match="writeback"):
        callback.on_train_batch_end(trainer, None, None, None, 0)
    assert (root / "resume-500.ckpt").read_bytes() == b"durable"
    assert (root / "resume-1000.ckpt").read_bytes() == b"durable"
    assert not (root / "resume-1500.ckpt").exists()


@pytest.mark.parametrize("refusal", ["completed", "lock"])
def test_rejected_resume_does_not_write_run_status_or_budget(tmp_path, monkeypatch, refusal):
    root = tmp_path / "existing"
    root.mkdir()
    (root / "run-status.json").write_text('{"status":"complete"}')
    (root / "budget.json").write_text('{"status":"closed"}')
    (root / "selected-checkpoint.json").write_text('{"step":8000}')
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    args = SimpleNamespace(
        authorize_projector_run=True, resume_checkpoint=root / "resume-8000.ckpt",
        output_base=tmp_path, gpu_cap_seconds=10800,
    )
    monkeypatch.setattr(experiment, "prepare_projector", lambda a: ({}, root, {}, {}, {}))
    monkeypatch.setattr(experiment.shutil, "disk_usage",
                        lambda p: SimpleNamespace(free=100 * 2**30))
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(experiment.torch.cuda, "mem_get_info", lambda: (16 * 2**30, 16 * 2**30))

    @contextmanager
    def lock(path):
        if refusal == "lock":
            raise RuntimeError("writer lock refused")
        yield

    monkeypatch.setattr(experiment, "writer", lock)
    with pytest.raises((ValueError, RuntimeError), match="completed|lock"):
        experiment.run_projector(args)
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


def test_actual_code_manifest_includes_reused_storage_modules(tmp_path):
    files = ["scripts/train_projector_pilot.py", "scripts/train_signer_pilot.py",
             "scripts/train_e3_pilot.py", "scripts/evaluate_e3_pilot.py", "pyproject.toml",
             "uv.lock", "tools/dinov3/storage.py", "tools/dinov3/identity.py"]
    for name in files:
        p = tmp_path / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(name)
    first = experiment.projector_code_hashes(tmp_path)
    (tmp_path / "tools/dinov3/storage.py").write_text("changed locking implementation")
    second = experiment.projector_code_hashes(tmp_path)
    assert first["tools/dinov3/storage.py"] != second["tools/dinov3/storage.py"]
    assert "tools/dinov3/identity.py" in second

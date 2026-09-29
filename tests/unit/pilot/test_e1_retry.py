"""Fresh E1 retry preserves failed first run, WSL worker count, and total budget."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.provenance import atomic_json, file_hash, read_json
from despamo.training.e1_budget import E1Ledger
from scripts import retrain_e1_wsl
from scripts.retrain_e1_wsl import build_retry, charge_loaded_model, run_retry
from scripts.train_signer_pilot import ROOT


def original_run(tmp_path):
    old = tmp_path / "e1-only-old"
    files = (
        "scripts/train_signer_pilot.py",
        "src/despamo/training/e1_budget.py",
        "src/despamo/training/pilot_factory.py",
        "src/despamo/training/pilot_replay.py",
        "src/despamo/training/pilot_runtime.py",
        "src/despamo/training/baseline_module.py",
        "src/despamo/evaluation/signer_pilot.py",
        "uv.lock",
    )
    identity = {
        "protocol_hash": "frozen",
        "run_policy": "signer-pilot-e1-only-bounded-v1",
        "code_hashes": {name: file_hash(ROOT / name) for name in files},
    }
    ledger = E1Ledger.create(old, identity, 1646.1)
    ledger.close()
    atomic_json(old / "run-status.json", {"status": "partial"})
    return old


def arguments(tmp_path):
    return SimpleNamespace(
        output_base=tmp_path,
        physical_batch=4,
        hf_cache=Path("/cache"),
        dino_root=Path("/dino"),
        motion_root=Path("/motion"),
        motion_manifest=Path("/motion.json"),
        annotation=Path("/annotations/train_info_ml.npy"),
    )


def test_retry_binds_old_closed_ledger_but_starts_new_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        retrain_e1_wsl.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    parent = original_run(tmp_path)
    original_bytes = (parent / "budget.json").read_bytes()
    identity, output = build_retry(arguments(tmp_path), {"protocol_hash": "frozen"}, parent)
    assert identity["run_policy"] == "signer-pilot-e1-fresh-retry-v2"
    assert identity["prior_gpu_seconds"] == read_json(parent / "budget.json")["gpu_seconds"]
    assert identity["parent_budget_hash"] == file_hash(parent / "budget.json")
    assert identity["parent_run_key"] == parent.name
    assert output.parent == tmp_path and output.name.startswith("e1-retry-")
    assert identity["config"]["data"]["num_workers"] == 0
    assert identity["config"]["data"]["batch_size"] == 4
    assert identity["config"]["trainer"]["max_steps"] == 4000
    assert (parent / "budget.json").read_bytes() == original_bytes
    assert not output.exists()
    output.mkdir()
    assert (
        build_retry(arguments(tmp_path), {"protocol_hash": "frozen"}, parent, resume=True)[1]
        == output
    )


def test_retry_rejects_active_or_foreign_old_run_and_batch2(tmp_path):
    parent = original_run(tmp_path)
    args = arguments(tmp_path)
    args.physical_batch = 2
    with pytest.raises(ValueError, match="batch"):
        build_retry(args, {"protocol_hash": "frozen"}, parent)
    args.physical_batch = 4
    recorded = read_json(parent / "budget.json")
    recorded["status"] = "running"
    atomic_json(parent / "budget.json", recorded)
    with pytest.raises(ValueError, match="closed"):
        build_retry(args, {"protocol_hash": "frozen"}, parent)
    recorded["status"] = "closed"
    atomic_json(parent / "budget.json", recorded)
    with pytest.raises(ValueError, match="protocol"):
        build_retry(args, {"protocol_hash": "other"}, parent)


def test_loaded_model_accounting_uses_resume_step_without_reset(tmp_path):
    ledger = E1Ledger.create(tmp_path / "new", {"protocol_hash": "frozen"}, 100.0)
    ledger.charge(step=1000, phase="checkpoint")
    assert charge_loaded_model(ledger) >= 100.0
    assert read_json(ledger.root / "budget.json")["optimizer_step"] == 1000


def test_retry_starts_fresh_step_zero_without_loading_predecessor(tmp_path, monkeypatch):
    parent = original_run(tmp_path)
    monkeypatch.setattr(
        retrain_e1_wsl.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    monkeypatch.setattr(retrain_e1_wsl, "writer", lambda *_: nullcontext())
    monkeypatch.setattr(retrain_e1_wsl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        retrain_e1_wsl.torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3)
    )
    monkeypatch.setattr(retrain_e1_wsl, "_model", lambda *_: SimpleNamespace(run_metadata=None))
    monkeypatch.setattr(
        retrain_e1_wsl, "shared_tensor_hash", lambda *_: retrain_e1_wsl.E1_SHARED_HASH
    )
    monkeypatch.setattr(retrain_e1_wsl.signal, "setitimer", lambda *_: None)

    class TinyTrainer:
        def __init__(self, **kw):
            self.callbacks = kw["callbacks"]
            self.global_step = 0

        def fit(self, model, datamodule, ckpt_path=None):
            assert ckpt_path is None
            assert datamodule.train_dataloader().num_workers == 0
            self.datamodule = datamodule
            self.global_step = 1
            self.callbacks[0].on_train_batch_end(self, model, None, None, 0)

    monkeypatch.setattr(retrain_e1_wsl.pl, "Trainer", TinyTrainer)
    protocol = {
        "protocol_hash": "frozen",
        "split_hash": "split",
        "split": {"groups": {"dev": {"clip_ids": [str(i) for i in range(582)]}}},
    }
    result = run_retry(
        arguments(tmp_path), protocol, {"train": list(range(5746)), "dev": [0] * 582}, parent
    )
    assert result["status"] == "partial" and result["global_step"] == 1
    new = tmp_path / result["run_key"]
    assert new != parent
    assert (
        read_json(new / "budget.json")["gpu_seconds"]
        >= read_json(parent / "budget.json")["gpu_seconds"]
    )
    assert read_json(parent / "run-status.json")["status"] == "partial"


def test_retry_can_resume_its_own_checkpoint_at_saved_step_without_reset(tmp_path, monkeypatch):
    parent = original_run(tmp_path)
    args = arguments(tmp_path)
    monkeypatch.setattr(
        retrain_e1_wsl.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    identity, runroot = build_retry(args, {"protocol_hash": "frozen"}, parent)
    ledger = E1Ledger.create(runroot, identity, identity["prior_gpu_seconds"])
    ledger.charge(step=500, phase="checkpoint")
    checkpoint = {
        "pilot_identity": identity,
        "run_metadata": identity,
        "global_step": 500,
        "optimizer_states": [{"state": {1: {"step": 500}}}],
        "lr_schedulers": [{"last_epoch": 500}],
        "pilot_rng": {"torch": torch.get_rng_state()},
        "pilot_sampler": {"batches_yielded": 500},
        "gpu_seconds_cumulative": ledger.last,
    }
    saved = runroot / "resume-500.ckpt"
    torch.save(checkpoint, saved)
    ledger.close()
    monkeypatch.setattr(retrain_e1_wsl, "writer", lambda *_: nullcontext())
    monkeypatch.setattr(retrain_e1_wsl.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        retrain_e1_wsl.torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3)
    )
    monkeypatch.setattr(retrain_e1_wsl, "_model", lambda *_: SimpleNamespace(run_metadata=None))
    monkeypatch.setattr(
        retrain_e1_wsl, "shared_tensor_hash", lambda *_: retrain_e1_wsl.E1_SHARED_HASH
    )
    monkeypatch.setattr(retrain_e1_wsl.signal, "setitimer", lambda *_: None)

    class ResumedTrainer:
        def __init__(self, **kw):
            self.callbacks = kw["callbacks"]
            self.global_step = 500

        def fit(self, model, datamodule, ckpt_path=None):
            assert ckpt_path == str(saved)
            self.datamodule = datamodule
            self.global_step = 501
            self.callbacks[0].on_train_batch_end(self, model, None, None, 0)

    monkeypatch.setattr(retrain_e1_wsl.pl, "Trainer", ResumedTrainer)
    args.resume_checkpoint = saved
    protocol = {
        "protocol_hash": "frozen",
        "split_hash": "split",
        "split": {"groups": {"dev": {"clip_ids": [str(i) for i in range(582)]}}},
    }
    result = run_retry(args, protocol, {"train": list(range(5746)), "dev": [0] * 582}, parent)
    assert result["global_step"] == 501
    assert read_json(runroot / "budget.json")["gpu_seconds"] >= checkpoint["gpu_seconds_cumulative"]

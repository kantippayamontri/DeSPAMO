"""Authorized fresh E1 v3 must preserve earlier spend and force WSL workers=0."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from despamo.appearance.provenance import atomic_json, file_hash, read_json
from despamo.training.e1_budget import E1Ledger
from scripts import retrain_e1_authorized
from scripts.retrain_e1_authorized import build_v3, charge_model_loaded, run_v3
from scripts.train_signer_pilot import ROOT


def previous_run(tmp_path, *, status="running", pid=999999):
    parent = tmp_path / "e1-retry-parent"
    code = (
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
        "run_policy": "signer-pilot-e1-fresh-retry-v2",
        "checkpoint_bound_code_hashes": {name: file_hash(ROOT / name) for name in code},
    }
    ledger = E1Ledger.create(parent, identity, 3771.0)
    ledger.charge(step=1000, phase="resume_snapshot_saved")
    record = read_json(parent / "budget.json")
    record.update(status=status, pid=pid)
    atomic_json(parent / "budget.json", record)
    return parent


def args(tmp_path):
    return SimpleNamespace(
        output_base=tmp_path,
        physical_batch=4,
        hf_cache=Path("/cache"),
        dino_root=Path("/dino"),
        motion_root=Path("/motion"),
        motion_manifest=Path("/motion-manifest.json"),
        annotation=Path("/annotations/train_info_ml.npy"),
        resume_checkpoint=None,
        authorize_new_six_hour_run=True,
    )


def test_v3_preserves_old_budget_and_starts_new_ledger_from_zero(tmp_path, monkeypatch):
    parent = previous_run(tmp_path)
    before = (parent / "budget.json").read_bytes()
    monkeypatch.setattr(
        retrain_e1_authorized.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    identity, output = build_v3(args(tmp_path), {"protocol_hash": "frozen"}, parent)
    assert identity["run_policy"] == "signer-pilot-e1-fresh-retry-v3"
    assert (
        identity["prior_recorded_gpu_seconds"] == read_json(parent / "budget.json")["gpu_seconds"]
    )
    assert identity["parent_budget_hash"] == file_hash(parent / "budget.json")
    assert identity["new_run_cap_seconds"] == 21600
    assert identity["overall_cap_seconds"] == 86400
    assert identity["prior_recorded_gpu_seconds"] + 21600 < 86400
    assert identity["config"]["data"]["num_workers"] == 0
    assert identity["config"]["data"]["batch_size"] == 4
    assert identity["config"]["trainer"]["max_steps"] == 4000
    assert output.name.startswith("e1-v3-") and not output.exists()
    new = E1Ledger.create(output, identity, 0.0)
    assert charge_model_loaded(new) < 10.0
    assert (parent / "budget.json").read_bytes() == before


def test_v3_rejects_live_parent_wrong_batch_or_missing_authorization(tmp_path, monkeypatch):
    parent = previous_run(tmp_path)
    monkeypatch.setattr(
        retrain_e1_authorized.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    chosen = args(tmp_path)
    chosen.physical_batch = 2
    with pytest.raises(ValueError, match="batch"):
        build_v3(chosen, {"protocol_hash": "frozen"}, parent)
    chosen.physical_batch = 4
    chosen.authorize_new_six_hour_run = False
    with pytest.raises(ValueError, match="authorization"):
        build_v3(chosen, {"protocol_hash": "frozen"}, parent)
    chosen.authorize_new_six_hour_run = True
    record = read_json(parent / "budget.json")
    record["pid"] = __import__("os").getpid()
    atomic_json(parent / "budget.json", record)
    with pytest.raises(ValueError, match="active"):
        build_v3(chosen, {"protocol_hash": "frozen"}, parent)


def test_model_load_does_not_rewind_resume_step(tmp_path):
    ledger = E1Ledger.create(tmp_path / "fresh", {"protocol_hash": "frozen"}, 0.0)
    ledger.charge(step=500, phase="checkpoint")
    charge_model_loaded(ledger)
    assert read_json(ledger.root / "budget.json")["optimizer_step"] == 500


def test_v3_trains_fresh_step_zero_worker_zero_without_parent_weights(tmp_path, monkeypatch):
    parent = previous_run(tmp_path)
    original_parent = (parent / "budget.json").read_bytes()
    monkeypatch.setattr(
        retrain_e1_authorized.shutil,
        "disk_usage",
        lambda *_: SimpleNamespace(free=200 * 1024**3),
    )
    monkeypatch.setattr(retrain_e1_authorized, "writer", lambda *_: nullcontext())
    monkeypatch.setattr(retrain_e1_authorized.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        retrain_e1_authorized.torch.cuda,
        "mem_get_info",
        lambda: (16 * 1024**3, 16 * 1024**3),
    )
    monkeypatch.setattr(
        retrain_e1_authorized, "_model", lambda *_: SimpleNamespace(run_metadata=None)
    )
    monkeypatch.setattr(
        retrain_e1_authorized,
        "shared_tensor_hash",
        lambda *_: retrain_e1_authorized.E1_SHARED_HASH,
    )
    monkeypatch.setattr(retrain_e1_authorized.signal, "setitimer", lambda *_: None)

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

    monkeypatch.setattr(retrain_e1_authorized.pl, "Trainer", TinyTrainer)
    protocol = {
        "protocol_hash": "frozen",
        "split_hash": "split",
        "split": {"groups": {"dev": {"clip_ids": [str(i) for i in range(582)]}}},
    }
    result = run_v3(
        args(tmp_path), protocol, {"train": list(range(5746)), "dev": [0] * 582}, parent
    )
    assert result["status"] == "partial" and result["global_step"] == 1
    new = tmp_path / result["run_key"]
    assert 0 <= read_json(new / "budget.json")["gpu_seconds"] < 10
    assert (parent / "budget.json").read_bytes() == original_parent

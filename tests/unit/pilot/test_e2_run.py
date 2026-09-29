"""E2-only runner keeps E1 matching, train-only factors and bounded GPU time."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.training.e1_budget import E1Ledger
from despamo.training.pilot_runtime import pilot_config
from scripts import train_e2_pilot
from scripts.train_e2_pilot import build_e2, charge_model_loaded, run_e2
from scripts.train_signer_pilot import E1_SHARED_HASH


def fixtures(tmp_path):
    e1 = tmp_path / "e1-v3"
    e1.mkdir()
    config = OmegaConf.to_container(
        pilot_config(
            tmp_path / "cache",
            tmp_path / "dino",
            tmp_path / "motion",
            tmp_path / "annotations",
            tmp_path / "dino/complete/manifest.json",
            tmp_path / "motion.json",
            4000,
            4,
            tmp_path,
        ),
        resolve=True,
    )
    identity = {
        "run_policy": "signer-pilot-e1-fresh-retry-v3",
        "protocol_hash": "frozen",
        "config": config,
        "initial_shared_hash": E1_SHARED_HASH,
    }
    atomic_json(e1 / "budget.json", {"status": "closed", "identity": identity, "gpu_seconds": 1000})
    atomic_json(
        e1 / "run-status.json",
        {"status": "complete", "global_step": 4000, "overall_recorded_seconds": 4000},
    )
    (e1 / "step-4000.ckpt").write_bytes(b"fake-complete-checkpoint")
    atomic_json(
        e1 / "selected-checkpoint.json",
        {"step": 4000, "checkpoint_hash": file_hash(e1 / "step-4000.ckpt")},
    )
    proto = {
        "protocol_hash": "frozen",
        "split_hash": "split",
        "records_hash": "records",
        "text_manifest_hash": "clip",
        "split": {
            "groups": {
                "train": {"clip_ids": [str(i) for i in range(5746)], "valid": 5641, "failed": 105},
                "dev": {"clip_ids": [f"dev-{i}" for i in range(582)]},
            }
        },
    }
    profile = {
        "status": "complete",
        "protocol_hash": "frozen",
        "variant": "E2_projector",
        "initial_shared_tensor_hash": E1_SHARED_HASH,
        "physical_batch": 4,
        "gpu_seconds": 180.0,
        "joint_seconds_per_step": 0.34,
        "dev_seconds_per_full_split": 5200.0,
        "input_identity": {
            "config": config,
            "factor_provenance": {
                "valid_count": 5641,
                "failed_count": 105,
                "records_hash": "records",
                "text_manifest_hash": "clip",
                "logical_train_ids_hash": digest(
                    tuple(proto["split"]["groups"]["train"]["clip_ids"])
                ),
                "supervision_policy": "qwen-schema98-unreviewed-v1",
                "human_review_status": "not_assessed",
            },
        },
    }
    saved = tmp_path / "profile-E2_projector-b4.json"
    atomic_json(saved, profile)
    args = SimpleNamespace(
        output_base=tmp_path,
        physical_batch=4,
        hf_cache=tmp_path / "cache",
        dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion",
        motion_manifest=tmp_path / "motion.json",
        annotation=tmp_path / "annotations/train_info_ml.npy",
        dataset=Path("/qwen"),
        text_manifest=Path("/clip"),
        resume_checkpoint=None,
        authorize_e2_six_hour_run=True,
    )
    return args, proto, e1, saved


def test_e2_identity_binds_selected_e1_and_complete_train_masks(tmp_path, monkeypatch):
    args, proto, e1, profile = fixtures(tmp_path)
    monkeypatch.setattr(
        train_e2_pilot.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200 * 1024**3)
    )
    identity, output = build_e2(args, proto, e1, profile)
    assert identity["run_policy"] == "signer-pilot-e2-projector-bounded-v1"
    assert identity["variant"] == "E2_projector" and identity["seed"] == 0
    assert identity["initial_shared_hash"] == E1_SHARED_HASH
    assert identity["factor_provenance"]["valid_count"] == 5641
    assert identity["factor_provenance"]["failed_count"] == 105
    assert identity["prior_other_gpu_seconds"] == 4000
    assert identity["e2_profile_gpu_seconds"] == 180
    assert identity["e2_profile_hash"] == file_hash(profile)
    assert output.name.startswith("e2-only-") and not output.exists()
    ledger = E1Ledger.create(output, identity, identity["e2_profile_gpu_seconds"])
    assert charge_model_loaded(ledger) >= 180


def test_e2_rejects_changed_shared_initialization_or_heldout_factor_policy(tmp_path, monkeypatch):
    args, proto, e1, path = fixtures(tmp_path)
    monkeypatch.setattr(
        train_e2_pilot.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200 * 1024**3)
    )
    changed = read_json(path)
    changed["initial_shared_tensor_hash"] = "other"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="initial"):
        build_e2(args, proto, e1, path)
    changed["initial_shared_tensor_hash"] = E1_SHARED_HASH
    changed["input_identity"]["factor_provenance"]["valid_count"] = 5746
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="factor"):
        build_e2(args, proto, e1, path)


def test_e2_refuses_batch2_or_missing_run_authorization(tmp_path):
    args, proto, e1, profile = fixtures(tmp_path)
    args.physical_batch = 2
    with pytest.raises(ValueError, match="batch"):
        build_e2(args, proto, e1, profile)
    args.physical_batch = 4
    args.authorize_e2_six_hour_run = False
    with pytest.raises(ValueError, match="authorization"):
        build_e2(args, proto, e1, profile)


def test_e2_runner_starts_fresh_with_factor_train_only_and_worker_zero(tmp_path, monkeypatch):
    args, protocol, e1, profile = fixtures(tmp_path)
    factor = read_json(profile)["input_identity"]["factor_provenance"]
    monkeypatch.setattr(
        train_e2_pilot.shutil, "disk_usage", lambda *_: SimpleNamespace(free=200 * 1024**3)
    )
    monkeypatch.setattr(train_e2_pilot, "writer", lambda *_: nullcontext())
    monkeypatch.setattr(train_e2_pilot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        train_e2_pilot.torch.cuda, "mem_get_info", lambda: (16 * 1024**3, 16 * 1024**3)
    )
    monkeypatch.setattr(train_e2_pilot, "_train_data", lambda *_: (list(range(5746)), list, factor))
    monkeypatch.setattr(train_e2_pilot, "_model", lambda *_: SimpleNamespace(run_metadata=None))
    monkeypatch.setattr(train_e2_pilot, "shared_tensor_hash", lambda *_: E1_SHARED_HASH)
    monkeypatch.setattr(train_e2_pilot.signal, "setitimer", lambda *_: None)

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

    monkeypatch.setattr(train_e2_pilot.pl, "Trainer", TinyTrainer)
    original_e1 = (e1 / "budget.json").read_bytes()
    result = run_e2(args, protocol, {"train": list(range(5746)), "dev": [0] * 582}, e1, profile)
    assert result["status"] == "partial" and result["global_step"] == 1
    root = tmp_path / result["run_key"]
    saved = read_json(root / "budget.json")
    assert saved["identity"]["factor_provenance"] == factor
    assert saved["identity"]["variant"] == "E2_projector"
    assert saved["gpu_seconds"] >= 180
    assert (e1 / "budget.json").read_bytes() == original_e1

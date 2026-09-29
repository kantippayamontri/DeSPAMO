from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.provenance import read_json
from despamo.data.batch import PhoenixSample
from despamo.training.e1_budget import E1Ledger
from despamo.training.pilot_replay import freeze_tier
from scripts import train_signer_pilot
from scripts.train_signer_pilot import (
    E1ResumeSnapshots,
    FullDevCheckpoints,
    checkpoint_steps,
    dev_profile_ids,
    e1_only_identity,
    e1_only_preflight,
    reject_failed_gate,
    require_frozen_tier,
    run_e1_only,
    validate_e1_resume_checkpoint,
)


def profile_fixture():
    return {
        "e1_seconds_per_step": 1.0,
        "e2_seconds_per_step": 1.5,
        "e3_spamo_seconds_per_step": 1.5,
        "dino_seconds_per_step": 0.3,
        "calibration_seconds_per_step": 0.3,
        "extract_seconds_per_frame": 0.04,
        "dev_seconds_per_full_split": 120.0,
        "final_and_probe_seconds": 1000.0,
    }


def test_frozen_budget_exactly_matches_profile_and_source_protocol():
    proto = "0" * 64
    receipt = {
        **freeze_tier(profile_fixture(), 85.0),
        "protocol_hash": proto,
        "source_profile_hashes": ["a", "b"],
        "physical_batch": 2,
    }
    require_frozen_tier(receipt, proto, 2)
    for change in (
        {"protocol_hash": "other"},
        {"spaMo_steps": 4000},
        {"physical_batch": 4},
        {"source_profile_hashes": []},
    ):
        bad = {**receipt, **change}
        with pytest.raises(ValueError, match="tier|profile"):
            require_frozen_tier(bad, proto, 2)
    bad = deepcopy(receipt)
    bad["profile"].pop("extract_seconds_per_frame")
    with pytest.raises(ValueError, match="tier|profile"):
        require_frozen_tier(bad, proto, 2)


def test_dev_checkpoint_schedule_uses_joint_phase_not_total_phase():
    assert checkpoint_steps(4000, 1000) == (1750, 2800, 4000)
    assert checkpoint_steps(6000, 2000) == (3000, 4400, 6000)
    assert checkpoint_steps(8000, 2000) == (3500, 5600, 8000)


def test_dev_profile_selection_spans_full_frozen_582_without_test_ids():
    dev = tuple(f"dev-{i:03}" for i in range(582))
    selected = dev_profile_ids(dev, count=24)
    assert len(selected) == len(set(selected)) == 24
    assert selected[0] == dev[0] and selected[-1] == dev[-1]
    assert set(selected) <= set(dev)
    assert dev_profile_ids(dev, count=582) == dev
    with pytest.raises(ValueError, match="profile"):
        dev_profile_ids(dev, count=583)


def test_failed_budget_gate_blocks_e1_even_if_a_tier_file_is_supplied():
    with pytest.raises(ValueError, match="budget|gate"):
        reject_failed_gate(
            {"status": "failed_20_gpu_hour_feasibility_gate", "protocol_hash": "frozen"},
            "frozen",
        )
    reject_failed_gate(
        {"status": "failed_20_gpu_hour_feasibility_gate", "protocol_hash": "other"},
        "frozen",
    )


def test_full_dev_callback_records_explicit_variant_without_callback_order_dependency(
    tmp_path, monkeypatch
):
    callback = FullDevCheckpoints(
        dev=["placeholder"],
        expected=("dev-id",),
        split_hash="frozen",
        output=tmp_path,
        steps=4000,
        warmup=1000,
        batch_size=2,
        variant="E1_frozen",
    )
    monkeypatch.setattr(
        train_signer_pilot,
        "score_pilot_batches",
        lambda *_args: {"split": "dev", "items": [{"clip_id": "dev-id"}], "metrics": {}},
    )
    trainer = SimpleNamespace(
        global_step=1750,
        callbacks=[],
        save_checkpoint=lambda path: path.write_bytes(b"checkpoint"),
    )
    model = SimpleNamespace(train=lambda: None)
    callback.on_train_batch_end(trainer, model, None, None, 0)
    assert read_json(tmp_path / "dev-1750.json")["model_variant"] == "E1_frozen"


def test_e1_dev_callback_charges_generation_batches_and_preserves_checkpoint_on_cap(
    tmp_path, monkeypatch
):
    run = tmp_path / "e1-only"
    ledger = E1Ledger.create(run, {"protocol_hash": "frozen"}, 100.0)
    callback = FullDevCheckpoints(
        dev=["placeholder"],
        expected=("dev-id",),
        split_hash="frozen",
        output=run,
        steps=4000,
        warmup=1000,
        batch_size=2,
        variant="E1_frozen",
        ledger=ledger,
    )

    def interrupted(*_args, on_batch_end):
        on_batch_end(1)
        raise RuntimeError("budget exhausted")

    monkeypatch.setattr(train_signer_pilot, "score_pilot_batches", interrupted)
    trainer = SimpleNamespace(
        global_step=1750,
        save_checkpoint=lambda path: path.write_bytes(b"complete-checkpoint"),
    )
    trained = []
    with pytest.raises(RuntimeError, match="budget exhausted"):
        callback.on_train_batch_end(
            trainer, SimpleNamespace(train=lambda: trained.append(True)), None, None, 0
        )
    assert (run / "step-1750.ckpt").is_file()
    assert not (run / "dev-1750.json").exists()
    assert trained == [True]
    assert read_json(run / "budget.json")["phase"] == "dev-1/1"


def test_dev_loader_does_not_consume_global_training_rng(tmp_path, monkeypatch):
    sample = PhoenixSample(
        "dev-id",
        "Signer03",
        "german.",
        "g",
        "en",
        "fr",
        "es",
        torch.ones(20, 2),
        torch.ones(3, 2),
    )
    callback = FullDevCheckpoints(
        [sample], ("dev-id",), "frozen", tmp_path, 4000, 1000, 4, "E1_frozen"
    )

    def evaluate(model, loader, expected, split, split_hash):
        assert [clip for batch in loader for clip in batch.clip_ids] == ["dev-id"]
        return {"items": [{"clip_id": "dev-id"}], "metrics": {"bleu4": 1.0}}

    monkeypatch.setattr(train_signer_pilot, "score_pilot_batches", evaluate)
    trainer = SimpleNamespace(
        global_step=1750,
        save_checkpoint=lambda path: path.write_bytes(b"checkpoint"),
    )
    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    callback.on_train_batch_end(trainer, SimpleNamespace(train=lambda: None), None, None, 0)
    assert torch.equal(before, torch.get_rng_state())


def test_e1_periodic_resume_snapshots_only_on_unique_completed_steps(tmp_path):
    run = tmp_path / "new-run"
    ledger = E1Ledger.create(run, {"protocol_hash": "frozen"}, 100.0)
    callback = E1ResumeSnapshots(run, ledger, interval=500)
    saved = []

    def save(path):
        saved.append(path)
        path.write_bytes(b"full-checkpoint")

    trainer = SimpleNamespace(global_step=499, save_checkpoint=save)
    callback.on_train_batch_end(trainer, None, None, None, 0)
    assert saved == []
    trainer.global_step = 500
    callback.on_train_batch_end(trainer, None, None, None, 1)
    callback.on_train_batch_end(trainer, None, None, None, 2)
    assert len(saved) == 1 and (run / "resume-500.ckpt").read_bytes() == b"full-checkpoint"
    trainer.global_step = 1000
    callback.on_train_batch_end(trainer, None, None, None, 3)
    assert len(saved) == 2 and (run / "resume-1000.ckpt").is_file()


def test_e1_only_identity_stays_distinct_from_failed_three_method_gate():
    config = {
        "seed": 0,
        "model": {"warm_up_steps": 1000},
        "data": {"batch_size": 4},
        "trainer": {"max_steps": 4000, "accumulate_grad_batches": 1},
    }
    identity = e1_only_identity("frozen", config, 1646.1, {"profile": "a" * 64})
    assert identity["run_policy"] == "signer-pilot-e1-only-bounded-v1"
    assert identity["budget_ceiling_seconds"] == 21600
    assert identity["prior_gpu_seconds"] == 1646.1
    assert identity["variant"] == "E1_frozen" and identity["protocol_hash"] == "frozen"
    with pytest.raises(ValueError, match="E1-only"):
        e1_only_identity("frozen", {**config, "data": {"batch_size": 2}}, 1646.1, {})


def test_e1_only_preflight_checks_receipts_storage_and_unique_run(tmp_path, monkeypatch):
    from tests.unit.pilot.test_e1_budget import receipts

    receipts(tmp_path)
    b4 = tmp_path / "profile-E1_frozen-b4.json"
    b4_record = read_json(b4)
    b4_record["initial_shared_tensor_hash"] = (
        "9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88"
    )
    b4.write_text(__import__("json").dumps(b4_record))
    monkeypatch.setattr(
        train_signer_pilot.shutil,
        "disk_usage",
        lambda *_args: SimpleNamespace(free=200 * 1024**3),
    )
    args = SimpleNamespace(
        output_base=tmp_path,
        physical_batch=4,
        hf_cache=tmp_path / "hf",
        dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion",
        motion_manifest=tmp_path / "motion.json",
        annotation=tmp_path / "train_info_ml.npy",
    )
    identity, runroot = e1_only_preflight(args, {"protocol_hash": "frozen"})
    assert runroot.parent == tmp_path and runroot.name.startswith("e1-only-")
    assert not runroot.exists() and identity["prior_gpu_seconds"] == pytest.approx(1645.9)
    runroot.mkdir()
    with pytest.raises(ValueError, match="exists"):
        e1_only_preflight(args, {"protocol_hash": "frozen"})
    assert e1_only_preflight(args, {"protocol_hash": "frozen"}, resume=True)[1] == runroot
    args.physical_batch = 2
    with pytest.raises(ValueError, match="batch"):
        e1_only_preflight(args, {"protocol_hash": "frozen"})
    args.physical_batch = 4
    monkeypatch.setattr(
        train_signer_pilot.shutil,
        "disk_usage",
        lambda *_args: SimpleNamespace(free=10),
    )
    with pytest.raises(ValueError, match="storage"):
        e1_only_preflight(args, {"protocol_hash": "frozen"}, resume=True)


def test_e1_only_runner_never_claims_full_run_when_trainer_stops_early(tmp_path, monkeypatch):
    from tests.unit.pilot.test_e1_budget import receipts

    receipts(tmp_path)
    b4 = tmp_path / "profile-E1_frozen-b4.json"
    record = read_json(b4)
    record["initial_shared_tensor_hash"] = train_signer_pilot.E1_SHARED_HASH
    b4.write_text(__import__("json").dumps(record))
    monkeypatch.setattr(
        train_signer_pilot.shutil,
        "disk_usage",
        lambda *_args: SimpleNamespace(free=200 * 1024**3),
    )
    monkeypatch.setattr(train_signer_pilot, "writer", lambda *_args: nullcontext())
    alarm_calls = []
    monkeypatch.setattr(
        train_signer_pilot.signal,
        "setitimer",
        lambda *values: alarm_calls.append(values),
    )
    monkeypatch.setattr(train_signer_pilot.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        train_signer_pilot.torch.cuda,
        "mem_get_info",
        lambda: (16 * 1024**3, 16 * 1024**3),
    )
    monkeypatch.setattr(
        train_signer_pilot,
        "_model",
        lambda *_args: SimpleNamespace(run_metadata=None),
    )
    monkeypatch.setattr(
        train_signer_pilot,
        "shared_tensor_hash",
        lambda *_args: train_signer_pilot.E1_SHARED_HASH,
    )

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.callbacks = kwargs["callbacks"]
            self.global_step = 0

        def fit(self, model, datamodule, ckpt_path=None):
            assert ckpt_path is None
            self.datamodule = datamodule
            self.global_step = 1
            self.callbacks[0].on_train_batch_end(self, model, None, None, 0)

    monkeypatch.setattr(train_signer_pilot.pl, "Trainer", FakeTrainer)
    args = SimpleNamespace(
        output_base=tmp_path,
        physical_batch=4,
        hf_cache=tmp_path / "hf",
        dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion",
        motion_manifest=tmp_path / "motion.json",
        annotation=tmp_path / "train_info_ml.npy",
    )
    protocol = {
        "protocol_hash": "frozen",
        "split_hash": "split",
        "split": {"groups": {"dev": {"clip_ids": [str(i) for i in range(582)]}}},
    }
    result = run_e1_only(args, protocol, {"train": list(range(5746)), "dev": [0] * 582})
    assert result["status"] == "partial" and result["global_step"] == 1
    runroot = tmp_path / result["run_key"]
    assert read_json(runroot / "budget.json")["gpu_seconds"] > 1200
    assert not (runroot / "selected-checkpoint.json").exists()
    assert alarm_calls[0][1] > 0 and alarm_calls[-1][1] == 0


def test_e1_resume_requires_latest_owned_full_optimizer_snapshot(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    identity = {"protocol_hash": "frozen", "run_policy": "signer-pilot-e1-only-bounded-v1"}
    checkpoint = {
        "pilot_identity": identity,
        "run_metadata": identity,
        "global_step": 500,
        "optimizer_states": [{"state": {1: {"step": 500}}}],
        "lr_schedulers": [{"last_epoch": 500}],
        "pilot_rng": {"torch": torch.get_rng_state()},
        "pilot_sampler": {"batches_yielded": 500},
        "gpu_seconds_cumulative": 100.0,
    }
    first = run / "resume-500.ckpt"
    torch.save(checkpoint, first)
    assert validate_e1_resume_checkpoint(first, run, identity, 101.0)["global_step"] == 500
    later = run / "resume-1000.ckpt"
    torch.save({**checkpoint, "global_step": 1000}, later)
    with pytest.raises(ValueError, match="latest"):
        validate_e1_resume_checkpoint(first, run, identity, 101.0)
    with pytest.raises(ValueError, match="identity|checkpoint"):
        validate_e1_resume_checkpoint(later, run, {"protocol_hash": "other"}, 101.0)


def test_resume_at_saved_dev_step_finishes_missing_full_report_before_training(
    tmp_path, monkeypatch
):
    from tests.unit.pilot.test_signer_pilot_evaluation import FakeModel

    sample = PhoenixSample(
        "dev-id",
        "Signer03",
        "eins.",
        "g",
        "en",
        "fr",
        "es",
        torch.ones(20, 2),
        torch.ones(3, 2),
    )
    checkpoint = tmp_path / "step-1750.ckpt"
    checkpoint.write_bytes(b"complete-checkpoint")
    callback = FullDevCheckpoints(
        [sample], ("dev-id",), "split", tmp_path, 4000, 1000, 4, "E1_frozen"
    )
    scored = []

    def score(*_args):
        scored.append(True)
        return {"items": [{"clip_id": "dev-id"}], "metrics": {"bleu4": 1.0}}

    monkeypatch.setattr(train_signer_pilot, "score_pilot_batches", score)
    trainer = SimpleNamespace(
        global_step=1750, save_checkpoint=lambda path: pytest.fail("overwrite")
    )
    callback.on_train_start(trainer, FakeModel())
    assert scored == [True] and read_json(tmp_path / "dev-1750.json")["checkpoint_step"] == 1750

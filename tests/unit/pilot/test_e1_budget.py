"""E1-only GPU-hour spending must persist before any long pilot job."""

import json
import os

import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from despamo.appearance.provenance import file_hash
from despamo.training.e1_budget import E1BudgetGuard, E1Ledger, initial_pilot_seconds
from despamo.training.pilot_factory import PilotDataModule

PRIOR = {
    "e3-profile-b2.json": 7.5,
    "e3-profile-b4-extract16.json": 13.3,
    "profile-E1_frozen-b2.json": 258.2,
    "profile-E1_frozen-b4.json": 166.9,
    "profile-timeout-e1-dev.json": 1200.0,
}


def receipts(tmp_path, protocol="frozen"):
    result = {}
    for name, seconds in PRIOR.items():
        path = tmp_path / name
        path.write_text(
            json.dumps(
                {
                    "status": "timed_out" if "timeout" in name else "complete",
                    "protocol_hash": protocol,
                    "gpu_seconds": seconds,
                }
            )
        )
        result[name] = path
    return result


def test_prior_spend_checks_exact_five_receipts_and_bytes(tmp_path):
    paths = receipts(tmp_path)
    total, hashes = initial_pilot_seconds(paths, "frozen")
    assert total == pytest.approx(sum(PRIOR.values()))
    assert hashes == {name: file_hash(path) for name, path in paths.items()}
    with pytest.raises(ValueError, match="profile|receipt"):
        initial_pilot_seconds({k: v for k, v in paths.items() if "timeout" not in k}, "frozen")
    with pytest.raises(ValueError, match="protocol"):
        initial_pilot_seconds(paths, "other")
    paths["profile-E1_frozen-b4.json"].write_text(
        '{"status":"complete","protocol_hash":"frozen","gpu_seconds":-1}'
    )
    with pytest.raises(ValueError, match="GPU|seconds"):
        initial_pilot_seconds(paths, "frozen")


class Clock:
    now = 0.0
    wall = 1000.0

    def monotonic(self):
        return self.now

    def time(self):
        return self.wall


def test_ledger_is_atomic_monotonic_and_stops_at_six_hours(tmp_path):
    clock = Clock()
    run = tmp_path / "new-run"
    identity = {"run": "e1-only", "protocol_hash": "frozen"}
    ledger = E1Ledger.create(run, identity, 100.0, monotonic=clock.monotonic, wall=clock.time)
    assert ledger.charge(step=0, phase="load") == 100.0
    clock.now, clock.wall = 9.0, 1009.0
    assert ledger.charge(step=1, phase="train") == 109.0
    saved = json.loads((run / "budget.json").read_text())
    assert saved["gpu_seconds"] == 109.0 and saved["optimizer_step"] == 1
    assert saved["pid"] == os.getpid()
    assert ledger.require_remaining(300, step=1, phase="checkpoint") == 109.0
    clock.now, clock.wall = 21201.0, 22201.0
    with pytest.raises(RuntimeError, match="budget"):
        ledger.require_remaining(300, step=1, phase="before_dev")
    clock.now, clock.wall = 9.0, 1009.0
    with pytest.raises(ValueError, match="exists"):
        E1Ledger.create(run, identity, 100.0, monotonic=clock.monotonic, wall=clock.time)
    clock.now = 21501.0
    with pytest.raises(RuntimeError, match="six|budget"):
        ledger.charge(step=2, phase="train")
    assert json.loads((run / "budget.json").read_text())["gpu_seconds"] >= 21600


def test_resume_rejects_foreign_identity_and_checkpoint_ledger_reset(tmp_path):
    clock = Clock()
    identity = {"run": "e1-only", "protocol_hash": "frozen"}
    run = tmp_path / "new-run"
    ledger = E1Ledger.create(run, identity, 100.0, monotonic=clock.monotonic, wall=clock.time)
    clock.now, clock.wall = 5.0, 1005.0
    ledger.charge(step=5, phase="train")
    ledger.close()
    with pytest.raises(ValueError, match="identity"):
        E1Ledger.resume(
            run,
            {"run": "other"},
            105.0,
            checkpoint_step=5,
            monotonic=clock.monotonic,
            wall=clock.time,
        )
    with pytest.raises(ValueError, match="checkpoint|budget"):
        E1Ledger.resume(
            run,
            identity,
            106.0,
            checkpoint_step=5,
            monotonic=clock.monotonic,
            wall=clock.time,
        )
    resumed = E1Ledger.resume(
        run,
        identity,
        105.0,
        checkpoint_step=5,
        monotonic=clock.monotonic,
        wall=clock.time,
    )
    assert resumed.charge(step=5, phase="resume") >= 105.0


def test_resume_older_safe_checkpoint_preserves_spend_while_replaying_steps(tmp_path):
    clock = Clock()
    identity = {"protocol_hash": "frozen"}
    root = tmp_path / "run"
    ledger = E1Ledger.create(root, identity, 100.0, monotonic=clock.monotonic, wall=clock.time)
    clock.now, clock.wall = 500.0, 1500.0
    ledger.charge(step=750, phase="train")
    ledger.close()
    replay = E1Ledger.resume(
        root,
        identity,
        400.0,
        checkpoint_step=500,
        monotonic=clock.monotonic,
        wall=clock.time,
    )
    assert replay.last >= 600 and replay.charge(step=501, phase="replayed_train") >= 600


def test_lightning_e1_guard_charges_each_step_checkpoint_and_budget_exhaustion(tmp_path):
    clock = Clock()
    identity = {"protocol_hash": "frozen", "run": "e1-only"}
    ledger = E1Ledger.create(
        tmp_path / "new-run", identity, 1200.0, monotonic=clock.monotonic, wall=clock.time
    )
    data = PilotDataModule(list(range(8)), 4, 0, 10, list)
    trainer = type("Trainer", (), {"datamodule": data, "global_step": 1})()
    callback = E1BudgetGuard(ledger)
    clock.now, clock.wall = 10.0, 1010.0
    callback.on_train_batch_end(trainer, None, None, None, 0)
    checkpoint = {}
    callback.on_save_checkpoint(trainer, None, checkpoint)
    assert checkpoint["gpu_seconds_cumulative"] >= 1210
    assert checkpoint["pilot_sampler"]["batches_yielded"] == data.state_dict()["batches_yielded"]
    clock.now, clock.wall = 20410.0, 21410.0
    with pytest.raises(RuntimeError, match="six-hour"):
        callback.on_train_batch_end(trainer, None, None, None, 1)
    assert json.loads((ledger.root / "budget.json").read_text())["gpu_seconds"] >= 21600


def test_e1_budget_lightning_resume_replays_identical_batches_and_weights(tmp_path):
    identity = {"protocol_hash": "frozen", "run_policy": "e1-only"}

    class Tiny(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.layer = nn.Linear(1, 1)
            self.seen = []

        def training_step(self, batch, index):
            self.seen.extend(int(x) for x in batch.flatten().tolist())
            return self.layer(batch).square().mean() + torch.rand(())

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 2),
            }

    def fit(root, steps, checkpoint=None):
        torch.manual_seed(0)
        data = PilotDataModule([torch.tensor([float(i)]) for i in range(13)], 2, 0, 5, torch.stack)
        model = Tiny()
        if checkpoint is None:
            ledger = E1Ledger.create(root, identity, 100.0)
        else:
            prior = torch.load(checkpoint, map_location="cpu", weights_only=False)
            ledger = E1Ledger.resume(
                root,
                identity,
                prior["gpu_seconds_cumulative"],
                checkpoint_step=prior["global_step"],
            )
        trainer = pl.Trainer(
            max_steps=steps,
            max_epochs=-1,
            callbacks=[E1BudgetGuard(ledger)],
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=data, ckpt_path=str(checkpoint) if checkpoint else None)
        return trainer, model, ledger

    full, full_model, full_ledger = fit(tmp_path / "full", 5)
    partial, partial_model, partial_ledger = fit(tmp_path / "partial", 2)
    path = partial_ledger.root / "resume-2.ckpt"
    partial.save_checkpoint(path)
    partial_ledger.close()
    resumed, resumed_model, resumed_ledger = fit(tmp_path / "partial", 5, checkpoint=path)
    assert partial_model.seen + resumed_model.seen == full_model.seen
    for key, value in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[key], value, rtol=0, atol=0)
    assert resumed_ledger.last >= partial_ledger.last
    assert resumed.global_step == full.global_step == 5

import json
from pathlib import Path

import pytest

from tools.spamo_repro.study import run_study
from tools.spamo_repro.train import epoch_groups


def test_epoch_groups_cover_train_once_with_444_updates_and_tail():
    ids = tuple(f"clip-{i}" for i in range(7096))
    groups = epoch_groups(ids, 1, seed=0)
    flat = [clip for group in groups for batch in group for clip in batch]
    assert len(groups) == 444
    assert [len(g) for g in groups[:-1]] == [2] * 443 and len(groups[-1]) == 1
    assert all(len(batch) == 8 for group in groups for batch in group)
    assert sorted(flat) == sorted(ids)
    assert epoch_groups(ids, 1, seed=0) == groups
    assert epoch_groups(ids, 2, seed=0) != groups


def receipts(root: Path, stage: str, epoch: int, bleu: float = 1.0) -> None:
    if stage == "train":
        (root / f"train-epoch-{epoch}.json").write_text(
            json.dumps({"epoch": epoch, "completed_updates": epoch * 444, "status": "complete"})
        )
    else:
        (root / "dev").mkdir(exist_ok=True)
        (root / "dev" / f"dev-epoch-{epoch}.json").write_text(
            json.dumps({"epoch": epoch, "complete": True, "clips": 519, "metrics": {"bleu4": bleu}})
        )


def test_parent_runs_sequential_train_then_dev_every_two_epochs_and_selects_best(tmp_path):
    calls = []

    def spawn(stage, epoch):
        calls.append((stage, epoch))
        receipts(tmp_path, stage, epoch, bleu=float(epoch % 7))
        return 0

    state = run_study(tmp_path, spawn=spawn, preflight=lambda stage, epoch: None)
    assert calls == [(s, e) for e in range(2, 41, 2) for s in ("train", "dev")]
    assert state["status"] == "complete" and state["completed_epochs"] == 40
    assert state["best"] == {"epoch": 6, "bleu4": 6.0}
    assert state["test_scored"] is False


def test_child_failure_holds_without_retry_or_next_worker(tmp_path):
    calls = []

    def spawn(stage, epoch):
        calls.append((stage, epoch))
        if stage == "dev":
            return 3
        receipts(tmp_path, stage, epoch)
        return 0

    with pytest.raises(RuntimeError, match="held"):
        run_study(tmp_path, spawn=spawn, preflight=lambda stage, epoch: None)
    assert calls == [("train", 2), ("dev", 2)]
    hold = json.loads((tmp_path / "manual-stop.json").read_text())
    assert hold["automatic_recovery_disabled"] is True
    with pytest.raises(RuntimeError, match="hold"):
        run_study(tmp_path, spawn=spawn, preflight=lambda stage, epoch: None)
    assert len(calls) == 2


def test_missing_or_incomplete_receipt_holds_before_dev(tmp_path):
    calls = []

    def spawn(stage, epoch):
        calls.append((stage, epoch))
        return 0

    with pytest.raises(RuntimeError, match="held"):
        run_study(tmp_path, spawn=spawn, preflight=lambda stage, epoch: None)
    assert calls == [("train", 2)]


def test_preflight_failure_prevents_worker(tmp_path):
    def preflight(stage, epoch):
        raise RuntimeError("low space")

    with pytest.raises(RuntimeError, match="held"):
        run_study(tmp_path, spawn=lambda s, e: pytest.fail("spawned"), preflight=preflight)


def test_restart_continues_after_last_recorded_dev(tmp_path):
    for epoch in (2, 4):
        receipts(tmp_path, "train", epoch)
        receipts(tmp_path, "dev", epoch, bleu=float(epoch))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "completed_epochs": 4,
                "best": {"epoch": 4, "bleu4": 4.0},
                "history": [],
                "test_scored": False,
            }
        )
    )
    calls = []

    def spawn(stage, epoch):
        calls.append((stage, epoch))
        receipts(tmp_path, stage, epoch, bleu=0.0)
        return 0

    state = run_study(tmp_path, spawn=spawn, preflight=lambda stage, epoch: None)
    assert calls[0] == ("train", 6)
    assert state["best"] == {"epoch": 4, "bleu4": 4.0}

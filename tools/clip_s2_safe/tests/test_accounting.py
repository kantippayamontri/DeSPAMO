import json
from pathlib import Path

import pytest

from tools.clip_s2_safe.tests.test_state import binding_fixture
from tools.dinov3.identity import digest

OWNER = dict(pid=99999999, boot_id="old", start_ticks="0", command=["dead"])


def setup(tmp_path, monkeypatch, mode="production"):
    from tools.clip_s2_safe import accounting as a

    _, c, b, *_ = binding_fixture(tmp_path)
    now = [100.0]
    monkeypatch.setattr(a.time, "time", lambda: now[0])
    scope = dict(
        approved=True,
        token="scope",
        mode=mode,
        max_train_updates=4,
        max_dev_clips=4,
        binding_hash=digest(b),
    )
    attempt = a.begin_attempt(c, b, scope, OWNER)
    return a, c, b, attempt, now


def job(stage="train", mode="production", start=2, target=4):
    return dict(
        stage=stage,
        mode=mode,
        stage_id=f"{stage}{start}-{target}",
        start_step=start,
        target_step=target,
        max_batches=None,
    )


def test_original_floor_and_stages_charged_once(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    j = job()
    a.begin_stage(c, attempt, j)
    now[0] = 110
    result = dict(completed_updates=4, durable_updates=4)
    a.finish_stage(c, attempt, j, result, status="complete")
    a.finish_stage(c, attempt, j, result, status="complete")
    d = job("dev", start=4, target=4)
    a.begin_stage(c, attempt, d)
    now[0] = 113
    a.finish_stage(
        c,
        attempt,
        d,
        dict(completed_updates=4, durable_updates=4, completed_clips=2),
        status="complete",
    )
    budget = json.loads((Path(c["run_root"]) / "budget.json").read_text())
    assert budget["charged_seconds"] == 510.5533489369991 and budget["completed_updates"] == 4
    assert budget["identity"] == b["original_identity"]


def test_probe_updates_are_provisional_and_cannot_auto_adopt(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch, "handoff_probe")
    j = job(mode="handoff_probe")
    a.begin_stage(c, attempt, j)
    now[0] = 107
    a.finish_stage(c, attempt, j, dict(completed_updates=4, durable_updates=4), status="complete")
    budget = json.loads((Path(c["run_root"]) / "budget.json").read_text())
    assert budget["completed_updates"] == 2 and budget["charged_seconds"] == 504.5533489369991
    assert budget["safe_runtime"]["executed_probe_updates"] == 2
    with pytest.raises(RuntimeError):
        a.adopt_probe(c, b, dict(step=4), None)


def test_attempt_caps_are_cumulative(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    j = job()
    a.begin_stage(c, attempt, j)
    now[0] = 101
    a.finish_stage(c, attempt, j, dict(completed_updates=4, durable_updates=4), status="complete")
    too = job(start=4, target=7)
    with pytest.raises(RuntimeError, match="cap"):
        a.begin_stage(c, attempt, too)


def test_backwards_clock_preserves_original_budget(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    j = job()
    a.begin_stage(c, attempt, j)
    before = (Path(c["run_root"]) / "budget.json").read_bytes()
    now[0] = 90
    with pytest.raises(ValueError):
        a.record_progress(c, attempt, j, dict(completed_updates=3, durable_updates=2))
    assert (Path(c["run_root"]) / "budget.json").read_bytes() == before


def test_reviewed_dead_replay_preserves_highwater_and_charges(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    j = job()
    a.begin_stage(c, attempt, j)
    now[0] = 110
    a.record_progress(c, attempt, j, dict(completed_updates=3, durable_updates=2))
    with pytest.raises(RuntimeError):
        a.reconcile_dead_attempt(c, b, reviewed=False)
    result = a.reconcile_dead_attempt(c, b, reviewed=True)
    assert result["completed_updates"] == 2 and result["charged_seconds"] == 507.5533489369991
    assert result["safe_runtime"]["replay_history"][-1]["completed_before"] == 3
    assert (
        a.reconcile_dead_attempt(c, b, reviewed=True)["charged_seconds"]
        == result["charged_seconds"]
    )


def test_torn_budget_after_write_reconciles_without_double_charge(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    j = job()
    a.begin_stage(c, attempt, j)
    now[0] = 105
    original = a.atomic_json

    def fail_head(path, value):
        if path.name == "accounting-head.json":
            raise OSError("torn head")
        return original(path, value)

    monkeypatch.setattr(a, "atomic_json", fail_head)
    with pytest.raises(OSError):
        a.finish_stage(
            c, attempt, j, dict(completed_updates=4, durable_updates=4), status="complete"
        )
    monkeypatch.setattr(a, "atomic_json", original)
    result = a.reconcile_dead_attempt(c, b, reviewed=True)
    assert result["charged_seconds"] == 502.5533489369991 and result["completed_updates"] == 4


def test_storage_guard_before_actions(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)

    def low(*args, **kwargs):
        raise ValueError("reserve")

    monkeypatch.setattr(a, "require_official_storage", low)
    with pytest.raises(ValueError, match="reserve"):
        a.require_storage(c, b, job())


def test_probe_storage_projects_bounded_outputs_not_long_study(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch, "handoff_probe")
    seen = {}

    def native(root, amount):
        seen["native"] = amount
        return {}

    def archive(root, amount):
        seen["archive"] = amount
        return {}

    monkeypatch.setattr(a, "require_official_storage", native)
    monkeypatch.setattr(a, "require_archive_storage", archive)
    j = job(mode="handoff_probe", start=2, target=3)
    a.require_storage(c, b, j)
    size = Path(c["anchor"]).stat().st_size
    assert seen["native"] == 3 * size + 64 * 1024**2 and seen["archive"] == size


def test_dev_storage_admits_outputs_without_admitting_future_training(tmp_path, monkeypatch):
    a, c, b, attempt, now = setup(tmp_path, monkeypatch)
    Path(c["anchor"]).write_bytes(b"x" * 1024**2)
    size = Path(c["anchor"]).stat().st_size
    capacity = 3 * size + 64 * 1024**2

    def native(root, amount):
        if amount > capacity:
            raise ValueError("projected outputs exceed capacity after reserve")
        return {"projected_bytes": amount}

    monkeypatch.setattr(a, "require_official_storage", native)
    monkeypatch.setattr(
        a, "require_archive_storage", lambda root, amount: {"projected_bytes": amount}
    )
    b["recipe"]["dev_epochs"] = list(range(2, 201, 2))
    result = a.require_storage(c, b, {**job(), "stage": "dev"})
    assert result["native"]["projected_bytes"] == capacity
    assert result["archive"]["projected_bytes"] == size
    with pytest.raises(ValueError, match="after reserve"):
        a.require_storage(c, b, {**job(), "stage": "train"})

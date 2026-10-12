import json
import os
import sys
from pathlib import Path

import pytest

from tools.clip_s2_safe.tests.test_state import binding_fixture
from tools.dinov3.identity import digest
from tools.dinov3.storage import atomic_json


def scope(c, b):
    return dict(
        approved=True,
        token="attempt",
        mode="production",
        max_train_updates=4,
        max_dev_clips=4,
        binding_hash=digest(b),
    )


def test_anchor_schedules_safe_dev_before_training(tmp_path):
    from tools.clip_s2_safe.study import pending_stage

    _, c, b, *_ = binding_fixture(tmp_path)
    pending = pending_stage(c, b, {})
    assert pending["stage"] == "dev" and pending["target_step"] == 2 and pending["epoch"] == 2


def test_actual_child_exit_not_handoff_file(tmp_path, monkeypatch):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_s2_safe import accounting
    from tools.clip_s2_safe import study as s
    from tools.clip_s2_safe.state import publish_checkpoint
    from tools.dinov3.official_runtime import process_owner

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    sampler.next_ids()
    sampler.next_ids()
    pub = publish_checkpoint(
        c,
        capture_state(m, o, b["original_identity"], 4, sampler.state_dict(), base_hash="frozen"),
        dict(last_update=4, ce_values=[]),
        mode="production",
    )
    attempt = accounting.begin_attempt(c, b, scope(c, b), process_owner(os.getpid()))
    job = dict(
        stage="train",
        mode="production",
        seed=0,
        stage_id="train2-4",
        token="attempt",
        start_step=2,
        target_step=4,
        binding_hash=digest(b),
    )
    marker = tmp_path / "exited"

    def command(c, job, approval):
        assert job.get("allow_owned_resume_pruning") is True
        value = dict(
            status="awaiting_dev",
            stage="train",
            mode="production",
            completed_updates=4,
            durable_updates=4,
            **pub,
        )
        script = (
            "import json,pathlib,time;pathlib.Path("
            + repr(job["result_path"])
            + ").write_text("
            + repr(json.dumps(value))
            + ");time.sleep(.15);pathlib.Path("
            + repr(str(marker))
            + ').write_text("done")'
        )
        return [sys.executable, "-c", script]

    monkeypatch.setattr(
        s, "require_scope", lambda *a, **k: {**scope(c, b), "allow_owned_resume_pruning": True}
    )
    monkeypatch.setattr(s, "worker_command", command)
    result = s.execute_job(c, b, attempt, job, tmp_path / "approval")
    assert marker.exists() and result["durable_updates"] == 4


def test_nonzero_child_never_retries_and_preserves_old_hold(tmp_path, monkeypatch):
    from tools.clip_s2_safe import accounting
    from tools.clip_s2_safe import study as s
    from tools.dinov3.official_runtime import process_owner

    _, c, b, *_ = binding_fixture(tmp_path)
    before = (Path(c["study_root"]) / "manual-stop.json").read_bytes()
    attempt = accounting.begin_attempt(c, b, scope(c, b), process_owner(os.getpid()))
    calls = []
    job = dict(
        stage="dev",
        mode="production",
        seed=0,
        stage_id="dev2",
        token="attempt",
        start_step=2,
        target_step=2,
        binding_hash=digest(b),
        max_batches=None,
    )

    def command(*a):
        calls.append(1)
        return [sys.executable, "-c", "raise SystemExit(3)"]

    monkeypatch.setattr(s, "require_scope", lambda *a, **k: scope(c, b))
    monkeypatch.setattr(s, "worker_command", command)
    with pytest.raises(RuntimeError):
        s.execute_job(c, b, attempt, job, tmp_path / "approval")
    assert len(calls) == 1 and (Path(c["runtime_root"]) / "manual-stop.json").exists()
    assert (Path(c["study_root"]) / "manual-stop.json").read_bytes() == before


def test_default_run_never_creates_or_loads_model(tmp_path):
    from tools.clip_s2_safe.study import run
    from tools.clip_s2_safe.tests.helpers import original_fixture

    c, *_ = original_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="scope"):
        run(c, None)


def test_selector_rejects_prior_diagnostic_without_admission(tmp_path):
    from tools.clip_s2_safe.study import select_checkpoint

    _, c, b, *_ = binding_fixture(tmp_path)
    root = Path(c["runtime_root"])
    atomic_json(
        root / "dev/epoch-2.json",
        dict(diagnostic_only=True, complete=True, epoch=2, step=2, metrics={"bleu4": 100.0}),
    )
    with pytest.raises(ValueError):
        select_checkpoint(c, b, {})

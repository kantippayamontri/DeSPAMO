import json
import sys
from pathlib import Path

import pytest

from tools.clip_plain.tests.helpers import config_at


def test_process_handoff_not_exit_barrier(tmp_path, monkeypatch):
    from tools.clip_plain import study as s
    from tools.clip_plain.policy import POLICY
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    run = Path(c["study_root"]) / "runs/seed0"
    run.mkdir(parents=True)
    identity = dict(policy=POLICY)
    atomic_json(run / "identity.json", identity)
    job = dict(
        stage="train",
        seed=0,
        epoch=2,
        run_root=str(run),
        attempt_token="attempt",
        result_path=str(run / "result.json"),
        progress_path=str(run / "progress.json"),
    )
    script = tmp_path / "child.py"
    exit_marker = tmp_path / "exited"
    script.write_text(
        "import json,time,pathlib; p=pathlib.Path("
        + repr(job["result_path"])
        + ");p.write_text(json.dumps(dict(status='awaiting_dev',stage='train',"
        + "completed_updates=2,durable_updates=2)));time.sleep(.15);pathlib.Path("
        + repr(str(exit_marker))
        + ").write_text('done')"
    )
    monkeypatch.setattr(s, "require_gpu_scope", lambda *a, **k: {})
    monkeypatch.setattr(s, "worker_command", lambda *a: [sys.executable, str(script)])
    r = s.execute_job(c, job, tmp_path / "approval")
    assert r["status"] == "awaiting_dev" and exit_marker.exists()
    assert json.loads((run / "budget.json").read_text())["charged_seconds"] > 0


def test_nonzero_child_holds_without_retry(tmp_path, monkeypatch):
    from tools.clip_plain import study as s
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    run = Path(c["study_root"]) / "runs/seed0"
    atomic_json(run / "identity.json", dict(policy="plain"))
    job = dict(
        stage="dev",
        seed=0,
        epoch=2,
        run_root=str(run),
        attempt_token="a",
        result_path=str(run / "result.json"),
        progress_path=str(run / "progress.json"),
    )
    monkeypatch.setattr(s, "require_gpu_scope", lambda *a, **k: {})
    calls = []

    def command(*a):
        calls.append(1)
        return [sys.executable, "-c", "raise SystemExit(3)"]

    monkeypatch.setattr(s, "worker_command", command)
    with pytest.raises(RuntimeError):
        s.execute_job(c, job, tmp_path / "approval")
    assert len(calls) == 1 and (Path(c["study_root"]) / "manual-stop.json").exists()
    assert json.loads((run / "budget.json").read_text())["status"] == "failed"


def test_pending_even_epoch_dev_before_training(tmp_path):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_plain.state import publish_epoch
    from tools.clip_plain.study import pending_stage
    from tools.clip_plain.tests.test_state import identity, tiny

    m, o = tiny()
    i = identity()
    i["recipe"]["dev_epochs"] = [2, 4]
    i["recipe"]["epochs"] = 4
    publish_epoch(tmp_path, capture_state(m, o, i, 2, {}, base_hash="f"), 2, dev_candidate=True)
    assert pending_stage(tmp_path, i, {})["stage"] == "dev"


def test_selection_uses_actual_best_not_patience_anchor(tmp_path, monkeypatch):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_plain import study as s
    from tools.clip_plain.state import publish_epoch
    from tools.clip_plain.tests.test_state import identity, tiny
    from tools.dinov3.identity import digest
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    run = Path(c["study_root"]) / "runs/seed0"
    m, o = tiny()
    i = identity()
    i["recipe"].update(
        dev_epochs=[2, 4],
        vt_warmup_steps=0,
        min_delta=0.10,
        steps_per_epoch=1,
        min_epochs=40,
        patience=10,
        epochs=200,
    )
    monkeypatch.setattr(s, "validate_report", lambda *a: None)
    monkeypatch.setattr(s, "archive_candidate", lambda *a: {})
    for epoch, score in [(2, 10.0), (4, 10.05)]:
        snap = capture_state(m, o, i, epoch, {}, base_hash="f")
        r = publish_epoch(run, snap, epoch, dev_candidate=True)
        atomic_json(
            run / "dev" / f"dev-epoch-{epoch}.json",
            dict(
                epoch=epoch,
                metrics=dict(bleu4=score),
                checkpoint_path=r["candidate"],
                checkpoint_sha256=r["candidate_sha256"],
                run_identity_hash=digest(i),
                evaluation_policy_hash="eval",
                data_identity_hash="data",
            ),
        )
    selected = s.select_checkpoint(c, run, i, {})
    assert selected["epoch"] == 4 and selected["metrics"]["bleu4"] == 10.05


def test_preflight_cannot_launch_worker(tmp_path, monkeypatch):
    from tools.clip_plain import study as s

    c = config_at(tmp_path)
    monkeypatch.setattr(s, "derive_package", lambda *a, **k: dict(root="planned"))
    result = s.run_study(c, None, check_only=True)
    assert result["gpu_launched"] is False


def test_parent_logger_failure_does_not_replace_child_fault(tmp_path, monkeypatch):
    from tools.clip_plain import study as s
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    run = Path(c["study_root"]) / "runs/seed0"
    atomic_json(run / "identity.json", dict(policy="plain"))
    job = dict(
        stage="dev",
        seed=0,
        epoch=2,
        run_root=str(run),
        attempt_token="fault",
        result_path=str(run / "result.json"),
        progress_path=str(run / "progress.json"),
    )
    monkeypatch.setattr(s, "require_gpu_scope", lambda *a, **k: {})
    monkeypatch.setattr(
        s, "worker_command", lambda *a: [sys.executable, "-c", "raise SystemExit(3)"]
    )
    original = s.record_stage

    def fail(*a, **k):
        if k.get("final_status") == "failed":
            raise OSError("budget disk full")
        return original(*a, **k)

    monkeypatch.setattr(s, "record_stage", fail)
    with pytest.raises(RuntimeError, match="exited3") as got:
        s.execute_job(c, job, tmp_path / "approval")
    assert any("budget disk full" in n for n in got.value.__notes__)


def test_forged_worker_update_count_held(tmp_path, monkeypatch):
    from tools.clip_plain import study as s
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    run = Path(c["study_root"]) / "runs/seed0"
    atomic_json(run / "identity.json", dict(policy="plain", recipe=dict(steps_per_epoch=1)))
    job = dict(
        stage="train",
        seed=0,
        epoch=2,
        run_root=str(run),
        attempt_token="forged",
        result_path=str(run / "result.json"),
        progress_path=str(run / "progress.json"),
    )
    monkeypatch.setattr(s, "require_gpu_scope", lambda *a, **k: {})
    command = (
        "import pathlib,json;pathlib.Path("
        + repr(job["result_path"])
        + ").write_text(json.dumps(dict(status='awaiting_dev',stage='train',"
        + "completed_updates=999,durable_updates=999)))"
    )
    monkeypatch.setattr(s, "worker_command", lambda *a: [sys.executable, "-c", command])
    with pytest.raises(ValueError, match="update"):
        s.execute_job(c, job, tmp_path / "approval")


def test_reconciliation_does_not_trust_tampered_job_file(tmp_path, monkeypatch):
    from tools.clip_plain.tests.test_policy import approval_fixture

    p, c, record, path = approval_fixture(tmp_path, monkeypatch)
    attempt = p.claim_attempt(c, path)
    job = dict(
        stage="dev",
        seed=0,
        epoch=2,
        run_root=str(Path(c["study_root"]) / "runs/seed0"),
        attempt_token=attempt["token"],
        parent_owner=attempt["owner"],
        max_batches=2,
    )
    jobpath = Path(job["run_root"]) / "jobs" / f"{attempt['token']}-dev-2/job.json"
    from tools.dinov3.storage import atomic_json

    atomic_json(jobpath, {**job, "seed": 1})
    with pytest.raises(RuntimeError, match="ticket"):
        p.require_gpu_scope(c, "dev", path, job=job)

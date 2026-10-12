import ast
import io
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType

import pytest

from scripts import run_projector_overnight as workflow
from scripts.run_projector_overnight import Study, baseline_decision, stage_allowance


def test_stage_allowance_never_exceeds_overall_or_stage_budget():
    assert stage_allowance(35000, 36000, 1800, 0) == 1000
    assert stage_allowance(0, 36000, 14400, 13000) == 1400
    with pytest.raises(RuntimeError, match="budget"):
        stage_allowance(36000, 36000, 1800, 0)
    with pytest.raises(ValueError):
        stage_allowance(-1, 36000, 1800, 0)


def test_completed_stage_is_not_executed_again(tmp_path):
    study = Study(tmp_path, 36000)
    output = tmp_path / "child-count.txt"
    command = [sys.executable, "-c", f"from pathlib import Path; p=Path({str(output)!r}); "
               "p.write_text(p.read_text()+'x' if p.exists() else 'x')"]
    study.run_stage("one", command, 10)
    spent = study.state["spent_seconds"]
    study.run_stage("one", command, 10)
    assert output.read_text() == "x"
    assert study.state["spent_seconds"] == spent
    state = json.loads((tmp_path / "study-state.json").read_text())
    assert state["stages"]["one"]["status"] == "complete"
    study.run_stage("one", command + ["--resume-checkpoint", "owned.ckpt"], 10)
    assert output.read_text() == "x"


def test_stage_failure_is_recorded_and_budget_charged(tmp_path):
    study = Study(tmp_path, 36000)
    with pytest.raises(RuntimeError, match="failed"):
        study.run_stage("bad", [sys.executable, "-c", "raise SystemExit(3)"], 10)
    assert study.state["stages"]["bad"]["status"] == "failed"
    assert study.state["stages"]["bad"]["exit_code"] == 3
    assert study.state["spent_seconds"] > 0


def test_baseline_rule_uses_primary_transfer_and_labels_mixed_replication():
    def analysis(seed, delta):
        return {"training_seed": seed, "metric_deltas": {"bleu4": delta}}
    assert baseline_decision([analysis(0, 1), analysis(1, 0.5)])["status"] == "replicated_two_seeds"
    mixed = baseline_decision([analysis(0, 1), analysis(1, -0.5)])
    assert mixed["projector"] == "residual" and mixed["status"] == "provisional_mixed_robustness"
    no_transfer = baseline_decision([analysis(0, -0.1), analysis(1, 1)])
    assert no_transfer["projector"] == "linear"
    with pytest.raises(ValueError):
        baseline_decision([analysis(0, 1)])


def test_refused_controller_lock_does_not_update_state(tmp_path, monkeypatch):
    Study(tmp_path, 36000)
    before = (tmp_path / "study-state.json").read_bytes()
    monkeypatch.setattr(sys, "argv", ["study", "--study-root", str(tmp_path)])

    def refuse(*args):
        raise BlockingIOError("study already running")

    monkeypatch.setattr(workflow.fcntl, "flock", refuse)
    with pytest.raises(BlockingIOError):
        workflow.main()
    assert (tmp_path / "study-state.json").read_bytes() == before


def test_study_itself_owns_lock_before_state_writes(tmp_path):
    with Study(tmp_path, 36000):
        before = (tmp_path / "study-state.json").read_bytes()
        with pytest.raises(BlockingIOError):
            Study(tmp_path, 36000)
        assert (tmp_path / "study-state.json").read_bytes() == before


def test_completed_stage_revalidates_before_skip(tmp_path):
    study = Study(tmp_path, 36000)
    command = [sys.executable, "-c", "pass"]
    study.run_stage("one", command, 10)

    def invalid():
        raise ValueError("completed report byte hash mismatch")

    with pytest.raises(ValueError, match="byte hash"):
        study.run_stage("one", command, 10, completion_validator=invalid)


def test_finished_artifacts_adopted_without_relaunch_or_budget(tmp_path):
    study = Study(tmp_path, 36000)
    forbidden = tmp_path / "relaunched"
    command = [sys.executable, "-c", f"open({str(forbidden)!r}, 'w').close()"]
    study.run_stage("train", command, 14400, completion_validator=lambda: True)
    assert not forbidden.exists()
    assert study.state["stages"]["train"]["status"] == "complete"
    assert study.state["spent_seconds"] == 0


def test_successful_child_requires_complete_artifacts(tmp_path):
    study = Study(tmp_path, 36000)
    with pytest.raises(RuntimeError, match="complet"):
        study.run_stage("train", [sys.executable, "-c", "pass"], 10,
                        completion_validator=lambda: False)
    assert study.state["stages"]["train"]["status"] != "complete"


def test_validation_error_after_child_invalidates_completion(tmp_path):
    study = Study(tmp_path, 36000)
    validations = 0

    def validate():
        nonlocal validations
        validations += 1
        if validations > 1:
            raise ValueError("selected checkpoint byte hash mismatch")
        return False

    with pytest.raises(ValueError, match="byte hash"):
        study.run_stage("one", [sys.executable, "-c", "pass"], 10,
                        completion_validator=validate)
    assert study.state["stages"]["one"]["status"] == "invalid"


def test_command_identity_ignores_only_resume_pointer(tmp_path):
    study = Study(tmp_path, 36000)
    command = [sys.executable, "-c", "pass", "--seed", "1"]
    study.run_stage("one", command, 10)
    with pytest.raises(ValueError, match="command changed"):
        study.run_stage("one", [*command[:-1], "0"], 10)


def _active(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def _wait_until(predicate, seconds=5):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("subprocess condition not reached")


def test_exception_after_spawn_kills_and_reaps_child(tmp_path, monkeypatch):
    study = Study(tmp_path, 36000)
    original_save = study.save
    injected = False

    def interrupt():
        nonlocal injected
        original_save()
        if not injected and study.state["stages"].get("one", {}).get("status") == "running":
            injected = True
            raise KeyboardInterrupt

    monkeypatch.setattr(study, "save", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt):
            study.run_stage("one", [sys.executable, "-c", "import time; time.sleep(30)"], 5)
        pid = study.state["stages"]["one"]["pid"]
        assert not _active(pid)
        assert study.state["stages"]["one"]["status"] == "interrupted"
        assert study.state["spent_seconds"] > 0
    finally:
        pid = study.state["stages"].get("one", {}).get("pid")
        if pid and _active(pid):
            os.killpg(pid, signal.SIGKILL)


def test_second_interrupt_cannot_abort_group_cleanup(tmp_path, monkeypatch):
    marker = tmp_path / "worker.pid"
    command = [sys.executable, "-c", "import os, time; "
               f"open({str(marker)!r}, 'w').write(str(os.getpid())); time.sleep(30)"]
    real_popen = subprocess.Popen
    original_stop = workflow.stop_group

    class InterruptWait(real_popen):
        interrupted = False

        def wait(self, *args, **kwargs):
            if isinstance(self.args, list) and "--signal=INT" in self.args and not self.interrupted:
                self.interrupted = True
                _wait_until(marker.exists)
                raise KeyboardInterrupt
            return super().wait(*args, **kwargs)

    def interrupt_cleanup(pid, grace=0):
        os.kill(os.getpid(), signal.SIGINT)
        original_stop(pid, grace)

    monkeypatch.setattr(workflow.subprocess, "Popen", InterruptWait)
    monkeypatch.setattr(workflow, "stop_group", interrupt_cleanup)
    with Study(tmp_path, 36000) as study:
        try:
            with pytest.raises(KeyboardInterrupt):
                study.run_stage("one", command, 2)
            assert not _active(int(marker.read_text()))
            assert not _active(study.state["stages"]["one"]["pid"])
            assert study.state["stages"]["one"]["status"] == "interrupted"
        finally:
            pid = study.state["stages"].get("one", {}).get("pid")
            if pid and _active(pid):
                os.killpg(pid, signal.SIGKILL)


def test_watchdog_hard_kill_precedes_allowance_and_records_all_reap_time(tmp_path):
    study = Study(tmp_path, 36000)
    markers = tmp_path / "signals.json"
    command = [sys.executable, "-c", "import json, os, signal, time; "
               f"p={str(markers)!r}; "
               "signal.signal(signal.SIGINT, lambda *args: "
               "open(p, 'w').write(json.dumps({'int_at': time.monotonic()}))); "
               "time.sleep(30)"]
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="failed"):
        study.run_stage("timeout", command, 2)
    end = time.monotonic()
    assert markers.exists()
    assert json.loads(markers.read_text())["int_at"] - start < 1.5
    assert end - start < 2.2
    assert study.state["stages"]["timeout"]["timed_out"]
    assert study.state["spent_seconds"] >= end - start - 0.1


def _start_controller(root, cap=3):
    payload = ("import json, os, signal, subprocess, sys, time; from pathlib import Path; "
               "signal.signal(signal.SIGINT, signal.SIG_IGN); "
               "child=subprocess.Popen([sys.executable, '-c', "
               "'import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
               "time.sleep(30)']); "
               f"Path({str(root / 'child-pids.json')!r}).write_text("
               "json.dumps([os.getpid(), child.pid])); time.sleep(30)")
    code = ("import sys; from pathlib import Path; "
            "from scripts.run_projector_overnight import Study; "
            "s=Study(Path(sys.argv[1]), 36000); "
            f"s.run_stage('one', [sys.executable, '-c', {payload!r}], int(sys.argv[2]))")
    controller = subprocess.Popen([sys.executable, "-c", code, str(root), str(cap)],
                                  cwd=workflow.ROOT, env=os.environ.copy())

    def running():
        path = root / "study-state.json"
        if not path.exists():
            return False
        return json.loads(path.read_text())["stages"].get("one", {}).get("status") == "running"

    _wait_until(running)
    _wait_until(lambda: (root / "child-pids.json").exists())
    return controller


def test_watchdog_survives_controller_sigkill(tmp_path):
    controller = _start_controller(tmp_path)
    state = json.loads((tmp_path / "study-state.json").read_text())
    pid = state["stages"]["one"]["pid"]
    workers = json.loads((tmp_path / "child-pids.json").read_text())
    try:
        controller.kill()
        controller.wait(timeout=2)
        _wait_until(lambda: not _active(pid), seconds=4)
        _wait_until(lambda: all(not _active(worker) for worker in workers), seconds=1)
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if _active(pid):
            os.killpg(pid, signal.SIGKILL)


def test_restart_stops_verified_child_and_accounts_before_new_work(tmp_path):
    controller = _start_controller(tmp_path, cap=5)
    pid = json.loads((tmp_path / "study-state.json").read_text())["stages"]["one"]["pid"]
    workers = json.loads((tmp_path / "child-pids.json").read_text())
    try:
        controller.kill()
        controller.wait(timeout=2)
        with Study(tmp_path, 36000) as restarted:
            assert not _active(pid)
            assert all(not _active(worker) for worker in workers)
            assert restarted.state["stages"]["one"]["status"] == "interrupted"
            assert restarted.state["spent_seconds"] > 0
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()
        if _active(pid):
            os.killpg(pid, signal.SIGKILL)


def test_recovery_rejects_reused_pid_without_signalling_it(tmp_path):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 start_new_session=True)
    study = Study(tmp_path, 36000)
    study.state["stages"]["one"] = {
        "status": "running", "pid": unrelated.pid, "command_hash": "not-this-process",
        "process_identity": {"start_ticks": "wrong", "command": ["wrong"]},
        "elapsed_seconds": 0.0, "last_wall": time.time(), "attempts": 1,
    }
    study.save()
    study.close()
    try:
        with pytest.raises(RuntimeError, match="identity"):
            Study(tmp_path, 36000)
        assert unrelated.poll() is None
        state = json.loads((tmp_path / "study-state.json").read_text())
        assert state["status"] == "incomplete"
    finally:
        unrelated.kill()
        unrelated.wait()


@pytest.mark.parametrize("newer_than_record", [False, True])
def test_legacy_child_requires_command_and_launch_time_identity(tmp_path, newer_than_record):
    command = [sys.executable, "-c", "import time; time.sleep(30)"]
    child = subprocess.Popen(command, start_new_session=True)
    if not newer_than_record:
        time.sleep(1.05)  # Legacy btime timestamps have one-second precision.
    with Study(tmp_path, 36000) as study:
        study.state["stages"]["legacy"] = {
            "status": "running", "pid": child.pid, "elapsed_seconds": 0.0,
            "last_wall": time.time() - (20 if newer_than_record else 0), "attempts": 1,
            "command_hash": workflow.command_signature(command),
        }
        study.save()
    try:
        if newer_than_record:
            with pytest.raises(RuntimeError, match="identity"):
                Study(tmp_path, 36000)
            assert child.poll() is None
        else:
            with Study(tmp_path, 36000) as restarted:
                assert child.wait(timeout=1) != 0
                assert restarted.state["stages"]["legacy"]["status"] == "interrupted"
                assert restarted.state["spent_seconds"] > 0
    finally:
        if child.poll() is None:
            child.kill()
        child.wait()


def test_delayed_reap_is_charged_uncapped_and_blocks_further_work(tmp_path, monkeypatch):
    real_popen = subprocess.Popen

    class SlowReap(real_popen):
        delayed = False

        def wait(self, *args, **kwargs):
            result = super().wait(*args, **kwargs)
            if isinstance(self.args, list) and "--signal=INT" in self.args and not self.delayed:
                self.delayed = True
                time.sleep(1.1)
            return result

    monkeypatch.setattr(workflow.subprocess, "Popen", SlowReap)
    study = Study(tmp_path, 1)
    with pytest.raises(RuntimeError, match="failed"):
        study.run_stage("late", [sys.executable, "-c", "pass"], 1)
    assert study.state["spent_seconds"] > 1.1
    assert study.state["stages"]["late"]["elapsed_seconds"] > 1.1
    with pytest.raises(RuntimeError, match="budget"):
        study.run_stage("next", [sys.executable, "-c", "pass"], 1)


def test_final_uncapped_charge_cannot_publish_complete_over_budget(tmp_path, monkeypatch):
    study = Study(tmp_path, 1)
    charge = study._charge

    def late_charge(record, elapsed, clock):
        # Model a delayed final reap debit, after the successful exit was observed.
        charge(record, 1.1 if record["status"] == "complete" else elapsed, clock)

    monkeypatch.setattr(study, "_charge", late_charge)
    with pytest.raises(RuntimeError, match="failed"):
        study.run_stage("late", [sys.executable, "-c", "pass"], 1)
    assert study.state["spent_seconds"] > 1
    assert study.state["stages"]["late"]["status"] == "failed"


def test_main_sigterm_terminates_child_and_publishes_incomplete(tmp_path):
    code = ("import sys; from scripts import run_projector_overnight as w; "
            "w.run_study=lambda s: s.run_stage('one', [sys.executable, '-c', "
            "'import time; time.sleep(30)'], 5); w.main()")
    controller = subprocess.Popen([sys.executable, "-c", code, "--study-root", str(tmp_path)],
                                  cwd=workflow.ROOT)
    path = tmp_path / "study-state.json"

    def running():
        return path.exists() and json.loads(path.read_text())["stages"].get(
            "one", {}).get("status") == "running"

    try:
        _wait_until(running)
        pid = json.loads(path.read_text())["stages"]["one"]["pid"]
        controller.terminate()
        assert controller.wait(timeout=5) == 143
        assert not _active(pid)
        state = json.loads(path.read_text())
        assert state["status"] == "incomplete"
        assert state["stages"]["one"]["status"] == "interrupted"
        assert state["spent_seconds"] > 0
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()


@pytest.fixture
def mocked_pipeline(tmp_path, monkeypatch):
    """Mock only model/data subprocess boundaries; run the controller and its ledger."""
    runs = tmp_path / "runs"
    runs.mkdir()
    monkeypatch.setattr(workflow, "BASE", runs)
    monkeypatch.setattr(workflow, "SEED0", {kind: runs / f"seed0-{kind}"
                                          for kind in ("linear", "residual")})
    study = Study(tmp_path / "study", 36000)
    expected = {kind: {"projector": kind, "seed": 1, "full_identity": "frozen"}
                for kind in ("linear", "residual")}
    original_stage = study.run_stage
    preflights = []
    generated = []

    def cpu_query(command):
        if str(workflow.ROOT / "scripts/evaluate_projector_pilot.py") in command:
            assert command[command.index("--mode") + 1] == "preflight"
            output = Path(command[command.index("--output") + 1])
            preflights.append(output.name)
            if output.exists():
                report = json.loads(output.read_text())
                if report.get("tampered"):
                    raise ValueError("registry completed report byte hash mismatch")
                return {"status": "ready", "registry_status": "complete", "output": str(output)}
            return {"status": "ready", "registry_status": "unclaimed", "output": str(output)}
        kind = command[command.index("--projector") + 1]
        return {"output": str(runs / f"seed1-{kind}"), "identity": expected[kind]}

    def stage(name, command, cap, completion_validator=None):
        if "--output" in command:
            output = Path(command[command.index("--output") + 1])
            seed = int(name[4])
            kind = "residual" if "residual" in name else "linear"
            report = {"training_seed": seed, "kind": kind,
                      "metrics": {"bleu4": 2 if kind == "residual" else 1}}
            child = [sys.executable, "-c", f"from pathlib import Path; "
                     f"Path({str(output)!r}).write_text({json.dumps(report)!r})"]
        elif name.endswith("-train"):
            kind = command[command.index("--projector") + 1]
            run = runs / f"seed1-{kind}"
            child = [sys.executable, "-c", f"from pathlib import Path; "
                     f"p=Path({str(run)!r}); p.mkdir(exist_ok=True); "
                     f"(p/'selected-checkpoint.json').write_text({json.dumps(expected[kind])!r})"]
            if run.exists() and "--resume-checkpoint" not in command:
                child = [sys.executable, "-c", "raise RuntimeError('existing output error')"]
        else:
            child = [sys.executable, "-c", "pass"]
        # Keep the declared command signature despite replacing only the expensive
        # executable at the subprocess boundary.
        real_watchdog = workflow.watchdog_command

        def guard(declared, available, gate):
            assert declared == command
            generated.append(name)
            return real_watchdog(child, available, gate)

        with monkeypatch.context() as patch:
            patch.setattr(workflow, "watchdog_command", guard)
            return original_stage(name, command, cap, completion_validator=completion_validator)

    def validate_training(study, run, identity):
        selected = run / "selected-checkpoint.json"
        if not selected.exists():
            return False
        if json.loads(selected.read_text()) != identity:
            raise ValueError("full training identity mismatch")
        return True

    # CPU analysis is pure arithmetic here; no existing training/model tests run.
    analysis = ModuleType("scripts.analyze_projector_study")
    def compare(linear, residual, **kwargs):
        return {
            "training_seed": linear["training_seed"],
            "metric_deltas": {"bleu4": residual["metrics"]["bleu4"] - linear["metrics"]["bleu4"]},
        }

    analysis.__dict__["compare_projector_reports"] = compare
    monkeypatch.setitem(sys.modules, "scripts.analyze_projector_study", analysis)
    monkeypatch.setattr(study, "cpu_query", cpu_query)
    monkeypatch.setattr(study, "run_stage", stage)
    monkeypatch.setattr(workflow, "validate_training_completion", validate_training, raising=False)
    yield study, runs, expected, preflights, generated
    study.close()


def test_pipeline_adopts_finished_training_and_revalidates_all_reports(mocked_pipeline):
    study, runs, expected, preflights, generated = mocked_pipeline
    for kind in ("linear", "residual"):
        run = runs / f"seed1-{kind}"
        run.mkdir()
        (run / "selected-checkpoint.json").write_text(json.dumps(expected[kind]))
    workflow.run_study(study)
    assert "seed1-linear-train" not in generated
    assert "seed1-residual-train" not in generated
    for name in (f"seed{seed}-{kind}-signer07.json" for seed in (0, 1)
                 for kind in ("linear", "residual")):
        assert preflights.count(name) >= 2  # admission and before analysis
    before = len(generated)
    workflow.run_study(study)
    assert len(generated) == before
    assert study.state["status"] == "complete"
    baseline = json.loads((study.root / "future-description-baseline.json").read_text())
    assert baseline["default_seed"] == 0
    assert baseline["supervision_policy"] == "qwen-schema98-unreviewed-v1"
    assert baseline["human_review_status"] == "not_assessed"


def test_completed_pipeline_report_tampering_blocks_restart(mocked_pipeline):
    study, _, _, _, generated = mocked_pipeline
    workflow.run_study(study)
    (study.root / "seed0-linear-signer07.json").write_text('{"tampered": true}')
    before = len(generated)
    with pytest.raises(ValueError, match="byte hash"):
        workflow.run_study(study)
    assert len(generated) == before


def test_pipeline_revalidates_after_last_stage_before_analysis(mocked_pipeline, monkeypatch):
    study, _, _, _, _ = mocked_pipeline
    original = study.run_stage

    def tamper_after_last(name, *args, **kwargs):
        result = original(name, *args, **kwargs)
        if name == "seed1-residual-heldout":
            (study.root / "seed0-linear-signer07.json").write_text('{"tampered": true}')
        return result

    monkeypatch.setattr(study, "run_stage", tamper_after_last)
    with pytest.raises(ValueError, match="byte hash"):
        workflow.run_study(study)
    assert not (study.root / "summary.json").exists()


def test_finished_training_full_identity_drift_blocks_adoption(mocked_pipeline):
    study, runs, _, _, generated = mocked_pipeline
    run = runs / "seed1-linear"
    run.mkdir()
    (run / "selected-checkpoint.json").write_text('{"full_identity": "changed"}')
    with pytest.raises(ValueError, match="identity"):
        workflow.run_study(study)
    assert "seed1-linear-train" not in generated


def test_pending_pair_resumes_latest_linear_then_binds_residual_control(
        mocked_pipeline, monkeypatch):
    study, runs, _, _, _ = mocked_pipeline
    linear = runs / "seed1-linear"
    linear.mkdir()
    for filename in ("resume-100.ckpt", "step-50.ckpt", "resume-200.ckpt", "not-a-step.ckpt"):
        (linear / filename).write_text("snapshot")
    recorded = []
    original = study.run_stage

    def record(name, command, cap, **kwargs):
        recorded.append((name, command, cap))
        return original(name, command, cap, **kwargs)

    monkeypatch.setattr(study, "run_stage", record)
    workflow.run_study(study)
    train = [(name, command, cap) for name, command, cap in recorded if name.endswith("-train")]
    assert [name for name, _, _ in train] == ["seed1-linear-train", "seed1-residual-train"]
    for _, command, cap in train:
        assert cap == 14400
        assert command[command.index("--seed") + 1] == "1"
        assert command[command.index("--steps") + 1] == "16000"
    assert train[0][1][-2:] == ["--resume-checkpoint", str(linear / "resume-200.ckpt")]
    assert train[1][1][-2:] == ["--matched-control-run", str(linear)]
    assert next(cap for name, _, cap in recorded if name == "seed1-initialization") == 600
    assert all(cap == 1800 for name, _, cap in recorded if name.endswith("-heldout"))
    assert all((linear / filename).exists() for filename in ("resume-100.ckpt", "step-50.ckpt"))


def test_crash_after_artifact_completion_recovers_charge_then_adopts(tmp_path):
    forbidden = tmp_path / "relaunched"
    command = [sys.executable, "-c", f"open({str(forbidden)!r}, 'w').close()"]
    with Study(tmp_path, 36000) as study:
        study.state["spent_seconds"] = 1.0
        study.state["stages"]["train"] = {
            "status": "running", "pid": None, "last_wall": time.time() - 0.25,
            "last_monotonic": time.monotonic() - 0.25, "boot_id": study.boot_id,
            "elapsed_seconds": 1.0, "attempts": 1,
            "command_hash": workflow.command_signature(command),
        }
        study.save()
    with Study(tmp_path, 36000) as restarted:
        restarted.run_stage("train", command, 14400, completion_validator=lambda: True)
        assert not forbidden.exists()
        assert restarted.state["spent_seconds"] >= 1.25
        assert restarted.state["stages"]["train"]["status"] == "complete"
        assert restarted.state["stages"]["train"]["attempts"] == 1


def test_unobserved_downtime_consumes_budget_conservatively(tmp_path):
    with Study(tmp_path, 1) as study:
        study.state["stages"]["old"] = {
            "status": "running", "pid": None, "last_wall": time.time() - 3,
            "elapsed_seconds": 0.0, "attempts": 1,
            "command_hash": workflow.command_signature([sys.executable, "-c", "pass"]),
        }
        study.save()
    with Study(tmp_path, 1) as restarted:
        assert restarted.state["spent_seconds"] >= 3
        with pytest.raises(RuntimeError, match="budget"):
            restarted.run_stage("new", [sys.executable, "-c", "pass"], 10)


def test_cross_boot_recovery_reserves_only_uncharged_allowance(tmp_path):
    command = [sys.executable, "-c", "pass"]
    with Study(tmp_path, 20) as study:
        record = {"status": "running", "pid": None, "elapsed_seconds": 0.0,
                  "reserved_seconds": 10.0, "attempts": 1,
                  "command_hash": workflow.command_signature(command)}
        study.state["stages"]["train"] = record
        study._charge(record, 2.0, time.monotonic())
        record["boot_id"] = "previous-boot"
        study.save()
    with Study(tmp_path, 20) as restarted:
        restarted.run_stage("train", command, 10, completion_validator=lambda: True)
        assert restarted.state["spent_seconds"] == 10
        assert restarted.state["stages"]["train"]["status"] == "complete"


def test_cpu_queries_charge_child_wall_and_parse_json(tmp_path):
    with Study(tmp_path, 36000) as study:
        result = study.cpu_query([sys.executable, "-c",
                                  'print(\'{"status": "ready", "x": null}\')'])
        assert result == {"status": "ready", "x": None}
        assert study.state["spent_seconds"] > 0
        assert study.state["stages"]["preflight-1"]["status"] == "complete"


def test_sigkill_before_pid_publication_cannot_launch_gpu_payload(tmp_path):
    marker = tmp_path / "forbidden-payload"
    command = [sys.executable, "-c", f"open({str(marker)!r}, 'w').close()"]
    code = """import os, signal, sys
from pathlib import Path
from scripts.run_projector_overnight import Study
study = Study(Path(sys.argv[1]), 36000)
save = study.save
def crash_before_pid_save():
    if study.state['stages'].get('one', {}).get('status') == 'running':
        os.kill(os.getpid(), signal.SIGKILL)
    save()
study.save = crash_before_pid_save
study.run_stage('one', """ + repr(command) + ", 3)"
    controller = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], cwd=workflow.ROOT)
    try:
        assert controller.wait(timeout=5) == -signal.SIGKILL
        with Study(tmp_path, 36000) as recovered:
            assert recovered.state["stages"]["one"]["status"] == "interrupted"
            assert recovered.state["spent_seconds"] > 0
        time.sleep(0.1)
        assert not marker.exists()
    finally:
        if controller.poll() is None:
            controller.kill()
            controller.wait()


def _execute_query_code(command, monkeypatch):
    """Execute the controller's CPU query code against mocked shared APIs."""
    output = io.StringIO()
    with monkeypatch.context() as patch:
        patch.setattr(sys, "argv", ["-c", *command[3:]])
        patch.setattr(sys, "stdout", output)
        exec(command[2], {})
    return ast.literal_eval(output.getvalue().splitlines()[-1])


def test_training_preflight_returns_full_prepared_identity(tmp_path, monkeypatch):
    trainer = ModuleType("scripts.train_projector_pilot")
    identity = {"seed": 1, "code_hashes": {"model.py": "frozen"}, "config": {"lr": 0.0003}}
    trainer.__dict__["prepare_projector"] = lambda args: (
        identity, tmp_path / "run", None, None, None)
    trainer.__dict__["main"] = lambda: trainer.prepare_projector(None)
    monkeypatch.setitem(sys.modules, "scripts.train_projector_pilot", trainer)
    import scripts

    monkeypatch.setattr(scripts, "train_projector_pilot", trainer, raising=False)
    ready = _execute_query_code(workflow.training_preflight_command(
        "linear", tmp_path / "profile.json", None), monkeypatch)
    assert ready["identity"] == identity
    assert ready["output"] == str(tmp_path / "run")


@pytest.mark.parametrize("changed", [False, True])
def test_training_completion_binds_full_identity_via_selection_validator(
        tmp_path, monkeypatch, changed):
    study = Study(tmp_path / "study", 36000)
    run = tmp_path / "run"
    run.mkdir()
    (run / "selected-checkpoint.json").write_text("{}")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"split": {"groups": {"dev": {"clip_ids": ["dev-1"]}}}}))
    monkeypatch.setitem(workflow.INPUTS, "protocol", protocol)
    evaluator = ModuleType("scripts.evaluate_projector_pilot")
    identity = {"seed": 1, "code_hashes": {"model.py": "frozen"}}

    def selection(root, protocol, dev_ids):
        assert root == run and dev_ids == ("dev-1",)
        return {"identity": identity}

    evaluator.__dict__["validate_projector_selection"] = selection
    monkeypatch.setitem(sys.modules, "scripts.evaluate_projector_pilot", evaluator)
    monkeypatch.setattr(study, "cpu_query",
                        lambda command: _execute_query_code(command, monkeypatch))
    expected = {**identity, "code_hashes": {"model.py": "changed"}} if changed else identity
    try:
        if changed:
            with pytest.raises(ValueError, match="full training identity"):
                workflow.validate_training_completion(study, run, expected)
        else:
            assert workflow.validate_training_completion(study, run, expected)
    finally:
        study.close()

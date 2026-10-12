"""Coordinator regressions; synthetic receipts and short CPU children only."""

import argparse
import importlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

import pytest


@pytest.fixture
def api():
    try:
        return importlib.import_module("scripts.run_motion_fusion_pipeline")
    except ModuleNotFoundError:
        pass
    class LazyAPI:
        def __getattr__(self, name):
            try:
                return getattr(importlib.import_module("scripts.run_motion_fusion_pipeline"), name)
            except ModuleNotFoundError:
                pytest.fail("Task4 coordinator missing")
    return LazyAPI()


def fixture_identity(api, tmp_path):
    source = tmp_path / "source.py"
    source.write_text("fixture source\n")
    return {"schema_version": 1, "policy": api.POLICY,
            "paths": {"pipeline_root": str(tmp_path / "pipeline"),
                      "extractor_python": sys.executable,
                      "training_root": str(tmp_path / "training"),
                      "output_base": str(tmp_path), "adapted_root": str(tmp_path / "adapted"),
                      "annotation": str(tmp_path / "annotation"),
                      "model_snapshot": str(tmp_path / "snapshot"),
                      "environment_lock": str(tmp_path / "lock")},
            "sources": {str(source): api.file_hash(source)},
            "authorization": {"extraction_cap_seconds": 18000,
                              "training_cap_seconds": 64800, "cap_transfer": False}}


def pipeline(api, tmp_path):
    identity = fixture_identity(api, tmp_path)
    return api.MotionFusionPipeline(Path(identity["paths"]["pipeline_root"]), identity)


def acquire_resume_before(q, deadline):
    """Wait on actual exclusive acquisition, retaining custody after success."""
    before = q.path.read_bytes()
    while True:
        try:
            return q.__enter__()
        except BlockingIOError:
            assert q.path.read_bytes() == before
            if time.monotonic() >= deadline:
                raise
            time.sleep(.005)


def test_bounded_resume_acquisition_rejects_persistent_foreign_lock(api, tmp_path, monkeypatch):
    with pipeline(api, tmp_path) as incumbent:
        before = incumbent.path.read_bytes()
        q = api.MotionFusionPipeline(incumbent.root, incumbent.identity)
        q.resume = True
        monkeypatch.setattr(api, "stop_owned", lambda *a, **k: pytest.fail("signalled holder"))
        monkeypatch.setattr(q, "save", lambda: pytest.fail("write under contention"))
        with pytest.raises(BlockingIOError):
            acquire_resume_before(q, time.monotonic() + .05)
        assert incumbent.path.read_bytes() == before


def admitted_fixture(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.state.update(status="closing_pending", external_worker={"fixture": True},
                       motion_package={"root": str(tmp_path / "cache")},
                       summary={"path": str(tmp_path / "summary.json"), "sha256": "fixture"})
        p.state["clocks"]["extraction"]["outer_seconds"] = 10.
        p.state["clocks"]["training"]["outer_closing_upper_bound_seconds"] = 20.
        p.state["terminal_closure"] = {
            "policy": "motion-fusion-pipeline-terminal-v2", "nonce": "fixture",
            "boot_id": api.boot_id(), "started_monotonic": 1., "deadline_monotonic": 5.,
            "upper_outer_seconds": 20., **{name + "_path": str(p.root / (
                "pipeline-" + name + "-fixture.json"))
                for name in ("admission", "seal", "failure")}}
        p.save()
    proof = {"policy": "motion-fusion-pipeline-admission-v2", "nonce": "fixture",
             "status": "complete", "identity_hash": api.digest(p.identity),
             "candidate_sha256": api.file_hash(p.path), "worker_returncode": 0,
             "worker": p.state["external_worker"], "boot_id": api.boot_id(),
             "worker_exited_monotonic": 2., "parent_checked_monotonic": 3.,
             "caller_deadline_monotonic": 5., "training_outer_upper_seconds": 20.}
    api.atomic_json(Path(p.state["terminal_closure"]["admission_path"]), proof)
    api.atomic_json(Path(p.state["terminal_closure"]["seal_path"]), {
        "proof": proof, "boot_id": api.boot_id(), "caller_cleanup_checked_monotonic": 4.})
    for path in (tmp_path / "motion-extraction-bootstrap-v1/.writer.lock",
                 tmp_path / "cache/.writer.lock", tmp_path / "training/.fusion-controller.lock",
                 tmp_path / "training/seed0/.study.lock", tmp_path / "training/seed1/.study.lock"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    return p


@pytest.mark.parametrize("cli", [False, True])
def test_public_completed_resume_is_readonly_without_dispatch(api, tmp_path, monkeypatch, cli):
    p = admitted_fixture(api, tmp_path)
    before = {str(path): path.read_bytes() for path in p.root.glob("*.json")}
    expected = api.admitted_pipeline_snapshot(p.root, p.identity, api.read_json(p.path))
    checks = []

    def readonly_query(python, code, path, timeout=300, **kwargs):
        checks.append(api.read_json(path))
        assert kwargs["pass_fds"]
        return expected["summary"]

    monkeypatch.setattr(api, "bounded_cpu_query", readonly_query)
    monkeypatch.setattr(api.os, "fork", lambda: pytest.fail("completed resume forked extraction"))
    monkeypatch.setattr(api.MotionFusionPipeline, "execute", lambda *a, **k:
                        pytest.fail("completed resume dispatched stage"))
    if cli:
        monkeypatch.setattr(api, "pipeline_identity", lambda args: p.identity)
        api.main(["--pipeline-root", str(p.root), "--resume",
                  "--authorize-motion-fusion-pipeline", *[v for k in (
                      "adapted-root", "annotation", "model-snapshot", "extractor-python",
                      "environment-lock") for v in ("--" + k, p.identity["paths"][
                          k.replace("-", "_")])]])
    else:
        q = api.MotionFusionPipeline(p.root, p.identity)
        q.resume = True
        result = q.run()
        assert result["status"] == "complete"
        assert result["extraction_seconds"] == 10. and result["training_seconds"] == 20.
    assert len(checks) == 1
    assert all(Path(path).read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("error", [ValueError("tampered completion/source"),
                                   TimeoutError("bounded readonly validation")])
def test_public_completed_resume_rejection_preserves_closed_bytes(
    api, tmp_path, monkeypatch, error,
):
    p = admitted_fixture(api, tmp_path)
    protected = [*p.root.glob("*.json"), tmp_path / "source.py"]
    for name in ("budget.json", "candidate.json", "admission.json", "seal.json"):
        path = tmp_path / "cache" / name
        path.write_text("closed immutable fixture\n")
        protected.append(path)
    before = {path: path.read_bytes() for path in protected}

    def reject(*args, **kwargs):
        raise error

    monkeypatch.setattr(api, "bounded_cpu_query", reject)
    monkeypatch.setattr(api.os, "fork", lambda: pytest.fail("rejected resume dispatched"))
    q = api.MotionFusionPipeline(p.root, p.identity)
    q.resume = True
    with pytest.raises(type(error), match=str(error)):
        q.run()
    assert all(path.read_bytes() == data for path, data in before.items())


@pytest.mark.parametrize("mutation", ["good", "source", "completion", "summary", "admission"])
def test_completed_resume_revalidates_package_summary_sources_without_rewriting(
    api, tmp_path, monkeypatch, mutation,
):
    from despamo.data import new_motion

    p = admitted_fixture(api, tmp_path)
    raw = api.read_json(p.path)
    query_path = p.root / "original-package-query.json"
    api.atomic_json(query_path, {"query_request_path": str(p.root / "original-clock-query.json")})
    raw.update(extraction_result_path=str(p.root / "result.json"), extraction_outer_receipt={},
               extraction_outer_request={}, clock_facts={})
    command = [sys.executable, "-c", "pass", str(query_path)]
    raw["stages"] = [{"name": "full-package-admission", "bucket": "extraction",
                      "command": command, "signature": api.digest(command), "status": "complete"}]
    stale_query = p.root / "prior-attempt-query.json"
    api.atomic_json(stale_query, {"query_request_path": str(p.root / "prior-clock-query.json")})
    old_command = [sys.executable, "-c", "pass", str(stale_query)]
    raw["stages"].insert(0, {"name": "full-package-admission", "bucket": "extraction",
        "command": old_command, "signature": api.digest(old_command), "status": "complete"})
    api.atomic_json(p.path, raw)
    proof_path = Path(raw["terminal_closure"]["admission_path"])
    seal_path = Path(raw["terminal_closure"]["seal_path"])
    proof = api.read_json(proof_path)
    proof["candidate_sha256"] = api.file_hash(p.path)
    api.atomic_json(proof_path, proof)
    seal = api.read_json(seal_path)
    seal["proof"] = proof
    api.atomic_json(seal_path, seal)
    projected = api.admitted_pipeline_snapshot(p.root, p.identity, raw)
    admissions = []

    def package_admission(root, request):
        admissions.append("package")
        assert request["query_request_path"].endswith("original-clock-query.json")
        return {"root": "foreign"} if mutation == "completion" else projected["motion_package"]

    def summary_admission(request):
        admissions.append("summary")
        if mutation == "admission":
            proof["worker_returncode"] = 1
            api.atomic_json(proof_path, proof)
        return {"changed": True} if mutation == "summary" else projected["summary"]

    monkeypatch.setattr(api, "validate_extraction_handoff", package_admission)
    monkeypatch.setattr(new_motion, "compact_motion_binding", lambda package: package)
    monkeypatch.setattr(api, "validate_final_summary", summary_admission)
    if mutation == "source":
        (tmp_path / "source.py").write_text("changed bytes")
    before = p.path.read_bytes()
    request = {"identity": p.identity, "candidate_path": str(p.path),
               "candidate_sha256": api.file_hash(p.path), "training_leases": []}
    if mutation == "good":
        assert api.validate_completed_pipeline(request) == projected["summary"]
        assert admissions == ["package", "summary"]
    else:
        with pytest.raises(ValueError):
            api.validate_completed_pipeline(request)
    assert p.path.read_bytes() == before


def test_separate_caps_and_overlapping_clocks(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.begin("extraction")
        p.observe_inner("extraction", 100)
        p.state["clocks"]["extraction"]["outer_seconds"] = 80
        assert p.charged("extraction") == 100
        p.state["clocks"]["extraction"]["outer_seconds"] = 130
        assert p.charged("extraction") == 130
        assert p.remaining("training") == 64800
        p.state["clocks"]["extraction"]["outer_seconds"] = 18001
        with pytest.raises(TimeoutError):
            p.check_cap("extraction")


@pytest.mark.parametrize("value", [-1, True, float("nan"), float("inf")])
def test_invalid_inner_clock_rejected(api, tmp_path, value):
    with pipeline(api, tmp_path) as p:
        with pytest.raises(ValueError):
            p.observe_inner("extraction", value)


def test_lock_conflict_and_corrupt_state_preserved(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        before = p.path.read_bytes()
        with pytest.raises((BlockingIOError, ValueError)):
            with pipeline(api, tmp_path):
                pass
        assert p.path.read_bytes() == before
    p.path.write_text('{"foreign": true}')
    with pytest.raises(ValueError):
        with pipeline(api, tmp_path):
            pass
    assert p.path.read_text() == '{"foreign": true}'


def test_resume_requires_flag_and_identity_no_reset(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.state["clocks"]["extraction"]["outer_seconds"] = 17980
        p.save()
    with pytest.raises(ValueError):
        with pipeline(api, tmp_path):
            pass
    q = pipeline(api, tmp_path)
    q.resume = True
    with q:
        assert q.remaining("extraction") == 20
        assert q.remaining("training") == 64800
    changed = fixture_identity(api, tmp_path)
    changed["paths"]["model_snapshot"] += "-new"
    with pytest.raises(ValueError):
        with api.MotionFusionPipeline(p.root, changed):
            pass


def test_source_drift_before_dispatch(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        Path(next(iter(p.identity["sources"]))).write_text("changed")
        with pytest.raises(ValueError):
            p.execute("extraction", "preflight", [sys.executable, "-c", "print('{}')"])
        assert not list(p.root.glob("*stdout*"))


def test_short_child_result_and_consumer_tail_charged(api, tmp_path):
    with pipeline(api, tmp_path) as p:
        start = time.monotonic()
        p.begin("extraction")

        def consume(path, receipt):
            assert json.loads(path.read_text()) == {"ok": True}
            assert receipt["returncode"] == 0
            time.sleep(.12)
            return "admitted"

        result = p.execute("extraction", "cpu", [sys.executable, "-c",
                           "import json; print(json.dumps({'ok': True}))"], consume)
        assert result == "admitted"
        assert p.charged("extraction") >= .12
        assert p.charged("extraction") <= time.monotonic() - start + .1


def test_nonzero_and_tail_timeout_never_complete(api, tmp_path):
    with pipeline(api, tmp_path) as p:
        p.begin("extraction")
        with pytest.raises(RuntimeError):
            p.execute("extraction", "bad", [sys.executable, "-c", "print('{}'); exit(7)"])
        assert p.state["stages"][-1]["status"] != "complete"
        assert p.state["stages"][-1]["returncode"] == 7


def test_owned_separate_session_grandchild_stopped(api, tmp_path):
    marker = tmp_path / "grandchild.json"
    command = [sys.executable, "-c",
               "import subprocess,sys,time,json; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'], "
               "start_new_session=True); "
               f"open({str(marker)!r},'w').write(json.dumps({{'pid':p.pid}})); "
               "time.sleep(20)"]
    with pipeline(api, tmp_path) as p:
        p.begin("extraction")
        with pytest.raises(TimeoutError):
            p.execute("extraction", "timeout", command, timeout=.5)
        pid = json.loads(marker.read_text())["pid"]
        assert api.process_identity(pid) is None
        assert p.state["stages"][-1]["status"] == "capped"


def test_pid_reuse_and_foreign_boot_not_signalled(api, monkeypatch):
    calls = []
    monkeypatch.setattr(api.os, "killpg", lambda *args: calls.append(args))
    monkeypatch.setattr(api, "process_identity", lambda pid: {
        "start_ticks": "new", "pgid": pid, "session": pid, "command": ["sleep"]})
    receipt = {"pid": 100, "boot_id": api.boot_id(), "identity": {
        "start_ticks": "old", "pgid": 100, "session": 100, "command": ["sleep"]}}
    with pytest.raises(ValueError):
        api.stop_owned(receipt)
    receipt["boot_id"] = "other-boot"
    with pytest.raises(ValueError):
        api.stop_owned(receipt)
    assert calls == []


@pytest.mark.parametrize("linux,windows", [(100, 10**15), (10**15, 100)])
def test_both_disks_fail_independently(api, tmp_path, linux, windows):
    with pytest.raises(ValueError):
        api.storage_admission(tmp_path, {}, linux_free=linux, windows_free=windows)


def test_projection_uses_allocated_credit_not_sparse_size(api, tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "step-5500.ckpt"
    with checkpoint.open("wb") as f:
        f.truncate(10**9)
    runs = {"0-reference": {"run": str(run), "complete": False}}
    projection = api.storage_admission(tmp_path, runs, linux_free=10**15,
                                      windows_free=10**15, checkpoint_bytes=1000)
    assert projection["remaining_training_bytes"] == 31000
    checkpoint.write_bytes(b"x" * 4096)
    projection = api.storage_admission(tmp_path, runs, linux_free=10**15,
                                      windows_free=10**15, checkpoint_bytes=1000)
    assert projection["remaining_training_bytes"] == 31000 - min(
        5000, checkpoint.stat().st_blocks * 512)


def test_gpu_overlap_rejected(api, monkeypatch):
    monkeypatch.setattr(api.subprocess, "run", lambda *a, **k: argparse.Namespace(
        stdout="GPU-owned,123\n", returncode=0))
    with pytest.raises(ValueError):
        api.require_gpu_idle()


def test_cli_help_and_missing_authorization_cpu_only(api):
    result = subprocess.run([sys.executable, str(Path(api.__file__)), "--help"],
                            capture_output=True, text=True, timeout=5)
    assert result.returncode == 0
    assert "--authorize-motion-fusion-pipeline" in result.stdout
    assert "--gpu-cap-seconds" not in result.stdout


def test_no_main_gpu_extractor_import(api):
    assert "tools.motion.extractor" not in sys.modules


def test_exit_zero_without_result_rejected(api, tmp_path):
    with pipeline(api, tmp_path) as p:
        with pytest.raises(ValueError):
            p.execute("extraction", "empty", [sys.executable, "-c", "pass"],
                      lambda path, receipt: api.delivered_result(path))


def test_reboot_reconciliation_does_not_refund(api, tmp_path, monkeypatch):
    p = pipeline(api, tmp_path)
    with p:
        p.begin("extraction")
        p.state["active"]["boot_id"] = "old-boot"
        p.state["active"]["wall"] = time.time() + 100
        p.save()
    q = pipeline(api, tmp_path)
    q.resume = True
    with q:
        assert q.charged("extraction") >= 18000
        assert q.state["status"] in {"interrupted", "capped"}
        assert q.charged("training") == 0


def test_uncertain_stage_receipt_rejects_without_relaunch(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.begin("extraction")
        p.state["stages"].append({"status": "launching", "bucket": "extraction",
                                  "command": ["unknown"], "receipt": None})
        p.save()
    before = p.path.read_bytes()
    q = pipeline(api, tmp_path)
    q.resume = True
    with pytest.raises(ValueError):
        with q:
            pass
    assert p.path.read_bytes() == before


def exact_result(api, mode, identity, root):
    result = dict.fromkeys(api.RESULT_KEYS)
    result.update(schema_version=1, policy="motion-extraction-result-v1", mode=mode,
                  status={"preflight": "ready", "profile": "profiled", "run": "complete"}[mode],
                  featurekey=api.digest(identity), identity=identity, root=str(root),
                  authorization=api.AUTHORIZATION, spent_seconds=5, remaining_seconds=17995,
                  admission_seconds=0)
    if mode == "run":
        result["package"] = {"fixture": True}
    return result


@pytest.mark.parametrize("mutation", ["failed", "capped", "feature", "hardware", "missingkey"])
def test_bad_result_never_dispatches_training(api, tmp_path, mutation):
    p = pipeline(api, tmp_path)
    extraction_identity = {"hardware": {"uuid": "fixture-gpu"}, "source": "fixture"}
    root = tmp_path / ("motion-features-" + api.digest(extraction_identity))
    with p:
        p.state["extraction_identity"] = extraction_identity
        result = exact_result(api, "run", extraction_identity, root)
        if mutation in {"failed", "capped"}:
            result.update(status=mutation, identity=None, featurekey=None, root=None, package=None)
        elif mutation == "feature":
            result["featurekey"] = "foreign"
        elif mutation == "hardware":
            result["identity"] = {**extraction_identity, "hardware": {"uuid": "other"}}
            result["featurekey"] = api.digest(result["identity"])
        else:
            del result["terminal_admission"]
        output = tmp_path / "result.jsonl"
        output.write_text(json.dumps(result) + "\n")
        with pytest.raises(ValueError):
            p.admit_result("run", output, {"returncode": 0})
        assert p.charged("training") == 0
        assert not any(s.get("name") == "new-fusion-controller" for s in p.state["stages"])


@pytest.mark.parametrize("exit_failure", [False, True, "delivery", "queued-ready",
                                        "queued-unknown", "queued-nonce",
                                        "queued-chronology", "queued-descriptor",
                                        "queued-empty-unknown"])
def test_success_cpu_pipeline_dispatches_new_mode_exactly_once(
    api, tmp_path, monkeypatch, exit_failure,
):
    p = pipeline(api, tmp_path)
    extraction_identity = {"source": "fixture", "hardware": {"uuid": "fixture-gpu"}}
    root = tmp_path / ("motion-features-" + api.digest(extraction_identity))
    graph = {"root": str(root), "timing_path": str(root / "complete/motion-timing.json")}
    dispatches = []
    study = tmp_path / "training"
    study.mkdir()
    for path in (study / ".fusion-controller.lock", study / "seed0/.study.lock",
                 study / "seed1/.study.lock"):
        path.parent.mkdir(exist_ok=True)
        path.touch()
    actual_execute = p.execute

    def extraction_command(mode):
        result = exact_result(api, mode, extraction_identity, root)
        return [sys.executable, "-c", f"print({json.dumps(result)!r})"]

    def cpu_execute(bucket, name, command, consume=None, **kwargs):
        if name == "new-fusion-controller":
            dispatches.append(command)
            (tmp_path / "dispatch.json").write_text(json.dumps(command))
            command = [sys.executable, "-c", "print(" + repr(json.dumps({
                "status": "complete", "summary": str(tmp_path / "training/summary.json")})) + ")"]
        return actual_execute(bucket, name, command, consume, **kwargs)

    def query(bucket, name, python, code, request):
        if name == "producer-projection":
            return {"status": "absent", "spent_seconds": 0}
        if name == "clock-admission":
            return {"status": "profiled" if p.state["status"] == "profiling" else "complete",
                    "spent_seconds": 5}
        if name == "final-six-run-admission":
            return {"path": str(tmp_path / "training/summary.json"), "sha256": "fixture",
                    "inner_seconds": 5}
        raise AssertionError(name)

    monkeypatch.setattr(p, "extraction_command", extraction_command)
    monkeypatch.setattr(p, "execute", cpu_execute)
    monkeypatch.setattr(p, "query", query)
    monkeypatch.setattr(p, "gpu_admission", lambda **kwargs: None)
    monkeypatch.setattr(p, "acquire_handoff_locks", lambda root: None)
    monkeypatch.setattr(p, "package_query", lambda: graph)
    delivered = tmp_path / "operator-receipt.json"

    def deliver(receipt):
        assert p.budget_guard.poll() is None
        for lock in (study / ".fusion-controller.lock", study / "seed0/.study.lock",
                     study / "seed1/.study.lock"):
            competitor = subprocess.run([sys.executable, "-c",
                "import fcntl,sys; f=open(sys.argv[1]); "
                "\ntry: fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)"
                "\nexcept BlockingIOError: sys.exit(17)", str(lock)], timeout=2)
            assert competitor.returncode == 17
        if exit_failure == "delivery":
            raise OSError("synthetic operator delivery failure")
        delivered.write_text(json.dumps(receipt))

    p.delivery = deliver
    if isinstance(exit_failure, str) and exit_failure.startswith("queued-"):
        real_waitpid = api.os.waitpid
        real_select = api.select.select
        owner_pid = os.getpid()
        closing_sent = tmp_path / "closing-sent"
        worker_exited = tmp_path / "worker-exited"
        actual_event = p.worker_event

        def event(value, leases=None):
            actual_event(value, leases)
            if value["kind"] == "closing":
                closing_sent.touch()
            if value["kind"] == "worker-return-ready" and exit_failure == "queued-unknown":
                actual_event({"kind": "unexpected-queued-packet"})
            if value["kind"] == "worker-return-ready" and exit_failure == "queued-empty-unknown":
                if p.worker_channel.type == api.socket.SOCK_SEQPACKET:
                    p.worker_channel.sendmsg([b""])
                else:
                    p.worker_channel.sendall(struct.pack("!I", 0))
                actual_event({"kind": "unexpected-queued-packet"})
            if value["kind"] == "worker-return-ready" and exit_failure in {
                    "queued-nonce", "queued-chronology", "queued-descriptor"}:
                packet = {"kind": "worker-return-ready", "nonce": p.external_nonce,
                          "boot_id": api.boot_id(), "sequence": p.worker_sequence + 1,
                          "sent_monotonic": time.monotonic()}
                ancillary = []
                if exit_failure == "queued-nonce":
                    packet["nonce"] = "foreign"
                elif exit_failure == "queued-chronology":
                    packet["sent_monotonic"] = 0
                else:
                    with (tmp_path / "source.py").open("r") as fd:
                        ancillary = [(api.socket.SOL_SOCKET, api.socket.SCM_RIGHTS,
                                      api.array.array("i", [fd.fileno()]))]
                        body = json.dumps(packet).encode()
                        wire = (body if p.worker_channel.type == api.socket.SOCK_SEQPACKET
                                else struct.pack("!I", len(body)) + body)
                        p.worker_channel.sendmsg([wire], ancillary)
                    return
                body = json.dumps(packet).encode()
                wire = (body if p.worker_channel.type == api.socket.SOCK_SEQPACKET
                        else struct.pack("!I", len(body)) + body)
                p.worker_channel.sendmsg([wire], ancillary)

        def waitpid(pid, flags):
            if os.getpid() == owner_pid and closing_sent.exists() and not worker_exited.exists():
                result = real_waitpid(pid, 0)
                worker_exited.touch()
                return result
            return real_waitpid(pid, flags)

        def select(readers, writers, errors, timeout=None):
            if (os.getpid() == owner_pid and closing_sent.exists()
                    and not worker_exited.exists()
                    and timeout == .01
                    and any(isinstance(fd, api.socket.socket) for fd in readers)):
                return [], [], []
            return real_select(readers, writers, errors, timeout)

        monkeypatch.setattr(p, "worker_event", event)
        monkeypatch.setattr(api.os, "waitpid", waitpid)
        monkeypatch.setattr(api.select, "select", select)
        if exit_failure != "queued-ready":
            with pytest.raises(ValueError, match="worker IPC"):
                p.run()
            assert not delivered.exists()
            assert not list(p.root.glob("pipeline-admission-*.json"))
            return
    if exit_failure is True:
        actual_exit = p.__exit__
        def failed_exit(*args):
            actual_exit(*args)
            raise OSError("synthetic failure after coordinator context cleanup")
        p.__exit__ = failed_exit
        with pytest.raises(RuntimeError, match="worker exit failed"):
            p.run()
        assert not delivered.exists()
        assert not list(p.root.glob("pipeline-admission-*.json"))
        q = api.MotionFusionPipeline(p.root, p.identity)
        q.resume = True
        acquire_resume_before(q, time.monotonic() + 3.)
        try:
            assert q.state["status"] != "complete"
        finally:
            q.__exit__(None, None, None)
        return
    if exit_failure == "delivery":
        with pytest.raises(OSError, match="operator delivery failure"):
            p.run()
        assert list(p.root.glob("pipeline-admission-*.json"))
        assert list(p.root.glob("pipeline-failure-*.json"))
        q = api.MotionFusionPipeline(p.root, p.identity)
        q.resume = True
        with q:
            assert q.state["status"] != "complete"
        return
    result = p.run()
    assert delivered.is_file()
    assert json.loads(delivered.read_text())["status"] == "complete"
    assert result["status"] == "complete"
    command = json.loads((tmp_path / "dispatch.json").read_text())
    assert command[command.index("--motion-package") + 1] == str(root)
    assert command[command.index("--motion-timing") + 1] == graph["timing_path"]
    assert command[command.index("--study-root") + 1] == str(tmp_path / "training")
    assert "--authorize-fusion-study" in command
    assert result["extraction_seconds"] == 5
    assert 5 <= result["training_seconds"] < 10
    # Closed success survives downtime without billing an already finished job.
    q = api.MotionFusionPipeline(p.root, p.identity)
    q.resume = True
    real_monotonic = api.time.monotonic
    monkeypatch.setattr(api.time, "monotonic", lambda: real_monotonic() + 70000)
    raw_before = p.path.read_bytes()
    with q:
        assert q.state["active"] is None
        assert q.remaining("training") > 64000
    assert p.path.read_bytes() == raw_before
    raw = api.read_json(p.path)
    proof_path = Path(raw["terminal_closure"]["admission_path"])
    proof = api.read_json(proof_path)
    original = proof_path.read_bytes()
    proof["worker_returncode"] = 1
    api.atomic_json(proof_path, proof)
    with pytest.raises(ValueError):
        api.admitted_pipeline_snapshot(p.root, p.identity, raw)
    proof_path.write_bytes(original)


def test_consumer_tail_timeout_is_owned_and_not_complete(api, tmp_path):
    script = """
import sys,time
from pathlib import Path
from scripts.run_motion_fusion_pipeline import MotionFusionPipeline,read_json
identity=read_json(Path(sys.argv[1]))
p=MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity)
with p:
    try:
        p.execute('extraction','tail',[sys.executable,'-c',"print('{}')"],
                  lambda *a:time.sleep(20),timeout=.3)
    except TimeoutError:
        pass
"""
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    child = subprocess.run([sys.executable, "-c", script, str(config)],
                           capture_output=True, text=True, timeout=5)
    assert child.returncode == 0, child.stderr
    state = api.read_json(Path(identity["paths"]["pipeline_root"]) / "state.json")
    assert state["stages"][-1]["status"] == "capped"
    assert state["clocks"]["extraction"]["outer_seconds"] >= .3


def test_disconnect_stops_orphan_separate_session_and_retains_lock(api, tmp_path):
    marker = tmp_path / "child.json"
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    script = """
import sys,time
from pathlib import Path
from scripts.run_motion_fusion_pipeline import MotionFusionPipeline,read_json
identity=read_json(Path(sys.argv[1]))
p=MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity)
cmd=[sys.executable,'-c',"import subprocess,sys,time,json; "
    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],"
    "start_new_session=True); "
    "open(sys.argv[1],'w').write(json.dumps({'pid':p.pid})); time.sleep(20)",sys.argv[2]]
with p:
    p.execute('extraction','disconnect',cmd,timeout=2)
"""
    child = subprocess.Popen([sys.executable, "-c", script, str(config), str(marker)])
    try:
        until = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert marker.exists()
        grandchild = json.loads(marker.read_text())["pid"]
        child.kill()
        child.wait(timeout=3)
        until = time.monotonic() + 3
        while api.process_identity(grandchild) is not None and time.monotonic() < until:
            time.sleep(.01)
        assert api.process_identity(grandchild) is None
        state_path = Path(identity["paths"]["pipeline_root"]) / "state.json"
        stage = api.read_json(state_path)["stages"][-1]
        watchdog = api.read_json(Path(stage["receipt"]))["watchdog"]["pid"]
        until = time.monotonic() + 3
        while api.process_identity(watchdog) is not None and time.monotonic() < until:
            time.sleep(.01)
        assert api.process_identity(watchdog) is None
        q = api.MotionFusionPipeline(Path(identity["paths"]["pipeline_root"]), identity)
        q.resume = True
        acquire_resume_before(q, time.monotonic() + 3.)
        try:
            assert q.state["stages"][-1]["status"] == "interrupted"
            assert q.charged("extraction") > 0
        finally:
            q.__exit__(None, None, None)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=3)


def test_identity_rejects_noncanonical_root_before_write(api, tmp_path):
    identity = fixture_identity(api, tmp_path)
    root = tmp_path / "unused/../pipeline"
    identity["paths"]["pipeline_root"] = str(root)
    with pytest.raises(ValueError):
        with api.MotionFusionPipeline(root, identity):
            pass
    assert not (tmp_path / "unused").exists()


# Only preparation/production-count dependencies are synthetic here. The actual
# Task2 public terminal validators execute in its dedicated Pillow interpreter.
TERMINAL_FIXTURE = r'''
import json,sys
from pathlib import Path
from types import SimpleNamespace
from despamo.appearance.provenance import atomic_json,digest,file_hash
from scripts.run_motion_fusion_pipeline import CLOCK_QUERY,process_identity,boot_id
from tools.motion import extractor as e
directory=Path(sys.argv[1]); change=sys.argv[2]
paths={'output_base':str(directory),'adapted_root':str(directory/'adapted'),
       'annotation':str(directory/'annotation'),'environment_lock':str(directory/'lock'),
       'model_snapshot':str(directory/'snapshot')}
launch={'schema_version':1,'policy':'motion-extraction-launch-v1','paths':paths,
        'authorization':e.AUTHORIZATION,'code':{},'interpreter':{}}
identity={'source':'synthetic','hardware':{'uuid':'fixture'}}
key=digest(identity); bootstrap=directory/'motion-extraction-bootstrap-v1'
root=directory/('motion-features-'+key); root.mkdir()
frames={'clips':{'c':{'frame_count':1}}}
atomic_json(Path(paths['adapted_root'])/'complete/frame_rows.json',frames)
observed={'schema_version':1,'files':[],'model_files':[],'hardware':identity['hardware']}
clock=e.ExtractionClock(bootstrap/'budget.json',key,resume=False,bootstrapkey=digest(launch),
                        clocks=lambda:(100.0,1000.0,boot_id()))
clock.start('run')
binding={'schema_version':1,'bootstrapkey':digest(launch),'featurekey':key,
         'root':str(root),'identity':identity}
atomic_json(bootstrap/'binding.json',binding)
atomic_json(bootstrap/'input-observation.json',observed)
atomic_json(bootstrap/'owner.json',{'schema_version':1,
    'policy':'motion-extraction-bootstrap-owner-v1','bootstrapkey':digest(launch),'launch':launch})
atomic_json(root/'owner.json',{'schema_version':1,'policy':'motion-extraction-owner-v2',
    'featurekey':key,'identity':identity,'authorization':e.AUTHORIZATION,
    'bootstrap_root':str(bootstrap),'bootstrapkey':digest(launch)})
atomic_json(root/'profile.json',{'samples':['c']})
clock.state['profile_sha256']=file_hash(root/'profile.json')
clock.state['binding']={'featurekey':key,'root':str(root),
                        'sha256':file_hash(bootstrap/'binding.json')}
terminal={'schema_version':1,'policy':'motion-terminal-close-v1','nonce':'fixture',
 'receipt_path':str(bootstrap/'terminal-fixture.json'),
 'failure_path':str(bootstrap/'terminal-failure-fixture.json'),
 'candidate_path':str(bootstrap/'terminal-candidate-fixture.json'),
 'admission_path':str(bootstrap/'terminal-admission-fixture.json'),
 'spent_upper_bound':5.,'closing_deadline_monotonic':105.,'parent_deadline_monotonic':110.,
 'boot_id':boot_id(),'clock_boot_id':boot_id(),'started_monotonic':100.,
 'started_wall':1000.,'observed_start_spend':0.}
clock.parent_guard=SimpleNamespace(closing_upper_bound=5.,binding=lambda:terminal)
clock.finish('complete'); atomic_json(bootstrap/'status.json',{'status':'complete'})
seal={**terminal,'candidate_sha256':file_hash(Path(terminal['candidate_path'])),
 'budget_sha256':file_hash(clock.path),'status_sha256':file_hash(bootstrap/'status.json'),
 'status':'complete','clock_status':'terminal_pending','featurekey':key,
 'bootstrapkey':digest(launch),'observed_before_receipt_write':101.}
atomic_json(Path(terminal['receipt_path']),seal)
proof={'schema_version':1,'policy':'motion-terminal-admission-v1','terminal_close':terminal,
 'candidate_sha256':seal['candidate_sha256'],'terminal_sha256':file_hash(
     Path(terminal['receipt_path'])),'budget_sha256':seal['budget_sha256'],
 'bootstrapkey':digest(launch),'featurekey':key,'status':'complete','spent_upper_bound':5.,
 'observed_spend_at_return_check':2.,'parent_checked_monotonic':102.,'boot_id':boot_id()}
atomic_json(Path(terminal['admission_path']),proof)
result=e._result('run','complete',identity,root,clock,bootstrap=bootstrap,package={'fixture':True})
result.update(spent_seconds=5.,remaining_seconds=17995.,terminal_admission=proof)
result_path=directory/'result.json'; atomic_json(result_path,result)
command=process_identity(e.os.getpid())['command']
outer_request={'command':command}
outer={'request_hash':digest(outer_request),'returncode':0,'status':'returned',
       'boot_id':boot_id(),'child':{'pid':e.os.getpid(),'boot_id':boot_id(),
                                  'identity':process_identity(e.os.getpid())}}
request={'identity':{'paths':paths},'result_path':str(result_path),
         'outer_request':outer_request,'outer_receipt':outer}
request_path=directory/'query.json'; atomic_json(request_path,request)
e.bootstrap_launch_binding=lambda args:launch
e.preparation_inputs=lambda args:observed
e.verify_evidence=lambda identity:None
e.production_gate=lambda *args:None
e.annotation_records=lambda value:value
e.np.load=lambda *args,**kwargs:SimpleNamespace(item=lambda:{})
e.valid_profile_samples=lambda *args,**kwargs:{'c':{}}
e.profile_ids=lambda frames:['c']
if change=='foreign-path':
    result['budget_path']=str(directory/'foreign.json'); atomic_json(result_path,result)
elif change=='foreign-cli':
    outer['child']['pid']+=1; atomic_json(request_path,request)
elif change=='hardware':
    e.preparation_inputs=lambda args:{**observed,'hardware':{'uuid':'other'}}
elif change=='missing-admission':
    Path(terminal['admission_path']).unlink()
elif change=='raw-complete':
    clock.state['status']='complete'; atomic_json(clock.path,clock.state)
elif change=='receipt':
    seal['nonce']='foreign'; atomic_json(Path(terminal['receipt_path']),seal)
elif change=='chronology':
    proof['parent_checked_monotonic']=99.; atomic_json(Path(terminal['admission_path']),proof)
elif change=='budget':
    clock.state['spent_seconds']=1.; atomic_json(clock.path,clock.state)
elif change=='foreign-nonce':
    proof['terminal_close']={**terminal,'nonce':'foreign'}
    atomic_json(Path(terminal['admission_path']),proof)
elif change=='failure-marker':
    atomic_json(Path(terminal['failure_path']),{'status':'failed'})
sys.argv=['query',str(request_path)]
exec(compile(CLOCK_QUERY,'actual-producer-query','exec'))
'''


@pytest.mark.parametrize("mutation", ["good", "foreign-path", "foreign-cli", "hardware",
                                     "missing-admission", "raw-complete", "receipt",
                                     "chronology", "budget", "foreign-nonce", "failure-marker"])
def test_actual_dedicated_public_admission_projection(api, tmp_path, mutation):
    python = Path(api.ROOT) / "tools/motion/.venv/bin/python"
    result = subprocess.run([str(python), "-c", TERMINAL_FIXTURE, str(tmp_path), mutation],
                            capture_output=True, text=True, timeout=15)
    if mutation == "good":
        assert result.returncode == 0, result.stderr
        facts = json.loads(result.stdout.splitlines()[-1])
        assert facts["status"] == "complete"
        assert facts["spent_seconds"] == 5
        assert api.read_json(tmp_path / "motion-extraction-bootstrap-v1/budget.json")[
            "status"] == "terminal_pending"
    else:
        assert result.returncode != 0
        assert "ValueError" in result.stderr or "FileNotFoundError" in result.stderr
    assert "tools.motion.extractor" not in sys.modules


def test_state_charge_cannot_be_lowered_below_completed_stage(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.execute("extraction", "cpu", [sys.executable, "-c", "pass"])
    state = api.read_json(p.path)
    state["clocks"]["extraction"]["outer_seconds"] = 0
    state["active"] = None
    api.atomic_json(p.path, state)
    original = p.path.read_bytes()
    q = pipeline(api, tmp_path)
    q.resume = True
    with pytest.raises(ValueError):
        with q:
            pass
    assert p.path.read_bytes() == original


@pytest.mark.parametrize("mutation", ["good", "marker", "source", "array", "claimed",
                                     "proof-during-consumer", "wrong-clock-package",
                                     "foreign-query-request"])
def test_handoff_independently_rebuilds_package_and_rechecks_proof(
    api, tmp_path, monkeypatch, mutation,
):
    # Reuse Task1's real small publication fixture, not its claimed descriptor.
    path = api.ROOT / "tests/unit/data/test_new_motion_package.py"
    spec = importlib.util.spec_from_file_location("task1_cpu_fixture", path)
    assert spec is not None and spec.loader is not None
    fixtures = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixtures)
    _, producer, root, identity, frames, annotation = fixtures.fixture(tmp_path)
    package = producer.publish_motion_package(root, identity, frames, annotation)
    adapted = tmp_path / "adapted"
    api.atomic_json(adapted / "complete/frame_rows.json", frames)
    bootstrap = tmp_path / "motion-extraction-bootstrap-v1"
    for name in ("budget.json", "binding.json", "input-observation.json", "admission.json"):
        api.atomic_json(bootstrap / name, {"fixture": name})
    facts = {"status": "complete", "spent_seconds": 5, "featurekey": api.digest(identity),
             "root": str(root), "bootstrapkey": None,
             "budget_sha256": api.file_hash(bootstrap / "budget.json"),
             "binding_sha256": api.file_hash(bootstrap / "binding.json"),
             "observation_sha256": api.file_hash(bootstrap / "input-observation.json"),
             "proof_bindings": {str(bootstrap / "admission.json"):
                                api.file_hash(bootstrap / "admission.json")}}
    result = exact_result(api, "run", identity, root)
    result["package"] = package
    if mutation == "claimed":
        result["package"] = {**package, "manifest_hash": "foreign"}
    api.atomic_json(tmp_path / "result.json", result)
    expected = {"identity": {"paths": {"adapted_root": str(adapted),
                "output_base": str(tmp_path), "extractor_python": sys.executable}},
                "result_path": str(tmp_path / "result.json"),
                "query_request_path": str(tmp_path / "query.json"),
                "clock_facts": {"status": "complete", "spent_seconds": 0}}
    api.atomic_json(tmp_path / "query.json", {"identity": expected["identity"],
                                             "result_path": expected["result_path"]})
    if mutation == "wrong-clock-package":
        facts["featurekey"] = "foreign"
    elif mutation == "foreign-query-request":
        api.atomic_json(tmp_path / "query.json", {"identity": expected["identity"],
                                                 "result_path": str(tmp_path / "foreign.json")})
    calls = []

    def query(*args):
        calls.append(args)
        return facts

    monkeypatch.setattr(api, "bounded_cpu_query", query)
    from scripts import train_fusion_study as runner

    def production_gate(*args):
        if mutation == "proof-during-consumer":
            api.atomic_json(bootstrap / "admission.json", {"changed": True})

    monkeypatch.setattr(runner, "validate_new_motion_production", production_gate)
    if mutation == "marker":
        (root / "complete/completion.json").write_text("{}")
    elif mutation == "source":
        (tmp_path / "sources/00.png").write_bytes(b"changed")
    elif mutation == "array":
        (root / "train/c.npy").write_bytes(b"changed")
    if mutation == "good":
        assert api.validate_extraction_handoff(root, expected) == package
        assert calls[0][1] == api.CLOCK_QUERY
    else:
        with pytest.raises(ValueError):
            api.validate_extraction_handoff(root, expected)


def test_extraction_allocation_credit_requires_valid_source_receipt(api, tmp_path):
    spec = importlib.util.spec_from_file_location(
        "task1_storage_fixture", api.ROOT / "tests/unit/data/test_new_motion_package.py")
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _, _, root, identity, frames, _ = fixture.fixture(tmp_path)
    api.atomic_json(root / "owner.json", {"identity": identity,
                                          "featurekey": api.digest(identity)})
    actual = api.remaining_extraction_demand(root, identity, frames)
    assert actual < 93232 * 1024 * 4
    (tmp_path / "sources/00.png").write_bytes(b"changed")
    with pytest.raises(ValueError):
        api.remaining_extraction_demand(root, identity, frames)


def test_training_gpu_lock_blocks_another_extractor_owner(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p, p.training_gpu_lock():
        import fcntl
        lock = tmp_path / ".motion-extraction-gpu.lock"
        with lock.open("r") as other:
            with pytest.raises(BlockingIOError):
                fcntl.flock(other.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("mutation", ["good", "summary-drift", "phase-cap", "run-cap",
                                     "graph", "references", "phase-policy", "controller-cap"])
def test_final_six_run_admission_read_only_and_drift_closed(api, tmp_path, monkeypatch, mutation):
    from scripts import run_fusion_study as controller
    spec = importlib.util.spec_from_file_location(
        "fusion_summary_cpu_fixture", api.ROOT / "tests/unit/pilot/test_fusion_controller.py")
    assert spec is not None and spec.loader is not None
    fixture = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture)
    results = fixture.rows()
    cache = tmp_path / ("motion-features-" + "a" * 64)
    graph = {"schema_version": 1, "graph_policy": "despamo-motion-package-graph-ref-v1",
             "package_policy": "despamo-motion-features-v1", "root": str(cache),
             "identity_hash": "a" * 64}
    for name in ("manifest", "timing", "annotation", "spatial_frame_rows"):
        graph[name + "_path"] = str(cache / (name + ".json"))
        graph[name + "_hash"] = "a" * 64
    graph.update(completion=str(cache / "complete/completion.json"), completion_hash="a" * 64,
                 manifest_path=str(cache / "complete/manifest.json"),
                 timing_path=str(cache / "complete/motion-timing.json"))
    root = tmp_path / "training"
    root.mkdir()
    (root / ".fusion-controller.lock").touch()
    accounting = []
    for row in results:
        row["identity"] = {"run_policy": controller.runner.NEW_RUN_POLICY,
            "motion_package": graph, "config": {"data": {
                "motion_root": graph["root"], "motion_manifest": graph["manifest_path"]}},
            "timing_hash": graph["timing_hash"],
            "bound_files": {graph["completion"]: graph["completion_hash"]}}
    sources = {"fixture": "a" * 64}
    for seed in (0, 1):
        phase = root / f"seed{seed}"
        phase.mkdir()
        (phase / ".study.lock").touch()
        stages = {arm: {"owned_arm": arm, "elapsed_seconds": 10, "status": "complete"}
                  for arm in controller.ARMS}
        state = {"study_policy": api.NEW_POLICY, "fusion_seed": seed,
                 "motion_package_root": graph["root"], "controller_sources": sources,
                 "budget_seconds": 32400, "run_cap_seconds": 10800, "spent_seconds": 30,
                 "stages": stages, "runs": {row["arm"]: {
                    "identity": row["identity"], "run": str(root / "runs" / row["arm"])}
                    for row in results if row["seed"] == seed}}
        api.atomic_json(phase / "study-state.json", state)
        accounting.append({"seed": seed, "spent_seconds": 30, "stages": stages})
    summary = controller.aggregate_fusion_results(results)
    summary.update(budget_seconds=64800, phase_accounting=accounting, spent_charged_seconds=60)
    summary["content_hash"] = api.digest(summary)
    api.atomic_json(root / "summary.json", summary)
    api.atomic_json(root / "state.json", {"status": "complete", "study_policy": api.NEW_POLICY,
        "summary_sha256": api.file_hash(root / "summary.json"), "motion_package": graph,
        "controller_sources": sources, "budget_seconds": 64800,
        "per_seed_budget_seconds": 32400, "per_run_budget_seconds": 10800})
    if mutation == "controller-cap":
        path = root / "state.json"
        value = api.read_json(path)
        value["budget_seconds"] = 64801
        api.atomic_json(path, value)
    if mutation in {"phase-cap", "run-cap", "phase-policy"}:
        path = root / "seed0/study-state.json"
        value = api.read_json(path)
        if mutation == "phase-cap":
            value["spent_seconds"] = 32401
        elif mutation == "run-cap":
            value["stages"]["reference"]["elapsed_seconds"] = 10801
            value["spent_seconds"] = 10821
        else:
            value["study_policy"] = "legacy"
        api.atomic_json(path, value)
    if mutation == "graph":
        path = root / "seed0/study-state.json"
        value = api.read_json(path)
        value["runs"]["reference"]["identity"]["motion_package"] = {
            **graph, "identity_hash": "b" * 64}
        api.atomic_json(path, value)
    if mutation == "references":
        results[-1]["items"][0]["reference"] = "changed"
    before = {str(p): p.read_bytes() for p in root.rglob("*.json")}

    def collect(phase, arm, run, expected):
        if mutation == "summary-drift":
            (root / "summary.json").write_text("{}")
        return next(row for row in results
                    if row["seed"] == phase.fusion_seed and row["arm"] == arm)

    monkeypatch.setattr(controller, "collect_result", collect)
    monkeypatch.setattr(controller.runner, "validate_motion_graph", lambda expected: None)
    monkeypatch.setattr(controller, "validate_summary_sources", lambda *args: None)
    owner_identity = fixture_identity(api, tmp_path)
    request = {"identity": owner_identity, "motion_package": graph}
    if mutation == "good":
        with api.MotionFusionPipeline(Path(owner_identity["paths"]["pipeline_root"]),
                                      owner_identity) as owner:
            owner.state["motion_package"] = graph
            owner.acquire_training_leases()
            request["training_leases"] = owner.training_bindings
            facts = api.validate_final_summary(request)
        assert facts["inner_seconds"] == 60
        assert facts["sha256"] == api.file_hash(root / "summary.json")
        assert {str(p): p.read_bytes() for p in root.rglob("*.json")} == before
    else:
        with pytest.raises(ValueError):
            api.validate_final_summary(request)


@pytest.mark.parametrize("field,value", [("schema_version", True), ("spent_seconds", True),
                                        ("admission_seconds", float("nan"))])
def test_success_result_numeric_fields_not_bool_or_nonfinite(api, tmp_path, field, value):
    with pipeline(api, tmp_path) as p:
        result = exact_result(api, "preflight", {"source": "fixture"}, tmp_path / "cache")
        result[field] = value
        output = tmp_path / "result.jsonl"
        output.write_text(json.dumps(result) + "\n")
        with pytest.raises(ValueError):
            p.admit_result("preflight", output, {})


def test_cumulative_watchdog_bounds_io_between_children(api, tmp_path):
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    code = """
import sys,time,json
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
p.CAPS={'extraction':1.,'training':64800}
identity=p.read_json(Path(sys.argv[1]))
with p.MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity) as pipeline:
    pipeline.begin('extraction')
    try:
        pipeline.check_sources=lambda:time.sleep(20)
        pipeline.execute('extraction','blocked-io',[sys.executable,'-c','pass'])
    except TimeoutError:
        print('bounded-source-io')
"""
    result = subprocess.run([sys.executable, "-c", code, str(config)],
                            capture_output=True, text=True, timeout=3)
    assert result.returncode == 0, result.stderr
    assert "bounded-source-io" in result.stdout


def test_hardware_drift_before_training_gpu_dispatch(api, tmp_path, monkeypatch):
    identity = fixture_identity(api, tmp_path)
    monkeypatch.setattr(api, "bounded_cpu_query", lambda *args: {"uuid": "other-gpu"})
    monkeypatch.setattr(api, "require_gpu_idle", lambda: None)
    monkeypatch.setattr(api, "storage_admission", lambda *args, **kwargs: {})
    with pytest.raises(ValueError):
        api.gpu_storage_query({"identity": identity, "extraction": False,
                               "extraction_identity": {"hardware": {"uuid": "expected-gpu"}}})


def test_identity_preserves_dedicated_venv_invocation_for_cpu_queries(api, tmp_path):
    args = argparse.Namespace(authorize_motion_fusion_pipeline=True,
        pipeline_root=api.DEFAULT_ROOT, adapted_root=api.BASE / "e3-features-fixture",
        annotation=tmp_path / "annotation", model_snapshot=tmp_path / "snapshot",
        environment_lock=api.ROOT / "tools/motion/uv.lock",
        extractor_python=api.ROOT / "tools/motion/.venv/bin/python")
    identity = api.pipeline_identity(args)
    result = api.bounded_cpu_query(identity["paths"]["extractor_python"],
        "import json,sys,PIL; print(json.dumps({'pillow':PIL.__version__,'prefix':sys.prefix}))",
        tmp_path / "unused", timeout=5)
    assert result["pillow"] == "11.3.0"
    assert result["prefix"] == str(api.ROOT / "tools/motion/.venv")


@pytest.mark.parametrize("selection_valid", [True, False])
def test_remaining_training_demand_waived_only_for_full_owned_completion(
    api, tmp_path, monkeypatch, selection_valid,
):
    from scripts import run_fusion_study as controller
    identity = fixture_identity(api, tmp_path)
    for seed in (0, 1):
        runs = {}
        for arm in controller.ARMS:
            run = tmp_path / "training/runs" / f"{seed}-{arm}"
            run.mkdir(parents=True)
            runs[arm] = {"run": str(run), "identity": {"run_key": run.name,
                "motion_package": {"root": str(tmp_path / "cache")}}}
        api.atomic_json(tmp_path / f"training/seed{seed}/study-state.json",
                        {"study_policy": api.NEW_POLICY, "runs": runs})
    monkeypatch.setattr(api, "bounded_cpu_query", lambda *args: {"uuid": "fixture"})
    monkeypatch.setattr(api, "require_gpu_idle", lambda: None)
    monkeypatch.setattr(controller.runner, "validate_motion_graph", lambda *args: None)
    monkeypatch.setattr(controller, "owned_status", lambda *args: {"status": "complete"})

    def selection(*args):
        if not selection_valid:
            raise ValueError("synthetic invalid selected/dev bytes")
        return {"status": "complete"}

    monkeypatch.setattr(controller, "completion_selection", selection)
    actual_storage = api.storage_admission
    monkeypatch.setattr(api, "storage_admission", lambda *args, **kwargs: actual_storage(
        *args, **kwargs, linux_free=10**15, windows_free=10**15))
    request = {"identity": identity, "extraction": False,
               "extraction_identity": {"hardware": {"uuid": "fixture"}}}
    if selection_valid:
        assert api.gpu_storage_query(request)["remaining_training_bytes"] == 0
    else:
        with pytest.raises(ValueError):
            api.gpu_storage_query(request)


def test_forward_wall_reboot_reconciliation_preserves_remaining_allowance(api, tmp_path):
    p = pipeline(api, tmp_path)
    with p:
        p.state["clocks"]["extraction"]["outer_seconds"] = 100
        p.begin("extraction")
        p.state["active"].update(boot_id="prior-boot", wall=time.time() - 10)
        p.save()
    q = api.MotionFusionPipeline(p.root, p.identity)
    q.resume = True
    with q:
        assert 110 <= q.charged("extraction") < 120
        assert 17880 < q.remaining("extraction") <= 17890
        assert q.remaining("training") == 64800


def test_handoff_writer_locks_held_until_pipeline_close(api, tmp_path):
    import fcntl
    bootstrap = tmp_path / "motion-extraction-bootstrap-v1"
    cache = tmp_path / "motion-features-fixture"
    for root in (bootstrap, cache):
        root.mkdir()
        (root / ".writer.lock").touch()
    with pipeline(api, tmp_path) as p:
        p.acquire_handoff_locks(cache)
        for root in (bootstrap, cache):
            with (root / ".writer.lock").open("r") as foreign:
                with pytest.raises(BlockingIOError):
                    fcntl.flock(foreign.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    for root in (bootstrap, cache):
        with (root / ".writer.lock").open("r") as foreign:
            fcntl.flock(foreign.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize("field,value", [("monotonic", True), ("wall", False),
                                        ("started_outer", 1000), ("bucket", "foreign")])
def test_corrupt_active_clock_cannot_be_reconciled_or_overwritten(api, tmp_path, field, value):
    p = pipeline(api, tmp_path)
    with p:
        p.begin("extraction")
        p.save()
    state = api.read_json(p.path)
    state["active"][field] = value
    api.atomic_json(p.path, state)
    before = p.path.read_bytes()
    q = api.MotionFusionPipeline(p.root, p.identity)
    q.resume = True
    with pytest.raises(ValueError):
        with q:
            pass
    assert p.path.read_bytes() == before


def test_receipt_fs_stall_parent_loss_still_stops_setsided_grandchild(api, tmp_path):
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    grand = tmp_path / "grand.json"
    stalled = tmp_path / "writer-stalled"
    code = r'''
import sys,time,json,subprocess
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
identity=p.read_json(Path(sys.argv[1])); grand=sys.argv[2]; stalled=sys.argv[3]
real_popen=p.subprocess.Popen
guardian=r"""
import sys,time,json
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
real=p.atomic_json
def write(path,value):
    if str(path).endswith('.receipt.json') and value.get('status')=='running':
        Path(sys.argv[4]).write_text('stalled')
        time.sleep(20)
    real(path,value)
p.atomic_json=write
arg=sys.argv[1]
request=json.loads(arg) if arg.startswith('{') else Path(arg)
raise SystemExit(p._watchdog(request,int(sys.argv[2]),int(sys.argv[3])))
"""
def spawn(command,*args,**kwargs):
    if '--watchdog' in command:
        command=[sys.executable,'-c',guardian,*command[3:],stalled]
    return real_popen(command,*args,**kwargs)
p.subprocess.Popen=spawn
cmd=[sys.executable,'-c',"import subprocess,sys,time,json; "
    "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],"
    "start_new_session=True); open(sys.argv[1],'w').write(json.dumps({'pid':c.pid})); "
    "time.sleep(20)",grand]
with p.MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity) as owner:
    owner.execute('extraction','stall',cmd,timeout=2)
'''
    parent = subprocess.Popen([sys.executable, "-c", code, str(config), str(grand), str(stalled)],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    owned = []
    try:
        until = time.monotonic() + 4
        while not (grand.exists() and stalled.exists()) and time.monotonic() < until:
            time.sleep(.01)
        assert grand.exists() and stalled.exists()
        pid = json.loads(grand.read_text())["pid"]
        owned.append({"pid": pid, "boot_id": api.boot_id(), "identity": api.process_identity(pid)})
        state = api.read_json(Path(identity["paths"]["pipeline_root"]) / "state.json")
        receipt = api.read_json(Path(state["stages"][-1]["receipt"]))
        owned.append(receipt["watchdog"])
        parent.kill()
        parent.wait(timeout=2)
        until = time.monotonic() + 3
        while api.process_identity(pid) is not None and time.monotonic() < until:
            time.sleep(.01)
        assert api.process_identity(pid) is None
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=2)
        for receipt in owned:
            if receipt["identity"] is not None:
                api.stop_owned(receipt)


def test_context_exit_failure_is_charged_under_armed_guard_and_not_admitted(api, tmp_path):
    p = pipeline(api, tmp_path)
    class FailedLease:
        def close(self):
            assert p.budget_guard is not None and p.budget_guard.poll() is None
            time.sleep(.05)
            raise OSError("synthetic close failure")
    p.__enter__()
    p.begin("training")
    p.state["status"] = "closing_pending"
    p.handoff_leases.append(FailedLease())
    with pytest.raises(OSError):
        p.__exit__(None, None, None)
    state = api.read_json(p.path)
    assert state["status"] == "failed"
    assert state["clocks"]["training"]["outer_seconds"] >= .05
    q = api.MotionFusionPipeline(p.root, p.identity)
    q.resume = True
    with q:
        assert q.state["status"] != "complete"


def test_owned_completed_producer_projection_without_local_pointer(api, tmp_path):
    prefix = TERMINAL_FIXTURE.split("sys.argv=['query',str(request_path)]")[0]
    code = prefix + r'''
from scripts.run_motion_fusion_pipeline import PROJECTION_QUERY
actual=e._result('profile','complete',identity,root,clock,bootstrap=bootstrap,
                 package={'fixture':True})
assert actual['mode']=='profile' and actual['status']=='complete' and actual['package'] is not None
atomic_json(request_path,{'identity':{'paths':paths},'expected_identity':identity})
sys.argv=['projection',str(request_path)]
exec(compile(PROJECTION_QUERY,'producer-immutable-projection','exec'))
'''
    result = subprocess.run([str(api.ROOT / "tools/motion/.venv/bin/python"), "-c", code,
                             str(tmp_path), "good"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0, result.stderr
    facts = json.loads(result.stdout.splitlines()[-1])
    assert facts["status"] == "complete" and facts["spent_seconds"] == 5
    assert api.read_json(tmp_path / "motion-extraction-bootstrap-v1/budget.json")[
        "status"] == "terminal_pending"
    with pipeline(api, tmp_path) as p:
        p.state["extraction_identity"] = {"source": "synthetic", "hardware": {"uuid": "fixture"}}
        p.query = lambda *args: facts
        assert p.extraction_modes() == (("run", "extracting"),)
        assert "--resume" in p.extraction_command("run")


def test_worker_context_exit_deadline_cannot_publish_or_resume_complete(api, tmp_path):
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    code = r'''
import sys,time,json
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
p.CAPS={'extraction':2,'training':2}
identity=p.read_json(Path(sys.argv[1]))
owner=p.MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity)
def stages():
    owner.begin('training')
    now=time.monotonic(); limit={'soft_deadline':now+.2,'hard_deadline':now+.4}
    owner.state['status']='closing_pending'
    owner.worker_event({'kind':'closing','limit':limit})
    owner.save()
owner._run_stages=stages
real_exit=owner.__exit__
def late(*args):
    real_exit(*args)
    time.sleep(20)
owner.__exit__=late
try:
    owner.run()
except (TimeoutError,RuntimeError):
    print('context-exit-rejected')
'''
    result = subprocess.run([sys.executable, "-c", code, str(config)],
                            capture_output=True, text=True, timeout=5)
    assert "context-exit-rejected" in result.stdout, result.stderr
    root = Path(identity["paths"]["pipeline_root"])
    assert not list(root.glob("pipeline-admission-*.json"))
    state = api.read_json(root / "state.json")
    assert state["status"] in {"failed", "capped"}
    assert state["clocks"]["training"]["outer_seconds"] >= .2


def test_external_guardian_eof_race_stops_worker_and_setsided_grandchild(api, tmp_path):
    import fcntl
    identity = fixture_identity(api, tmp_path)
    config = tmp_path / "identity.json"
    config.write_text(json.dumps(identity))
    grand = tmp_path / "grand.json"
    worker_code = r'''
import sys,json
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
identity=p.read_json(Path(sys.argv[1]))
cmd=[sys.executable,'-c',"import subprocess,sys,time,json; "
    "c=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'],"
    "start_new_session=True); open(sys.argv[1],'w').write(json.dumps({'pid':c.pid})); "
    "time.sleep(20)",sys.argv[2]]
with p.MotionFusionPipeline(Path(identity['paths']['pipeline_root']),identity) as owner:
    owner.execute('extraction','eof-worker',cmd,timeout=8)
'''
    worker = subprocess.Popen([sys.executable, "-c", worker_code, str(config), str(grand)],
                              start_new_session=True, stdout=subprocess.DEVNULL,
                              stderr=subprocess.DEVNULL)
    guardian = None
    caller = None
    owned = []
    rd, wr = os.pipe()
    ack_rd, ack_wr = os.pipe()
    lock_path = tmp_path / "custody.lock"
    lock_path.touch()
    lease = lock_path.open("r")
    fcntl.flock(lease.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        until = time.monotonic() + 4
        while not grand.exists() and time.monotonic() < until:
            time.sleep(.01)
        assert grand.exists()
        grand_pid = json.loads(grand.read_text())["pid"]
        owned.append({"pid": grand_pid, "boot_id": api.boot_id(),
                      "identity": api.process_identity(grand_pid)})
        caller = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"],
                                  pass_fds=(wr,), start_new_session=True)
        os.close(wr)
        wr = -1
        request = {"parent": {"pid": caller.pid, "boot_id": api.boot_id(),
                              "identity": api.process_identity(caller.pid)},
                   "owned_worker": {"pid": worker.pid, "boot_id": api.boot_id(),
                                    "identity": api.process_identity(worker.pid)},
                   "soft_deadline": time.monotonic() + 5,
                   "hard_deadline": time.monotonic() + 6, "nonce": "eof-race",
                   "failure_path": str(tmp_path / "failure.json")}
        guardian_code = r'''
import sys,json,time,signal
from pathlib import Path
from scripts import run_motion_fusion_pipeline as p
r=json.loads(sys.argv[1]); control=int(sys.argv[2]); ack=int(sys.argv[3])
read=p.os.read; write=p.atomic_json
def race(fd,size):
    if fd==control:
        p.stop_owned(r['parent'],signal.SIGKILL)
    return read(fd,size)
def delayed(path,value):
    if str(path)==r['failure_path']:
        Path(sys.argv[4]).write_text('persisting')
        time.sleep(.3)
    write(path,value)
p.os.read=race
p.atomic_json=delayed
p.select.select=lambda *args:([control],[],[])
raise SystemExit(p._budget_monitor(r,control,ack))
'''
        guardian = subprocess.Popen([sys.executable, "-c", guardian_code, json.dumps(request),
            str(rd), str(ack_wr), str(tmp_path / "persisting")],
            pass_fds=(rd, ack_wr, lease.fileno()), start_new_session=True,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.close(rd)
        rd = -1
        os.close(ack_wr)
        ack_wr = -1
        lease.close()
        assert os.read(ack_rd, 1) == b"1"
        until = time.monotonic() + 3
        while not (tmp_path / "persisting").exists() and guardian.poll() is None:
            assert time.monotonic() < until
            time.sleep(.005)
        assert (tmp_path / "persisting").exists()
        with lock_path.open("r") as competitor:
            with pytest.raises(BlockingIOError):
                fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert guardian.wait(timeout=3) == 1
        caller.wait(timeout=2)
        worker.wait(timeout=2)
        until = time.monotonic() + 2
        while api.process_identity(grand_pid) is not None and time.monotonic() < until:
            time.sleep(.01)
        assert api.process_identity(grand_pid) is None
        assert api.read_json(tmp_path / "failure.json")["reason"] == "caller disconnected"
        with lock_path.open("r") as competitor:
            fcntl.flock(competitor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for process in (guardian, caller, worker):
            if process is not None and process.poll() is None:
                receipt = {"pid": process.pid, "boot_id": api.boot_id(),
                           "identity": api.process_identity(process.pid)}
                if receipt["identity"] is not None:
                    api.stop_owned(receipt)
                process.wait(timeout=2)
        for receipt in owned:
            if receipt["identity"] is not None:
                api.stop_owned(receipt)
        lease.close()
        for fd in (rd, wr, ack_rd, ack_wr):
            if fd >= 0:
                os.close(fd)


@pytest.mark.parametrize("wire", [b"\0\0\0\0", struct.pack("!I", 65537), b"\0\0",
                                  struct.pack("!I", 5) + b"{}", struct.pack("!I", 1) + b"["])
def test_stream_frame_rejects_zero_oversize_and_truncated_data(api, wire):
    reader, writer = api.socket.socketpair(api.socket.AF_UNIX, api.socket.SOCK_STREAM)
    try:
        writer.sendall(wire)
        writer.shutdown(api.socket.SHUT_WR)
        with pytest.raises(ValueError, match="worker IPC"):
            api.receive_worker_frame(reader, time.monotonic() + .5)
    finally:
        reader.close()
        writer.close()


def test_stream_frames_fragmented_data_then_real_boundary_eof(api):
    reader, writer = api.socket.socketpair(api.socket.AF_UNIX, api.socket.SOCK_STREAM)
    body = b'{"kind":"ready"}'
    wire = struct.pack("!I", len(body)) + body
    try:
        for byte in wire:
            writer.send(bytes([byte]))
        writer.shutdown(api.socket.SHUT_WR)
        packet, fds = api.receive_worker_frame(reader, time.monotonic() + .5)
        assert packet == {"kind": "ready"} and fds == []
        assert api.receive_worker_frame(reader, time.monotonic() + .5) is None
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize("partial", [b"\0", struct.pack("!I", 40) + b"{"])
def test_stream_partial_read_has_own_bounded_deadline(api, partial):
    reader, writer = api.socket.socketpair(api.socket.AF_UNIX, api.socket.SOCK_STREAM)
    try:
        writer.sendall(partial)
        started = time.monotonic()
        with pytest.raises(TimeoutError):
            api.receive_worker_frame(reader, started + .1)
        assert time.monotonic() - started < .5
    finally:
        reader.close()
        writer.close()


@pytest.mark.parametrize("condition", ["header-zero", "payload-fd", "partial-fd",
                                      "too-many-fds"])
def test_stream_rejected_frames_close_received_ancillary_fds(api, tmp_path, condition):
    reader, writer = api.socket.socketpair(api.socket.AF_UNIX, api.socket.SOCK_STREAM)
    path = tmp_path / "lease"
    path.touch()
    with path.open("r") as lease:
        before = len(list(Path("/proc/self/fd").iterdir()))
        ancillary = [(api.socket.SOL_SOCKET, api.socket.SCM_RIGHTS,
                      api.array.array("i", [lease.fileno()] * (20 if condition == "too-many-fds"
                                                             else 1)))]
        if condition == "payload-fd":
            writer.sendall(struct.pack("!I", 2))
            writer.sendmsg([b"{}"], ancillary)
        else:
            wire = b"\0" if condition == "partial-fd" else struct.pack("!I", 0)
            writer.sendmsg([wire], ancillary)
        writer.shutdown(api.socket.SHUT_WR)
        try:
            with pytest.raises(ValueError, match="worker IPC"):
                api.receive_worker_frame(reader, time.monotonic() + .5)
            assert len(list(Path("/proc/self/fd").iterdir())) == before
        finally:
            reader.close()
            writer.close()


def test_stream_frame_lease_roundtrip_and_sender_deadline(api, tmp_path):
    reader, writer = api.socket.socketpair(api.socket.AF_UNIX, api.socket.SOCK_STREAM)
    path = tmp_path / "lease"
    path.touch()
    fds = []
    try:
        with path.open("r") as lease:
            api.send_worker_frame(writer, {"kind": "lease"}, time.monotonic() + .5,
                                  [lease.fileno()])
            packet, fds = api.receive_worker_frame(reader, time.monotonic() + .5)
            assert packet == {"kind": "lease"}
            assert len(fds) == 1
            assert os.fstat(fds[0]).st_ino == os.fstat(lease.fileno()).st_ino
        writer.setsockopt(api.socket.SOL_SOCKET, api.socket.SO_SNDBUF, 4096)
        with pytest.raises(TimeoutError):
            api.send_worker_frame(writer, {"fill": "x" * 60000}, time.monotonic() + .1)
    finally:
        for fd in fds:
            os.close(fd)
        reader.close()
        writer.close()

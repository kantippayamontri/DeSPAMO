import importlib
import json


def module():
    return importlib.import_module("scripts.watch_official_benchmark")


def setup(tmp_path,status="running",live=True,error=""):
    root=tmp_path/"study"
    (root/"controller").mkdir(parents=True)
    owner=dict(pid=123,boot_id="boot",start_ticks="1",command=["controller"])
    state=dict(status=status,batch_size=8,owner=owner,stages={})
    if status=="failed":
        state.update(error=error)
    (root/"controller/state.json").write_text(json.dumps(state))
    def reader(pid):
        return owner if live else None
    return root,reader


def test_running_controller_never_restarts(tmp_path):
    root,reader=setup(tmp_path)
    health=module().inspect_study(root,reader)
    assert health["state"]=="running" and not health["restart_allowed"]


def test_completed_study_stops_monitor(tmp_path):
    root,reader=setup(tmp_path,status="complete",live=False)
    health=module().inspect_study(root,reader)
    assert health["state"]=="complete" and not health["restart_allowed"]


def test_orphan_worker_waits_without_killing_or_duplicate_restart(tmp_path):
    root,_=setup(tmp_path,live=False)
    worker=dict(pid=456,boot_id="boot",start_ticks="2",command=["worker"])
    state=json.loads((root/"controller/state.json").read_text())
    state["stages"]={"seed0-reference":dict(status="running",owner=worker,command=["worker"])}
    (root/"controller/state.json").write_text(json.dumps(state))
    health=module().inspect_study(root,lambda pid:worker if pid==456 else None)
    assert health["state"]=="orphan_worker_active" and not health["restart_allowed"]


def test_pid_reuse_blocks_recovery(tmp_path):
    root,_=setup(tmp_path)
    def reader(pid):
        return dict(pid=123,boot_id="other",start_ticks="different",command=["unrelated"])
    health=module().inspect_study(root,reader)
    assert health["state"]=="owner_mismatch" and not health["restart_allowed"]


def test_transient_interruption_retry_only_after_all_owned_processes_exit(tmp_path):
    root,reader=setup(tmp_path,status="failed",live=False,error="KeyboardInterrupt signal15")
    health=module().inspect_study(root,reader)
    assert health["restart_allowed"]


def test_unknown_code_failure_not_speculatively_restarted(tmp_path):
    root,reader=setup(tmp_path,status="failed",live=False,error="ValueError checkpoint corrupted")
    health=module().inspect_study(root,reader)
    assert health["state"]=="needs_agent" and not health["restart_allowed"]

import json
from pathlib import Path

import pytest


def source(tmp_path):
    root = tmp_path / "runtime"
    root.mkdir()
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"runtime_root": str(root), "run_root": str(tmp_path / "run")}))
    (root / "state.json").write_text(json.dumps({"status": "running", "owner": {"pid": 1}}))
    return config, root


def test_no_authorization_cannot_place_hold(tmp_path):
    from tools.checkpoint_storage.handoff import request_handoff

    config, root = source(tmp_path)
    with pytest.raises(RuntimeError, match="authorization"):
        request_handoff(config, tmp_path / "control", authorize_handoff=False)
    assert not (root / "manual-stop.json").exists()


def test_completed_source_never_requests_new_training(tmp_path):
    from tools.checkpoint_storage.handoff import request_handoff

    config, root = source(tmp_path)
    (root / "state.json").write_text('{"status":"seed0_complete","owner":{"pid":1}}')
    assert (
        request_handoff(config, tmp_path / "control", authorize_handoff=True)["status"]
        == "source_already_complete"
    )
    assert not (root / "manual-stop.json").exists()


def test_live_parent_or_device_fault_cannot_seal(tmp_path, monkeypatch):
    from tools.checkpoint_storage import handoff as h

    config, root = source(tmp_path)
    monkeypatch.setattr(h, "owner_active", lambda owner: True)
    with pytest.raises(RuntimeError, match="live|healthy"):
        h.seal_handoff(config, tmp_path / "control")
    monkeypatch.setattr(h, "owner_active", lambda owner: False)
    (root / "state.json").write_text(
        '{"status":"held","owner":{"pid":1},"error":"CUDA illegal instruction"}'
    )
    with pytest.raises(RuntimeError, match="fault|operator|healthy"):
        h.seal_handoff(config, tmp_path / "control")


def test_seal_consumed_operator_boundary_preserves_cost_and_step(tmp_path, monkeypatch):
    from tools.checkpoint_storage import handoff as h
    from tools.checkpoint_storage.inventory import file_hash

    config, root = source(tmp_path)
    cfg = json.loads(config.read_text())
    run = Path(cfg["run_root"])
    run.mkdir()
    (run / "budget.json").write_text(
        json.dumps(
            {
                "completed_updates": 4,
                "charged_seconds": 501.5,
                "safe_runtime": {"durable_step": 4, "executed_probe_updates": 1},
            }
        )
    )
    (root / "state.json").write_text(
        '{"status":"held","owner":{"pid":1},"error":"Safe runtime is held"}'
    )
    (root / "migration.json").write_text('{"sources":{}}')
    (root / "accounting-head.json").write_text('{"sequence":17}')
    folder = root / "checkpoints/production"
    folder.mkdir(parents=True)
    (folder / "resume-step-4.pt").write_bytes(b"full-healthy")
    (folder / "candidate-step-4.pt").write_bytes(b"model-only")
    approval = tmp_path / "approval.json"
    approval.write_text('{"consumed":true}')
    directory = h._directory(cfg, tmp_path / "control")
    directory.mkdir(parents=True)
    (directory / "request.json").write_text(
        json.dumps({"scope": str(approval), "scope_token": "old"})
    )
    monkeypatch.setattr(h, "owner_active", lambda owner: False)
    monkeypatch.setattr(h, "reconcile_source", lambda *a: None)
    result = h.seal_handoff(config, tmp_path / "control")
    assert result["completed_updates"] == result["durable_updates"] == 4
    assert result["charged_seconds"] == 501.5 and result["probe_updates"] == 1
    assert result["latest_resume_sha256"] == file_hash(folder / "resume-step-4.pt")
    assert result["old_scope_consumed"] and result["owners_exited"]

"""Reviewed next-worker barrier; never interrupt an admitted device operation."""

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from tools.checkpoint_storage.inventory import file_hash, owner_active
from tools.dinov3.storage import atomic_json, writer


def _directory(config: dict, control_root: Path) -> Path:
    return control_root / "handoffs" / hashlib.sha256(config["run_root"].encode()).hexdigest()


def request_handoff(source_config: Path, control_root: Path, *, authorize_handoff: bool) -> dict:
    if not authorize_handoff:
        raise RuntimeError("Explicit handoff authorization required")
    config = json.loads(source_config.read_text())
    root = Path(config["runtime_root"])
    state = json.loads((root / "state.json").read_text())
    if state["status"] == "seed0_complete":
        return {"status": "source_already_complete", "source_config": str(source_config)}
    if state["status"] != "running" or not owner_active(state["owner"]):
        raise RuntimeError("Healthy exact live parent required for operator hold")
    owner = state["owner"]
    command = owner["command"]
    approval = Path(command[command.index("--approval") + 1])
    scope = json.loads(approval.read_text())
    claim = json.loads((root / "attempts" / scope["token"] / "claim.json").read_text())
    if (
        claim["owner"] != owner
        or claim["approval_sha256"] != file_hash(approval)
        or scope["consumed"]
    ):
        raise ValueError("Parent/scope claim mismatch")
    binding = json.loads((root / "migration.json").read_text())
    if any(file_hash(Path(p)) != sha for p, sha in binding["sources"].items()):
        raise ValueError("Source drift before operator hold")
    record = {
        "status": "operator_hold_requested",
        "source_config": str(source_config),
        "source_config_sha256": file_hash(source_config),
        "source_binding": binding,
        "owner": owner,
        "scope": str(approval),
        "scope_token": scope["token"],
        "requested_wall": time.time(),
        "source_runtime": str(root),
    }
    with writer(_directory(config, control_root)):
        if (root / "manual-stop.json").exists():
            raise RuntimeError("Existing runtime hold cannot be replaced")
        atomic_json(_directory(config, control_root) / "request.json", record)
        atomic_json(
            root / "manual-stop.json",
            {
                "status": "operator_storage_handoff",
                "automatic_recovery_disabled": True,
                "request": str(_directory(config, control_root) / "request.json"),
                "scope_token": scope["token"],
                "owner": owner,
            },
        )
    return record


def reconcile_source(source_config: Path, config: dict) -> None:
    original_worktree = source_config.resolve().parents[2]
    project = Path("/home/kan/Research/DeSpaMo")
    code = (
        "from pathlib import Path;from tools.clip_s2_safe.policy import load_config;"
        "from tools.clip_s2_safe.state import read_binding;"
        "from tools.clip_s2_safe.accounting import reconcile_dead_attempt;"
        f"c=load_config(Path({str(source_config)!r}));"
        "reconcile_dead_attempt(c,read_binding(c),reviewed=True)"
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(project / "src") + ":" + str(original_worktree) + ":" + str(project),
    }
    subprocess.run(
        [str(project / ".venv/bin/python"), "-B", "-c", code],
        cwd=original_worktree,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def seal_handoff(source_config: Path, control_root: Path) -> dict:
    config = json.loads(source_config.read_text())
    root = Path(config["runtime_root"])
    state = json.loads((root / "state.json").read_text())
    if owner_active(state.get("owner") or {}):
        raise RuntimeError("Parent still live; healthy boundary not sealed")
    if state["status"] not in {"held", "seed0_complete"}:
        raise RuntimeError("Healthy completed/operator-held state required")
    request_path = _directory(config, control_root) / "request.json"
    if state["status"] == "held" and (
        state.get("error") != "Safe runtime is held" or not request_path.exists()
    ):
        raise RuntimeError("Non-operator or device fault blocks handoff")
    request = json.loads(request_path.read_text()) if request_path.exists() else None
    if request:
        scope = json.loads(Path(request["scope"]).read_text())
        if not scope["consumed"]:
            raise RuntimeError("Prior GPU scope not consumed")
        token_root = root / "attempts" / request["scope_token"]
        for owner_file in token_root.rglob("owner.json"):
            if owner_active(json.loads(owner_file.read_text())["owner"]):
                raise RuntimeError("Child still live")
    reconcile_source(source_config, config)
    budget = json.loads((Path(config["run_root"]) / "budget.json").read_text())
    if budget["completed_updates"] != budget["safe_runtime"]["durable_step"]:
        raise ValueError("Unsaved high-water/replay needs separate review")
    step = budget["completed_updates"]
    latest = root / "checkpoints/production" / f"resume-step-{step}.pt"
    if not latest.exists():
        raise ValueError("Healthy durable full resume missing")
    candidate = root / "checkpoints/production" / f"candidate-step-{step}.pt"
    binding = json.loads((root / "migration.json").read_text())
    if any(file_hash(Path(p)) != sha for p, sha in binding["sources"].items()):
        raise ValueError("Source drift at handoff seal")
    result = {
        "status": "sealed",
        "source_config": config,
        "source_config_path": str(source_config),
        "original_binding": binding,
        "latest_resume": str(latest),
        "latest_resume_sha256": file_hash(latest),
        "latest_candidate": str(candidate),
        "latest_candidate_sha256": file_hash(candidate) if candidate.exists() else None,
        "completed_updates": step,
        "durable_updates": step,
        "charged_seconds": budget["charged_seconds"],
        "probe_updates": budget["safe_runtime"]["executed_probe_updates"],
        "original_budget_sha256": file_hash(Path(config["run_root"]) / "budget.json"),
        "prior_head": json.loads((root / "accounting-head.json").read_text()),
        "reports": [
            {"path": str(p), "sha256": file_hash(p)}
            for p in sorted((root / "dev").glob("epoch-*.json"))
        ],
        "owners_exited": True,
        "old_scope_consumed": True,
        "sealed_wall": time.time(),
        "source_completed": state["status"] == "seed0_complete",
    }
    with writer(Path(config["run_root"])), writer(_directory(config, control_root)):
        atomic_json(_directory(config, control_root) / "sealed.json", result)
    return result

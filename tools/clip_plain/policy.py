"""Immutable variant definitions and CPU-only, fail-closed authorization."""

import json
import os
import re
import time
from pathlib import Path

from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import atomic_json, writer

PROJECT = Path("/home/kan/Research/DeSpaMo")
GPU_HOLD = Path(
    "/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3/manual-stop.json"
)
POLICY = "official-clip-global224-safe-dev-spamo200-v1"
FEATURE_POLICY = "official-frozen-clip-global224-spatial-v1"
COUNTS = {"train": 7096, "dev": 519, "test": 642}


def evaluation_policy() -> dict:
    return dict(
        id="clip-plain-cached-b1-beam5-v1",
        batch_size=1,
        beam_size=5,
        max_length=64,
        do_sample=False,
        use_cache=True,
        telemetry_version=1,
    )


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    canonical = json.loads(Path(__file__).with_name("config.json").read_text())
    if config != canonical:
        raise ValueError("Exact reviewed plain-CLIP configuration required")
    return config


def recipe(config: dict) -> dict:
    result = dict(config["recipe"])
    every = result.pop("dev_every")
    result["dev_epochs"] = list(range(every, result["epochs"] + 1, every))
    return result


def source_hashes() -> dict[str, str]:
    runtime = Path(__file__).resolve().parents[2]
    paths = set(Path(__file__).parent.glob("*.py"))
    paths.update((PROJECT / "src").rglob("*.py"))
    for directory in ["benchmark", "followup", "dinov3"]:
        paths.update((runtime / "tools" / directory).glob("*.py"))
    paths.add(runtime / "tools/clip_s2/guards.py")
    paths.update([PROJECT / "pyproject.toml", PROJECT / "uv.lock"])
    return {str(p): sha256_file(p) for p in sorted(paths)}


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owner_alive(owner: dict | None) -> bool:
    return bool(owner) and process_owner(owner["pid"]) == owner


def require_exclusive_gpu() -> None:
    """Call only after CPU-only scope validation; never after a device fault."""
    from tools.benchmark.feature_package import other_gpu_pids

    if other_gpu_pids():
        raise RuntimeError("Another GPU compute worker is active")


def require_gpu_scope(
    config: dict, stage: str, approval: Path | None, *, job: dict | None = None
) -> dict:
    if approval is None:
        raise RuntimeError("Explicit reviewed GPU approval required")
    if (Path(config["study_root"]) / "manual-stop.json").exists():
        raise RuntimeError("Plain study safety hold is active")
    record = json.loads(approval.read_text())
    driver = Path(record.get("driver_review", "/nonexistent"))
    token = record.get("token", "")
    if (
        record.get("approved") is not True
        or record.get("consumed", False)
        or not re.fullmatch(r"[a-zA-Z0-9_-]+", token)
        or record.get("boot_id") != boot_id()
        or record.get("expires_at", 0) <= time.time()
        or record.get("config_sha256") != digest(config)
        or record.get("source_hashes") != source_hashes()
        or not GPU_HOLD.is_file()
        or record.get("hold_sha256") != sha256_file(GPU_HOLD)
        or stage not in record.get("stages", [])
        or stage not in {"train", "dev", "study"}
        or not driver.is_file()
        or sha256_file(driver) != record.get("driver_review_sha256")
        or not set(record.get("seeds", [])).issubset(config["seeds"])
        or len(record.get("seeds", [])) != len(set(record.get("seeds", [])))
        or not record.get("seeds")
    ):
        raise RuntimeError("GPU scope is stale, consumed or mismatched")
    if job is not None:
        if "run_root" in job:
            expected = Path(config["study_root"]) / "runs" / f"seed{job['seed']}"
            ticket = expected / "jobs" / f"{token}-{stage}-{job['epoch']}" / "job.json"
            if (
                Path(job["run_root"]).resolve() != expected.resolve()
                or not ticket.is_file()
                or json.loads(ticket.read_text()) != job
            ):
                raise RuntimeError("Child ticket file or run path changed")
        claim = Path(config["study_root"]) / "attempts" / token / "claim.json"
        if not claim.is_file():
            raise RuntimeError("Parent attempt not claimed")
        claimed = json.loads(claim.read_text())
        if (
            claimed["owner"] != job.get("parent_owner")
            or not owner_alive(claimed["owner"])
            or job.get("attempt_token") != token
            or job.get("seed") not in record["seeds"]
            or claimed["approval_sha256"] != sha256_file(approval)
            or job.get("max_batches") != record.get("max_batches")
            and stage == "dev"
        ):
            raise RuntimeError("Child ticket not bound to exact live parent/scope")
    return record


def claim_attempt(config: dict, approval: Path) -> dict:
    raw = json.loads(approval.read_text())
    stage = "study" if "study" in raw.get("stages", []) else raw.get("stages", [""])[0]
    record = require_gpu_scope(config, stage, approval)
    root = Path(config["study_root"]) / "attempts" / record["token"]
    with writer(root):
        path = root / "claim.json"
        if path.exists():
            raise RuntimeError("Attempt already claimed; no repeat")
        result = dict(
            token=record["token"],
            owner=process_owner(os.getpid()),
            approval_sha256=sha256_file(approval),
            started_wall=time.time(),
        )
        atomic_json(path, result)
        return result

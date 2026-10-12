"""CPU-only provenance and fail-closed scope checks; never clears legacy hold."""

import json
import math
import os
import re
import time
from importlib.metadata import version
from pathlib import Path

from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import atomic_json, writer


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    canonical = json.loads(Path(__file__).with_name("config.json").read_text())
    if config != canonical:
        raise ValueError("Exact reviewed S² safe-resume configuration required")
    return config


def evaluation_policy() -> dict:
    return dict(
        id="clip-s2-cached-b1-beam5-v1",
        batch_size=1,
        use_cache=True,
        beam_size=5,
        max_length=64,
        do_sample=False,
        telemetry_version=1,
    )


def recipe(config: dict) -> dict:
    r = dict(config["recipe"])
    if "dev_every" in r:
        every = r.pop("dev_every")
        r["dev_epochs"] = list(range(every, r["epochs"] + 1, every))
    return r


def boot_id() -> str:
    return Path("/proc/sys/kernel/random/boot_id").read_text().strip()


def owner_alive(owner: dict | None) -> bool:
    return bool(owner) and process_owner(owner["pid"]) == owner


def source_hashes() -> dict[str, str]:
    here = Path(__file__).resolve().parent
    canonical = json.loads((here / "config.json").read_text())
    original = json.loads(Path(canonical["anchor"]).with_suffix(".json").read_text())["identity"]
    paths = {Path(p) for p in original["code_hashes"]}
    paths.update(here.glob("*.py"))
    paths.update([here.parent / "clip_plain/generation.py", here.parent / "clip_plain/policy.py"])
    return {str(p): sha256_file(p) for p in sorted(paths)}


def verify_original(config: dict) -> dict:
    anchor = Path(config["anchor"])
    receipt = anchor.with_suffix(".json")
    if (
        anchor.is_symlink()
        or receipt.is_symlink()
        or sha256_file(anchor) != config["anchor_sha256"]
        or sha256_file(receipt) != config["anchor_receipt_sha256"]
    ):
        raise ValueError("Original anchor payload/receipt changed")
    row = json.loads(receipt.read_text())
    identity = row["identity"]
    if (
        row["step"] != config["anchor_step"]
        or row["bytes"] != anchor.stat().st_size
        or identity["seed"] != 0
        or identity["arm"] != "reference"
        or identity["policy"] != config["original_policy"]
        or identity["recipe"] != recipe(config)
        or identity["frozen_fingerprint"] != config["frozen_fingerprint"]
        or len(identity["code_hashes"]) != config["expected_source_files"]
        or identity["runtime"] != {n: version(n) for n in ["torch", "transformers", "peft"]}
        or sha256_file(Path(config["study_config"])) != identity["config_sha256"]
    ):
        raise ValueError("Original schema/recipe/runtime mismatch")
    for table in [identity["code_hashes"], identity["base_files"], identity["data"]["bindings"]]:
        if any(sha256_file(Path(p)) != sha for p, sha in table.items()):
            raise ValueError("Original source/base/data drift")
    hold = Path(config["study_root"]) / "manual-stop.json"
    if sha256_file(hold) != config["initial_hold_sha256"]:
        raise ValueError("Legacy hold changed; review required")
    h = json.loads(hold.read_text())
    if h["status"] != "stopped_by_user" or not h["automatic_recovery_disabled"]:
        raise ValueError("Legacy unsafe runner must remain blocked")
    budget_path = Path(config["run_root"]) / "budget.json"
    budget = json.loads(budget_path.read_text())
    budget_sha = sha256_file(budget_path)
    if (
        budget["identity"] != identity
        or budget["charged_seconds"] < config["initial_charged_seconds"]
        or budget["completed_updates"] < config["anchor_step"]
    ):
        raise ValueError("Original accounting identity/floor changed")
    if budget_sha != config["initial_budget_sha256"]:
        head = Path(config["runtime_root"]) / "accounting-head.json"
        if not head.is_file() or json.loads(head.read_text())["budget_sha256"] != budget_sha:
            raise ValueError("Original budget changed without committed runtime journal")
    proof = Path(config["full_diagnostic"])
    if sha256_file(proof) != config["full_diagnostic_sha256"]:
        raise ValueError("Full dev prerequisite changed")
    d = json.loads(proof.read_text())
    if (
        d["status"] != "completed"
        or d["processed_clips"] != config["expected_dev_clips"]
        or d["checkpoint_sha256"] != config["anchor_sha256"]
        or d["checkpoint_step"] != config["anchor_step"]
        or d["dev_ids_sha256"] != identity["data"]["ordered_ids"]["dev"]
        or d["batch_size"] != 1
        or d["use_cache"] is not True
        or d["beam_size"] != 5
        or d["max_length"] != 64
        or d["optimizer_loaded"] is not False
    ):
        raise ValueError("Prerequisite is not matching complete cached S² dev")
    return dict(identity=identity, anchor_receipt=row, budget=budget, hold=h)


def require_scope(
    config: dict, stage: str, approval: Path | None, *, job: dict | None = None
) -> dict:
    if approval is None:
        raise RuntimeError("Explicit separately reviewed GPU scope required")
    root = Path(config["runtime_root"])
    if (root / "manual-stop.json").exists():
        raise RuntimeError("Safe runtime is held")
    record = json.loads(approval.read_text())
    binding = json.loads((root / "migration.json").read_text())
    hold = Path(config["study_root"]) / "manual-stop.json"
    driver = Path(record.get("driver_review", "/missing"))
    expires = record.get("expires_at", 0)
    if (
        record.get("approved") is not True
        or record.get("consumed", False)
        or type(record.get("seed")) is not int
        or record.get("seed") != 0
        or record.get("mode") not in ["production", "handoff_probe"]
        or record.get("boot_id") != boot_id()
        or not isinstance(expires, (int, float))
        or not math.isfinite(expires)
        or expires <= time.time()
        or not re.fullmatch(r"[A-Za-z0-9_-]+", record.get("token", ""))
        or stage not in record.get("stages", [])
        or stage not in ["train", "dev", "supervisor", "adopt"]
        or record.get("binding_hash") != digest(binding)
        or record.get("source_hashes") != source_hashes()
        or binding["sources"] != source_hashes()
        or binding["config_hash"] != digest(config)
        or record.get("config_hash") != digest(config)
        or record.get("hold_sha256") != sha256_file(hold)
        or not driver.is_file()
        or sha256_file(driver) != record.get("driver_review_sha256")
        or type(record.get("max_train_updates")) is not int
        or record["max_train_updates"] < 0
        or type(record.get("max_dev_clips")) is not int
        or record["max_dev_clips"] < 0
    ):
        raise RuntimeError("GPU scope stale, mismatched or not authorized")
    if job is not None:
        claim = root / "attempts" / record["token"] / "claim.json"
        path = Path(job.get("ticket_path", "/missing"))
        if (
            not claim.is_file()
            or not path.is_file()
            or path.is_symlink()
            or (root / "attempts" / record["token"]).resolve() not in path.resolve().parents
            or json.loads(path.read_text()) != job
        ):
            raise RuntimeError("Missing/mutated parent ticket")
        parent = json.loads(claim.read_text())
        if (
            parent["owner"] != job.get("parent_owner")
            or not owner_alive(parent["owner"])
            or parent["approval_sha256"] != sha256_file(approval)
            or job.get("seed") != 0
            or type(job.get("seed")) is not int
            or job.get("allow_owned_resume_pruning", False)
            != record.get("allow_owned_resume_pruning", False)
            or job.get("mode") != record["mode"]
            or job.get("binding_hash") != record["binding_hash"]
            or job.get("token") != record["token"]
            or job.get("stage") != stage
        ):
            raise RuntimeError("Child is not bound to exact live parent/scope")
    return record


def claim_scope(config: dict, approval: Path) -> dict:
    record = require_scope(config, "supervisor", approval)
    root = Path(config["runtime_root"]) / "attempts" / record["token"]
    with writer(root):
        if (root / "claim.json").exists():
            raise RuntimeError("Scope token already claimed; no retry")
        row = dict(
            token=record["token"],
            owner=process_owner(os.getpid()),
            approval_sha256=sha256_file(approval),
        )
        atomic_json(root / "claim.json", row)
        return row


def require_exclusive_gpu() -> None:
    from tools.benchmark.feature_package import other_gpu_pids

    if other_gpu_pids():
        raise RuntimeError("Another GPU compute worker exists")

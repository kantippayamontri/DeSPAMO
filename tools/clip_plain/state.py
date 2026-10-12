"""Plain-run checkpoint identity, immutable lineage and parent-owned accounting."""

import json
import math
import os
import time
from importlib.metadata import version
from pathlib import Path

from tools.benchmark.compact_checkpoint import (
    common_fingerprint,
    load_snapshot,
    restore_rng,
    save_snapshot,
)
from tools.clip_plain.policy import POLICY, evaluation_policy, owner_alive, recipe, source_hashes
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _copy_file, require_archive_storage
from tools.dinov3.official_runtime import process_owner, require_official_storage
from tools.dinov3.storage import atomic_json, writer
from tools.followup.recipe import PromptPool


def base_files(cache: str) -> dict:
    root = (
        Path(cache)
        / "models--google--flan-t5-xl/snapshots/7d6315df2c2fb742f0f5b556879d730926ca9001"
    )
    return {str(p): sha256_file(p) for p in sorted(root.iterdir()) if p.is_file()}


def make_identity(config: dict, admission: dict, seed: int, model, base_hash: str) -> dict:
    return dict(
        policy=POLICY,
        seed=seed,
        arm="reference",
        recipe=recipe(config),
        data=admission["identity"],
        code_hashes=source_hashes(),
        config_sha256=digest(config),
        base_files=base_files(config["cache"]),
        frozen_fingerprint=base_hash,
        common_initialization=common_fingerprint(model),
        runtime={n: version(n) for n in ["torch", "transformers", "peft"]},
        evaluation_policy_hash=digest(evaluation_policy()),
        decoder={
            **evaluation_policy(),
            "exemplar_id": PromptPool(admission["datasets"]["train"].records, seed).eval_id,
        },
    )


def ensure_lineage(config: dict) -> dict:
    old = Path(config["historical_root"])
    budgets = sorted((old / "runs").glob("*/budget.json"))
    if len(budgets) != 1:
        raise ValueError("Exact retained S² run required")
    budget = json.loads(budgets[0].read_text())
    checkpoint = budgets[0].parent / "resume-epoch-2.pt"
    if budget["completed_updates"] != 1774 or budget["charged_seconds"] != 497.5533489369991:
        raise ValueError("Historical accounting changed; review before new variant")
    row = dict(
        historical_root=str(old),
        historical_updates=1774,
        historical_charge_seconds=budget["charged_seconds"],
        budget_sha256=sha256_file(budgets[0]),
        checkpoint_sha256=sha256_file(checkpoint),
        hold_sha256=sha256_file(old / "manual-stop.json"),
        parent_identity_sha256=config["parent_identity_sha256"],
    )
    if (
        row["checkpoint_sha256"]
        != "d70cb46fbfd181f7a04c1c1ae082370031216a7799141684260a6db6813215c9"
    ):
        raise ValueError("Historical checkpoint changed")
    path = Path(config["study_root"]) / "lineage.json"
    if path.exists() and json.loads(path.read_text()) != row:
        raise ValueError("Historical lineage drift")
    if not path.exists():
        atomic_json(path, row)
    return row


def load_weights(path: Path, model, identity: dict, base_hash: str) -> dict:
    if identity.get("policy") != POLICY:
        raise ValueError("Plain policy required; S² snapshots cannot resume")
    return load_snapshot(path, model, None, identity, base_hash=base_hash, replay=False)


def restore_training(state: dict, model, optimizer, sampler) -> None:
    if not {"optimizer", "sampler", "rng"}.issubset(state):
        raise ValueError("Full plain training checkpoint required")
    optimizer.load_state_dict(state["optimizer"])
    sampler.load_state_dict(state["sampler"])
    model.rng.setstate(state["prompt_rng"])
    restore_rng(state["rng"])


def publish_epoch(run_root: Path, state: dict, epoch: int, *, dev_candidate: bool) -> dict:
    identity = state["identity"]
    if (
        identity.get("policy") != POLICY
        or state["step"] != epoch * identity["recipe"]["steps_per_epoch"]
    ):
        raise ValueError("Plain epoch/identity mismatch")
    path = run_root / f"resume-epoch-{epoch}.pt"
    receipt = save_snapshot(path, state)
    result = dict(
        checkpoint=str(path), checkpoint_sha256=receipt["sha256"], step=state["step"], epoch=epoch
    )
    if dev_candidate:
        candidate = run_root / f"candidate-epoch-{epoch}.pt"
        slim = {
            k: state[k]
            for k in [
                "schema_version",
                "identity",
                "step",
                "weights",
                "frozen_fingerprint",
                "prompt_rng",
            ]
        }
        r = save_snapshot(candidate, slim)
        result.update(candidate=str(candidate), candidate_sha256=r["sha256"])
    return result


def archive_candidate(config: dict, candidate: Path) -> dict:
    run = Path(config["study_root"]) / "runs"
    if candidate.is_symlink() or candidate.parent.parent.resolve() != run.resolve():
        raise ValueError("Foreign candidate path")
    receipt = json.loads(candidate.with_suffix(".json").read_text())
    if receipt["identity"].get("policy") != POLICY or sha256_file(candidate) != receipt["sha256"]:
        raise ValueError("Candidate integrity mismatch")
    require_archive_storage(Path(config["archive_root"]), candidate.stat().st_size + 4096)
    target = Path(config["archive_root"]) / candidate.parent.name / candidate.name
    if target.exists():
        if sha256_file(target) != receipt["sha256"]:
            raise ValueError("Archived candidate differs")
    else:
        _copy_file(candidate, target)
    atomic_json(target.with_suffix(".json"), receipt)
    return dict(path=str(target), sha256=sha256_file(target))


def prune_owned_resumes(run_root: Path, identity: dict, *, allowed: bool) -> list[str]:
    if not allowed:
        return []
    if identity.get("policy") != POLICY:
        raise ValueError("Refuse pruning foreign experiment")
    paths = sorted(run_root.glob("resume-epoch-*.pt"), key=lambda p: int(p.stem.split("-")[-1]))
    for p in paths:
        r = json.loads(p.with_suffix(".json").read_text())
        if p.is_symlink() or r["identity"] != identity or sha256_file(p) != r["sha256"]:
            raise ValueError("Refuse pruning mismatched checkpoint")
    removed = []
    for p in paths[:-2]:
        p.unlink()
        p.with_suffix(".json").unlink()
        removed.append(p.name)
    return removed


def begin_stage(run_root: Path, identity: dict, attempt: dict) -> dict:
    with writer(run_root / "accounting"):
        path = run_root / "budget.json"
        b: dict = (
            json.loads(path.read_text())
            if path.exists()
            else dict(
                identity=identity,
                charged_seconds=0.0,
                completed_updates=0,
                durable_updates=0,
                attempts={},
            )
        )
        if b["identity"] != identity or b.get("status") == "running":
            raise RuntimeError("Ledger identity/live attempt conflict")
        if attempt["token"] in b["attempts"]:
            raise RuntimeError("Stage token already used")
        row = dict(
            stage=attempt["stage"],
            started_wall=time.time(),
            base_charge=b["charged_seconds"],
            status="running",
        )
        b["attempts"][attempt["token"]] = row
        b.update(
            status="running",
            phase=attempt["stage"],
            owner=process_owner(os.getpid()),
            duration_limit_seconds=None,
        )
        atomic_json(path, b)
        return b


def record_stage(
    run_root: Path, attempt: dict, progress: dict, *, final_status: str | None = None
) -> dict:
    with writer(run_root / "accounting"):
        path = run_root / "budget.json"
        b = json.loads(path.read_text())
        a = b["attempts"][attempt["token"]]
        if a["status"] != "running":
            return b
        elapsed = time.time() - a["started_wall"]
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Backward/invalid wall clock; preserve accounting")
        updates = progress["completed_updates"]
        durable = progress["durable_updates"]
        if (
            type(updates) is not int
            or type(durable) is not int
            or updates < b["completed_updates"]
            or durable > updates
        ):
            raise ValueError("Invalid update accounting")
        b.update(
            charged_seconds=max(b["charged_seconds"], a["base_charge"] + elapsed),
            completed_updates=updates,
            durable_updates=durable,
        )
        if final_status is not None:
            a.update(status=final_status, finished_wall=time.time())
            b["status"] = final_status
        atomic_json(path, b)
        return b


def reconcile_abandoned_stage(run_root: Path, *, reviewed: bool) -> dict:
    b = json.loads((run_root / "budget.json").read_text())
    if not reviewed or owner_alive(b.get("owner")):
        raise RuntimeError("Manual review and dead owner required")
    for token, a in b["attempts"].items():
        if a["status"] == "running":
            b = record_stage(
                run_root,
                dict(token=token),
                dict(
                    completed_updates=b["completed_updates"], durable_updates=b["durable_updates"]
                ),
                final_status="interrupted",
            )
    if b["completed_updates"] > b["durable_updates"]:
        with writer(run_root / "accounting"):
            b = json.loads((run_root / "budget.json").read_text())
            b.setdefault("replay_history", []).append(
                dict(
                    completed_before=b["completed_updates"],
                    durable_replay=b["durable_updates"],
                    charged_floor=b["charged_seconds"],
                    reviewed_wall=time.time(),
                )
            )
            b["completed_updates"] = b["durable_updates"]
            atomic_json(run_root / "budget.json", b)
    return b


def cost_summary(config: dict) -> dict:
    lineage = json.loads((Path(config["study_root"]) / "lineage.json").read_text())
    charges = sum(
        json.loads(p.read_text())["charged_seconds"]
        for p in (Path(config["study_root"]) / "runs").glob("*/budget.json")
    )
    prep = Path(config["feature_base"]) / "preparation-budget.json"
    preparation = json.loads(prep.read_text())["charged_seconds"] if prep.exists() else 0.0
    return dict(
        historical_charge_seconds=lineage["historical_charge_seconds"],
        new_charged_seconds=charges,
        preparation_seconds=preparation,
        total_charged_seconds=lineage["historical_charge_seconds"] + charges + preparation,
        historical_updates=lineage["historical_updates"],
    )


def require_storage(config: dict, run_root: Path, *, prune_allowed: bool) -> dict:
    paths = list(run_root.glob("resume-epoch-*.pt"))
    size = max((p.stat().st_size for p in paths), default=1024**3)
    candidates = list(run_root.glob("candidate-epoch-*.pt"))
    small = max((p.stat().st_size for p in candidates), default=size)
    native = (4 if prune_allowed else 400) * size + 2 * small + 64 * 1024**2
    archive = 200 * small
    return dict(
        native=require_official_storage(Path(config["study_root"]), native),
        archive=require_archive_storage(Path(config["archive_root"]), archive),
    )

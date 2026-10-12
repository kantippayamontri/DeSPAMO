"""Parent-owned original-budget transactions; provisional probe progress is separate."""

import copy
import json
import math
import time
from pathlib import Path

from tools.clip_s2_safe.policy import owner_alive, require_scope
from tools.clip_s2_safe.state import checkpoint_bundle, read_binding, verify_binding
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_archive import _copy_file, require_archive_storage
from tools.dinov3.official_runtime import require_official_storage
from tools.dinov3.storage import atomic_json, writer


def _path(config: dict, token: str) -> Path:
    return Path(config["runtime_root"]) / "attempts" / token / "attempt.json"


def _budget(config: dict, binding: dict) -> dict:
    path = Path(config["run_root"]) / "budget.json"
    b = json.loads(path.read_text())
    if (
        b["identity"] != binding["original_identity"]
        or not math.isfinite(b["charged_seconds"])
        or b["charged_seconds"] < binding["initial_charged_seconds"]
    ):
        raise ValueError("Authoritative original budget identity/floor changed")
    root = Path(config["runtime_root"])
    head = root / "accounting-head.json"
    if (root / "pending-budget.json").exists():
        raise RuntimeError("Torn transaction requires explicit reconciliation")
    expected = (
        json.loads(head.read_text())["budget_sha256"]
        if head.exists()
        else binding["initial_budget_sha256"]
    )
    if sha256_file(path) != expected:
        raise ValueError("Unjournalled original budget change")
    return b


def _commit(
    config: dict, binding: dict, before: dict, after: dict, attempt: dict, reason: str
) -> dict:
    root = Path(config["runtime_root"])
    budget_path = Path(config["run_root"]) / "budget.json"
    if digest(json.loads(budget_path.read_text())) != digest(before):
        raise ValueError("Concurrent budget writer")
    head = root / "accounting-head.json"
    seq = json.loads(head.read_text())["sequence"] + 1 if head.exists() else 1
    intent = root / "journal" / f"{seq:08}.json"
    if intent.exists():
        raise RuntimeError("Orphan journal intention requires manual reconciliation")
    row = dict(
        schema_version=1,
        sequence=seq,
        binding_hash=digest(binding),
        before_sha256=sha256_file(budget_path),
        after_budget=after,
        after_content_hash=digest(after),
        attempt=attempt,
        attempt_path=str(_path(config, attempt["token"])),
        reason=reason,
    )
    atomic_json(intent, row)
    atomic_json(
        root / "pending-budget.json", dict(intent=str(intent), intent_sha256=sha256_file(intent))
    )
    atomic_json(budget_path, after)
    atomic_json(_path(config, attempt["token"]), attempt)
    atomic_json(
        head,
        dict(
            sequence=seq,
            binding_hash=digest(binding),
            budget_sha256=sha256_file(budget_path),
            intent=str(intent),
            intent_sha256=sha256_file(intent),
        ),
    )
    atomic_json(intent.with_suffix(".commit.json"), dict(budget_sha256=sha256_file(budget_path)))
    if "adoption" in attempt:
        entry = attempt["adoption"]
        atomic_json(root / "adoptions" / f"{entry['checkpoint_sha256']}.json", entry)
    (root / "pending-budget.json").unlink()
    return after


def _elapsed(start: float) -> float:
    value = time.time() - start
    if not math.isfinite(value) or value < 0:
        raise ValueError("Backward/invalid wall clock; preserve budget")
    return value


def begin_attempt(config: dict, binding: dict, scope: dict, owner: dict) -> dict:
    verify_binding(binding)
    if (
        scope.get("approved") is not True
        or scope["binding_hash"] != digest(binding)
        or scope["mode"] not in ["production", "handoff_probe"]
    ):
        raise RuntimeError("Reviewed exact attempt scope required")
    root = Path(config["runtime_root"])
    with writer(root / "accounting"):
        b = _budget(config, binding)
        if b["status"] == "running" and owner_alive(b.get("owner")):
            raise RuntimeError("Original budget still live")
        if _path(config, scope["token"]).exists():
            raise RuntimeError("Accounting token already used")
        for p in (root / "attempts").glob("*/attempt.json"):
            if json.loads(p.read_text())["status"] == "active":
                raise RuntimeError("Earlier active attempt must be reviewed")
        attempt = dict(
            token=scope["token"],
            binding_hash=digest(binding),
            mode=scope["mode"],
            scope=scope,
            owner=owner,
            status="active",
            stages={},
            active_stage=None,
            train_used=0,
            dev_used=0,
            started_wall=time.time(),
            last_charge_wall=time.time(),
        )
        after = copy.deepcopy(b)
        after["owner"] = owner
        after.setdefault(
            "safe_runtime",
            dict(
                binding_hash=digest(binding),
                durable_step=b["completed_updates"],
                executed_probe_updates=0,
                replay_history=[],
            ),
        )
        if after["safe_runtime"]["binding_hash"] != digest(binding):
            raise ValueError("Ledger belongs to different safe runtime")
        _commit(config, binding, b, after, attempt, "attempt_start")
        return attempt


def begin_stage(config: dict, attempt: dict, job: dict) -> dict:
    binding = read_binding(config)
    root = Path(config["runtime_root"])
    with writer(root / "accounting"):
        b = _budget(config, binding)
        a = json.loads(_path(config, attempt["token"]).read_text())
        sid = job["stage_id"]
        if (
            a["status"] != "active"
            or a["active_stage"] is not None
            or sid in a["stages"]
            or job["mode"] != a["mode"]
            or job["stage"] not in ["train", "dev"]
        ):
            raise RuntimeError("Stage ownership/mode/reuse conflict")
        delta = job["target_step"] - job["start_step"] if job["stage"] == "train" else 0
        clips = (
            config["expected_dev_clips"]
            if job.get("max_batches") is None
            else min(job["max_batches"], config["expected_dev_clips"])
        )
        if (
            delta < 0
            or a["train_used"] + delta > a["scope"]["max_train_updates"]
            or job["stage"] == "dev"
            and a["dev_used"] + clips > a["scope"]["max_dev_clips"]
        ):
            raise RuntimeError("Cumulative attempt cap exceeded")
        if job["mode"] == "production" and job["start_step"] != b["completed_updates"]:
            raise ValueError("Stage starts outside authoritative progress")
        after = copy.deepcopy(b)
        after["charged_seconds"] += _elapsed(a["last_charge_wall"])
        row = dict(
            stage=job["stage"],
            mode=job["mode"],
            start_step=job["start_step"],
            target_step=job["target_step"],
            started_wall=time.time(),
            base_charge=after["charged_seconds"],
            status="running",
            highwater=job["start_step"],
            durable_step=job["start_step"],
            completed_clips=0,
        )
        a["stages"][sid] = row
        a["active_stage"] = sid
        after.update(status="running", phase=job["stage"], owner=a["owner"])
        _commit(config, binding, b, after, a, "stage_start")
        return row


def _progress(
    config: dict, binding: dict, b: dict, a: dict, job: dict, progress: dict, status: str | None
) -> tuple[dict, dict]:
    sid = job["stage_id"]
    stage = a["stages"][sid]
    if stage["status"] != "running":
        return b, a
    elapsed = _elapsed(stage["started_wall"])
    step = progress.get("completed_updates", stage["highwater"])
    durable = progress.get("durable_updates", stage["durable_step"])
    clips = progress.get("completed_clips", stage["completed_clips"])
    if (
        type(step) is not int
        or type(durable) is not int
        or type(clips) is not int
        or step < stage["highwater"]
        or step > stage["target_step"]
        or durable > step
        or durable < stage["durable_step"]
        or clips < stage["completed_clips"]
    ):
        raise ValueError("Worker progress count regression or overflow")
    if job["stage"] == "dev" and (
        step != stage["start_step"] or a["dev_used"] + clips > a["scope"]["max_dev_clips"]
    ):
        raise ValueError("Dev progress changes training or exceeds cap")
    after = copy.deepcopy(b)
    after["charged_seconds"] = max(after["charged_seconds"], stage["base_charge"] + elapsed)
    stage.update(highwater=step, durable_step=durable, completed_clips=clips)
    if a["mode"] == "production":
        after["completed_updates"] = step
        after["safe_runtime"]["durable_step"] = durable
    if status is not None:
        if status == "complete" and step != stage["target_step"]:
            raise ValueError("Successful stage did not reach target")
        stage["status"] = status
        stage["finished_wall"] = time.time()
        a["active_stage"] = None
        a["last_charge_wall"] = time.time()
        if job["stage"] == "train":
            used = step - stage["start_step"]
            a["train_used"] += used
            if a["mode"] == "handoff_probe":
                after["safe_runtime"]["executed_probe_updates"] += used
        else:
            a["dev_used"] += clips
        after["status"] = "interrupted" if status != "complete" else "stage_complete"
        after["phase"] = "stage_finalize"
        if status != "complete":
            a["status"] = "held"
    return after, a


def record_progress(config: dict, attempt: dict, job: dict, progress: dict) -> dict:
    binding = read_binding(config)
    with writer(Path(config["runtime_root"]) / "accounting"):
        b = _budget(config, binding)
        a = json.loads(_path(config, attempt["token"]).read_text())
        after, a = _progress(config, binding, b, a, job, progress, None)
        if after == b:
            return b
        return _commit(config, binding, b, after, a, "progress")


def finish_stage(config: dict, attempt: dict, job: dict, result: dict, *, status: str) -> dict:
    binding = read_binding(config)
    with writer(Path(config["runtime_root"]) / "accounting"):
        b = _budget(config, binding)
        a = json.loads(_path(config, attempt["token"]).read_text())
        if a["stages"][job["stage_id"]]["status"] != "running":
            return b
        after, a = _progress(config, binding, b, a, job, result, status)
        return _commit(config, binding, b, after, a, "stage_finish")


def finish_attempt(config: dict, attempt: dict, *, status: str) -> dict:
    binding = read_binding(config)
    with writer(Path(config["runtime_root"]) / "accounting"):
        b = _budget(config, binding)
        a = json.loads(_path(config, attempt["token"]).read_text())
        if a["active_stage"] is not None:
            raise RuntimeError("Active stage must finish first")
        if a["status"] == "complete":
            return b
        after = copy.deepcopy(b)
        after["charged_seconds"] += _elapsed(a["last_charge_wall"])
        a["last_charge_wall"] = time.time()
        a["status"] = status
        after["status"] = "interrupted" if status != "complete" else "stage_complete"
        return _commit(config, binding, b, after, a, "attempt_finish")


def _recover_transaction(config: dict, binding: dict) -> None:
    root = Path(config["runtime_root"])
    pending = root / "pending-budget.json"
    head = root / "accounting-head.json"
    if pending.exists():
        p = json.loads(pending.read_text())
        intent = Path(p["intent"])
        if sha256_file(intent) != p["intent_sha256"]:
            raise ValueError("Pending intention changed")
    else:
        seq = json.loads(head.read_text())["sequence"] + 1 if head.exists() else 1
        intent = root / "journal" / f"{seq:08}.json"
        if not intent.exists():
            return
    row = json.loads(intent.read_text())
    current = Path(config["run_root"]) / "budget.json"
    if row["binding_hash"] != digest(binding) or owner_alive(row["attempt"]["owner"]):
        raise RuntimeError("Recovery requires matching binding and dead parent")
    if sha256_file(current) == row["before_sha256"]:
        atomic_json(current, row["after_budget"])
    elif digest(json.loads(current.read_text())) != row["after_content_hash"]:
        raise ValueError("Torn transaction budget is neither before nor after")
    atomic_json(Path(row["attempt_path"]), row["attempt"])
    atomic_json(
        head,
        dict(
            sequence=row["sequence"],
            binding_hash=digest(binding),
            budget_sha256=sha256_file(current),
            intent=str(intent),
            intent_sha256=sha256_file(intent),
        ),
    )
    atomic_json(intent.with_suffix(".commit.json"), dict(budget_sha256=sha256_file(current)))
    if "adoption" in row["attempt"]:
        entry = row["attempt"]["adoption"]
        atomic_json(root / "adoptions" / f"{entry['checkpoint_sha256']}.json", entry)
    if pending.exists():
        pending.unlink()


def reconcile_dead_attempt(config: dict, binding: dict, *, reviewed: bool) -> dict:
    if not reviewed:
        raise RuntimeError("Explicit dead-attempt CPU reconciliation review required")
    root = Path(config["runtime_root"])
    with writer(root / "accounting"):
        _recover_transaction(config, binding)
    for path in (root / "attempts").glob("*/attempt.json"):
        a = json.loads(path.read_text())
        if a["status"] == "active":
            if owner_alive(a["owner"]):
                raise RuntimeError("Live parent cannot be reconciled")
            sid = a["active_stage"]
            if sid is not None:
                row = a["stages"][sid]
                job = dict(stage_id=sid, stage=row["stage"], mode=a["mode"])
                finish_stage(
                    config,
                    a,
                    job,
                    dict(
                        completed_updates=row["highwater"],
                        durable_updates=row["durable_step"],
                        completed_clips=row["completed_clips"],
                    ),
                    status="interrupted",
                )
            else:
                finish_attempt(config, a, status="interrupted")
    with writer(root / "accounting"):
        b = _budget(config, binding)
        if b["completed_updates"] > b["safe_runtime"]["durable_step"]:
            a = json.loads(path.read_text())
            after = copy.deepcopy(b)
            after["safe_runtime"]["replay_history"].append(
                dict(
                    completed_before=b["completed_updates"],
                    durable_step=b["safe_runtime"]["durable_step"],
                    charged_floor=b["charged_seconds"],
                )
            )
            after["completed_updates"] = b["safe_runtime"]["durable_step"]
            b = _commit(config, binding, b, after, a, "reviewed_replay")
        return b


def adopt_probe(config: dict, binding: dict, probe: dict, approval: Path | None) -> dict:
    scope = require_scope(config, "adopt", approval)
    assert approval is not None
    if scope.get("approve_adoption") is not True or scope.get(
        "probe_checkpoint_sha256"
    ) != probe.get("checkpoint_sha256"):
        raise RuntimeError("Exact separately reviewed probe adoption required")
    path = Path(probe["checkpoint"])
    bundle = checkpoint_bundle(path, binding)
    if (
        path.parent.name != "handoff_probe"
        or bundle["receipt"]["sha256"] != probe["checkpoint_sha256"]
    ):
        raise ValueError("Probe checkpoint binding mismatch")
    with writer(Path(config["runtime_root"]) / "accounting"):
        b = _budget(config, binding)
        if b["completed_updates"] != binding["anchor_step"]:
            raise ValueError("Adoption requires unchanged production anchor position")
        proof = Path(probe.get("handoff", "/missing"))
        if not proof.is_file():
            raise RuntimeError("Completed train/exit/dev handoff evidence required")
        if (
            Path(config["runtime_root"]) / "validation" not in proof.parents
            or sha256_file(proof) != probe.get("handoff_sha256")
            or scope.get("handoff_sha256") != probe.get("handoff_sha256")
        ):
            raise ValueError("Adoption handoff scope/path/hash mismatch")
        h = json.loads(proof.read_text())
        if (
            h.get("status") != "completed"
            or h.get("binding_hash") != digest(binding)
            or h.get("checkpoint_sha256") != probe["checkpoint_sha256"]
            or h.get("end_step") != bundle["receipt"]["step"]
        ):
            raise ValueError("Completed handoff checkpoint mismatch")
        old_attempt = json.loads(_path(config, h["token"]).read_text())
        if (
            old_attempt.get("status") != "complete"
            or old_attempt.get("mode") != "handoff_probe"
            or old_attempt["binding_hash"] != digest(binding)
        ):
            raise RuntimeError("Probe handoff was not completed in authoritative journal")
        dev = Path(h["dev_report"])
        if sha256_file(dev) != h["dev_report_sha256"]:
            raise ValueError("Handoff dev report changed")
        d = json.loads(dev.read_text())
        if (
            d.get("diagnostic_only") is not True
            or d.get("metrics") is not None
            or d.get("binding_hash") != digest(binding)
            or d.get("step") != h["end_step"]
            or not d.get("items")
        ):
            raise ValueError("Handoff dev did not validate the checkpoint")
        gate = Path(config["runtime_root"]) / "production-anchor-dev-gate.json"
        if not gate.is_file():
            raise RuntimeError("Production anchor dev must precede adoption")
        g = json.loads(gate.read_text())
        report = Path(g["report"])
        if g["binding_hash"] != digest(binding) or sha256_file(report) != g["report_sha256"]:
            raise ValueError("Production anchor gate changed")
        from tools.clip_s2_safe.evaluate import admit_original, validate_report

        r = json.loads(report.read_text())
        admission = admit_original(config, binding)
        validate_report(
            r,
            dict(
                mode="production",
                epoch=binding["anchor_step"] // binding["recipe"]["steps_per_epoch"],
                target_step=binding["anchor_step"],
                checkpoint_sha256=r["checkpoint_sha256"],
                max_batches=None,
            ),
            binding,
            admission,
        )
        root = Path(config["runtime_root"]) / "checkpoints/production"
        target = root / path.name
        for suffix in [".pt", ".json", ".guard.json"]:
            source = path.with_suffix(suffix)
            dest = target.with_suffix(suffix)
            if dest.exists() and sha256_file(dest) != sha256_file(source):
                raise ValueError("Adoption destination conflict")
            if not dest.exists():
                _copy_file(source, dest)
        runtime = json.loads(path.with_suffix(".runtime.json").read_text())
        runtime.update(mode="production", adoption_approval_sha256=sha256_file(approval))
        atomic_json(target.with_suffix(".runtime.json"), runtime)
        after = copy.deepcopy(b)
        after["completed_updates"] = bundle["receipt"]["step"]
        after["safe_runtime"]["durable_step"] = bundle["receipt"]["step"]
        a = dict(
            token=scope["token"],
            owner={"pid": 0, "boot_id": "none", "start_ticks": "0", "command": []},
            status="complete",
            mode="production",
            stages={},
            active_stage=None,
            adoption=dict(
                binding_hash=digest(binding),
                checkpoint_sha256=probe["checkpoint_sha256"],
                approval_sha256=sha256_file(approval),
                handoff_sha256=probe["handoff_sha256"],
            ),
        )
        _commit(config, binding, b, after, a, "explicit_probe_adoption")
    record = json.loads(approval.read_text())
    record["consumed"] = True
    atomic_json(approval, record)
    return dict(checkpoint=str(target), step=bundle["receipt"]["step"])


def require_storage(config: dict, binding: dict, job: dict) -> dict:
    size = Path(binding["anchor"]).stat().st_size
    if job.get("stage") == "dev":
        # Current candidate, selected copy and atomic staging; future train guards stay separate.
        native = 3 * size + 64 * 1024**2
        archive = size
    elif job.get("mode") == "handoff_probe":
        start, target = job["start_step"], job["target_step"]
        per = binding["recipe"]["steps_per_epoch"]
        snapshots = max(1, target // per - start // per + int(target % per != 0))
        # Full resume + model-only candidate per stop, plus one atomic staging file.
        native = size * (2 * snapshots + 1) + 64 * 1024**2
        archive = size * snapshots
    else:
        owned = (
            2 if job.get("allow_owned_resume_pruning") else len(binding["recipe"]["dev_epochs"]) * 2
        )
        native = max(64 * 1024**2, size * (owned + 2))
        archive = size * len(binding["recipe"]["dev_epochs"])
    return dict(
        native=require_official_storage(Path(config["runtime_root"]), native),
        archive=require_archive_storage(Path(config["archive_root"]), archive),
    )

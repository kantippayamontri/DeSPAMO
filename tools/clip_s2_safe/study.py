"""Same-root CPU supervisor: sequential owned workers, safe selection, no retry."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from tools.clip_s2_safe import accounting
from tools.clip_s2_safe.evaluate import admit_original, validate_probe_report, validate_report
from tools.clip_s2_safe.policy import (
    claim_scope,
    load_config,
    require_scope,
    verify_original,
)
from tools.clip_s2_safe.state import (
    archive_candidate,
    checkpoint_bundle,
    make_anchor_candidate,
    make_binding,
    prune_owned_resumes,
    read_binding,
)
from tools.clip_s2_safe.train import next_boundary
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import _atomic, atomic_json, writer
from tools.followup.recipe import stop_state


def _job_for_report(report: dict) -> dict:
    return dict(
        mode="production",
        epoch=report["epoch"],
        target_step=report["step"],
        checkpoint_sha256=report["checkpoint_sha256"],
        max_batches=None,
    )


def admitted_reports(config: dict, binding: dict, admission: dict) -> list[dict]:
    rows = []
    for path in sorted(
        (Path(config["runtime_root"]) / "dev").glob("epoch-*.json"),
        key=lambda p: int(p.stem.split("-")[-1]),
    ):
        r = json.loads(path.read_text())
        if r.get("diagnostic_only") is not False or r.get("complete") is not True:
            raise ValueError("Diagnostic/partial report cannot select")
        validate_report(r, _job_for_report(r), binding, admission)
        bundle = checkpoint_bundle(Path(r["checkpoint_path"]), binding)
        if (
            bundle["receipt"]["step"] != r["step"]
            or bundle["receipt"]["sha256"] != r["checkpoint_sha256"]
        ):
            raise ValueError("Report candidate drift")
        rows.append(r)
    stop_state(rows, binding["recipe"])
    return rows


def pending_stage(config: dict, binding: dict, admission: dict) -> dict:
    root = Path(config["runtime_root"])
    paths = list((root / "checkpoints/production").glob("resume-step-*.pt"))
    latest = (
        max(paths, key=lambda p: checkpoint_bundle(p, binding)["receipt"]["step"])
        if paths
        else Path(binding["anchor"])
    )
    bundle = checkpoint_bundle(latest, binding)
    step = bundle["receipt"]["step"]
    recipe = binding["recipe"]
    per = recipe["steps_per_epoch"]
    rows = (
        admitted_reports(config, binding, admission)
        if list((root / "dev").glob("epoch-*.json"))
        else []
    )
    if stop_state(rows, recipe)["stop"]:
        return dict(stage="complete", start_step=step, target_step=step)
    covered = {r["epoch"] for r in rows}
    for epoch in recipe["dev_epochs"]:
        target = epoch * per
        if target > step:
            break
        if epoch not in covered:
            if target == binding["anchor_step"]:
                candidate = make_anchor_candidate(config, binding)
            else:
                candidate = root / "checkpoints/production" / f"candidate-step-{target}.pt"
            cp = checkpoint_bundle(candidate, binding)
            return dict(
                stage="dev",
                start_step=target,
                target_step=target,
                epoch=epoch,
                checkpoint=str(candidate),
                checkpoint_sha256=cp["receipt"]["sha256"],
                max_batches=None,
            )
    return dict(
        stage="train",
        start_step=step,
        target_step=next_boundary(step, recipe),
        checkpoint=str(latest),
        checkpoint_sha256=bundle["receipt"]["sha256"],
        max_batches=None,
    )


def worker_command(config: dict, job: dict, approval: Path) -> list[str]:
    module = "train" if job["stage"] == "train" else "evaluate"
    return [
        sys.executable,
        "-B",
        "-m",
        f"tools.clip_s2_safe.{module}",
        "--config",
        str(Path(__file__).with_name("config.json")),
        "--job",
        job["ticket_path"],
        "--approval",
        str(approval),
    ]


def fail_hold(config: dict, binding: dict, error: BaseException, approval: Path | None) -> None:
    atomic_json(
        Path(config["runtime_root"]) / "manual-stop.json",
        dict(
            status="held",
            binding_hash=digest(binding),
            automatic_recovery_disabled=True,
            error_type=type(error).__name__,
            error=str(error),
            wall_timestamp=time.time(),
        ),
    )
    if approval is not None and approval.is_file():
        r = json.loads(approval.read_text())
        r["consumed"] = True
        r["outcome"] = "failed_no_retry"
        atomic_json(approval, r)


def _validate_result(config: dict, binding: dict, job: dict, result: dict) -> None:
    if (
        result.get("stage") != job["stage"]
        or result.get("mode") != job["mode"]
        or result.get("completed_updates") != job["target_step"]
        or result.get("durable_updates") != job["target_step"]
        or result.get("status") not in ["complete", "awaiting_dev"]
    ):
        raise ValueError("Worker result stage/mode/update mismatch")
    if job["stage"] == "train":
        path = Path(result["checkpoint"])
        bundle = checkpoint_bundle(path, binding)
        if (
            bundle["receipt"]["step"] != job["target_step"]
            or bundle["receipt"]["sha256"] != result["checkpoint_sha256"]
            or path.parent.name != job["mode"]
        ):
            raise ValueError("Train result snapshot mismatch")
        if "candidate" not in result:
            raise ValueError("Training handoff candidate missing")
        cp = checkpoint_bundle(Path(result["candidate"]), binding)
        if (
            cp["receipt"]["sha256"] != result["candidate_sha256"]
            or cp["receipt"]["step"] != job["target_step"]
        ):
            raise ValueError("Training candidate mismatch")
    else:
        path = Path(result["report"])
        if (
            Path(config["runtime_root"]) not in path.parents
            or sha256_file(path) != result["report_sha256"]
        ):
            raise ValueError("Dev result report path/hash mismatch")
        report = json.loads(path.read_text())
        if (
            len(report["items"]) != result["completed_clips"]
            or report["step"] != job["target_step"]
        ):
            raise ValueError("Dev result count/step mismatch")
        if job["mode"] == "production" and job.get("max_batches") is None:
            validate_report(report, job, binding, admit_original(config, binding))
        else:
            validate_probe_report(report, job, binding, admit_original(config, binding))


def execute_job(config: dict, binding: dict, attempt: dict, job: dict, approval: Path) -> dict:
    allowed_scope = require_scope(config, job["stage"], approval)
    root = Path(config["runtime_root"]) / "attempts" / attempt["token"] / "jobs" / job["stage_id"]
    with writer(root):
        if (root / "job.json").exists():
            raise RuntimeError("Stage ticket already exists; no retry")
        job = {
            **job,
            "allow_owned_resume_pruning": allowed_scope.get("allow_owned_resume_pruning", False),
            "parent_owner": attempt["owner"],
            "token": attempt["token"],
            "ticket_path": str(root / "job.json"),
            "result_path": str(root / "result.json"),
            "progress_path": str(root / "progress.json"),
        }
        atomic_json(root / "job.json", job)
        require_scope(config, job["stage"], approval, job=job)
        accounting.begin_stage(config, attempt, job)
        progress = dict(
            completed_updates=job["start_step"],
            durable_updates=job["start_step"],
            completed_clips=0,
        )
        child = None
        last_flush = time.monotonic()
        try:
            env = {
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "0",
                "HF_HUB_OFFLINE": "1",
                "OMP_NUM_THREADS": "2",
                "PYTHONDONTWRITEBYTECODE": "1",
                "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
            }
            with (root / "worker.log").open("x") as handle:
                child = subprocess.Popen(
                    worker_command(config, job, approval),
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                atomic_json(root / "owner.json", dict(owner=process_owner(child.pid)))
                while child.poll() is None:
                    if (root / "progress.json").exists():
                        progress = json.loads((root / "progress.json").read_text())
                    if time.monotonic() - last_flush >= 10:
                        accounting.record_progress(config, attempt, job, progress)
                        last_flush = time.monotonic()
                    time.sleep(0.1)
                child.wait()
            if child.returncode != 0:
                message = (root / "worker.log").read_text()[-1600:]
                raise RuntimeError(
                    f"{job['stage']} worker exited{child.returncode}; no retry\n{message}"
                )
            if not (root / "result.json").is_file():
                raise RuntimeError("Exited worker missing durable handoff")
            result = json.loads((root / "result.json").read_text())
            _validate_result(config, binding, job, result)
            accounting.finish_stage(config, attempt, job, result, status="complete")
            return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                try:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                except Exception as observer:
                    error.add_note(f"Owned child stop failed: {observer}")
            if (root / "progress.json").exists():
                try:
                    progress = json.loads((root / "progress.json").read_text())
                except Exception as observer:
                    error.add_note(f"Failure progress read failed: {observer}")
            try:
                accounting.finish_stage(config, attempt, job, progress, status="failed")
            except Exception as observer:
                error.add_note(f"Failure accounting failed: {observer}")
            try:
                fail_hold(config, binding, error, approval)
            except Exception as observer:
                error.add_note(f"Failure hold publication failed: {observer}")
            raise


def select_checkpoint(config: dict, binding: dict, admission: dict) -> dict:
    rows = admitted_reports(config, binding, admission)
    if not rows:
        raise ValueError("No admitted production reports")
    for r in rows:
        archive_candidate(config, Path(r["checkpoint_path"]), binding)
    best = max(rows, key=lambda r: (r["metrics"]["bleu4"], -r["epoch"]))
    root = Path(config["runtime_root"])
    source = Path(best["checkpoint_path"])
    selected = root / "selected" / f"epoch-{best['epoch']}.pt"
    if selected.exists():
        if sha256_file(selected) != best["checkpoint_sha256"]:
            raise ValueError("Native selected weights differ")
    else:
        _atomic(selected, lambda h: h.write(source.read_bytes()))
    for suffix in [".json", ".guard.json", ".runtime.json"]:
        target = selected.with_suffix(suffix)
        value = json.loads(source.with_suffix(suffix).read_text())
        if target.exists() and json.loads(target.read_text()) != value:
            raise ValueError("Selected sidecar conflict")
        if not target.exists():
            atomic_json(target, value)
    pointer = dict(
        epoch=best["epoch"],
        checkpoint=str(selected),
        checkpoint_sha256=best["checkpoint_sha256"],
        metrics=best["metrics"],
        binding_hash=digest(binding),
        original_identity_hash=binding["original_identity_hash"],
        evaluation_policy_hash=binding["evaluation_policy_hash"],
    )
    atomic_json(root / "selected-checkpoint.json", pointer)
    atomic_json(root / "stopping.json", stop_state(rows, binding["recipe"]))
    anchor_epoch = binding["anchor_step"] // binding["recipe"]["steps_per_epoch"]
    original = next((r for r in rows if r["epoch"] == anchor_epoch), None)
    if original is not None:
        report = root / "dev" / f"epoch-{anchor_epoch}.json"
        atomic_json(
            root / "production-anchor-dev-gate.json",
            dict(
                binding_hash=digest(binding), report=str(report), report_sha256=sha256_file(report)
            ),
        )
    return pointer


def _job(binding: dict, pending: dict, mode: str) -> dict:
    return dict(
        schema_version=1,
        seed=0,
        mode=mode,
        binding_hash=digest(binding),
        stage_id=f"{pending['stage']}-{pending['start_step']}-{pending['target_step']}",
        **pending,
    )


def run(config: dict, approval: Path | None, *, check_only: bool = False) -> dict:
    if check_only:
        original = verify_original(config)
        binding = make_binding(config, original)
        return dict(
            status="cpu_preflight",
            binding_hash=digest(binding),
            anchor_step=binding["anchor_step"],
            gpu_launched=False,
        )
    scope = require_scope(config, "supervisor", approval)
    assert approval is not None
    b = read_binding(config)
    verify_original(config)
    admission = admit_original(config, b)
    root = Path(config["runtime_root"])
    # Lock original run as well: old writers cannot overlap while new parent owns authority.
    with writer(Path(config["run_root"])), writer(root / "supervisor"):
        claimed = claim_scope(config, approval)
        attempt = accounting.begin_attempt(config, b, scope, claimed["owner"])
        state: dict = dict(
            status="running",
            binding_hash=digest(b),
            owner=claimed["owner"],
            mode=scope["mode"],
            results=[],
        )
        atomic_json(root / "state.json", state)
        try:
            if scope["mode"] == "handoff_probe":
                cp = Path(scope["start_checkpoint"])
                bundle = checkpoint_bundle(cp, b)
                if bundle["receipt"]["sha256"] != scope["start_checkpoint_sha256"]:
                    raise ValueError("Probe start checkpoint changed")
                start = bundle["receipt"]["step"]
                target = start + scope["max_train_updates"]
                if target > b["recipe"]["total_steps"] or target <= start:
                    raise ValueError("Probe update bound invalid")
                pending = dict(
                    stage="train",
                    start_step=start,
                    target_step=target,
                    checkpoint=str(cp),
                    checkpoint_sha256=bundle["receipt"]["sha256"],
                    max_batches=None,
                )
                result = execute_job(
                    config, b, attempt, _job(b, pending, "handoff_probe"), approval
                )
                state["results"].append(result)
                pending = dict(
                    stage="dev",
                    start_step=target,
                    target_step=target,
                    epoch=None,
                    checkpoint=result["candidate"],
                    checkpoint_sha256=result["candidate_sha256"],
                    max_batches=scope["max_dev_clips"],
                )
                state["results"].append(
                    execute_job(config, b, attempt, _job(b, pending, "handoff_probe"), approval)
                )
                state["status"] = "diagnostic_complete"
            else:
                while True:
                    pending = pending_stage(config, b, admission)
                    if pending["stage"] == "complete":
                        state["status"] = "seed0_complete"
                        break
                    current = json.loads(_path_attempt(config, attempt).read_text())
                    remaining_updates = scope["max_train_updates"] - current["train_used"]
                    remaining_clips = scope["max_dev_clips"] - current["dev_used"]
                    if (
                        pending["stage"] not in scope["stages"]
                        or pending["stage"] == "train"
                        and pending["target_step"] - pending["start_step"] > remaining_updates
                        or pending["stage"] == "dev"
                        and config["expected_dev_clips"] > remaining_clips
                    ):
                        state.update(status="scope_complete", next_pending=pending)
                        break
                    result = execute_job(
                        config, b, attempt, _job(b, pending, "production"), approval
                    )
                    state["results"].append(result)
                    if result["stage"] == "dev":
                        select_checkpoint(config, b, admission)
                    prune_owned_resumes(
                        config, b, allowed=scope.get("allow_owned_resume_pruning", False)
                    )
                    atomic_json(root / "state.json", state)
            accounting.finish_attempt(config, attempt, status="complete")
            if scope["mode"] == "handoff_probe":
                train_result, dev_result = state["results"]
                evidence = dict(
                    status="completed",
                    token=attempt["token"],
                    binding_hash=digest(b),
                    start_step=start,
                    end_step=target,
                    checkpoint=train_result["checkpoint"],
                    checkpoint_sha256=train_result["checkpoint_sha256"],
                    dev_report=dev_result["report"],
                    dev_report_sha256=dev_result["report_sha256"],
                )
                atomic_json(root / "validation" / attempt["token"] / "handoff.json", evidence)
            record = json.loads(approval.read_text())
            record["consumed"] = True
            record["outcome"] = state["status"]
            atomic_json(approval, record)
            atomic_json(root / "state.json", state)
            return state
        except BaseException as error:
            try:
                fail_hold(config, b, error, approval)
            except Exception as observer:
                error.add_note(f"Failure hold publication failed: {observer}")
            state.update(status="held", error=str(error))
            try:
                atomic_json(root / "state.json", state)
            except Exception as observer:
                error.add_note(f"Supervisor evidence failed: {observer}")
            raise


def _path_attempt(config: dict, attempt: dict) -> Path:
    return Path(config["runtime_root"]) / "attempts" / attempt["token"] / "attempt.json"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--approval", type=Path)
    a = p.parse_args()
    print(json.dumps(run(load_config(a.config), a.approval, check_only=a.mode == "preflight")))


if __name__ == "__main__":
    main()

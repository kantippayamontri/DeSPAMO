"""GPU-free parent sequences train/dev workers with fail-stop ownership barriers."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

from tools.benchmark.compact_checkpoint import save_snapshot
from tools.clip_plain.data import admit_data
from tools.clip_plain.evaluate import validate_report
from tools.clip_plain.features import derive_package
from tools.clip_plain.policy import (
    claim_attempt,
    load_config,
    recipe,
    require_gpu_scope,
    source_hashes,
)
from tools.clip_plain.state import (
    archive_candidate,
    begin_stage,
    cost_summary,
    ensure_lineage,
    prune_owned_resumes,
    record_stage,
    require_storage,
)
from tools.clip_plain.train import next_dev_epoch
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import process_owner
from tools.dinov3.storage import _atomic, atomic_json, writer
from tools.followup.recipe import stop_state


def recover_candidate(run_root: Path, identity: dict, epoch: int, *, reviewed: bool) -> dict:
    if not reviewed:
        raise RuntimeError("Explicit CPU handoff recovery review required")
    path = run_root / f"resume-epoch-{epoch}.pt"
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (
        path.is_symlink()
        or receipt["identity"] != identity
        or sha256_file(path) != receipt["sha256"]
    ):
        raise ValueError("Recovery snapshot mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["identity"] != identity or state["step"] != receipt["step"]:
        raise ValueError("Recovery internal identity mismatch")
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
    if not candidate.exists():
        save_snapshot(candidate, slim)
    else:
        saved = json.loads(candidate.with_suffix(".json").read_text())
        if saved["identity"] != identity or sha256_file(candidate) != saved["sha256"]:
            raise ValueError("Recovery candidate conflict")
    return dict(candidate=str(candidate), step=state["step"])


def bind_initialized_budget(run_root: Path) -> None:
    with writer(run_root / "accounting"):
        path = run_root / "budget.json"
        budget = json.loads(path.read_text())
        identity = json.loads((run_root / "identity.json").read_text())
        if budget["identity"].get("policy") == "uninitialized":
            if budget["identity"]["seed"] != identity["seed"]:
                raise ValueError("Initial seed changed")
            budget["identity"] = identity
            atomic_json(path, budget)
        elif budget["identity"] != identity:
            raise ValueError("Ledger/run identity mismatch")


def pending_stage(run_root: Path, identity: dict, admission: dict) -> dict:
    checkpoints = sorted(
        run_root.glob("resume-epoch-*.pt"), key=lambda p: int(p.stem.split("-")[-1])
    )
    if not checkpoints:
        return dict(stage="train", epoch=identity["recipe"]["dev_epochs"][0], completed_updates=0)
    cp = checkpoints[-1]
    r = json.loads(cp.with_suffix(".json").read_text())
    if cp.is_symlink() or r["identity"] != identity or sha256_file(cp) != r["sha256"]:
        raise ValueError("Latest owned resume changed")
    epoch = r["step"] // identity["recipe"]["steps_per_epoch"]
    common = dict(checkpoint=str(cp), checkpoint_sha256=r["sha256"], completed_updates=r["step"])
    if epoch in identity["recipe"]["dev_epochs"]:
        report = run_root / "dev" / f"dev-epoch-{epoch}.json"
        if not report.exists():
            candidate = run_root / f"candidate-epoch-{epoch}.pt"
            if not candidate.exists():
                raise ValueError("Missing candidate; reviewed CPU handoff recovery required")
            receipt = json.loads(candidate.with_suffix(".json").read_text())
            if (
                receipt["identity"] != identity
                or receipt["step"] != r["step"]
                or sha256_file(candidate) != receipt["sha256"]
            ):
                raise ValueError("Candidate mismatch")
            return dict(
                stage="dev",
                epoch=epoch,
                checkpoint=str(candidate),
                checkpoint_sha256=receipt["sha256"],
                completed_updates=r["step"],
            )
        job = dict(
            seed=identity["seed"],
            epoch=epoch,
            checkpoint_sha256=json.loads((run_root / f"candidate-epoch-{epoch}.json").read_text())[
                "sha256"
            ],
            run_identity_hash=digest(identity),
            evaluation_policy_hash=identity["evaluation_policy_hash"],
            data_identity_hash=digest(identity["data"]),
            max_batches=None,
        )
        validate_report(json.loads(report.read_text()), job, admission)
        rows = [
            json.loads(p.read_text())
            for p in sorted(
                (run_root / "dev").glob("dev-epoch-*.json"),
                key=lambda p: int(p.stem.split("-")[-1]),
            )
        ]
        if stop_state(rows, identity["recipe"])["stop"]:
            return dict(stage="complete", epoch=epoch, **common)
    return dict(stage="train", epoch=next_dev_epoch(r["step"], identity["recipe"]), **common)


def worker_command(config: dict, job: dict, approval: Path, job_path: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        f"tools.clip_plain.{('evaluate' if job['stage'] == 'dev' else 'train')}",
        "--config",
        config["config_path"],
        "--job",
        str(job_path),
        "--approval",
        str(approval),
    ]


def hold_study(config: dict, error: BaseException, approval: Path | None) -> None:
    root = Path(config["study_root"])
    atomic_json(
        root / "manual-stop.json",
        dict(
            status="held",
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


def execute_job(config: dict, job: dict, approval: Path) -> dict:
    require_gpu_scope(
        {k: v for k, v in config.items() if k != "config_path"}, job["stage"], approval
    )
    run = Path(job["run_root"])
    stage = job["stage"]
    root = run / "jobs" / f"{job['attempt_token']}-{stage}-{job['epoch']}"
    with writer(root):
        result_path = Path(job["result_path"])
        job_path = root / "job.json"
        if job_path.exists() or result_path.exists():
            raise RuntimeError("Stage attempt already exists")
        atomic_json(job_path, job)
        require_gpu_scope(
            {k: v for k, v in config.items() if k != "config_path"}, stage, approval, job=job
        )
        identity = (
            json.loads((run / "identity.json").read_text())
            if (run / "identity.json").exists()
            else dict(policy="uninitialized", seed=job["seed"])
        )
        attempt = dict(token=f"{job['attempt_token']}-{stage}-{job['epoch']}", stage=stage)
        begin_stage(run, identity, attempt)
        child = None
        progress = dict(
            completed_updates=job.get("completed_updates", 0),
            durable_updates=job.get("completed_updates", 0),
        )
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
                    worker_command(config, job, approval, job_path),
                    cwd=Path(__file__).resolve().parents[2],
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                owner = process_owner(child.pid)
                atomic_json(root / "owner.json", dict(owner=owner))
                while child.poll() is None:
                    p = Path(job["progress_path"])
                    if p.exists():
                        progress = json.loads(p.read_text())
                    record_stage(run, attempt, progress)
                    time.sleep(0.1)
                child.wait()
            if child.returncode != 0:
                raise RuntimeError(f"{stage} worker exited{child.returncode}; no retry")
            if not result_path.is_file():
                raise RuntimeError("Exited worker missing durable handoff")
            result = json.loads(result_path.read_text())
            if result["stage"] != stage or result["status"] not in ["complete", "awaiting_dev"]:
                raise ValueError("Invalid stage result")
            bind_initialized_budget(run)
            actual_identity = json.loads((run / "identity.json").read_text())
            if "recipe" in actual_identity:
                expected = job["epoch"] * actual_identity["recipe"]["steps_per_epoch"]
                if result["completed_updates"] != expected or result["durable_updates"] != expected:
                    raise ValueError("Worker update count differs from declared epoch")
            record_stage(run, attempt, result, final_status="complete")
            return result
        except BaseException as error:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
            try:
                record_stage(run, attempt, progress, final_status="failed")
            except Exception as observer:
                error.add_note(f"Failure accounting failed: {observer}")
            try:
                hold_study(config, error, approval)
            except Exception as observer:
                error.add_note(f"Failure hold publication failed: {observer}")
            raise


def select_checkpoint(config: dict, run_root: Path, identity: dict, admission: dict) -> dict:
    reports = sorted(
        [json.loads(p.read_text()) for p in (run_root / "dev").glob("dev-epoch-*.json")],
        key=lambda r: r["epoch"],
    )
    for report in reports:
        cp = Path(report["checkpoint_path"])
        receipt = json.loads(cp.with_suffix(".json").read_text())
        job = dict(
            seed=identity["seed"],
            epoch=report["epoch"],
            checkpoint_sha256=receipt["sha256"],
            run_identity_hash=digest(identity),
            evaluation_policy_hash=identity["evaluation_policy_hash"],
            data_identity_hash=report["data_identity_hash"],
            max_batches=None,
        )
        validate_report(report, job, admission)
        if (
            cp.parent.resolve() != run_root.resolve()
            or receipt["identity"] != identity
            or sha256_file(cp) != receipt["sha256"]
        ):
            raise ValueError("Foreign/mutated selection candidate")
        archive_candidate(config, cp)
    stopping = stop_state(reports, identity["recipe"])
    best = max(reports, key=lambda r: (r["metrics"]["bleu4"], -r["epoch"]))
    cp = Path(best["checkpoint_path"])
    selected = run_root / f"selected-epoch-{best['epoch']}.pt"
    if selected.exists():
        if sha256_file(selected) != best["checkpoint_sha256"]:
            raise ValueError("Selected native bytes differ")
    else:
        _atomic(selected, lambda h: h.write(cp.read_bytes()))
    atomic_json(selected.with_suffix(".json"), json.loads(cp.with_suffix(".json").read_text()))
    pointer = dict(
        epoch=best["epoch"],
        checkpoint=str(selected),
        checkpoint_sha256=best["checkpoint_sha256"],
        metrics=best["metrics"],
        identity=identity,
        evaluation_policy_hash=identity["evaluation_policy_hash"],
    )
    atomic_json(run_root / "selected-checkpoint.json", pointer)
    atomic_json(run_root / "stopping.json", stopping)
    return pointer


def run_study(config: dict, approval: Path | None, *, check_only: bool = False) -> dict:
    if check_only:
        return dict(
            status="cpu_preflight",
            feature=derive_package(config, check_only=True),
            gpu_launched=False,
        )
    scope = require_gpu_scope(config, "study", approval)
    assert approval is not None
    lineage = ensure_lineage(config)
    admission = admit_data(config)
    attempt = claim_attempt(config, approval)
    root = Path(config["study_root"])
    with writer(root / "controller"):
        state: dict = dict(
            status="running",
            owner=process_owner(os.getpid()),
            source_hashes=source_hashes(),
            attempt=attempt,
            stages=[],
        )
        atomic_json(root / "controller/state.json", state)
        try:
            selections = []
            for seed in scope["seeds"]:
                run = root / "runs" / f"seed{seed}"
                require_storage(
                    config, run, prune_allowed=scope.get("allow_owned_resume_pruning", False)
                )
                while True:
                    identity = (
                        json.loads((run / "identity.json").read_text())
                        if (run / "identity.json").exists()
                        else dict(recipe=recipe(config))
                    )
                    if (
                        identity.get("code_hashes") is not None
                        and identity["code_hashes"] != source_hashes()
                    ):
                        raise ValueError("Run source drift")
                    pending = pending_stage(run, identity, admission)
                    if pending["stage"] == "complete":
                        break
                    stage = pending["stage"]
                    if stage not in scope["stages"]:
                        raise RuntimeError("Pending stage not approved")
                    token = attempt["token"]
                    directory = run / "jobs" / f"{token}-{stage}-{pending['epoch']}"
                    job: dict = dict(
                        schema_version=1,
                        seed=seed,
                        run_root=str(run),
                        attempt_token=token,
                        parent_owner=attempt["owner"],
                        result_path=str(directory / "result.json"),
                        progress_path=str(directory / "progress.json"),
                        **pending,
                    )
                    if stage == "dev":
                        job.update(
                            split="dev",
                            run_identity_hash=digest(identity),
                            evaluation_policy_hash=identity["evaluation_policy_hash"],
                            data_identity_hash=digest(admission["identity"]),
                            max_batches=scope.get("max_batches"),
                        )
                    runtime_config = {
                        **config,
                        "config_path": str(Path(__file__).with_name("config.json")),
                    }
                    result = execute_job(runtime_config, job, approval)
                    state["stages"].append(result)
                    atomic_json(root / "controller/state.json", state)
                    if result.get("diagnostic_only"):
                        state["status"] = "diagnostic_complete"
                        atomic_json(root / "controller/state.json", state)
                        record = json.loads(approval.read_text())
                        record["consumed"] = True
                        atomic_json(approval, record)
                        return state
                    identity = json.loads((run / "identity.json").read_text())
                    if stage == "dev":
                        select_checkpoint(config, run, identity, admission)
                    prune_owned_resumes(
                        run, identity, allowed=scope.get("allow_owned_resume_pruning", False)
                    )
                selections.append(select_checkpoint(config, run, identity, admission))
            state.update(
                status="complete"
                if set(scope["seeds"]) == set(config["seeds"])
                else "scope_complete",
                selections=selections,
                lineage=lineage,
                cost=cost_summary(config),
                test_scored=False,
            )
            atomic_json(root / "controller/state.json", state)
            atomic_json(root / "summary.json", state)
            if state["status"] == "complete":
                atomic_json(root / "controller/frozen-selections.json", selections)
            r = json.loads(approval.read_text())
            r["consumed"] = True
            atomic_json(approval, r)
            return state
        except BaseException as error:
            hold_study(config, error, approval)
            state.update(status="held", error=str(error))
            atomic_json(root / "controller/state.json", state)
            raise


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["preflight", "run"], default="preflight")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--approval", type=Path)
    a = p.parse_args()
    print(
        json.dumps(run_study(load_config(a.config), a.approval, check_only=a.mode == "preflight"))
    )


if __name__ == "__main__":
    main()

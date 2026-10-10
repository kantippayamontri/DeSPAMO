"""Sequential six-run controller, once-only test and owned20minute recovery monitor."""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from pathlib import Path

from tools.benchmark.feature_package import other_gpu_pids
from tools.benchmark.run_translation_study import execute_stage
from tools.benchmark.translation_data import admit_data
from tools.dinov3.identity import sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import process_owner, require_official_storage
from tools.dinov3.storage import atomic_json, writer
from tools.followup.evaluation import score_run
from tools.followup.recipe import schedule
from tools.followup.train import ARCHIVE, BASE, PROJECT, code_hashes, model_files

WORKTREE = Path(__file__).resolve().parents[2]
ARMS = ("reference", "aligned_fixed", "aligned_learned")


def runtime_env() -> dict:
    return {
        **os.environ,
        "PYTHONPATH": f"{PROJECT / 'src'}:{WORKTREE}:{PROJECT}",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def validate_selections(rows: list[dict]) -> None:
    if (
        len(rows) != 6
        or {(r["seed"], r["arm"]) for r in rows} != {(s, a) for s in (0, 1) for a in ARMS}
        or any(not r.get("selection_hash") for r in rows)
    ):
        raise ValueError("exact six frozen selections required")


def recovery_allowed(status: str, error: str, *, manual_stop: bool, any_live: bool) -> bool:
    return (
        not manual_stop
        and not any_live
        and status not in ("complete", "stopped_by_user")
        and bool(
            re.search(
                r"signal15|KeyboardInterrupt|ConnectionResetError|BrokenPipeError|"
                r"Input/output error|CUDA out of memory",
                error,
            )
        )
    )


def failure_evidence(state: dict, root: Path) -> str:
    failed = [name for name, stage in state["stages"].items() if stage["status"] == "failed"]
    text = state.get("error", "")
    if failed:
        path = root / f"{failed[-1]}.log"
        if path.exists():
            latest = path.read_text()[-10000:].rsplit("Traceback (most recent call last):", 1)[-1]
            text += "\n" + latest
    return text


def accepted_profile() -> dict:
    receipt = json.loads((BASE / "profiles/accepted.json").read_text())
    p = Path(receipt["path"])
    if sha256_file(p) != receipt["sha256"]:
        raise ValueError("accepted profile bytes changed")
    profile = json.loads(p.read_text())
    if (
        profile["identity"]["source"] != code_hashes()
        or profile["identity"]["recipe"] != schedule(8)
        or profile["status"] != "complete"
        or not profile["generation_equal"]
        or not profile["fresh_base_reload"]
        or profile["gpu_free_bytes"] < 2 * 1024**3
        or max(r["peak_reserved_bytes"] for r in profile["results"].values()) > 21 * 1024**3
    ):
        raise ValueError("accepted ICL batch8 profile/source/margin changed")
    return profile


def archive_final(run: Path) -> None:
    files: dict[str, dict] = {
        str(p.relative_to(run)): dict(sha256=sha256_file(p), bytes=p.stat().st_size)
        for p in run.rglob("*")
        if p.is_file() and p.name != ".writer.lock" and "archive-receipts" not in p.parts
    }
    require_archive_storage(ARCHIVE, sum(v["bytes"] for v in files.values()))
    dest = ARCHIVE / run.name / "final-payload"
    manifest = dest / "archive-manifest.json"
    if not manifest.exists():
        for name in files:
            _copy_file(run / name, dest / name)
        for name, v in files.items():
            if sha256_file(dest / name) != v["sha256"]:
                raise ValueError("final archive verification failed")
        _archive_json(manifest, dict(files=files, source_root=str(run)))
    if json.loads(manifest.read_text())["files"] != files or any(
        sha256_file(dest / n) != v["sha256"] for n, v in files.items()
    ):
        raise ValueError("completed archive changed")
    atomic_json(
        run / "archive-receipts/final.json",
        dict(manifest=str(manifest), sha256=sha256_file(manifest)),
    )


def controller() -> None:
    if (BASE / "manual-stop.json").exists():
        raise ValueError("study intentionally stopped; no automatic restart")
    accepted_profile()
    require_official_storage(BASE, 6 * 1024**3)
    require_archive_storage(ARCHIVE, 60 * 1024**3)
    python = str(PROJECT / ".venv/bin/python")
    with writer(BASE / "controller"):
        path = BASE / "controller/state.json"
        state: dict = json.loads(path.read_text()) if path.exists() else dict(stages={})
        if state.get("source", code_hashes()) != code_hashes():
            raise ValueError("controller source identity changed")
        state.update(
            status="running",
            batch_size=8,
            source=code_hashes(),
            recipe=schedule(8),
            owner=process_owner(os.getpid()),
        )
        atomic_json(path, state)

        def stop(signum, frame):
            raise KeyboardInterrupt(f"signal{signum}")

        signal.signal(signal.SIGTERM, stop)
        try:
            for seed in (0, 1):
                for arm in ARMS:
                    command = [
                        python,
                        "-B",
                        "-m",
                        "tools.followup.train",
                        "--mode",
                        "run",
                        "--seed",
                        str(seed),
                        "--arm",
                        arm,
                        "--batch-size",
                        "8",
                        "--authorize-translation",
                    ]
                    execute_stage(state, path, f"seed{seed}-{arm}", command, runtime_env())
            selections = []
            for p in (BASE / "runs").glob("*/complete.json"):
                result = json.loads(p.read_text())
                identity = result["selected"]["identity"]
                selections.append(
                    dict(
                        seed=identity["seed"],
                        arm=identity["arm"],
                        run_root=str(p.parent),
                        selection_hash=sha256_file(p.parent / "selected-checkpoint.json"),
                    )
                )
            validate_selections(selections)
            frozen = BASE / "controller/frozen-selections.json"
            if frozen.exists() and json.loads(frozen.read_text()) != selections:
                raise ValueError("frozen selection inventory changed")
            atomic_json(frozen, selections)
            for selected in selections:
                command = [
                    python,
                    "-B",
                    "-m",
                    "tools.followup.study",
                    "--mode",
                    "test",
                    "--run",
                    selected["run_root"],
                    "--authorize-study",
                ]
                execute_stage(
                    state,
                    path,
                    f"test-seed{selected['seed']}-{selected['arm']}",
                    command,
                    runtime_env(),
                )
            rows = []
            for selected in selections:
                report = json.loads(
                    (Path(selected["run_root"]) / "official-test/report.json").read_text()
                )
                complete = json.loads((Path(selected["run_root"]) / "complete.json").read_text())
                rows.append(
                    dict(
                        **selected,
                        dev_metrics=report["selection"]["metrics"],
                        test_metrics=report["metrics"],
                        selected_epoch=report["selection"]["epoch"],
                        stopped_epoch=complete["stopped_epoch"],
                        stopped_updates=complete["stopped_updates"],
                    )
                )
            summary = dict(
                status="complete",
                protocol="official-mixed-signer",
                recipe=schedule(8),
                runs=rows,
                profile_sha256=sha256_file(
                    Path(json.loads((BASE / "profiles/accepted.json").read_text())["path"])
                ),
                means={
                    a: {
                        m: sum(r["test_metrics"][m] for r in rows if r["arm"] == a) / 2
                        for m in rows[0]["test_metrics"]
                    }
                    for a in ARMS
                },
            )
            atomic_json(BASE / "summary.json", summary)
            for selected in selections:
                archive_final(Path(selected["run_root"]))
            state.update(status="complete", summary_sha256=sha256_file(BASE / "summary.json"))
            atomic_json(path, state)
        except BaseException as error:
            for stage in state["stages"].values():
                owner = stage.get("owner")
                if stage["status"] == "running" and owner and process_owner(owner["pid"]) == owner:
                    os.kill(owner["pid"], signal.SIGTERM)
                    deadline = time.monotonic() + 60
                    while process_owner(owner["pid"]) == owner and time.monotonic() < deadline:
                        time.sleep(0.2)
            state.update(status="failed", error=str(error))
            if isinstance(error, KeyboardInterrupt):
                state["status"] = "stopped_by_user"
                atomic_json(
                    BASE / "manual-stop.json", dict(status="stopped_by_user", error=str(error))
                )
            atomic_json(path, state)
            raise


def verify_recovery() -> None:
    accepted_profile()
    admission = admit_data()
    sources, bases = code_hashes(), model_files()
    for p in (BASE / "runs").glob("*/budget.json"):
        b = json.loads(p.read_text())
        i = b["identity"]
        if (
            i["code_hashes"] != sources
            or i["base_files"] != bases
            or i["data"] != admission["identity"]
            or i["recipe"] != schedule(8)
        ):
            raise ValueError("recovery source/base/data/recipe mismatch")
        for q in p.parent.glob("resume-epoch-*.pt"):
            receipt = json.loads(q.with_suffix(".json").read_text())
            if (
                receipt["identity"] != i
                or receipt["sha256"] != sha256_file(q)
                or receipt["bytes"] != q.stat().st_size
                or receipt["step"] > b["completed_updates"]
            ):
                raise ValueError("recovery resume verification failed")
    if other_gpu_pids():
        raise ValueError("recovery requires idle GPU")
    require_official_storage(BASE, 6 * 1024**3)
    require_archive_storage(ARCHIVE, 60 * 1024**3)


def monitor() -> None:
    root = BASE / "watchdog"
    with writer(root):
        history = root / "history.jsonl"
        previous = (
            json.loads((root / "status.json").read_text())
            if (root / "status.json").exists()
            else {}
        )
        recoveries = previous.get("recoveries", [])
        while True:
            state = json.loads((BASE / "controller/state.json").read_text())
            owners = [state.get("owner")] + [
                v.get("owner") for v in state["stages"].values() if v["status"] == "running"
            ]
            live, mismatch = False, False
            for owner in owners:
                current = process_owner(owner["pid"]) if owner else None
                live |= current is not None
                mismatch |= current is not None and current != owner
            action, error = "observe_only", None
            manual = (BASE / "manual-stop.json").exists()
            if mismatch:
                action, error = "needs_agent", "PID ownership mismatch"
            elif state["status"] == "complete" or manual:
                action = "stop_monitor"
            elif not live:
                known = recovery_allowed(
                    state["status"],
                    failure_evidence(state, BASE / "controller"),
                    manual_stop=manual,
                    any_live=False,
                )
                if state["status"] == "running":
                    known = True  # Missing owner, after explicit ownership/source checks.
                if known and len(recoveries) < 3:
                    try:
                        verify_recovery()
                        fresh = json.loads((BASE / "controller/state.json").read_text())
                        if (
                            fresh != state
                            or (BASE / "manual-stop.json").exists()
                            or any(process_owner(o["pid"]) is not None for o in owners if o)
                        ):
                            raise ValueError("recovery ownership/manual-stop changed")
                        with writer(BASE / "controller"):
                            pass
                        with (root / "recovery-controller.log").open("a") as handle:
                            child = subprocess.Popen(
                                [
                                    str(PROJECT / ".venv/bin/python"),
                                    "-B",
                                    "-m",
                                    "tools.followup.study",
                                    "--mode",
                                    "controller",
                                    "--authorize-study",
                                ],
                                cwd=WORKTREE,
                                env=runtime_env(),
                                stdout=handle,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                        recoveries.append(dict(wall=time.time(), pid=child.pid))
                        action = "restart_verified_controller"
                    except Exception as failure:
                        action, error = "needs_agent", str(failure)
                else:
                    action, error = "needs_agent", "unknown failure or recovery limit"
            status = dict(
                checked_wall=time.time(),
                monitor_owner=process_owner(os.getpid()),
                controller_status=state["status"],
                active_stages=[n for n, v in state["stages"].items() if v["status"] == "running"],
                action=action,
                error=error,
                recoveries=recoveries,
                interval_seconds=1200,
            )
            atomic_json(root / "status.json", status)
            with history.open("a") as handle:
                handle.write(json.dumps(status) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            if action == "stop_monitor":
                return
            time.sleep(1200)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("controller", "watch", "test"), required=True)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--authorize-study", action="store_true")
    args = parser.parse_args(argv)
    if not args.authorize_study:
        raise ValueError("explicit follow-up study authorization required")
    if args.mode == "controller":
        controller()
    elif args.mode == "watch":
        monitor()
    else:
        if args.run is None:
            raise ValueError("test run root required")
        score_run(args.run)


if __name__ == "__main__":
    main()

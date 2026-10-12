"""Detached extraction -> verification/profile -> two reference runs; dev-only finish."""

import argparse
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from tools.benchmark.feature_package import other_gpu_pids
from tools.benchmark.run_translation_study import execute_stage
from tools.clip_s2.data import admit_data
from tools.clip_s2.report import compare_dev
from tools.clip_s2.train import (
    ARCHIVE,
    BASE,
    PREDECESSOR,
    PROJECT,
    code_hashes,
    load_config,
    model_files,
)
from tools.dinov3.identity import sha256_file
from tools.dinov3.official_archive import _archive_json, _copy_file, require_archive_storage
from tools.dinov3.official_runtime import process_owner, require_official_storage
from tools.dinov3.storage import atomic_json, writer
from tools.followup.study import failure_evidence
from tools.followup.study import recovery_allowed as inherited_recovery_allowed

WORKTREE = Path(__file__).resolve().parents[2]
FEATURE_BASE = Path("/home/kan/datasets/despamo/official-phoenix14t-clip-s2-features-v1")
EXTRACTION_CONFIG = FEATURE_BASE / "extraction-inputs.json"


def recovery_allowed(status: str, error: str, *, manual_stop: bool, any_live: bool) -> bool:
    """Inherited transient list plus explicit in-process corruption faults."""
    if inherited_recovery_allowed(status, error, manual_stop=manual_stop, any_live=any_live):
        return True
    return (
        not manual_stop
        and not any_live
        and status not in ("complete", "stopped_by_user")
        and "RecoverableFault" in error
    )


def runtime_env(extraction: bool = False) -> dict:
    return {
        **os.environ,
        "PYTHONPATH": str(WORKTREE) if extraction else f"{PROJECT / 'src'}:{WORKTREE}:{PROJECT}",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
    }


def owner_alive(owner: dict | None) -> bool:
    """Same process only: PID, boot, start ticks and command must all match."""
    return bool(owner) and process_owner(owner["pid"]) == owner


def owner_conflict(owner: dict | None) -> bool:
    """A different live process holding the PID in the same boot is ambiguous; a PID
    from an earlier boot cannot be the recorded owner and is simply dead."""
    if not owner:
        return False
    current = process_owner(owner["pid"])
    return current is not None and current != owner and current["boot_id"] == owner["boot_id"]


def predecessor_allowed(status: str, explicit_stop: bool, any_live: bool) -> bool:
    return not any_live and (status == "complete" or status == "stopped_by_user" and explicit_stop)


FIRST_ATTEMPT = Path(
    "/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v1"
)


EARLIER_ATTEMPTS = (
    FIRST_ATTEMPT,
    Path("/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v2"),
)


def supersedes(target: str) -> bool:
    """User-approved switch named the first CLIP attempt; later versions continue it."""
    return target in {str(p) for p in EARLIER_ATTEMPTS} | {str(BASE)}


def preflight() -> None:
    for root in (PREDECESSOR, *EARLIER_ATTEMPTS):
        state = json.loads((root / "controller/state.json").read_text())
        manual = root / "manual-stop.json"
        record = json.loads(manual.read_text()) if manual.exists() else {}
        approved = manual.exists() and (
            root in EARLIER_ATTEMPTS or supersedes(record.get("superseding_study", ""))
        )
        owners = [state.get("owner")] + [
            v.get("owner") for v in state["stages"].values() if v["status"] == "running"
        ]
        watch = root / "watchdog/status.json"
        if watch.exists():
            owners.append(json.loads(watch.read_text()).get("monitor_owner"))
        live = any(owner_alive(o) for o in owners)
        if not predecessor_allowed(state["status"], approved, live):
            raise ValueError(f"{root.name} must be complete or explicitly stopped, owners exited")
    if other_gpu_pids():
        raise RuntimeError("GPU already owned")
    require_official_storage(BASE, 16 * 1024**3)
    require_archive_storage(ARCHIVE, 40 * 1024**3)


def validate_selections(rows: list[dict]) -> None:
    if (
        len(rows) != 2
        or {(r["seed"], r["arm"]) for r in rows} != {(0, "reference"), (1, "reference")}
        or any(not r.get("selection_hash") for r in rows)
    ):
        raise ValueError("exact reference seeds0/1 selections required")


def freeze_selections(path: Path, rows: list[dict]) -> None:
    validate_selections(rows)
    ordered = sorted(rows, key=lambda r: r["seed"])
    if path.exists() and json.loads(path.read_text()) != ordered:
        raise ValueError("frozen dev selections changed")
    atomic_json(path, ordered)


def archive_dev(run: Path, archive: Path) -> None:
    if not (run / "complete.json").is_file():
        raise ValueError("completed training required")
    if (run / "official-test").exists():
        raise ValueError("dev-only study must not create test artifacts")
    files: dict[str, dict] = {
        str(p.relative_to(run)): dict(sha256=sha256_file(p), bytes=p.stat().st_size)
        for p in run.rglob("*")
        if p.is_file() and p.name != ".writer.lock" and "archive-receipts" not in p.parts
    }
    require_archive_storage(archive, sum(v["bytes"] for v in files.values()))
    dest = archive / run.name / "dev-final-payload"
    manifest = dest / "archive-manifest.json"
    if not manifest.exists():
        for name in files:
            _copy_file(run / name, dest / name)
        for name, r in files.items():
            if sha256_file(dest / name) != r["sha256"]:
                raise ValueError("archive copy hash mismatch")
        _archive_json(manifest, dict(files=files, source_root=str(run)))
    if json.loads(manifest.read_text())["files"] != files or any(
        sha256_file(dest / n) != r["sha256"] for n, r in files.items()
    ):
        raise ValueError("dev final archive changed")
    atomic_json(
        run / "archive-receipts/dev-final.json",
        dict(manifest=str(manifest), sha256=sha256_file(manifest)),
    )


def bind_study_inputs(package: dict, name: str = "study-inputs.json") -> Path:
    root = Path(package["root"])
    values = dict(
        spatial_root=str(root),
        spatial_manifest_sha256=sha256_file(root / "complete/manifest.json"),
        feature_identity_sha256=sha256_file(root / "complete/identity.json"),
        predecessor_root=str(PREDECESSOR),
        study_root=str(BASE),
        archive_root=str(ARCHIVE),
        seeds=[0, 1],
        arm="reference",
        test_scoring=False,
    )
    path = BASE / name
    if path.exists() and json.loads(path.read_text()) != values:
        raise ValueError("immutable study inputs changed")
    atomic_json(path, values)
    return path


def controller() -> None:
    if (BASE / "manual-stop.json").exists():
        raise ValueError("new study intentionally stopped")
    preflight()
    python = str(PROJECT / ".venv/bin/python")
    extractor = str(PROJECT / "tools/motion/.venv/bin/python")
    with writer(BASE / "controller"):
        path = BASE / "controller/state.json"
        state: dict = json.loads(path.read_text()) if path.exists() else dict(stages={})
        if state.get("source", code_hashes()) != code_hashes():
            raise ValueError("pipeline source drift")
        config_sha = sha256_file(EXTRACTION_CONFIG)
        if state.get("extraction_config_sha256", config_sha) != config_sha:
            raise ValueError("extraction inputs drift")
        state.update(
            status="running",
            source=code_hashes(),
            extraction_config_sha256=config_sha,
            owner=process_owner(os.getpid()),
            test_scoring=False,
        )
        atomic_json(path, state)

        def stop(signum, frame):
            raise KeyboardInterrupt(f"signal{signum}")

        signal.signal(signal.SIGTERM, stop)
        try:
            for mode in ("run", "verify"):
                command = [
                    extractor,
                    "-B",
                    "-m",
                    "tools.clip_s2.extract",
                    "--mode",
                    mode,
                    "--config",
                    str(EXTRACTION_CONFIG),
                    "--authorize-extraction",
                ]
                execute_stage(state, path, f"extract-{mode}", command, runtime_env(True))
            package = json.loads((FEATURE_BASE / "package-result.json").read_text())
            config = bind_study_inputs(package)
            profile_command = [
                python,
                "-B",
                "-m",
                "tools.clip_s2.train",
                "--mode",
                "profile",
                "--config",
                str(config),
                "--authorize-translation",
            ]
            execute_stage(state, path, "training-profile", profile_command, runtime_env())
            for seed in (0, 1):
                command = [
                    python,
                    "-B",
                    "-m",
                    "tools.clip_s2.train",
                    "--mode",
                    "run",
                    "--seed",
                    str(seed),
                    "--config",
                    str(config),
                    "--authorize-translation",
                ]
                execute_stage(state, path, f"seed{seed}-reference", command, runtime_env())
            rows = []
            for p in (BASE / "runs").glob("*/complete.json"):
                d = json.loads(p.read_text())
                i = d["selected"]["identity"]
                rows.append(
                    dict(
                        seed=i["seed"],
                        arm=i["arm"],
                        run_root=str(p.parent),
                        selection_hash=sha256_file(p.parent / "selected-checkpoint.json"),
                    )
                )
            freeze_selections(BASE / "controller/frozen-selections.json", rows)
            report = compare_dev(BASE, PREDECESSOR)
            atomic_json(BASE / "comparison-dev.json", report)
            for row in rows:
                archive_dev(Path(row["run_root"]), ARCHIVE)
            atomic_json(
                BASE / "summary.json",
                dict(
                    status="complete",
                    test_scored=False,
                    runs=sorted(rows, key=lambda r: r["seed"]),
                    comparison=report["summary"],
                ),
            )
            state.update(status="complete", summary_sha256=sha256_file(BASE / "summary.json"))
            atomic_json(path, state)
        except BaseException as error:
            for stage in state["stages"].values():
                owner = stage.get("owner")
                if stage["status"] == "running" and owner and process_owner(owner["pid"]) == owner:
                    os.kill(owner["pid"], signal.SIGTERM)
                    deadline = time.monotonic() + 90
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


def verify_recovery(state: dict) -> None:
    if state["source"] != code_hashes() or state["extraction_config_sha256"] != sha256_file(
        EXTRACTION_CONFIG
    ):
        raise ValueError("recovery source/extraction input drift")
    if (BASE / "manual-stop.json").exists():
        raise ValueError("study intentionally stopped")
    preflight()
    bases = model_files()
    for p in (BASE / "runs").glob("*/budget.json"):
        b = json.loads(p.read_text())
        i = b["identity"]
        c = load_config(BASE / "study-inputs.json")
        admission = admit_data(Path(c["spatial_root"]), c["spatial_manifest_sha256"])
        if (
            i["code_hashes"] != code_hashes()
            or i["base_files"] != bases
            or i["data"] != admission["identity"]
        ):
            raise ValueError("recovery base/data/source identity mismatch")
        for q in p.parent.glob("resume-epoch-*.pt"):
            r = json.loads(q.with_suffix(".json").read_text())
            if (
                r["identity"] != i
                or r["sha256"] != sha256_file(q)
                or r["bytes"] != q.stat().st_size
                or r["step"] > b["completed_updates"]
            ):
                raise ValueError("native resume recovery verification failed")


def monitor() -> None:
    root = BASE / "watchdog"
    with writer(root):
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
            live, reused = False, False
            for owner in owners:
                live |= owner_alive(owner)
                reused |= owner_conflict(owner)
            action, error = "observe_only", None
            manual = (BASE / "manual-stop.json").exists()
            if reused:
                action, error = "needs_agent", "PID ownership mismatch"
            elif state["status"] == "complete" or manual:
                action = "stop_monitor"
            elif not live:
                eligible = state["status"] == "running" or recovery_allowed(
                    state["status"],
                    failure_evidence(state, BASE / "controller"),
                    manual_stop=False,
                    any_live=False,
                )
                if eligible and len(recoveries) < 3:
                    try:
                        verify_recovery(state)
                        if (
                            any(owner_alive(o) or owner_conflict(o) for o in owners)
                            or (BASE / "manual-stop.json").exists()
                        ):
                            raise ValueError("ownership changed during recovery")
                        with writer(BASE / "controller"):
                            pass
                        with (root / "recovery-controller.log").open("a") as h:
                            child = subprocess.Popen(
                                [
                                    str(PROJECT / ".venv/bin/python"),
                                    "-B",
                                    "-m",
                                    "tools.clip_s2.study",
                                    "--mode",
                                    "controller",
                                    "--authorize-study",
                                ],
                                cwd=WORKTREE,
                                env=runtime_env(),
                                stdout=h,
                                stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL,
                                start_new_session=True,
                            )
                        recoveries.append(dict(wall=time.time(), pid=child.pid))
                        action = "restart_verified_controller"
                    except Exception as e:
                        action, error = "needs_agent", str(e)
                else:
                    action, error = "needs_agent", "unknown failure or recovery limit"
            status = dict(
                checked_wall=time.time(),
                action=action,
                error=error,
                recoveries=recoveries,
                controller_status=state["status"],
                active_stages=[n for n, v in state["stages"].items() if v["status"] == "running"],
                monitor_owner=process_owner(os.getpid()),
                interval_seconds=1200,
            )
            atomic_json(root / "status.json", status)
            with (root / "history.jsonl").open("a") as h:
                h.write(json.dumps(status) + "\n")
                h.flush()
                os.fsync(h.fileno())
            if action == "stop_monitor":
                return
            time.sleep(1200)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("preflight", "controller", "watch"), required=True)
    p.add_argument("--authorize-study", action="store_true")
    a = p.parse_args()
    if a.mode != "preflight" and not a.authorize_study:
        raise ValueError("explicit two-run pipeline authorization required")
    if a.mode == "preflight":
        preflight()
        print("pipeline preflight passed")
    elif a.mode == "controller":
        controller()
    else:
        monitor()


if __name__ == "__main__":
    main()

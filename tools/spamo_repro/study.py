"""CPU parent: sequential train-to-boundary child, actual exit, fresh dev child."""

import json
import math
import time
from collections.abc import Callable
from pathlib import Path

from tools.dinov3.storage import atomic_json

DEV_EPOCHS = tuple(range(2, 41, 2))


def _read(path: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Missing receipt: {path}")
    return json.loads(path.read_text())


def _hold(root: Path, error: BaseException, stage: str, epoch: int) -> None:
    atomic_json(
        root / "manual-stop.json",
        {
            "status": "held",
            "automatic_recovery_disabled": True,
            "stage": stage,
            "epoch": epoch,
            "error_type": type(error).__name__,
            "error": str(error),
            "wall_timestamp": time.time(),
        },
    )


def should_stop(history: list[float], patience: int | None) -> bool:
    """Lightning EarlyStopping(mode=max, min_delta=0): stop after `patience`
    consecutive checks without strict improvement over the running best."""
    if patience is None or len(history) <= patience:
        return False
    best, wait = float("-inf"), 0
    for value in history:
        if value > best:
            best, wait = value, 0
        else:
            wait += 1
    return wait >= patience


def _check_train(root: Path, epoch: int, per_epoch: int = 444) -> None:
    receipt = _read(root / f"train-epoch-{epoch}.json")
    if (
        receipt.get("status") != "complete"
        or receipt.get("epoch") != epoch
        or receipt.get("completed_updates") != epoch * per_epoch
    ):
        raise ValueError("Incomplete training receipt")


def _check_dev(root: Path, epoch: int) -> float:
    report = _read(root / "dev" / f"dev-epoch-{epoch}.json")
    bleu = report.get("metrics", {}).get("bleu4")
    if (
        report.get("epoch") != epoch
        or report.get("complete") is not True
        or report.get("clips") != 519
        or not isinstance(bleu, (int, float))
        or not math.isfinite(bleu)
    ):
        raise ValueError("Incomplete dev report")
    return float(bleu)


def run_study(
    root: Path,
    *,
    spawn: Callable[[str, int], int],
    preflight: Callable[[str, int], None],
    dev_epochs: tuple[int, ...] = DEV_EPOCHS,
    patience: int | None = None,
) -> dict:
    """Never retry; any child, receipt or preflight failure writes a durable hold."""
    if (root / "manual-stop.json").exists():
        raise RuntimeError("Study hold exists; explicit recovery required")
    path = root / "state.json"
    state = (
        _read(path)
        if path.exists()
        else {
            "status": "running",
            "completed_epochs": 0,
            "best": None,
            "history": [],
            "test_scored": False,
        }
    )
    if state.get("test_scored") is not False:
        raise ValueError("Training supervisor never scores test")
    for epoch in dev_epochs:
        if epoch <= state["completed_epochs"]:
            continue
        for stage in ("train", "dev"):
            try:
                preflight(stage, epoch)
                code = spawn(stage, epoch)
                if code != 0:
                    raise RuntimeError(f"{stage} child exited {code}")
                if stage == "train":
                    _check_train(root, epoch)
                else:
                    bleu = _check_dev(root, epoch)
            except BaseException as error:
                _hold(root, error, stage, epoch)
                raise RuntimeError(f"Study held at {stage} epoch {epoch}: {error}") from error
        if state["best"] is None or bleu > state["best"]["bleu4"]:
            state["best"] = {"epoch": epoch, "bleu4": bleu}
        state["completed_epochs"] = epoch
        state["history"].append({"epoch": epoch, "bleu4": bleu})
        atomic_json(path, state)
        if should_stop([h["bleu4"] for h in state["history"]], patience):
            state["stop_reason"] = f"early_stop_patience_{patience}"
            break
    state["status"] = "complete"
    atomic_json(path, state)
    return state


def child_command(config: Path, root: Path, stage: str, epoch: int, probe: int | None) -> list[str]:
    import sys

    command = [
        sys.executable,
        "-B",
        "-m",
        "tools.spamo_repro.worker",
        "--config",
        str(config),
        "--root",
        str(root),
        "--stage",
        stage,
        "--epoch",
        str(epoch),
    ]
    return command + ([] if probe is None else ["--probe", str(probe)])


def spawn_child(config: Path, root: Path, stage: str, epoch: int, probe: int | None = None) -> int:
    import os
    import subprocess

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "0",
        "LD_LIBRARY_PATH": "/usr/lib/wsl/lib",
        "HF_HUB_OFFLINE": "1",
        "OMP_NUM_THREADS": "2",
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    log = (root / f"{stage}-epoch-{epoch}{'-probe' if probe else ''}.log").open("ab")
    child = subprocess.Popen(
        child_command(config, root, stage, epoch, probe),
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    atomic_json(
        root / "child.json",
        {"pid": child.pid, "stage": stage, "epoch": epoch, "started": time.time()},
    )
    code = child.wait()
    # Actual exit observed before any next worker can start.
    atomic_json(
        root / "child.json",
        {
            "pid": child.pid,
            "stage": stage,
            "epoch": epoch,
            "exit_code": code,
            "exited": time.time(),
        },
    )
    return code


def preflight_check(root: Path, checkpoint_root: Path) -> None:
    import shutil
    import subprocess

    reserve = 50 * 1024**3
    if shutil.disk_usage(root).free < reserve + 20 * 1024**3:
        raise RuntimeError("Native storage below reserve plus run demand")
    if shutil.disk_usage(checkpoint_root).free < reserve + 10 * 1024**3:
        raise RuntimeError("B storage below reserve plus checkpoint demand")
    apps = subprocess.check_output(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True, timeout=30
    )
    if apps.strip():
        raise RuntimeError(f"Another GPU compute process is active: {apps.strip()}")


def main() -> None:
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    ckpt = Path(config["checkpoint_root"])
    ckpt.mkdir(parents=True, exist_ok=True)
    claim = args.root / "parent.json"
    if claim.exists():
        old = json.loads(claim.read_text())
        if Path(f"/proc/{old['pid']}").exists():
            raise RuntimeError("Parent already live")
    atomic_json(
        claim,
        {
            "pid": os.getpid(),
            "started": time.time(),
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        },
    )
    from tools.spamo_repro.policy import recipe_for, validate_config

    validate_config(config)
    recipe = recipe_for(config)
    every = recipe["dev_every"]
    run_study(
        args.root,
        dev_epochs=tuple(range(every, recipe["max_epochs"] + 1, every)),
        patience=recipe["patience_checks"],
        spawn=lambda stage, epoch: spawn_child(args.config, args.root, stage, epoch),
        preflight=lambda stage, epoch: preflight_check(args.root, ckpt),
    )


if __name__ == "__main__":
    main()

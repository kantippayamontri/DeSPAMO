"""One user-authorized same-root recovery using the unchanged original controller."""

import json
import os
from datetime import UTC, datetime
from pathlib import Path

from tools.clip_s2.study import controller, owner_alive
from tools.clip_s2.train import BASE, code_hashes
from tools.dinov3.official_runtime import process_owner

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, writer

BOOT = "21ceb469-0d6b-494a-b4fa-dd12d94f2931"
RECOVERY = BASE / "recovery/driver61742-original-recipe-2026-10-11"


def main() -> None:
    if Path("/proc/sys/kernel/random/boot_id").read_text().strip() != BOOT:
        raise RuntimeError("Recovery authorization belongs to a different boot")
    scope = json.loads((RECOVERY / "authorization.json").read_text())
    if (
        scope.get("training_resume_approved") is not True
        or scope.get("automatic_retries_allowed") is not False
        or scope.get("launcher_sha256") != sha256_file(Path(__file__))
    ):
        raise RuntimeError("Exact one-attempt recovery scope required")
    hold = BASE / "manual-stop.json"
    if sha256_file(hold) != scope["manual_stop_before_sha256"]:
        raise ValueError("Stop receipt changed after authorization")
    state = json.loads((BASE / "controller/state.json").read_text())
    owners = [state.get("owner")] + [v.get("owner") for v in state["stages"].values()]
    watch = json.loads((BASE / "watchdog/status.json").read_text())
    owners.append(watch.get("monitor_owner"))
    if any(owner_alive(owner) for owner in owners):
        raise RuntimeError("Existing controller/worker/watchdog still owned and live")
    if state["source"] != code_hashes():
        raise ValueError("Original source closure changed")
    original_hold = json.loads(hold.read_text())
    with writer(RECOVERY):
        started = RECOVERY / "attempt-started.json"
        if started.exists():
            raise RuntimeError("Single recovery attempt already claimed; no automatic retry")
        atomic_json(
            started,
            dict(utc=datetime.now(UTC).isoformat(), owner=process_owner(os.getpid())),
        )
        outcome = "failed"
        try:
            # Preserve the exact historical stop bytes; controller requires its absence.
            history = RECOVERY / "manual-stop-before-resume.json"
            if history.exists():
                raise RuntimeError("Historical stop receipt already exists")
            os.rename(hold, history)
            if sha256_file(history) != scope["manual_stop_before_sha256"]:
                raise ValueError("Historical stop-byte verification failed")
            print(
                json.dumps(
                    dict(
                        phase="original_controller_recovery",
                        checkpoint_updates=1774,
                        train_batch=8,
                        dev_batch=4,
                        dev_use_cache=False,
                        driver="617.42",
                        automatic_retry=False,
                    )
                ),
                flush=True,
            )
            controller()
            outcome = "complete"
        finally:
            finished = datetime.now(UTC).isoformat()
            restored_hold = {
                **original_hold,
                "reason": f"User-authorized driver617.42 recovery ended: {outcome}; no retry",
                "automatic_recovery_disabled": True,
                "gpu_validation_approved": False,
                "recovery_attempt": dict(status=outcome, finished_utc=finished, root=str(RECOVERY)),
            }
            atomic_json(hold, restored_hold)
            atomic_json(
                RECOVERY / "outcome.json",
                dict(status=outcome, finished_utc=finished, automatic_retry=False),
            )


if __name__ == "__main__":
    main()

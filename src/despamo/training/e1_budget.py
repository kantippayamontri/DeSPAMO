"""Durable E1-only GPU-time accounting, independent of model/checkpoint code."""

import math
import os
import time
from collections.abc import Callable
from pathlib import Path

from despamo.appearance.provenance import atomic_json, file_hash, read_json
from despamo.training.pilot_factory import PilotRunState

CEILING_SECONDS = 6 * 3600
PRIOR_RECEIPTS = frozenset(
    {
        "e3-profile-b2.json",
        "e3-profile-b4-extract16.json",
        "profile-E1_frozen-b2.json",
        "profile-E1_frozen-b4.json",
        "profile-timeout-e1-dev.json",
    }
)


def initial_pilot_seconds(paths: dict[str, Path], protocol_hash: str) -> tuple[float, dict]:
    if set(paths) != PRIOR_RECEIPTS or len({path.resolve() for path in paths.values()}) != 5:
        raise ValueError("pilot profile receipt inventory incomplete/duplicated")
    total = 0.0
    hashes = {}
    for name, path in sorted(paths.items()):
        record = read_json(path)
        seconds = record.get("gpu_seconds")
        if record.get("protocol_hash") != protocol_hash:
            raise ValueError("pilot profile protocol mismatch")
        if record.get("status") != ("timed_out" if "timeout" in name else "complete"):
            raise ValueError("pilot profile receipt status mismatch")
        if not isinstance(seconds, (float, int)) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("pilot profile GPU seconds invalid")
        total += seconds
        hashes[name] = file_hash(path)
    if total >= CEILING_SECONDS:
        raise ValueError("previous pilot GPU spend exceeds E1-only ceiling")
    return total, hashes


class E1Ledger:
    """Elapsed wall time while GPU job runs; persist monotonic total to external JSON."""

    def __init__(
        self,
        root: Path,
        identity: dict,
        charged: float,
        step: int,
        monotonic: Callable[[], float],
        wall: Callable[[], float],
    ) -> None:
        self.root = root
        self.identity = identity
        self.charged = charged
        self.step = step
        self.monotonic = monotonic
        self.wall = wall
        self.started = monotonic()
        self.last = charged

    @classmethod
    def create(
        cls,
        root: Path,
        identity: dict,
        initial_seconds: float,
        *,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> "E1Ledger":
        if (
            root.exists()
            or root.is_symlink()
            or not root.parent.is_dir()
            or not identity.get("protocol_hash")
            or not 0 <= initial_seconds < CEILING_SECONDS
        ):
            raise ValueError("E1 ledger root exists or invalid identity/budget")
        root.mkdir()
        result = cls(root, identity, initial_seconds, 0, monotonic, wall)
        result._write("created")
        return result

    @classmethod
    def resume(
        cls,
        root: Path,
        identity: dict,
        checkpoint_seconds: float,
        *,
        checkpoint_step: int,
        monotonic: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> "E1Ledger":
        payload = read_json(root / "budget.json")
        if payload.get("identity") != identity:
            raise ValueError("E1 resume ledger identity mismatch")
        recorded = payload.get("gpu_seconds")
        if (
            not isinstance(recorded, (int, float))
            or not isinstance(checkpoint_seconds, (int, float))
            or not math.isfinite(recorded)
            or not math.isfinite(checkpoint_seconds)
            or not 0 <= checkpoint_seconds <= recorded < CEILING_SECONDS
            or type(payload.get("optimizer_step")) is not int
            or type(checkpoint_step) is not int
            or not 0 <= checkpoint_step <= payload["optimizer_step"]
        ):
            raise ValueError("E1 resume checkpoint/budget mismatch")
        if payload.get("status") not in ("closed", "paused"):
            # An unclean stop may have spent GPU time since the last heartbeat.
            elapsed = wall() - payload["wall_timestamp"]
            if elapsed < 0:
                raise ValueError("E1 resume wall clock moved backwards")
            recorded += elapsed
            if recorded >= CEILING_SECONDS:
                raise ValueError("E1 resume unrecorded GPU time exceeded budget")
        result = cls(root, identity, recorded, checkpoint_step, monotonic, wall)
        result._write("resumed")
        return result

    def _write(self, status: str, phase: str | None = None) -> None:
        atomic_json(
            self.root / "budget.json",
            {
                "identity": self.identity,
                "gpu_seconds": self.last,
                "optimizer_step": self.step,
                "pid": os.getpid(),
                "phase": phase,
                "status": status,
                "wall_timestamp": self.wall(),
                "ceiling_seconds": CEILING_SECONDS,
            },
        )

    def charge(self, *, step: int, phase: str) -> float:
        if type(step) is not int or step < self.step or not phase:
            raise ValueError("E1 ledger optimizer step/phase moved backwards")
        elapsed = self.monotonic() - self.started
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("E1 ledger monotonic clock moved backwards")
        self.last = max(self.last, self.charged + elapsed)
        self.step = step
        self._write("running", phase)
        if self.last >= CEILING_SECONDS:
            raise RuntimeError("E1 six-hour GPU budget exhausted")
        return self.last

    def require_remaining(self, reserve_seconds: float, *, step: int, phase: str) -> float:
        if not math.isfinite(reserve_seconds) or reserve_seconds < 0:
            raise ValueError("E1 budget reserve must be finite and nonnegative")
        spent = self.charge(step=step, phase=phase)
        if spent + reserve_seconds >= CEILING_SECONDS:
            raise RuntimeError("E1 budget lacks time reserve for next operation")
        return spent

    def close(self) -> None:
        self.last = max(self.last, self.charged + max(0.0, self.monotonic() - self.started))
        self._write("closed")


class E1BudgetGuard(PilotRunState):
    """Charge GPU wall time at each optimizer step and before every checkpoint."""

    def __init__(self, ledger: E1Ledger):
        super().__init__(ledger.identity, ledger.last)
        self.ledger = ledger

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ledger.charge(step=trainer.global_step, phase="train")

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        super().on_save_checkpoint(trainer, pl_module, checkpoint)
        checkpoint["gpu_seconds_cumulative"] = self.ledger.charge(
            step=trainer.global_step, phase="checkpoint"
        )

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if checkpoint.get("gpu_seconds_cumulative", float("inf")) > self.ledger.last:
            raise ValueError("E1 resume checkpoint exceeds durable ledger budget")
        super().on_load_checkpoint(trainer, pl_module, checkpoint)

    def on_exception(self, trainer, pl_module, exception):
        try:
            self.ledger.charge(step=trainer.global_step, phase="exception")
        except RuntimeError:
            # A cap breach is recorded before its original exception escapes.
            pass

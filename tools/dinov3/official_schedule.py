"""One-based pre-update clocks for official 40-epoch DINO adaptation."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields


@dataclass(frozen=True)
class AdaptSchedule:
    train_clips: int = 7096
    batch_size: int = 24
    epochs: int = 40
    calibration_epochs: int = 5
    lora_warmup_epochs: int = 2
    grl_ramp_epochs: int = 2
    checkpoint_epochs: int = 10

    def __post_init__(self):
        if any(type(v) is not int or v <= 0 for v in asdict(self).values()):
            raise ValueError("schedule fields must be positive integers")
        joint = self.epochs - self.calibration_epochs
        if (self.train_clips < self.batch_size or joint <= self.lora_warmup_epochs
                or self.grl_ramp_epochs > joint or self.checkpoint_epochs > self.epochs
                or self.epochs % self.checkpoint_epochs):
            raise ValueError("invalid schedule horizon/warmup/checkpoint arithmetic")

    @property
    def steps_per_epoch(self) -> int:
        return self.train_clips // self.batch_size

    @property
    def total_steps(self) -> int:
        return self.epochs * self.steps_per_epoch

    @property
    def calibration_steps(self) -> int:
        return self.calibration_epochs * self.steps_per_epoch

    @property
    def lora_warmup_steps(self) -> int:
        return self.lora_warmup_epochs * self.steps_per_epoch

    @property
    def grl_ramp_steps(self) -> int:
        return self.grl_ramp_epochs * self.steps_per_epoch

    @property
    def milestone_steps(self) -> tuple[int, ...]:
        return tuple(e * self.steps_per_epoch for e in
                     range(self.checkpoint_epochs, self.epochs + 1, self.checkpoint_epochs))

    def state_dict(self, completed_updates: int) -> dict:
        if type(completed_updates) is not int or not 0 <= completed_updates <= self.total_steps:
            raise ValueError("invalid completed-update counter")
        next_update = completed_updates + 1 if completed_updates < self.total_steps else None
        return dict(schema_version=1, schedule=asdict(self), completed_updates=completed_updates,
                    applied_update=completed_updates,
                    next_update=next_update)

    @classmethod
    def restore(cls, state: dict) -> tuple[AdaptSchedule, int | None]:
        if (type(state.get("schema_version")) is not int or state["schema_version"] != 1
                or type(state.get("applied_update")) is not int
                or (state.get("next_update") is not None
                    and type(state["next_update"]) is not int)
                or set(state.get("schedule", {})) != {f.name for f in fields(cls)}):
            raise ValueError("schedule state schema/parameters mismatch")
        schedule = cls(**state["schedule"])
        completed = state.get("completed_updates")
        if type(completed) is not int or schedule.state_dict(completed) != state:
            raise ValueError("schedule applied/completed/next counters mismatch")
        return schedule, state["next_update"]


def _validate(k: int, schedule: AdaptSchedule) -> None:
    if type(k) is not int or not 1 <= k <= schedule.total_steps:
        raise ValueError("applied update must be an integer in 1..total_steps")


def head_factor(k: int, schedule: AdaptSchedule) -> float:
    _validate(k, schedule)
    if k <= schedule.calibration_steps:
        return k / schedule.calibration_steps
    progress = (k - schedule.calibration_steps) / (
        schedule.total_steps - schedule.calibration_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def lora_factor(k: int, schedule: AdaptSchedule) -> float:
    _validate(k, schedule)
    if k <= schedule.calibration_steps:
        return 0.0
    joint = k - schedule.calibration_steps
    if joint <= schedule.lora_warmup_steps:
        return joint / schedule.lora_warmup_steps
    progress = (joint - schedule.lora_warmup_steps) / (
        schedule.total_steps - schedule.calibration_steps - schedule.lora_warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))


def grl_factor(k: int, schedule: AdaptSchedule) -> float:
    _validate(k, schedule)
    return max(0.0, min((k - schedule.calibration_steps) / schedule.grl_ramp_steps, 1.0))

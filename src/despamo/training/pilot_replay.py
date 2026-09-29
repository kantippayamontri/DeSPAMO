"""Exact physical-batch replay and measured-budget feasibility gates."""

import random
from collections.abc import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


class PilotBatchSampler(Sampler[list[int]]):
    """Single-worker drop-last permutations, advanced *before* each yielded batch."""

    def __init__(self, items: int, batch_size: int, seed: int, total_batches: int) -> None:
        if (
            type(items) is not int
            or type(batch_size) is not int
            or type(total_batches) is not int
            or items < batch_size
            or batch_size < 1
            or total_batches < 1
            or seed != 0
        ):
            raise ValueError("invalid pilot sampler configuration")
        self.items, self.batch_size, self.seed = items, batch_size, seed
        self.total_batches = total_batches
        self.generator = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(items, generator=self.generator).tolist()
        self.cursor = 0
        self.batches_yielded = 0

    def __len__(self) -> int:
        return self.total_batches

    def __iter__(self) -> Iterator[list[int]]:
        while self.batches_yielded < self.total_batches:
            if self.cursor + self.batch_size > self.items:
                self.order = torch.randperm(self.items, generator=self.generator).tolist()
                self.cursor = 0
            batch = self.order[self.cursor : self.cursor + self.batch_size]
            self.cursor += self.batch_size
            self.batches_yielded += 1
            yield batch

    def state_dict(self) -> dict:
        return {
            "items": self.items,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "total_batches": self.total_batches,
            "order": list(self.order),
            "cursor": self.cursor,
            "batches_yielded": self.batches_yielded,
            "generator_state": self.generator.get_state().clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            any(
                state.get(key) != getattr(self, key)
                for key in ("items", "batch_size", "seed", "total_batches")
            )
            or sorted(state.get("order", [])) != list(range(self.items))
            or type(state.get("cursor")) is not int
            or state["cursor"] < 0
            or state["cursor"] > self.items
            or state["cursor"] % self.batch_size
            or type(state.get("batches_yielded")) is not int
            or not 0 <= state["batches_yielded"] <= self.total_batches
            or not isinstance(state.get("generator_state"), torch.Tensor)
        ):
            raise ValueError("pilot sampler identity/state mismatch")
        self.generator.set_state(state["generator_state"])
        self.order = list(state["order"])
        self.cursor = state["cursor"]
        self.batches_yielded = state["batches_yielded"]


def capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available():
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("CUDA RNG device count changed")
        torch.cuda.set_rng_state_all(state["cuda"])
    elif state["cuda"]:
        raise ValueError("checkpoint requires CUDA RNG")


def verify_resume(
    checkpoint: dict, identity: dict, expected_step: int, spent_gpu_seconds: float
) -> None:
    if checkpoint.get("pilot_identity") != identity or not identity.get("protocol_hash"):
        raise ValueError("pilot resume identity mismatch")
    if checkpoint.get("global_step") != expected_step or expected_step < 1:
        raise ValueError("pilot resume step mismatch")
    if not checkpoint.get("optimizer_states") or not checkpoint.get("lr_schedulers"):
        raise ValueError("pilot resume missing optimizer/scheduler")
    if not checkpoint.get("pilot_rng") or not checkpoint.get("pilot_sampler"):
        raise ValueError("pilot resume missing RNG/sampler")
    checkpoint_seconds = checkpoint.get("gpu_seconds_cumulative")
    if (
        not isinstance(checkpoint_seconds, (int, float))
        or checkpoint_seconds < 0
        or spent_gpu_seconds < checkpoint_seconds
        or spent_gpu_seconds > 24 * 3600
    ):
        raise ValueError("pilot resume budget reset/exceeded")


PROFILE_KEYS = (
    "e1_seconds_per_step",
    "e2_seconds_per_step",
    "e3_spamo_seconds_per_step",
    "dino_seconds_per_step",
    "calibration_seconds_per_step",
    "extract_seconds_per_frame",
    "dev_seconds_per_full_split",
    "final_and_probe_seconds",
)


def _valid_profile_value(value: object) -> bool:
    return isinstance(value, (int, float)) and 0 < value < float("inf")


def freeze_tier(profile: dict[str, float], spent_gpu_seconds: float) -> dict:
    """Fail closed: every component must be measured before any E1 run."""
    if (
        set(profile) != set(PROFILE_KEYS)
        or not all(_valid_profile_value(value) for value in profile.values())
        or not 0 <= spent_gpu_seconds < 24 * 3600
    ):
        raise ValueError("incomplete or invalid GPU profile")
    for total, warm, dino in ((8000, 2000, 2000), (6000, 2000, 1000), (4000, 1000, 500)):
        estimate = (
            spent_gpu_seconds
            + total
            * (
                profile["e1_seconds_per_step"]
                + profile["e2_seconds_per_step"]
                + profile["e3_spamo_seconds_per_step"]
            )
            + dino * profile["dino_seconds_per_step"]
            + 200 * profile["calibration_seconds_per_step"]
            + 827354 * profile["extract_seconds_per_frame"]
            + 9 * profile["dev_seconds_per_full_split"]
            + profile["final_and_probe_seconds"]
        )
        if estimate <= 20 * 3600:
            return {
                "spaMo_steps": total,
                "warmup_steps": warm,
                "dino_steps": dino,
                "estimated_gpu_seconds": estimate,
                "profile": dict(profile),
                "spent_gpu_seconds_at_freeze": spent_gpu_seconds,
            }
    raise ValueError("no pilot tier fits measured 20 GPU-hour gate")

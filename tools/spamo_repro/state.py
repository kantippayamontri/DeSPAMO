"""Compact full replay state and immutable file-flushed/rehashed B publication."""

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch

from tools.benchmark.compact_checkpoint import (
    capture_state as original_capture,
)
from tools.benchmark.compact_checkpoint import (
    mutable_keys,
    restore_rng,
)
from tools.checkpoint_storage.inventory import file_hash, no_symlinks
from tools.checkpoint_storage.publish import directory_fsync, flush_file, publish_manifest
from tools.dinov3.identity import digest
from tools.dinov3.storage import writer


def _cpu_copy(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_copy(item) for item in value)
    return copy.deepcopy(value)


def _finite(value) -> None:
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu" or not torch.isfinite(value).all():
            raise ValueError("Nonfinite or non-CPU checkpoint tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _finite(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite(item)


def capture_state(
    model, optimizer, identity: dict, step: int, sampler: dict, *, base_hash=None
) -> dict:
    if type(step) is not int or step < 0:
        raise ValueError("Nonnegative completed optimizer update required")
    # Only healthy execution calls this function. Error paths never capture CUDA
    # RNG, move tensors, save state or invoke device cleanup.
    state = _cpu_copy(original_capture(model, optimizer, identity, step, sampler, base_hash))
    _finite(state)
    return state


def publish_state(control: Path, path: Path, state: dict) -> dict:
    _finite(state)
    no_symlinks(path)
    no_symlinks(control)
    marker = path.with_suffix(".json")
    with writer(control / "checkpoint-leases" / digest(str(path))):
        if path.exists() or marker.exists():
            raise ValueError("Immutable checkpoint already exists")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".partial-", delete=False
        ) as handle:
            partial = Path(handle.name)
            torch.save(state, handle)
            flush_file(handle)
        sha, size = file_hash(partial), partial.stat().st_size
        if path.exists():
            raise ValueError("Checkpoint appeared during publication")
        os.replace(partial, path)
        supported = directory_fsync(path.parent)
        if file_hash(path) != sha or path.stat().st_size != size:
            raise ValueError("Final checkpoint bytes/hash mismatch")
        receipt = {
            "schema_version": 1,
            "path": str(path),
            "sha256": sha,
            "bytes": size,
            "step": state["step"],
            "identity": state["identity"],
            "frozen_fingerprint": state["frozen_fingerprint"],
            "durability": "file-fsync-rehashed-atomic-rename",
            "directory_fsync_supported": supported,
        }
        publish_manifest(marker, receipt)
        if json.loads(marker.read_text()) != receipt:
            raise ValueError("Published checkpoint receipt mismatch")
        return receipt


def restore_state(receipt: dict, model, optimizer, identity: dict, base_hash: str) -> dict:
    path = Path(receipt["path"])
    no_symlinks(path)
    if receipt.get("identity") != identity:
        raise ValueError("Checkpoint identity mismatch")
    if (
        json.loads(path.with_suffix(".json").read_text()) != receipt
        or path.stat().st_size != receipt["bytes"]
        or file_hash(path) != receipt["sha256"]
    ):
        raise ValueError("Checkpoint receipt mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if (
        state.get("schema_version") != 1
        or state["identity"] != identity
        or state["frozen_fingerprint"] != base_hash
        or receipt["frozen_fingerprint"] != base_hash
        or set(state["weights"]) != mutable_keys(model)
        or state["step"] != receipt["step"]
        or not {"optimizer", "sampler", "rng", "prompt_rng"} <= state.keys()
    ):
        raise ValueError("Full checkpoint identity/base/schema mismatch")
    _finite(state)
    current = model.state_dict()
    for name, value in state["weights"].items():
        if value.shape != current[name].shape or value.dtype != current[name].dtype:
            raise ValueError("Checkpoint overlay tensor mismatch")
    for params in state["optimizer"]["state"].values():
        if not 0 < params["step"].item() <= state["step"] or (params["exp_avg_sq"] < 0).any():
            raise ValueError("Checkpoint optimizer history invalid")
    model.load_state_dict(state["weights"], strict=False)
    optimizer.load_state_dict(state["optimizer"])
    if state["prompt_rng"] is not None:
        model.rng.setstate(state["prompt_rng"])
    restore_rng(state["rng"])
    return state

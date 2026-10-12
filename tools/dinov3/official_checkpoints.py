"""Atomic, receipt-bound milestone exports and exact official adaptation resumes."""

import json
import math
import random
import re
from pathlib import Path

import numpy as np
import torch

from tools.dinov3.identity import sha256_file
from tools.dinov3.official_schedule import AdaptSchedule
from tools.dinov3.storage import _atomic, _fsync_directory, atomic_json

POLICY = "official-phoenix14t-merged-dino-adaptation-v1"


def _finite(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _tensors(value) -> None:
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("snapshot non-finite tensor")
    elif isinstance(value, dict):
        for item in value.values():
            _tensors(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _tensors(item)


def validate_state(state: dict) -> None:
    required = {"schema_version", "identity", "epoch", "completed_updates", "applied_update",
                "model", "heads", "optimizer", "schedule", "sampler", "rng", "charged_seconds"}
    if (not required <= state.keys() or type(state["schema_version"]) is not int
            or state["schema_version"] != 1
            or state["identity"].get("policy") != POLICY or not _finite(state["charged_seconds"])):
        raise ValueError("snapshot schema/policy/accounting mismatch")
    schedule, _ = AdaptSchedule.restore(state["schedule"])
    step, epoch = state["completed_updates"], state["epoch"]
    if (type(epoch) is not int or type(step) is not int or step not in schedule.milestone_steps
            or epoch * schedule.steps_per_epoch != step or state["applied_update"] != step
            or state["schedule"]["completed_updates"] != step
            or state["sampler"].get("batches_yielded") != step
            or state["sampler"].get("cursor") != (
                ((step-1) % schedule.steps_per_epoch+1)*schedule.batch_size)
            or state["sampler"].get("batch_size") != schedule.batch_size):
        raise ValueError("snapshot epoch/schedule/sampler mismatch")
    for name in ("model", "heads"):
        if not state[name] or any(not isinstance(t, torch.Tensor) for t in state[name].values()):
            raise ValueError("snapshot model/head state missing")
    if any(not k.endswith(("down.weight", "up.weight")) for k in state["model"]):
        raise ValueError("snapshot contains non-LoRA base state")
    opt = state["optimizer"]
    if not isinstance(opt.get("state"), dict) or not opt.get("param_groups"):
        raise ValueError("snapshot optimizer missing")
    parameters = [i for group in opt["param_groups"] for i in group["params"]]
    if (set(opt["state"]) != set(parameters) or len(set(parameters)) != len(parameters)
            or any(not {"step", "exp_avg", "exp_avg_sq"} <= s.keys()
                   for s in opt["state"].values())):
        raise ValueError("snapshot optimizer state incomplete")
    rng = state["rng"]
    if not {"python", "numpy", "torch", "cuda"} <= rng.keys():
        raise ValueError("snapshot RNG missing")
    try:
        random.Random().setstate(rng["python"])
        np.random.RandomState().set_state(rng["numpy"])
        torch.Generator().set_state(rng["torch"])
        if not isinstance(rng["cuda"], list) or any(
                not isinstance(t, torch.Tensor) or t.dtype != torch.uint8 for t in rng["cuda"]):
            raise ValueError("CUDA RNG malformed")
    except (ValueError, TypeError, RuntimeError, IndexError) as error:
        raise ValueError("snapshot RNG invalid") from error
    _tensors(state)


def _owned(run: Path, identity: dict) -> None:
    if (identity.get("run_key") != run.name or identity.get("run_root") != str(run.resolve())
            or any(p.is_symlink() for p in (run, *run.parents))):
        raise ValueError("snapshot run path/identity ownership mismatch")


def _receipt(path: Path) -> dict:
    receipt = path.with_suffix(".json")
    if path.is_symlink() or receipt.is_symlink() or not path.is_file() or not receipt.is_file():
        raise ValueError("complete owned snapshot and receipt required")
    data = json.loads(receipt.read_text())
    if data.get("sha256") != sha256_file(path) or data.get("bytes") != path.stat().st_size:
        raise ValueError("snapshot receipt/hash mismatch")
    return data


def publish_milestone(run: Path, state: dict, epoch: int) -> dict:
    validate_state(state)
    _owned(run, state["identity"])
    if epoch != state["epoch"]:
        raise ValueError("snapshot publication epoch mismatch")
    names = [run/f"{kind}-epoch-{epoch}.pt" for kind in ("model", "resume")]
    if any(p.exists() or p.is_symlink() or p.with_suffix(".json").exists() for p in names):
        raise ValueError("snapshot publication already exists")
    compact = {k: state[k] for k in ("schema_version", "identity", "epoch", "completed_updates",
                                   "applied_update", "model", "heads", "charged_seconds")}
    result: dict = dict(epoch=epoch, completed_updates=state["completed_updates"])
    for kind, path, value in zip(("model", "resume"), names, (compact, state), strict=True):
        _atomic(path, lambda handle, payload=value: torch.save(payload, handle))
        receipt = dict(kind=kind, epoch=epoch, completed_updates=state["completed_updates"],
                       identity=state["identity"], sha256=sha256_file(path),
                       bytes=path.stat().st_size)
        atomic_json(path.with_suffix(".json"), receipt)
        _receipt(path)
        result[kind] = receipt
    atomic_json(run/f"milestone-epoch-{epoch}.json", result)
    return result


def _complete_epochs(run: Path) -> list[int]:
    for model in run.glob("model-epoch-*.pt"):
        match = re.fullmatch(r"model-epoch-(\d+)\.pt", model.name)
        if not match:
            raise ValueError("invalid owned model filename")
        milestone = run/f"milestone-epoch-{int(match.group(1))}.json"
        if (not milestone.is_file() or milestone.is_symlink()
                or json.loads(milestone.read_text()).get("model") != _receipt(model)):
            raise ValueError("incomplete model milestone publication")
    epochs = []
    for path in run.glob("resume-epoch-*.pt"):
        match = re.fullmatch(r"resume-epoch-(\d+)\.pt", path.name)
        if not match:
            raise ValueError("invalid owned resume filename")
        epoch = int(match.group(1))
        receipt = _receipt(path)
        model_receipt = _receipt(run/f"model-epoch-{epoch}.pt")
        milestone = run/f"milestone-epoch-{epoch}.json"
        if (not milestone.is_file() or milestone.is_symlink()
                or json.loads(milestone.read_text()).get("resume") != receipt
                or json.loads(milestone.read_text()).get("model") != model_receipt):
            raise ValueError("incomplete milestone transaction")
        epochs.append(epoch)
    return sorted(epochs)


def validate_official_resume(path: Path, identity: dict, ledger: dict) -> dict:
    run = path.parent
    _owned(run, identity)
    epochs = _complete_epochs(run)
    if not epochs or path.name != f"resume-epoch-{epochs[-1]}.pt":
        raise ValueError("latest complete owned resume required")
    receipt = _receipt(path)
    if receipt["identity"] != identity:
        raise ValueError("resume receipt identity mismatch")
    state = torch.load(path, map_location="cpu", weights_only=False)
    validate_state(state)
    if (state["identity"] != identity or state["epoch"] != epochs[-1]
            or not _finite(ledger.get("charged_seconds"))
            or state["charged_seconds"] > ledger["charged_seconds"]
            or state["completed_updates"] > ledger.get("completed_updates", -1)):
        raise ValueError("resume identity/accounting rewind mismatch")
    return state


def prune_owned_resumes(run: Path, keep: int = 2) -> list[Path]:
    if type(keep) is not int or keep != 2:
        raise ValueError("latest-two resume retention required")
    epochs = _complete_epochs(run)
    if not epochs:
        return []
    identity = _receipt(run/f"resume-epoch-{epochs[-1]}.pt")["identity"]
    _owned(run, identity)
    deleted = []
    for epoch in epochs[:-keep]:
        path = run/f"resume-epoch-{epoch}.pt"
        if _receipt(path)["identity"] != identity:
            raise ValueError("resume prune ownership mismatch")
        path.unlink()
        path.with_suffix(".json").unlink()
        deleted.append(path)
    _fsync_directory(run)
    return deleted

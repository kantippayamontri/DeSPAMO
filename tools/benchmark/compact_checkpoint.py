"""Compact mutable-model/AdamW/RNG state bound to immutable frozen base tensors."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import _atomic, atomic_json


def frozen_fingerprint(model) -> str:
    hasher=hashlib.sha256()
    for name,p in model.named_parameters():
        if not p.requires_grad:
            hasher.update(str((name,str(p.dtype),list(p.shape))).encode())
            array=p.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
            hasher.update(memoryview(array))
    return hasher.hexdigest()


def mutable_keys(model) -> set[str]:
    state=model.state_dict()
    return {n for n,p in model.named_parameters() if p.requires_grad}|{
        n for n,_ in model.named_buffers() if n in state}


def common_fingerprint(model) -> str:
    state=model.state_dict()
    hasher=hashlib.sha256()
    for name in sorted(mutable_keys(model)):
        if name.startswith(("visual_adapter.corrector.","visual_adapter.gate.")):
            continue
        value=state[name].detach().cpu().contiguous()
        hasher.update(str((name,str(value.dtype),list(value.shape))).encode())
        hasher.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return hasher.hexdigest()


def capture_rng() -> dict:
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(rng: dict):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"]:
        torch.cuda.set_rng_state_all(rng["cuda"])


def capture_state(model,optimizer,identity: dict,step: int,sampler: dict,
                  base_hash: str | None=None,full: bool=True) -> dict:
    state=model.state_dict()
    weights={n:state[n].detach().cpu().clone() for n in sorted(mutable_keys(model))}
    result=dict(schema_version=1,identity=identity,step=step,weights=weights,
                frozen_fingerprint=base_hash or frozen_fingerprint(model),
                prompt_rng=model.rng.getstate() if hasattr(model,"rng") else None)
    if full:
        result.update(optimizer=optimizer.state_dict(),sampler=sampler,rng=capture_rng())
    return result


def save_snapshot(path: Path,state: dict) -> dict:
    if path.exists() or path.is_symlink() or path.with_suffix(".json").exists():
        raise ValueError("owned immutable checkpoint already exists")
    if any(not torch.isfinite(t).all() for t in state["weights"].values()):
        raise ValueError("non-finite compact weights")
    _atomic(path,lambda handle:torch.save(state,handle))
    receipt=dict(sha256=sha256_file(path),bytes=path.stat().st_size,step=state["step"],
                 identity=state["identity"],frozen_fingerprint=state["frozen_fingerprint"])
    atomic_json(path.with_suffix(".json"),receipt)
    return receipt


def load_snapshot(path: Path,model,optimizer,identity: dict,*,base_hash: str | None=None,
                  replay: bool=True) -> dict:
    receipt=json.loads(path.with_suffix(".json").read_text())
    if (path.is_symlink() or receipt["sha256"]!=sha256_file(path)
            or receipt["bytes"]!=path.stat().st_size or receipt["identity"]!=identity):
        raise ValueError("compact checkpoint receipt/identity mismatch")
    state=torch.load(path,map_location="cpu",weights_only=False)
    if (state.get("schema_version")!=1 or state["identity"]!=identity
            or state["frozen_fingerprint"]!=(base_hash or frozen_fingerprint(model))
            or set(state["weights"])!=mutable_keys(model) or state["step"]!=receipt["step"]):
        raise ValueError("compact checkpoint base/schema/mutable keys mismatch")
    current=model.state_dict()
    for n,value in state["weights"].items():
        if (not isinstance(value,torch.Tensor) or value.shape!=current[n].shape
                or value.dtype!=current[n].dtype or not torch.isfinite(value).all()):
            raise ValueError("compact checkpoint tensor mismatch")
    result=model.load_state_dict(state["weights"],strict=False)
    if result.unexpected_keys or set(result.missing_keys)!=set(current)-mutable_keys(model):
        raise ValueError("compact model overlay mismatch")
    if state["prompt_rng"] is not None:
        model.rng.setstate(state["prompt_rng"])
    if replay:
        if not {"optimizer","sampler","rng"}<=state.keys():
            raise ValueError("full compact resume required")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"])
    return state

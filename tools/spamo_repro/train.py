import random

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from tools.spamo_repro.recipe import joint_for, joint_update, lr_at_update, lr_for


def epoch_groups(ids: tuple[str, ...], epoch: int, *, seed: int = 0) -> list[list[tuple[str, ...]]]:
    """Shuffle once per epoch; drop no clips; final group may contain one microbatch."""
    if type(epoch) is not int or epoch < 1 or len(ids) != 7096 or len(set(ids)) != len(ids):
        raise ValueError("Complete original train epoch required")
    order = list(ids)
    random.Random(seed * 1_000_003 + epoch).shuffle(order)
    batches = [tuple(order[i : i + 8]) for i in range(0, len(order), 8)]
    return [batches[i : i + 2] for i in range(0, len(batches), 2)]


def accumulated_update(
    model, optimizer, batches, update: int, total: int, loss_fn, *, recipe: dict | None = None
):
    if not 1 <= len(batches) <= 2:
        raise ValueError("Original batch8/accumulation2 group required")
    if recipe is None:
        joint, lr = joint_update(update), lr_at_update(update, total)
    else:
        joint, lr = joint_for(update, recipe), lr_for(update, recipe)
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    ces, vts = [], []
    for batch in batches:
        vt, ce = loss_fn(model, batch, joint)
        if joint != (ce is not None):
            raise ValueError("Loss callback does not match approved VT/joint phase")
        loss = vt if ce is None else vt + ce
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite original SpaMo loss")
        (loss / 2).backward()
        vts.append(float(vt.detach()))
        if ce is not None:
            ces.append(float(ce.detach()))
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("Nonfinite original SpaMo gradient")
    norm = clip_grad_norm_(params, 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("Nonfinite original SpaMo norm")
    optimizer.step()
    return {
        "optimizer_update": update,
        "lr": optimizer.param_groups[0]["lr"],
        "grad_norm": float(norm),
        "VT": sum(vts) / len(vts),
        "CE": None if not ces else sum(ces) / len(ces),
    }

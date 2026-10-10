"""Same model initialization with approved pooling, ICL and nonzero-floor updates."""

from typing import Any

import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from tools.benchmark.translation_model import build_model as original_model
from tools.followup.recipe import group_lr


def build_model(seed: int, arm: str, cache: str):
    model, base_hash = original_model(seed, arm, cache)
    model.use_in_context = True
    model.num_in_context = 3
    model.vt_pooling = "legacy_mean"
    model.warm_up_steps = 4000
    model.language_model.max_text_length = 1024
    # Checkpoint transformer activations only; visual BatchNorm is never recomputed.
    transformer: Any = model.language_model.model
    transformer.gradient_checkpointing_enable()
    transformer.enable_input_require_grads()
    transformer.config.use_cache = False
    return model, base_hash


def validate_prompt_lengths(model, pool) -> dict:
    lengths = [len(model.language_model.tokenizer.encode(pool.render(r))) for r in pool.rows]
    if max(lengths) > model.language_model.max_text_length:
        raise ValueError("multilingual ICL would be silently truncated")
    return dict(
        max_tokens=max(lengths),
        limit=model.language_model.max_text_length,
        longest_id=pool.rows[max(range(len(lengths)), key=lengths.__getitem__)]["fileid"],
    )


def update(
    model,
    optimizer,
    batch,
    k: int,
    recipe: dict,
    pool,
    *,
    force_joint: bool = False,
    prompt_override: list[str] | None = None,
) -> dict:
    for group in optimizer.param_groups:
        group["lr"] = group_lr(k, recipe, group["name"])
    optimizer.zero_grad(set_to_none=True)
    exemplar_ids = []
    with torch.autocast(batch.spatial.device.type, dtype=torch.bfloat16):
        visual, valid = model.visual_adapter(
            batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
        )
        text, mask = model.language_model.target_embeddings(batch.texts)
        vt = model.vt_align(visual.float(), valid, text.float(), mask, "legacy_mean")
        ce = None
        if force_joint or k > recipe["vt_warmup_steps"]:
            if prompt_override is None:
                prompts, exemplar_ids = pool.training(batch.clip_ids, model.rng)
            else:
                prompts = prompt_override
            ce = model.language_model.translation_loss(visual, valid, prompts, batch.texts)
        loss = vt if ce is None else ce + vt
    if not torch.isfinite(loss):
        raise FloatingPointError("non-finite translation loss")
    loss.backward()
    params = [p for p in model.parameters() if p.requires_grad]
    if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in params):
        raise FloatingPointError("non-finite translation gradients")
    norm = clip_grad_norm_(params, 1.0)
    if not torch.isfinite(norm):
        raise FloatingPointError("non-finite translation norm")
    optimizer.step()
    return dict(
        loss=float(loss.detach()),
        vt_loss=float(vt.detach()),
        translation_loss=None if ce is None else float(ce.detach()),
        grad_norm=float(norm),
        adapter_lr=group_lr(k, recipe, "adapter"),
        flan_lr=group_lr(k, recipe, "flan"),
        step=k,
        exemplar_ids=exemplar_ids,
    )

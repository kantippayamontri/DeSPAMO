"""Paper optimizer construction; pretrained model construction remains separate."""

import torch

from despamo.models.visual_adapter import SpaMoVisualAdapter
from tools.spamo_repro.policy import validate_config

# Upstream training ICL is in-batch derangement (train references only, never
# own). Evaluation keeps the clean fixed train exemplar.


def build_visual_adapter(config: dict) -> SpaMoVisualAdapter:
    validate_config(config)
    return SpaMoVisualAdapter(2048, 1024, 768, 2048)


def configure_training(transformer) -> None:
    """Checkpoint transformer activations only, never visual BatchNorm."""
    transformer.train()
    transformer.gradient_checkpointing_enable()
    transformer.enable_input_require_grads()
    transformer.config.use_cache = False


def build_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=0.01,
    )


def derangement_prompts(clip_ids: tuple[str, ...], pool, rng) -> list[str]:
    """Upstream training ICL: each clip gets another in-batch clip's train pairs."""
    if len(clip_ids) < 2:
        return pool.training(clip_ids, rng)[0]
    for _ in range(1000):
        order = list(clip_ids)
        rng.shuffle(order)
        if all(a != b for a, b in zip(order, clip_ids, strict=True)):
            return [pool.render(pool.by_id[c]) for c in order]
    raise ValueError("No in-batch derangement found")


def paper_losses(model, batch, joint: bool, pool, *, icl: str = "pool"):
    """Original padded-mean VT; transformer forward only after SA warmup.

    Target token embeddings belong to the frozen base. VT therefore supplies no
    LoRA gradients; absent gradients also prevent AdamW decay/state creation for
    those parameters during warmup. Prompts use reference-disjoint train rows.
    """
    with torch.autocast(batch.spatial.device.type, dtype=torch.bfloat16):
        visual, valid = model.visual_adapter(
            batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
        )
        text, mask = model.language_model.target_embeddings(batch.texts)
        vt = model.vt_align(visual.float(), valid, text.float(), mask, "legacy_mean")
        ce = None
        if joint:
            if icl == "derangement":
                prompts = derangement_prompts(batch.clip_ids, pool, model.rng)
            else:
                prompts, _ = pool.training(batch.clip_ids, model.rng)
            ce = model.language_model.translation_loss(visual, valid, prompts, batch.texts)
    return vt, ce

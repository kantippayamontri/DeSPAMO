"""Explicit cached deterministic generation without mutating training configuration."""

import json
from collections.abc import Callable
from pathlib import Path

import torch

GPU_HOLD = Path(
    "/home/kan/datasets/despamo/official-phoenix14t-translation-clip-s2-spamo200-v3/"
    "manual-stop.json"
)


def guard_device(device: torch.device, *, gpu_reviewed: bool = False) -> None:
    if device.type == "cpu":
        return
    if device.type != "cuda":
        raise ValueError("CPU or CUDA inference device required")
    if not gpu_reviewed:
        raise RuntimeError("GPU validation requires explicit host-mitigation review")
    if GPU_HOLD.exists():
        record = json.loads(GPU_HOLD.read_text())
        if record.get("gpu_validation_approved") is not True:
            raise RuntimeError("GPU safety hold remains active")


@torch.no_grad()
def generate_cached_text(
    language,
    visual: torch.Tensor,
    visual_mask: torch.Tensor,
    prompts: list[str],
    *,
    beam_size: int = 5,
    max_length: int = 64,
    gpu_reviewed: bool = False,
    on_start: Callable[[dict], None] | None = None,
) -> list[str]:
    guard_device(visual.device, gpu_reviewed=gpu_reviewed)
    if type(beam_size) is not int or beam_size < 1 or type(max_length) is not int or max_length < 2:
        raise ValueError("positive beam count and output limit>=2 required")
    if visual.ndim != 3 or not visual.shape[0] or len(prompts) != visual.shape[0]:
        raise ValueError("visual batch and prompt count must match")
    if visual_mask.dtype != torch.bool or visual_mask.shape != visual.shape[:2]:
        raise ValueError("boolean visual mask matching batch/time required")
    if visual_mask.device != visual.device or not visual_mask.any(1).all():
        raise ValueError("each sample must have valid visual tokens on the same device")
    if not torch.isfinite(visual[visual_mask]).all():
        raise ValueError("non-finite valid visual token")
    modes = [(module, module.training) for module in language.modules()]
    try:
        language.eval()
        joint, mask = language._joint_inputs(visual, visual_mask, prompts)
        if on_start is not None:
            on_start(
                dict(
                    joint_shape=list(joint.shape),
                    joint_lengths=mask.sum(1).tolist(),
                    beam_size=beam_size,
                    effective_beam_sequences=len(prompts) * beam_size,
                    max_length=max_length,
                    use_cache=True,
                    do_sample=False,
                )
            )
        generated = language.model.generate(
            inputs_embeds=joint,
            attention_mask=mask,
            num_beams=beam_size,
            max_length=max_length,
            do_sample=False,
            use_cache=True,
        )
        texts = language.tokenizer.batch_decode(generated, skip_special_tokens=True)
        if len(texts) != len(prompts):
            raise ValueError("generated prediction count differs from batch")
        return [text.lower() for text in texts]
    finally:
        # Boolean state restoration only: no CUDA RNG calls after a device fault.
        for module, flag in modes:
            module.training = flag

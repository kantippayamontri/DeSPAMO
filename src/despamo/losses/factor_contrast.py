"""Masked symmetric multi-positive CLIP-space alignment."""

import torch
from torch.nn import functional as F


def multi_positive(
    visual: torch.Tensor, text: torch.Tensor, labels: tuple[str, ...], valid: torch.Tensor
) -> torch.Tensor:
    if visual.ndim != 2 or visual.shape != text.shape or len(labels) != len(visual):
        raise ValueError("factor contrast shape mismatch")
    if valid.shape != visual.shape[:1] or valid.dtype != torch.bool:
        raise ValueError("factor contrast mask mismatch")
    indices = valid.nonzero(as_tuple=True)[0]
    active = [labels[i] for i in indices.tolist()]
    if len(active) < 2 or len(set(active)) < 2:
        return visual.sum() * 0.0
    v, t = visual[indices].float(), text[indices].detach().float()
    if not torch.isfinite(v).all() or not torch.isfinite(t).all():
        raise FloatingPointError("non-finite valid factor vector")
    logits = F.normalize(v, dim=-1) @ F.normalize(t, dim=-1).T / 0.07
    positive = torch.tensor(
        [[left == right for right in active] for left in active],
        device=logits.device,
        dtype=logits.dtype,
    )
    row = positive / positive.sum(dim=1, keepdim=True)
    col = positive.T / positive.T.sum(dim=1, keepdim=True)
    return -0.5 * (
        (row * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
        + (col * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean()
    )

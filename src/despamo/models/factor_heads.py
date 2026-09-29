"""Seven independent nuisance/adaptation heads on projected spatial rows."""

import torch
from torch import nn

from despamo.appearance.schema import FACTORS, STABLE
from despamo.losses.factor_contrast import multi_positive


class Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.alpha = alpha
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.alpha * gradient, None


def reverse(value: torch.Tensor, alpha: float) -> torch.Tensor:
    return Reverse.apply(value, alpha)


def select_representations(
    projected: torch.Tensor, mask: torch.Tensor, rows: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    if projected.ndim != 3 or mask.shape != projected.shape[:2] or rows.ndim != 2:
        raise ValueError("factor projected spatial shape mismatch")
    mask, rows = mask.to(projected.device), rows.to(projected.device)
    if (mask.sum(dim=1) == 0).any():
        raise ValueError("empty spatial sequence")
    if ((rows < -1) | (rows >= projected.shape[1])).any():
        raise ValueError("frame row outside projected sequence")
    safe = rows.clamp_min(0)
    if ((rows >= 0) & ~mask.gather(1, safe)).any():
        raise ValueError("frame target aligned to padding")
    clip = (projected * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True)
    frames = projected.gather(1, safe.unsqueeze(-1).expand(-1, -1, projected.shape[-1]))
    return clip, frames


class FactorHeads(nn.Module):
    def __init__(self, width: int, text_width: int, enabled: list[str]):
        super().__init__()
        if not enabled or not set(enabled) <= set(FACTORS) or len(set(enabled)) != len(enabled):
            raise ValueError("unknown, duplicate or empty factors")
        self.enabled = tuple(enabled)
        self.heads = nn.ModuleDict({f: nn.Linear(width, text_width) for f in FACTORS})
        for factor in FACTORS:
            self.heads[factor].requires_grad_(factor in enabled)

    def forward(self, projected, mask, batch, alpha: float):
        clip, frames = select_representations(projected, mask, batch.rows)
        losses, counts = {}, {}
        for factor in self.enabled:
            visual = reverse(clip, alpha) if factor in STABLE else frames
            output = self.heads[factor](visual).reshape(-1, self.heads[factor].out_features)
            text = batch.vectors[factor].to(output.device).reshape_as(output)
            valid = batch.masks[factor].to(output.device).reshape(-1)
            if factor not in STABLE:
                valid = valid & (batch.rows.to(output.device).reshape(-1) >= 0)
            losses[factor] = multi_positive(output, text, batch.labels[factor], valid)
            counts[factor] = int(valid.sum())
        return losses, counts

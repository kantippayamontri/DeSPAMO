import torch
from torch import nn
from torch.nn import functional as F


def pooled_sequence(features: torch.Tensor, mask: torch.Tensor, mode: str) -> torch.Tensor:
    if features.ndim != 3:
        raise ValueError(f"features must have shape (batch, time, channels), got {features.shape}")
    if mask.ndim != 2 or mask.shape != features.shape[:2]:
        raise ValueError(f"mask shape must be {features.shape[:2]}, got {mask.shape}")
    if features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must have non-empty batch and time dimensions")
    if mask.dtype != torch.bool:
        raise ValueError(f"mask must be boolean, got {mask.dtype}")

    if mode == "legacy_mean":
        return features.mean(dim=1)
    if mode == "masked_mean":
        if not mask.any(dim=1).all():
            raise ValueError("masked_mean requires at least one valid token; no valid tokens found")
        weights = mask.unsqueeze(-1).to(features.dtype)
        valid_features = features.masked_fill(~mask.unsqueeze(-1), 0)
        return valid_features.sum(dim=1) / weights.sum(dim=1)
    raise ValueError(f"unsupported pooling mode: {mode}")


class VTAlignLoss(nn.Module):
    def __init__(self, initial_logit_scale: float = 2.6592) -> None:
        super().__init__()
        self.logit_scale = nn.Parameter(torch.tensor(initial_logit_scale))

    def forward(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        text: torch.Tensor,
        text_mask: torch.Tensor,
        mode: str,
    ) -> torch.Tensor:
        visual_embedding = F.normalize(pooled_sequence(visual, visual_mask, mode), dim=-1)
        text_embedding = F.normalize(pooled_sequence(text, text_mask, mode), dim=-1)
        logits = text_embedding @ visual_embedding.T * self.logit_scale.exp()
        labels = torch.arange(logits.shape[0], device=logits.device)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

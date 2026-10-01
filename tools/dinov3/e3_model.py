"""Pinned DINOv3 Q/V LoRA and independent masked factor supervision."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers import DINOv3ViTModel

FACTORS = ("biometric", "clothing", "hair", "background", "left_hand", "right_hand", "mouth")
NUISANCE = frozenset(FACTORS[:4])


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, *, rank: int = 8, alpha: int = 8):
        super().__init__()
        if not isinstance(base, nn.Linear) or rank != 8 or alpha != 8:
            raise ValueError("unsupported E3 DINO LoRA configuration")
        self.base = base.requires_grad_(False)
        self.down = nn.Linear(base.in_features, rank, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        self.up = nn.Linear(rank, base.out_features, bias=False).to(
            device=base.weight.device, dtype=base.weight.dtype
        )
        nn.init.normal_(self.down.weight, std=1 / rank)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.up(self.down(value)) * self.scale


def attach_late_lora(model: DINOv3ViTModel) -> tuple[str, ...]:
    if (not isinstance(model, DINOv3ViTModel) or len(model.layer) != 24
            or model.config.hidden_size != 1024):
        raise ValueError("unsupported E3 pinned ViT-L model")
    targets = tuple(f"layer.{index}.attention.{target}_proj"
                    for index in range(20, 24) for target in ("q", "v"))
    if any(not isinstance(getattr(model.layer[index].attention, f"{target}_proj"), nn.Linear)
           for index in range(20, 24) for target in ("q", "v")):
        raise ValueError("E3 DINO LoRA already attached or projection mismatch")
    model.requires_grad_(False)
    for index in range(20, 24):
        for target in ("q", "v"):
            attention = model.layer[index].attention
            name = f"{target}_proj"
            setattr(attention, name, LoRALinear(getattr(attention, name)))
    return targets


def scale_reference_loss(adapted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    if (adapted.ndim != 2 or adapted.shape != teacher.shape or adapted.shape[-1] % 2
            or not torch.isfinite(adapted).all() or not torch.isfinite(teacher).all()):
        raise ValueError("E3 cosine teacher shape/values invalid")
    left, right = adapted.float().chunk(2, dim=-1)
    original_left, original_right = teacher.detach().float().chunk(2, dim=-1)
    return 0.5 * ((1 - F.cosine_similarity(left, original_left, dim=-1)).mean()
                  + (1 - F.cosine_similarity(right, original_right, dim=-1)).mean())


def relational_structure_loss(adapted: torch.Tensor,
                              teacher: torch.Tensor) -> dict[str, torch.Tensor]:
    """Preserve the pairwise geometry of frozen DINO, not just per-vector direction.

    `pattern` (Term A) matches the standardised off-diagonal cosine pattern and is
    scale-invariant. `spread` (Term B) matches the dispersion of those cosines, which
    Term A cannot see; it is what punishes collapsing every clip into one narrow cone.
    """
    if (adapted.ndim != 2 or adapted.shape != teacher.shape or adapted.shape[-1] % 2
            or adapted.shape[0] < 2
            or not torch.isfinite(adapted).all() or not torch.isfinite(teacher).all()):
        raise ValueError("E3 relational structure shape/values invalid")
    mask = ~torch.eye(adapted.shape[0], dtype=torch.bool, device=adapted.device)
    pattern = adapted.new_zeros(())
    spread = adapted.new_zeros(())
    half = adapted.shape[-1] // 2
    for lo, hi in ((0, half), (half, adapted.shape[-1])):
        student = F.normalize(adapted[:, lo:hi].float(), dim=-1)
        frozen = F.normalize(teacher[:, lo:hi].detach().float(), dim=-1)
        sa = (student @ student.T)[mask]
        st = (frozen @ frozen.T)[mask]
        # Population std: Bessel's correction would leave a 1/n_pairs floor in `pattern`,
        # a constant that shifts with batch size and never reaches zero.
        sa_std, st_std = sa.std(unbiased=False), st.std(unbiased=False)
        za = (sa - sa.mean()) / (sa_std + 1e-8)
        zt = (st - st.mean()) / (st_std + 1e-8)
        pattern = pattern + (1 - (za * zt).mean())
        spread = spread + (torch.log(sa_std + 1e-8) - torch.log(st_std + 1e-8)) ** 2
    return {"pattern": 0.5 * pattern, "spread": 0.5 * spread}


class _Reverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, alpha):
        ctx.alpha = alpha
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.alpha * gradient, None


def _contrast(visual, text, labels, active):
    positions = active.nonzero(as_tuple=True)[0]
    selected = [labels[i] for i in positions.tolist()]
    if len(selected) < 2 or len(set(selected)) < 2:
        return visual.sum() * 0.0
    v = F.normalize(visual[positions].float(), dim=-1)
    t = F.normalize(text[positions].detach().float(), dim=-1)
    logits = v @ t.T / 0.07
    positives = torch.tensor([[left == right for right in selected] for left in selected],
                             device=logits.device, dtype=logits.dtype)
    row = positives / positives.sum(dim=1, keepdim=True)
    col = positives.T / positives.T.sum(dim=1, keepdim=True)
    return -0.5 * ((row * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
                   + (col * F.log_softmax(logits.T, dim=1)).sum(dim=1).mean())


class DINOFactorHeads(nn.Module):
    def __init__(self, visual_width: int = 2048, text_width: int = 768):
        super().__init__()
        self.heads = nn.ModuleDict({factor: nn.Linear(visual_width, text_width)
                                    for factor in FACTORS})

    def forward(self, features, targets, masks, labels, alpha: float):
        batch, frames, width = features.shape
        if (frames != 5 or width != self.heads[FACTORS[0]].in_features
                or targets.shape != (batch, 19, self.heads[FACTORS[0]].out_features)
                or masks.shape != (batch, 19) or masks.dtype != torch.bool
                or len(labels) != batch or any(len(row) != 19 for row in labels)
                or not 0 <= alpha <= 1):
            raise ValueError("E3 factor target shape/ramp invalid")
        result = {}
        for index, factor in enumerate(FACTORS):
            if factor in NUISANCE:
                visual = self.heads[factor](_Reverse.apply(features.mean(dim=1), alpha))
                text, valid = targets[:, index], masks[:, index]
                descriptions = tuple(row[index] for row in labels)
            else:
                columns = [4 + frame * 3 + index - 4 for frame in range(5)]
                visual = self.heads[factor](features).reshape(batch * 5, -1)
                text = targets[:, columns].reshape(batch * 5, -1)
                valid = masks[:, columns].reshape(-1)
                descriptions = tuple(row[column] for row in labels for column in columns)
            result[factor] = _contrast(visual, text, descriptions, valid)
        return result

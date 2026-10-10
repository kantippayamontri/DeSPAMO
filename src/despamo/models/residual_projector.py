"""Opt-in zero-output nonlinear refinements of the original SpaMo projections."""

import torch
from torch import nn

from despamo.models.visual_adapter import SpaMoVisualAdapter


class ResidualMLPProjector(nn.Module):
    def __init__(self, base: nn.Linear, hidden_dim: int) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear) or type(hidden_dim) is not int or hidden_dim < 1:
            raise ValueError("residual projector requires a linear base and positive hidden width")
        self.base = base
        self.refinement = nn.Sequential(
            nn.LayerNorm(base.in_features),
            nn.Linear(base.in_features, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, base.out_features),
        ).to(device=base.weight.device, dtype=base.weight.dtype)
        output = self.refinement[3]
        assert isinstance(output, nn.Linear) and output.bias is not None
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.base(value) + self.refinement(value)


def enable_residual_projectors(
    adapter: SpaMoVisualAdapter, hidden_dim: int = 1024, seed: int = 1
) -> None:
    if type(hidden_dim) is not int or hidden_dim < 1:
        raise ValueError("residual hidden width must be a positive integer")
    if not isinstance(adapter.spatial_projector, nn.Linear) or not isinstance(
        adapter.motion_projector, nn.Linear
    ):
        raise ValueError("residual projectors already attached or base is not linear")
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        adapter.add_module("spatial_projector",
                           ResidualMLPProjector(adapter.spatial_projector, hidden_dim))
        adapter.add_module("motion_projector",
                           ResidualMLPProjector(adapter.motion_projector, hidden_dim))

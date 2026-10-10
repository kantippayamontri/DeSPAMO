"""Opt-in spatial MLP and from-scratch temporal query connector study arms."""

import math

import torch
from torch import nn

from despamo.models.visual_adapter import SpaMoVisualAdapter

ARM_CONFIGS = {
    "reference": {
        "spatial": [2048, 768], "motion": [1024, 768],
        "output": [768, 2048, 2048], "output_activation": "GELU",
        "normalization": "original_tcn_batchnorm", "token_policy": "valid_tcn_sequence",
        "position_policy": "none", "temporal": "K5/P2/K5/P2",
    },
    "plain_mlp": {
        "spatial": [2048, 1024, 768], "spatial_activation": "GELU",
        "hidden_dim": 1024, "spatial_normalization": "none", "spatial_dropout": 0.0,
        "spatial_residual": False, "motion": [1024, 768], "output": [768, 2048, 2048],
        "output_activation": "GELU", "normalization": "original_tcn_batchnorm",
        "token_policy": "valid_tcn_sequence", "position_policy": "none",
        "temporal": "K5/P2/K5/P2",
    },
    "qformer": {
        "spatial": [2048, 768], "motion": [1024, 768], "width": 768,
        "num_queries": 32, "num_layers": 2, "num_heads": 12, "ff_dim": 3072,
        "dropout": 0.1, "output_dim": 2048, "activation": "GELU",
        "normalization": "pre_norm_self_cross_ff; memory_layernorm; final_query_layernorm",
        "position_policy": "fixed_sinusoidal_zero_based_tcn_time_before_memory_layernorm",
        "padding_policy": "zero_before_position_and_norm; key_padding_mask=~valid",
        "token_policy": "32_queries_all_valid", "temporal": "K5/P2/K5/P2",
        "pretrained": False, "memory_features": "temporal_frame_summaries",
    },
}


class QueryConnectorBlock(nn.Module):
    """Pre-normalized query self-attention, visual cross-attention and FFN."""

    def __init__(self, width: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width)
        self.cross_norm = nn.LayerNorm(width)
        self.ff_norm = nn.LayerNorm(width)
        self.self_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True,
        )
        self.cross_attention = nn.MultiheadAttention(
            width, num_heads, dropout=dropout, batch_first=True,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(width, ff_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(ff_dim, width),
        )
        self.residual_dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, memory: torch.Tensor, valid: torch.Tensor
    ) -> torch.Tensor:
        normalized = self.self_norm(queries)
        attended, _ = self.self_attention(normalized, normalized, normalized, need_weights=False)
        queries = queries + self.residual_dropout(attended)
        attended, _ = self.cross_attention(
            self.cross_norm(queries), memory, memory, key_padding_mask=~valid, need_weights=False,
        )
        queries = queries + self.residual_dropout(attended)
        return queries + self.residual_dropout(self.feed_forward(self.ff_norm(queries)))


class MaskedQueryConnector(nn.Module):
    """Learned queries over positioned, normalized valid TCN memory [B, L, C]."""

    def __init__(
        self, width: int = 768, output_dim: int = 2048, num_queries: int = 32,
        num_layers: int = 2, num_heads: int = 12, ff_dim: int = 3072, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        dimensions = (width, output_dim, num_queries, num_layers, num_heads, ff_dim)
        if any(type(value) is not int or value < 1 for value in dimensions):
            raise ValueError("query connector dimensions must be positive integers")
        if width % num_heads or not 0 <= dropout < 1:
            raise ValueError("query width must divide into heads and dropout must be in [0, 1)")
        self.width = width
        self.queries = nn.Parameter(torch.empty(num_queries, width))
        nn.init.normal_(self.queries, std=0.02)
        self.memory_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList([
            QueryConnectorBlock(width, num_heads, ff_dim, dropout) for _ in range(num_layers)
        ])
        self.query_norm = nn.LayerNorm(width)
        self.output_projection = nn.Linear(width, output_dim)

    def _position_encoding(
        self, length: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        # Compute angles in FP32 even under BF16 autocast; cast only the final encoding.
        positions = torch.arange(length, device=device, dtype=torch.float32)[:, None]
        frequencies = torch.exp(
            torch.arange(0, self.width, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / self.width)
        )
        angles = positions * frequencies
        encoding = torch.empty(length, self.width, device=device, dtype=torch.float32)
        encoding[:, 0::2] = angles.sin()
        encoding[:, 1::2] = angles[:, :self.width // 2].cos()
        return encoding.to(dtype=dtype)

    def forward(
        self, memory: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if memory.ndim != 3 or valid.ndim != 2 or valid.dtype != torch.bool:
            raise ValueError("memory must be [B, L, C] and valid must be boolean [B, L]")
        if memory.shape[:2] != valid.shape or memory.shape[2] != self.width:
            raise ValueError("memory and valid dimensions must match connector width")
        if memory.device != valid.device or not memory.is_floating_point():
            raise ValueError("memory must be floating point on the same device as valid")
        if memory.shape[0] == 0 or memory.shape[1] == 0 or not valid.any(dim=1).all():
            raise ValueError("each visual memory sequence must contain valid tokens")
        padding = ~valid[..., None]
        memory = memory.masked_fill(padding, 0)
        memory = memory + self._position_encoding(memory.shape[1], memory.device, memory.dtype)
        memory = self.memory_norm(memory.masked_fill(padding, 0)).masked_fill(padding, 0)
        queries = self.queries.to(dtype=memory.dtype)[None].expand(memory.shape[0], -1, -1)
        for block in self.blocks:
            queries = block(queries, memory, valid)
        tokens = self.output_projection(self.query_norm(queries))
        return tokens, torch.ones(tokens.shape[:2], dtype=torch.bool, device=tokens.device)


class ConnectorVisualAdapter(SpaMoVisualAdapter):
    """Reuse the already initialized adapter, replacing only its chosen branch."""

    def __init__(self, base: SpaMoVisualAdapter, arm: str, replacement: nn.Module) -> None:
        # Do not call the base constructor: shared weights were initialized by the caller.
        nn.Module.__init__(self)
        self.spatial_projector = replacement if arm == "plain_mlp" else base.spatial_projector
        self.motion_projector = base.motion_projector
        self.temporal_encoder = base.temporal_encoder
        self.multimodal_projector = replacement if arm == "qformer" else base.multimodal_projector
        self.arm = arm
        self.training = base.training
        replacement.train(base.training)

    def _project_tokens(
        self, encoded: torch.Tensor, valid: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.arm == "qformer":
            return self.multimodal_projector(encoded, valid)
        return super()._project_tokens(encoded, valid)


def enable_connector_arm(adapter: SpaMoVisualAdapter, arm: str, seed: int) -> SpaMoVisualAdapter:
    """Attach a seeded study branch without changing common weights or global RNG."""
    if arm not in ARM_CONFIGS:
        raise ValueError(f"unknown connector arm: {arm}")
    if arm == "reference":
        return adapter
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    spatial = adapter.spatial_projector
    if isinstance(adapter, ConnectorVisualAdapter) or not isinstance(spatial, nn.Linear):
        raise ValueError("connector arm requires an unmodified linear base adapter")
    output = adapter.multimodal_projector
    if not isinstance(output, nn.Sequential) or not isinstance(output[0], nn.Linear):
        raise ValueError("connector arm requires the original output MLP")
    output_layer = output[0]
    assert isinstance(output_layer, nn.Linear)
    if arm == "qformer" and spatial.out_features != 768:
        raise ValueError("study qformer requires adapter width 768")
    with torch.random.fork_rng():
        torch.manual_seed(100 + seed)
        if arm == "plain_mlp":
            replacement = nn.Sequential(
                nn.Linear(spatial.in_features, 1024),
                nn.GELU(),
                nn.Linear(1024, spatial.out_features),
            )
        else:
            replacement = MaskedQueryConnector(output_dim=output_layer.out_features)
        replacement = replacement.to(device=spatial.weight.device, dtype=spatial.weight.dtype)
        return ConnectorVisualAdapter(adapter, arm, replacement)

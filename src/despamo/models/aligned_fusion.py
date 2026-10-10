"""Opt-in, within-clip motion alignment on the matched plain-MLP backbone.

Runtime timestamps require runner admission of full-source spatial rows 0..N-1.
Masks/counts alone cannot establish extraction chronology or detect renumbered crops.
"""

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from despamo.models.temporal import TemporalConv
from despamo.models.visual_adapter import SpaMoVisualAdapter, lengths_to_mask

FUSION_ARMS = ("reference", "aligned_fixed", "aligned_learned")
FUSION_CONFIGS = {
    arm: {
        "spatial": [2048, 1024, 768], "spatial_activation": "GELU",
        "motion": [1024, 768], "output": [768, 2048, 2048],
        "output_activation": "GELU", "temporal": "K5/P2/K5/P2",
        "token_policy": "valid_tcn_sequence", "runtime_source_rows": "full_source_0_to_N_minus_1",
        "window": 16, "stride": 8, "timing_dtype": "float32",
        "support_policy": "real_window_union_no_extrapolation",
        "corrector": None if arm == "reference" else [768, 768, 768],
        "corrector_activation": None if arm == "reference" else "GELU",
        "corrector_final_init": None if arm == "reference" else "zero_weight_and_bias",
        "gate": {"reference": None, "aligned_fixed": 0.5,
                 "aligned_learned": "sigmoid_linear_1536_to_1_zero_init"}[arm],
        "added_normalization": "none", "added_dropout": 0.0, "added_loss": "none",
        "added_module_seed_offset": 200,
    }
    for arm in FUSION_ARMS
}


def _timing_vector(value: torch.Tensor, name: str, device: torch.device) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.ndim != 1 or value.dtype == torch.bool or value.is_complex():
        raise ValueError(f"{name} must be a real numeric vector")
    if value.device != device:
        raise ValueError(f"{name} must be on the motion device")
    timing = value.to(dtype=torch.float32)
    if not torch.isfinite(timing).all() or (timing < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative timestamps")
    if (timing[1:] <= timing[:-1]).any():
        raise ValueError(f"{name} must be strictly increasing in FP32")
    return timing


def interpolate_motion(
    motion: torch.Tensor,
    source_indices: torch.Tensor,
    window_starts: torch.Tensor,
    window_ends: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interpolate one clip's [M,C] motion onto original [S] source timestamps.

    Windows have inclusive real-frame support. Values clamp to first/last centre
    only inside the union of windows; uncovered rows (including gaps) are zero.
    Timing is FP32; feature values/weights retain motion dtype/device/autograd.
    Source timestamps may be sparse or fractional, but must be ordered uniquely.
    """
    if not isinstance(motion, torch.Tensor):
        raise TypeError("motion must be a tensor")
    if motion.ndim != 2 or not motion.is_floating_point() or 0 in motion.shape:
        raise ValueError("motion must be nonempty floating point [M,C]")
    if not torch.isfinite(motion).all():
        raise ValueError("motion must contain finite features")
    source = _timing_vector(source_indices, "source_indices", motion.device)
    starts = _timing_vector(window_starts, "window_starts", motion.device)
    ends = _timing_vector(window_ends, "window_ends", motion.device)
    if starts.numel() != motion.shape[0] or ends.shape != starts.shape:
        raise ValueError("window vectors must have one entry per motion row")
    if (ends < starts).any():
        raise ValueError("window ends must not precede starts")
    # Half-sums avoid overflowing when individually finite FP32 times are large.
    centres = starts * 0.5 + ends * 0.5
    if (centres[1:] <= centres[:-1]).any():
        raise ValueError("window centres must be strictly increasing in FP32")
    window = torch.searchsorted(starts, source, right=True) - 1
    supported = (window >= 0) & (source <= ends[window.clamp_min(0)])
    if motion.shape[0] == 1:
        values = motion.expand(source.numel(), -1)
    else:
        left = (torch.searchsorted(centres, source, right=True) - 1).clamp(
            0, motion.shape[0] - 2,
        )
        right = left + 1
        weight = ((source - centres[left]) / (centres[right] - centres[left])).clamp(0, 1)
        weight = weight.to(dtype=motion.dtype)[:, None]
        values = (1 - weight) * motion[left] + weight * motion[right]
    return values.masked_fill(~supported[:, None], 0), supported


def _validate_backbone(base: SpaMoVisualAdapter) -> None:
    if not isinstance(base, SpaMoVisualAdapter) or isinstance(base, TimeAlignedFusionAdapter):
        raise ValueError("fusion requires an unattached plain-MLP SpaMoVisualAdapter")
    spatial = base.spatial_projector
    motion = base.motion_projector
    output = base.multimodal_projector
    if not (
        isinstance(spatial, nn.Sequential) and len(spatial) == 3
        and isinstance(spatial[0], nn.Linear) and isinstance(spatial[1], nn.GELU)
        and isinstance(spatial[2], nn.Linear)
        and (spatial[0].in_features, spatial[0].out_features,
             spatial[2].in_features, spatial[2].out_features) == (2048, 1024, 1024, 768)
        and isinstance(motion, nn.Linear)
        and (motion.in_features, motion.out_features) == (1024, 768)
        and isinstance(base.temporal_encoder, TemporalConv)
        and isinstance(output, nn.Sequential) and len(output) == 3
        and isinstance(output[0], nn.Linear) and isinstance(output[1], nn.GELU)
        and isinstance(output[2], nn.Linear)
        and (output[0].in_features, output[0].out_features,
             output[2].in_features, output[2].out_features) == (768, 2048, 2048, 2048)
    ):
        raise ValueError("fusion requires the matched 2048/1024/768 plain-MLP backbone")
    temporal = base.temporal_encoder.temporal_conv
    expected_types = (nn.Conv1d, nn.BatchNorm1d, nn.ReLU, nn.MaxPool1d) * 2
    if len(temporal) != 8 or any(
        type(layer) is not kind for layer, kind in zip(temporal, expected_types, strict=True)
    ):
        raise ValueError("fusion requires the original K5/P2/K5/P2 TCN")
    for index in (0, 4):
        conv = temporal[index]
        pool = temporal[index + 3]
        if (
            (conv.in_channels, conv.out_channels, conv.kernel_size, conv.stride,
             conv.padding, conv.dilation, conv.groups) != (768, 768, (5,), (1,), (0,), (1,), 1)
            or temporal[index + 1].num_features != 768
            or (pool.kernel_size, pool.stride, pool.padding, pool.dilation, pool.ceil_mode)
            != (2, 2, 0, 1, False)
        ):
            raise ValueError("fusion requires the original width768 K5/P2/K5/P2 TCN")


class TimeAlignedFusionAdapter(SpaMoVisualAdapter):
    """Enrich valid full-source spatial frames, then retain original motion tokens."""

    def __init__(
        self, base: SpaMoVisualAdapter, arm: str, corrector: nn.Module, gate: nn.Linear | None,
    ) -> None:
        # Common modules already exist; never reinitialize backbone parameters.
        nn.Module.__init__(self)
        self.spatial_projector = base.spatial_projector
        self.motion_projector = base.motion_projector
        self.temporal_encoder = base.temporal_encoder
        self.multimodal_projector = base.multimodal_projector
        self.corrector = corrector
        self.gate = gate
        self.arm = arm
        self.training = base.training
        corrector.train(base.training)
        if gate is not None:
            gate.train(base.training)

    def forward(
        self,
        spatial: torch.Tensor,
        spatial_mask: torch.Tensor,
        motion: torch.Tensor,
        motion_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        for features, mask, width in ((spatial, spatial_mask, 2048), (motion, motion_mask, 1024)):
            if not isinstance(features, torch.Tensor) or not isinstance(mask, torch.Tensor):
                raise TypeError("features and masks must be tensors")
            if features.ndim != 3 or features.shape[2] != width or not features.is_floating_point():
                raise ValueError(f"features must be floating point [B,T,{width}]")
            if mask.dtype != torch.bool or mask.shape != features.shape[:2]:
                raise ValueError("masks must be boolean [B,T] matching features")
            if mask.device != features.device or features.device != spatial.device:
                raise ValueError("all features and masks must share a device")
            if features.shape[0] == 0 or not mask.any(dim=1).all():
                raise ValueError("each clip must contain nonempty spatial and motion sequences")
            if (mask[:, 1:] & ~mask[:, :-1]).any():
                raise ValueError("runtime masks must be valid prefixes of full-source rows")
            if not torch.isfinite(features[mask]).all():
                raise ValueError("valid features must be finite")
        if spatial.shape[0] != motion.shape[0]:
            raise ValueError("spatial and motion batch sizes must match")
        frame_counts = spatial_mask.sum(dim=1)
        motion_counts = motion_mask.sum(dim=1)
        expected = (torch.div(frame_counts - 16, 8, rounding_mode="floor") + 1).clamp_min(1)
        if not torch.equal(motion_counts, expected):
            raise ValueError("motion counts must match full-window16/stride8 real-frame policy")
        # Mask before projection: NaN/Inf padding cannot poison weight gradients.
        spatial = self.spatial_projector(spatial.masked_fill(~spatial_mask[..., None], 0))
        motion = self.motion_projector(motion.masked_fill(~motion_mask[..., None], 0))
        fused = []
        for i in range(spatial.shape[0]):
            s, m = spatial[i, spatial_mask[i]], motion[i, motion_mask[i]]
            source = torch.arange(s.shape[0], device=s.device)
            starts = torch.arange(m.shape[0], device=s.device) * 8
            ends = (starts + 15).clamp_max(s.shape[0] - 1)
            aligned, supported = interpolate_motion(m, source, starts, ends)
            gates = (torch.full_like(s[:, :1], 0.5) if self.gate is None
                     else self.gate(torch.cat((s, aligned), dim=-1)).sigmoid())
            correction = (gates * self.corrector(aligned)).masked_fill(~supported[:, None], 0)
            fused.append(torch.cat((s + correction, m), dim=0))
        lengths = frame_counts + motion_counts
        padded = pad_sequence(fused, batch_first=True)
        encoded, output_lengths = self.temporal_encoder(padded.transpose(1, 2), lengths)
        valid = lengths_to_mask(output_lengths, encoded.shape[1], encoded.device)
        return self._project_tokens(encoded, valid)


def enable_aligned_fusion(base: SpaMoVisualAdapter, arm: str, seed: int) -> SpaMoVisualAdapter:
    """Attach seeded modules, preserving common modules and CPU/all-CUDA RNG state."""
    if arm not in FUSION_ARMS:
        raise ValueError(f"unknown fusion arm: {arm}")
    if type(seed) is not int or seed < 0:
        raise ValueError("training seed must be a nonnegative integer")
    _validate_backbone(base)
    if arm == "reference":
        return base
    weight = base.motion_projector.weight
    with torch.random.fork_rng():
        torch.manual_seed(200 + seed)
        # Construct corrector first in both arms: gate draws cannot change its initialization.
        corrector = nn.Sequential(nn.Linear(768, 768), nn.GELU(), nn.Linear(768, 768))
        output_layer = corrector[-1]
        assert isinstance(output_layer, nn.Linear) and output_layer.bias is not None
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        corrector = corrector.to(device=weight.device, dtype=weight.dtype)
        gate = None
        if arm == "aligned_learned":
            gate = nn.Linear(1536, 1).to(device=weight.device, dtype=weight.dtype)
            nn.init.zeros_(gate.weight)
            nn.init.zeros_(gate.bias)
    return TimeAlignedFusionAdapter(base, arm, corrector, gate)


def fusion_frame_stats(
    supported: torch.Tensor,
    gates: torch.Tensor | None,
    corrections: torch.Tensor | None,
    *,
    arm: str,
) -> dict[str, int | float | None]:
    """Pure, JSON-ready sufficient statistics for Task3/4 aggregation.

    supported: bool [S] or [B,S], already excluding padding/uncovered frames.
    gates: floating point with exactly supported.shape (squeeze final singleton).
    corrections: floating point [*supported.shape,C], APPLIED gated corrections,
    i.e. gate * R(aligned_motion), not raw R. All tensors share a device.
    Only supported rows must be finite; gates there must be in [0,1].
    correction_norm sums per-frame L2 norms (not a batch Frobenius norm).
    Empty aligned sets have zero sums/counts/norm and null extrema; reference
    takes None/None and retains coverage with zero gate count and null gate/norm statistics.
    No forward caches, hooks, module/RNG mutation, or retained autograd graph.
    """
    if arm not in FUSION_ARMS:
        raise ValueError(f"unknown fusion arm: {arm}")
    if not isinstance(supported, torch.Tensor) or supported.dtype != torch.bool:
        raise ValueError("supported must be a boolean tensor")
    if supported.ndim not in (1, 2):
        raise ValueError("supported must have shape [S] or [B,S]")
    stats: dict[str, int | float | None] = {
        "supported_frame_count": supported.sum().item(), "gate_count": 0, "gate_sum": None,
        "gate_sumsq": None, "gate_min": None, "gate_max": None,
        "gate_below_0_05": None, "gate_above_0_95": None, "correction_norm": None,
    }
    if arm == "reference":
        if gates is not None or corrections is not None:
            raise ValueError("reference gate/correction statistics are not applicable")
        return stats
    if not isinstance(gates, torch.Tensor) or not isinstance(corrections, torch.Tensor):
        raise ValueError("aligned statistics require gates and applied corrections")
    if gates.shape != supported.shape or corrections.shape[:-1] != supported.shape:
        raise ValueError("gate/correction shapes must match supported frames")
    if corrections.shape[-1] == 0 or not gates.is_floating_point() \
            or not corrections.is_floating_point():
        raise ValueError("gates/corrections must be floating point with nonempty feature width")
    if gates.device != supported.device or corrections.device != supported.device:
        raise ValueError("statistics tensors must share a device")
    with torch.no_grad():
        g, c = gates[supported].double(), corrections[supported].double()
        if not torch.isfinite(g).all() or not torch.isfinite(c).all():
            raise ValueError("supported gate/correction values must be finite")
        if ((g < 0) | (g > 1)).any():
            raise ValueError("supported gates must be in [0,1]")
        count = g.numel()
        stats.update({
            "supported_frame_count": count, "gate_count": count,
            "gate_sum": g.sum().item(), "gate_sumsq": g.square().sum().item(),
            "gate_min": g.min().item() if count else None,
            "gate_max": g.max().item() if count else None,
            "gate_below_0_05": (g < 0.05).sum().item(),
            "gate_above_0_95": (g > 0.95).sum().item(),
            "correction_norm": torch.linalg.vector_norm(c, dim=-1).sum().item(),
        })
    return stats

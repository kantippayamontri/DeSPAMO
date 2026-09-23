import io

import pytest
import torch

from despamo.models.visual_adapter import SpaMoVisualAdapter


def test_visual_adapter_fuses_only_valid_tokens() -> None:
    adapter = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    spatial = torch.randn(2, 20, 2048)
    motion = torch.randn(2, 4, 1024)
    spatial_mask = torch.ones(2, 20, dtype=torch.bool)
    motion_mask = torch.ones(2, 4, dtype=torch.bool)
    spatial_mask[1, :4] = False
    motion_mask[1, 3] = False

    tokens, mask = adapter(spatial, spatial_mask, motion, motion_mask)

    assert tokens.shape == (2, 3, 2048)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, True], [True, False, False]]


def test_visual_adapter_ignores_masked_input_values() -> None:
    torch.manual_seed(2)
    adapter = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    spatial = torch.randn(1, 20, 4)
    motion = torch.randn(1, 4, 3)
    spatial_mask = torch.ones(1, 20, dtype=torch.bool)
    motion_mask = torch.ones(1, 4, dtype=torch.bool)
    spatial_mask[0, 2] = False
    motion_mask[0, 1] = False
    altered_spatial = spatial.clone()
    altered_motion = motion.clone()
    altered_spatial[~spatial_mask] = 10_000
    altered_motion[~motion_mask] = -10_000

    expected, expected_mask = adapter(spatial, spatial_mask, motion, motion_mask)
    actual, actual_mask = adapter(altered_spatial, spatial_mask, altered_motion, motion_mask)

    assert expected_mask.tolist() == [[True, True]]
    assert torch.equal(actual_mask, expected_mask)
    torch.testing.assert_close(actual, expected)


def test_visual_adapter_rejects_short_fused_sequences() -> None:
    adapter = SpaMoVisualAdapter(4, 3, 8, 6)
    spatial = torch.randn(2, 20, 4)
    motion = torch.randn(2, 4, 3)
    spatial_mask = torch.ones(2, 20, dtype=torch.bool)
    motion_mask = torch.ones(2, 4, dtype=torch.bool)
    spatial_mask[1, :9] = False  # 11 spatial + 4 motion = 15 fused tokens

    with pytest.raises(ValueError, match="at least 16 tokens"):
        adapter(spatial, spatial_mask, motion, motion_mask)


def test_visual_adapter_backpropagates_only_through_valid_inputs() -> None:
    adapter = SpaMoVisualAdapter(4, 3, 8, 6)
    spatial = torch.randn(1, 20, 4, requires_grad=True)
    motion = torch.randn(1, 4, 3, requires_grad=True)
    spatial_mask = torch.ones(1, 20, dtype=torch.bool)
    motion_mask = torch.ones(1, 4, dtype=torch.bool)
    spatial_mask[0, -1] = False
    motion_mask[0, -1] = False

    tokens, mask = adapter(spatial, spatial_mask, motion, motion_mask)
    tokens[mask].square().sum().backward()

    assert spatial.grad is not None and spatial.grad[spatial_mask].abs().sum() > 0
    assert motion.grad is not None and motion.grad[motion_mask].abs().sum() > 0
    assert torch.count_nonzero(spatial.grad[~spatial_mask]) == 0
    assert torch.count_nonzero(motion.grad[~motion_mask]) == 0
    for name in (
        "spatial_projector.weight",
        "motion_projector.weight",
        "temporal_encoder.temporal_conv.0.weight",
        "multimodal_projector.2.weight",
    ):
        grad = dict(adapter.named_parameters())[name].grad
        assert grad is not None and grad.abs().sum() > 0, name


def test_visual_adapter_checkpoint_round_trip() -> None:
    torch.manual_seed(3)
    source = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    spatial = torch.randn(1, 20, 2048)
    motion = torch.randn(1, 4, 1024)
    spatial_mask = torch.ones(1, 20, dtype=torch.bool)
    motion_mask = torch.ones(1, 4, dtype=torch.bool)
    expected, _ = source(spatial, spatial_mask, motion, motion_mask)
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)

    actual, _ = restored(spatial, spatial_mask, motion, motion_mask)

    torch.testing.assert_close(actual, expected)


def test_visual_adapter_preserves_shared_checkpoint_keys() -> None:
    adapter = SpaMoVisualAdapter(4, 3, 8, 6)
    temporal_keys = {
        f"temporal_encoder.temporal_conv.{index}.{name}"
        for index, names in (
            (0, ("weight", "bias")),
            (1, ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")),
            (4, ("weight", "bias")),
            (5, ("weight", "bias", "running_mean", "running_var", "num_batches_tracked")),
        )
        for name in names
    }
    projector_keys = {
        f"{module}.{param}"
        for module in (
            "spatial_projector",
            "motion_projector",
            "multimodal_projector.0",
            "multimodal_projector.2",
        )
        for param in ("weight", "bias")
    }

    assert set(adapter.state_dict()) == temporal_keys | projector_keys

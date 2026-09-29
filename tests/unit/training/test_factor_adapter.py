import torch

from despamo.models.factor_adapter import FactorVisualAdapter
from despamo.models.visual_adapter import SpaMoVisualAdapter


def test_spatial_tap_keeps_baseline_state_and_tokens():
    torch.manual_seed(7)
    base = SpaMoVisualAdapter(4, 2, 8, 6).eval()
    tapped = FactorVisualAdapter(4, 2, 8, 6).eval()
    tapped.load_state_dict(base.state_dict(), strict=True)
    spatial = torch.randn(2, 20, 4)
    spatial_mask = torch.tensor([[True] * 20, [True] * 17 + [False] * 3])
    motion = torch.randn(2, 3, 2)
    motion_mask = torch.ones(2, 3, dtype=torch.bool)
    args = (spatial, spatial_mask, motion, motion_mask)
    expected, expected_mask = base(*args)
    actual, actual_mask, projected = tapped.forward_with_spatial(*args)
    torch.testing.assert_close(actual, expected)
    assert torch.equal(actual_mask, expected_mask)
    torch.testing.assert_close(projected, tapped.spatial_projector(spatial))
    assert set(base.state_dict()) == set(tapped.state_dict())
    torch.testing.assert_close(tapped(*args)[0], expected)

import io

import pytest
import torch

from despamo.models.residual_projector import enable_residual_projectors
from despamo.models.visual_adapter import SpaMoVisualAdapter


def _inputs():
    spatial, motion = torch.randn(2, 20, 4), torch.randn(2, 8, 3)
    sm, mm = torch.ones(2, 20, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool)
    sm[1, -2:] = False
    mm[1, -1:] = False
    return spatial, sm, motion, mm


def test_residual_initialization_preserves_mapping_base_parameters_and_rng():
    torch.manual_seed(7)
    adapter = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    inputs = _inputs()
    expected, expected_mask = adapter(*inputs)
    spatial_base, motion_base = adapter.spatial_projector, adapter.motion_projector
    weights = [p.detach().clone() for p in adapter.parameters()]
    old_parameters = list(adapter.parameters())
    before_rng = torch.get_rng_state().clone()
    enable_residual_projectors(adapter, hidden_dim=12)
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert adapter.spatial_projector.base is spatial_base
    assert adapter.motion_projector.base is motion_base
    for before, param in zip(weights, old_parameters, strict=True):
        torch.testing.assert_close(param, before, rtol=0, atol=0)
    actual, actual_mask = adapter(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(actual_mask, expected_mask)


def test_residual_output_then_hidden_layers_receive_gradients_without_padding_leakage():
    torch.manual_seed(8)
    adapter = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    enable_residual_projectors(adapter, hidden_dim=12)
    spatial, sm, motion, mm = _inputs()
    spatial.requires_grad_()
    motion.requires_grad_()
    optimizer = torch.optim.SGD(adapter.parameters(), lr=0.1)
    for iteration in range(2):
        optimizer.zero_grad(set_to_none=True)
        spatial.grad = motion.grad = None
        tokens, mask = adapter(spatial, sm, motion, mm)
        tokens[mask].square().mean().backward()
        for projector in (adapter.spatial_projector, adapter.motion_projector):
            assert projector.refinement[3].weight.grad.abs().sum() > 0
            if iteration:
                assert projector.refinement[1].weight.grad.abs().sum() > 0
        assert torch.count_nonzero(spatial.grad[~sm]) == 0
        assert torch.count_nonzero(motion.grad[~mm]) == 0
        optimizer.step()
    altered_s, altered_m = spatial.detach().clone(), motion.detach().clone()
    altered_s[~sm], altered_m[~mm] = 10000, -10000
    actual, _ = adapter(altered_s, sm, altered_m, mm)
    expected, _ = adapter(spatial.detach(), sm, motion.detach(), mm)
    torch.testing.assert_close(actual, expected)


def test_residual_checkpoint_round_trip_is_strict():
    torch.manual_seed(9)
    source = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    enable_residual_projectors(source, hidden_dim=12)
    with torch.no_grad():
        source.spatial_projector.refinement[3].weight.fill_(0.01)
    inputs = _inputs()
    expected, _ = source(*inputs)
    saved = io.BytesIO()
    torch.save(source.state_dict(), saved)
    saved.seek(0)
    restored = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    enable_residual_projectors(restored, hidden_dim=12)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    actual, _ = restored(*inputs)
    torch.testing.assert_close(actual, expected)


def test_residual_attachment_rejects_duplicate_or_invalid_width():
    adapter = SpaMoVisualAdapter(4, 3, 8, 6)
    with pytest.raises(ValueError, match="width"):
        enable_residual_projectors(adapter, hidden_dim=0)
    enable_residual_projectors(adapter, hidden_dim=12)
    with pytest.raises(ValueError, match="already|linear"):
        enable_residual_projectors(adapter, hidden_dim=12)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA RNG control requires CUDA")
def test_residual_initialization_preserves_cuda_dropout_rng():
    adapter = SpaMoVisualAdapter(4, 3, 8, 6)
    torch.cuda.manual_seed_all(123)
    before = [state.clone() for state in torch.cuda.get_rng_state_all()]
    enable_residual_projectors(adapter, hidden_dim=12)
    assert all(torch.equal(old, new) for old, new in
               zip(before, torch.cuda.get_rng_state_all(), strict=True))

import copy
import io
import json
import math

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from despamo.models import connector_adapter as connectors
from despamo.models.visual_adapter import SpaMoVisualAdapter, lengths_to_mask


def _inputs():
    spatial, motion = torch.randn(2, 20, 4), torch.randn(2, 8, 3)
    sm, mm = torch.ones(2, 20, dtype=torch.bool), torch.ones(2, 8, dtype=torch.bool)
    sm[1, -2:] = False
    mm[1, -1:] = False
    return spatial, sm, motion, mm


def _connector():
    return connectors.MaskedQueryConnector(
        width=24, output_dim=16, num_queries=4, num_layers=2,
        num_heads=4, ff_dim=48, dropout=0.1,
    )


def test_reference_exact_original_forward_and_state_keys():
    adapter = SpaMoVisualAdapter(4, 3, 8, 6).eval()
    inputs = _inputs()
    spatial, sm, motion, mm = inputs
    s, m = adapter.spatial_projector(spatial), adapter.motion_projector(motion)
    fused = [torch.cat((s[i, sm[i]], m[i, mm[i]])) for i in range(2)]
    lengths = torch.tensor([len(x) for x in fused])
    encoded, out_lengths = adapter.temporal_encoder(
        pad_sequence(fused, batch_first=True).transpose(1, 2), lengths,
    )
    expected = adapter.multimodal_projector(encoded)
    keys = set(adapter.state_dict())
    assert connectors.enable_connector_arm(adapter, "reference", 0) is adapter
    actual, mask = adapter(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mask, lengths_to_mask(out_lengths, expected.shape[1], expected.device))
    assert set(adapter.state_dict()) == keys


@pytest.mark.parametrize("arm", ["plain_mlp", "qformer"])
def test_attachment_preserves_common_modules_mode_weights_and_rng(arm):
    base = SpaMoVisualAdapter(4, 3, 768 if arm == "qformer" else 8, 16).eval()
    state = {k: v.clone() for k, v in base.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    result = connectors.enable_connector_arm(base, arm, 1)
    assert isinstance(result, SpaMoVisualAdapter)
    assert not result.training
    assert all(not module.training for module in result.modules())
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(x, y) for x, y in
               zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    changed = "spatial_projector." if arm == "plain_mlp" else "multimodal_projector."
    for key, value in state.items():
        if not key.startswith(changed):
            torch.testing.assert_close(result.state_dict()[key], value, rtol=0, atol=0)
    assert result.motion_projector is base.motion_projector
    assert result.temporal_encoder is base.temporal_encoder
    if arm == "plain_mlp":
        assert result.multimodal_projector is base.multimodal_projector
    else:
        assert result.spatial_projector is base.spatial_projector


def test_plain_spatial_only_nonlinear_mapping_and_gradients():
    adapter = connectors.enable_connector_arm(SpaMoVisualAdapter(4, 3, 8, 6), "plain_mlp", 0)
    layers = list(adapter.spatial_projector.children())
    assert [type(x) for x in layers] == [nn.Linear, nn.GELU, nn.Linear]
    assert (layers[0].in_features, layers[0].out_features, layers[2].out_features) == (4, 1024, 8)
    x = torch.randn(2, 4)
    assert not torch.allclose(adapter.spatial_projector(x) + adapter.spatial_projector(-x),
                              2 * adapter.spatial_projector(torch.zeros_like(x)))
    tokens, mask = adapter(*_inputs())
    assert tokens.shape == (2, 4, 6)
    tokens[mask].square().mean().backward()
    for layer in (layers[0], layers[2], adapter.motion_projector):
        assert isinstance(layer, nn.Linear)
        assert layer.weight.grad is not None and layer.weight.grad.abs().sum() > 0


def test_query_padding_input_dependence_order_and_gradients():
    model = _connector().eval()
    memory = torch.randn(2, 6, 24, requires_grad=True)
    valid = torch.tensor([[True] * 6, [True] * 3 + [False] * 3])
    tokens, mask = model(memory, valid)
    assert tokens.shape == (2, 4, 16) and mask.all() and mask.dtype == torch.bool
    assert torch.isfinite(tokens).all()
    altered = memory.detach().clone()
    altered[~valid] = float("nan")
    torch.testing.assert_close(model(altered, valid)[0], tokens, rtol=0, atol=0)
    extended = torch.cat((memory.detach(), torch.full((2, 3, 24), float("inf"))), dim=1)
    extended_mask = torch.cat((valid, torch.zeros(2, 3, dtype=torch.bool)), dim=1)
    torch.testing.assert_close(model(extended, extended_mask)[0], tokens, atol=1e-6, rtol=1e-5)
    changed = memory.detach().clone()
    changed[valid] += torch.randn_like(changed[valid])
    assert not torch.allclose(model(changed, valid)[0], tokens)
    reversed_memory = memory.detach().clone()
    reversed_memory[0] = reversed_memory[0].flip(0)
    assert not torch.allclose(model(reversed_memory, valid)[0][0], tokens[0])
    tokens.square().mean().backward()
    assert memory.grad is not None
    assert torch.count_nonzero(memory.grad[~valid]) == 0
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
    for block in model.blocks:
        assert isinstance(block, connectors.QueryConnectorBlock)
        weight = block.cross_attention.in_proj_weight
        assert weight is not None and weight.grad is not None
        grad = weight.grad.view(3, 4, 6, 24)
        assert (grad.abs().sum(dim=(0, 2, 3)) > 0).all()


def test_positions_exact_sinusoidal():
    model = _connector()
    positions = model._position_encoding(3, torch.device("cpu"), torch.float32)
    for p in range(3):
        for i in range(12):
            angle = p / (10000 ** (2 * i / 24))
            assert positions[p, 2 * i].item() == pytest.approx(math.sin(angle), abs=1e-6)
            assert positions[p, 2 * i + 1].item() == pytest.approx(math.cos(angle), abs=1e-6)


@pytest.mark.parametrize("memory,valid", [
    (torch.randn(2, 3, 24), torch.zeros(2, 3, dtype=torch.bool)),
    (torch.randn(2, 3, 24), torch.tensor([[True] * 3, [False] * 3])),
    (torch.randn(2, 0, 24), torch.zeros(2, 0, dtype=torch.bool)),
    (torch.randn(0, 3, 24), torch.ones(0, 3, dtype=torch.bool)),
    (torch.randn(2, 3, 24), torch.ones(2, 3)),
    (torch.randn(2, 3, 24), torch.ones(2, 3, 1, dtype=torch.bool)),
    (torch.randn(2, 3, 24), torch.ones(1, 3, dtype=torch.bool)),
    (torch.randn(2, 24), torch.ones(2, 3, dtype=torch.bool)),
    (torch.randn(2, 3, 23), torch.ones(2, 3, dtype=torch.bool)),
    (torch.ones(2, 3, 24, dtype=torch.int64), torch.ones(2, 3, dtype=torch.bool)),
])
def test_invalid_memory_rejected_before_attention(memory, valid):
    with pytest.raises(ValueError):
        _connector()(memory, valid)


@pytest.mark.parametrize("arm", ["plain_mlp", "qformer"])
def test_strict_adapter_checkpoint_and_deterministic_seed(arm):
    width = 768 if arm == "qformer" else 8
    base = SpaMoVisualAdapter(4, 3, width, 16).eval()
    source = connectors.enable_connector_arm(base, arm, 0)
    restored = connectors.enable_connector_arm(copy.deepcopy(base), arm, 0)
    for key, value in source.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[key], rtol=0, atol=0)
    saved = io.BytesIO()
    torch.save(source.state_dict(), saved)
    saved.seek(0)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    inputs = _inputs()
    expected, mask = source(*inputs)
    actual, actual_mask = restored(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mask, actual_mask)
    if arm == "qformer":
        assert actual.shape == (2, 32, 16) and actual_mask.all()


def test_actual_query_constants_and_serializable_metadata():
    model = connectors.MaskedQueryConnector()
    tokens, valid = model(torch.randn(2, 5, 768), torch.ones(2, 5, dtype=torch.bool))
    assert tokens.shape == (2, 32, 2048) and valid.all() and torch.isfinite(tokens).all()
    assert model.queries.shape == (32, 768)
    assert len(model.blocks) == 2
    block = model.blocks[0]
    assert isinstance(block, connectors.QueryConnectorBlock)
    assert block.cross_attention.num_heads == 12
    hidden = block.feed_forward[0]
    assert isinstance(hidden, nn.Linear) and hidden.out_features == 3072
    assert model.output_projection.out_features == 2048
    assert set(connectors.ARM_CONFIGS) == {"reference", "plain_mlp", "qformer"}
    json.dumps(connectors.ARM_CONFIGS)
    with pytest.raises(ValueError):
        connectors.enable_connector_arm(SpaMoVisualAdapter(4, 3, 8, 6), "qformer", 0)
    with pytest.raises(ValueError):
        connectors.enable_connector_arm(SpaMoVisualAdapter(4, 3, 8, 6), "unknown", 0)


def test_query_adapter_gradient_path_spatial_motion_and_connector():
    adapter = connectors.enable_connector_arm(SpaMoVisualAdapter(4, 3, 768, 16), "qformer", 1)
    spatial, sm, motion, mm = _inputs()
    spatial.requires_grad_()
    motion.requires_grad_()
    tokens, valid = adapter(spatial, sm, motion, mm)
    tokens[valid].square().mean().backward()
    assert spatial.grad is not None and spatial.grad[sm].abs().sum() > 0
    assert motion.grad is not None and motion.grad[mm].abs().sum() > 0
    assert torch.count_nonzero(spatial.grad[~sm]) == 0
    assert torch.count_nonzero(motion.grad[~mm]) == 0
    for module in (adapter.spatial_projector, adapter.motion_projector,
                   adapter.temporal_encoder, adapter.multimodal_projector):
        gradients = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients if g is not None) > 0


def test_direct_dtype_and_seed_variation():
    connector = _connector().double().eval()
    tokens, _ = connector(torch.randn(1, 3, 24, dtype=torch.float64),
                          torch.ones(1, 3, dtype=torch.bool))
    assert tokens.dtype == torch.float64 and torch.isfinite(tokens).all()
    base = SpaMoVisualAdapter(4, 3, 8, 6)
    first = connectors.enable_connector_arm(base, "plain_mlp", 0)
    second = connectors.enable_connector_arm(base, "plain_mlp", 1)
    assert not torch.equal(next(first.spatial_projector.parameters()),
                           next(second.spatial_projector.parameters()))
    with pytest.raises(ValueError):
        connectors.enable_connector_arm(first, "plain_mlp", 0)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA BF16 smoke requires CUDA")
def test_bfloat16_cuda_single_batch():
    model = _connector().cuda()
    memory = torch.randn(1, 5, 24, device="cuda", requires_grad=True)
    valid = torch.tensor([[True, True, True, False, False]], device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, mask = model(memory, valid)
        loss = tokens.float().square().mean()
    loss.backward()
    assert tokens.dtype == torch.bfloat16 and mask.all() and torch.isfinite(tokens).all()
    assert memory.grad is not None
    assert torch.isfinite(memory.grad).all()
    assert torch.count_nonzero(memory.grad[~valid]) == 0

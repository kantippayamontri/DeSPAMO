"""Task2 contracts: time support, opt-in parity, gradients, and diagnostics."""

import copy
import importlib
import json

import pytest
import torch
from torch import nn

from despamo.models.connector_adapter import enable_connector_arm
from despamo.models.visual_adapter import SpaMoVisualAdapter


@pytest.fixture(scope="module")
def fusion():
    # Import inside the fixture so RED reports the missing module for each contract.
    return importlib.import_module("despamo.models.aligned_fusion")


@pytest.fixture(scope="module", autouse=True)
def small_cpu_thread_pool():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


def _base():
    return enable_connector_arm(
        SpaMoVisualAdapter(2048, 1024, 768, 2048), "plain_mlp", 0,
    ).eval()


def _inputs():
    spatial = torch.randn(2, 25, 2048)
    motion = torch.randn(2, 2, 1024)
    sm = torch.arange(25)[None] < torch.tensor([25, 17])[:, None]
    mm = torch.tensor([[True, True], [True, False]])
    return spatial, sm, motion, mm


def test_interpolation_at_frame10(fusion):
    values, valid = fusion.interpolate_motion(
        torch.tensor([[0.0], [8.0]]), torch.tensor([10, 24]),
        torch.tensor([0, 8]), torch.tensor([15, 23]),
    )
    torch.testing.assert_close(values, torch.tensor([[2.5], [0.0]]), rtol=0, atol=0)
    assert valid.tolist() == [True, False]
    assert valid.dtype == torch.bool


def test_centres_boundaries_and_original_sparse_indices(fusion):
    values, valid = fusion.interpolate_motion(
        torch.tensor([[0.0], [8.0]]), torch.tensor([0, 7.5, 15.5, 20, 23, 24]),
        torch.tensor([0, 8]), torch.tensor([15, 23]),
    )
    assert valid.tolist() == [True, True, True, True, True, False]
    assert values[:, 0].tolist() == [0, 0, 8, 8, 8, 0]
    values, valid = fusion.interpolate_motion(
        torch.tensor([[0.0], [8.0]]), torch.tensor([102, 110, 124]),
        torch.tensor([100, 108]), torch.tensor([115, 123]),
    )
    assert valid.tolist() == [True, True, False]
    assert values[:, 0].tolist() == [0, 2.5, 0]


def test_single_short_window_and_uncovered_gap(fusion):
    values, valid = fusion.interpolate_motion(
        torch.tensor([[3.0]]), torch.tensor([0, 4, 9, 10, 15]),
        torch.tensor([0]), torch.tensor([9]),
    )
    assert values[:, 0].tolist() == [3, 3, 3, 0, 0]
    assert valid.tolist() == [True, True, True, False, False]
    values, valid = fusion.interpolate_motion(
        torch.tensor([[2.0], [6.0]]), torch.tensor([0, 2, 3, 7, 8, 10, 11]),
        torch.tensor([0, 8]), torch.tensor([2, 10]),
    )
    assert valid.tolist() == [True, True, False, False, True, True, False]
    assert torch.count_nonzero(values[~valid]) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
def test_interpolation_gradient_and_feature_dtype(fusion, dtype):
    motion = torch.tensor([[0.0], [8.0]], dtype=dtype, requires_grad=True)
    values, _ = fusion.interpolate_motion(
        motion, torch.tensor([10, 24]), torch.tensor([0, 8]), torch.tensor([15, 23]),
    )
    assert values.dtype == dtype and values.device == motion.device
    values.sum().backward()
    torch.testing.assert_close(
        motion.grad, torch.tensor([[0.6875], [0.3125]], dtype=dtype), rtol=0, atol=0,
    )


@pytest.mark.parametrize("field,replacement", [
    (0, [[0.0], [8.0]]),
    (0, torch.ones(2)),
    (0, torch.ones(2, 1, dtype=torch.int64)),
    (0, torch.empty(0, 1)),
    (0, torch.empty(2, 0)),
    (0, torch.tensor([[float("nan")], [8.0]])),
    (1, torch.tensor([[10]])),
    (1, torch.tensor([True])),
    (1, torch.tensor([float("inf")])),
    (1, torch.tensor([-1])),
    (1, torch.tensor([10, 9])),
    (1, torch.tensor([10, 10])),
    (2, torch.tensor([0])),
    (2, torch.tensor([8, 0])),
    (2, torch.tensor([0, 0])),
    (3, torch.tensor([15, 7])),
    (3, torch.tensor([15, float("nan")])),
])
def test_invalid_interpolation_inputs_rejected(fusion, field, replacement):
    args = [torch.tensor([[0.0], [8.0]]), torch.tensor([10]),
            torch.tensor([0, 8]), torch.tensor([15, 23])]
    args[field] = replacement
    with pytest.raises((TypeError, ValueError)):
        fusion.interpolate_motion(*args)


@pytest.mark.parametrize("arm", ["aligned_fixed", "aligned_learned"])
@pytest.mark.parametrize("training", [False, True])
def test_zero_initialized_forward_matches_plain_mlp(fusion, arm, training):
    base = _base().train(training)
    adapter = fusion.enable_aligned_fusion(copy.deepcopy(base), arm, 0)
    inputs = _inputs()
    expected, expected_mask = base(*inputs)
    actual, mask = adapter(*inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(mask, expected_mask)
    assert adapter.training == training
    assert all(module.training == training for module in adapter.modules())


def test_mask_and_token_count_match_reference(fusion):
    base = _base()
    assert fusion.enable_aligned_fusion(base, "reference", 0) is base
    expected, valid = base(*_inputs())
    assert expected.shape == (2, 3, 2048)
    assert valid.tolist() == [[True, True, True], [True, False, False]]
    for arm in ("aligned_fixed", "aligned_learned"):
        adapter = fusion.enable_aligned_fusion(base, arm, 0)
        # A nonzero correction must retain the original motion stream and token policy.
        with torch.no_grad():
            adapter.corrector[-1].bias.fill_(2)
        spatial, sm, motion, mm = _inputs()
        captured = []
        handle = adapter.temporal_encoder.register_forward_pre_hook(
            lambda module, args, captured=captured: captured.append(args[0].transpose(1, 2)),
        )
        actual, mask = adapter(spatial, sm, motion, mm)
        handle.remove()
        assert actual.shape == expected.shape and torch.equal(mask, valid)
        s = base.spatial_projector(spatial[0])
        torch.testing.assert_close(captured[0][0, :24], s[:24] + 1)
        torch.testing.assert_close(captured[0][0, 24], s[24])
        torch.testing.assert_close(captured[0][0, 25:27], base.motion_projector(motion[0]))


def test_shared_rng_and_correction_match_fixed_learned(fusion):
    base = _base()
    common = {name: value.clone() for name, value in base.state_dict().items()}
    rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    fixed = fusion.enable_aligned_fusion(base, "aligned_fixed", 1)
    learned = fusion.enable_aligned_fusion(base, "aligned_learned", 1)
    assert torch.equal(rng, torch.get_rng_state())
    if cuda_rng:
        assert all(torch.equal(x, y) for x, y in
                   zip(cuda_rng, torch.cuda.get_rng_state_all(), strict=True))
    for adapter in (fixed, learned):
        for name, value in common.items():
            torch.testing.assert_close(adapter.state_dict()[name], value, rtol=0, atol=0)
        for name in ("spatial_projector", "motion_projector", "temporal_encoder",
                     "multimodal_projector"):
            assert getattr(adapter, name) is getattr(base, name)
    for x, y in zip(fixed.corrector.parameters(), learned.corrector.parameters(), strict=True):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    with torch.random.fork_rng():
        torch.manual_seed(201)  # Training seed 1 plus the specified offset 200.
        expected_first_layer = nn.Linear(768, 768)
    torch.testing.assert_close(fixed.corrector[0].weight, expected_first_layer.weight,
                               rtol=0, atol=0)
    assert torch.count_nonzero(learned.gate.weight) == 0
    assert torch.count_nonzero(learned.gate.bias) == 0
    other = fusion.enable_aligned_fusion(base, "aligned_fixed", 2)
    assert not torch.equal(other.corrector[0].weight, fixed.corrector[0].weight)
    container = nn.Module()
    container.visual_adapter = learned
    named = dict(container.named_parameters())
    assert all(name.startswith("visual_adapter.") for name in named)
    assert {"visual_adapter.gate.weight", "visual_adapter.corrector.2.weight"} <= named.keys()
    optimizer = torch.optim.AdamW(container.parameters(), lr=3e-4)
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == {
        id(p) for p in container.parameters()
    }


@pytest.mark.parametrize("arm", ["aligned_fixed", "aligned_learned"])
def test_padding_does_not_change_valid_outputs(fusion, arm):
    adapter = fusion.enable_aligned_fusion(_base(), arm, 0)
    with torch.no_grad():
        adapter.corrector[-1].bias.fill_(1)
    spatial, sm, motion, mm = _inputs()
    expected, mask = adapter(spatial, sm, motion, mm)
    spatial[~sm] = float("nan")
    motion[~mm] = float("inf")
    actual, actual_mask = adapter(spatial, sm, motion, mm)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    spatial = torch.cat((spatial, torch.full((2, 7, 2048), float("nan"))), dim=1)
    motion = torch.cat((motion, torch.full((2, 5, 1024), float("inf"))), dim=1)
    sm = torch.cat((sm, torch.zeros(2, 7, dtype=torch.bool)), dim=1)
    mm = torch.cat((mm, torch.zeros(2, 5, dtype=torch.bool)), dim=1)
    extended, extended_mask = adapter(spatial, sm, motion, mm)
    torch.testing.assert_close(extended[mask], expected[mask], rtol=0, atol=0)
    assert torch.equal(actual_mask, mask) and torch.equal(extended_mask, mask)


def test_within_clip_isolation_and_correction_motion_projector_gradient(fusion):
    adapter = fusion.enable_aligned_fusion(_base(), "aligned_learned", 0)
    with torch.no_grad():
        adapter.corrector[-1].weight.copy_(torch.eye(768))
    spatial, sm, motion, mm = _inputs()
    motion.requires_grad_()
    captured = []
    handle = adapter.temporal_encoder.register_forward_pre_hook(
        lambda module, args: captured.append(args[0].transpose(1, 2)),
    )
    original, _ = adapter(spatial, sm, motion, mm)
    # Loss uses enriched spatial stream only, not appended motion tokens.
    captured[0][0, :24].square().mean().backward()
    assert adapter.motion_projector.weight.grad.abs().sum() > 0
    motion_grad = motion.grad
    assert motion_grad is not None
    assert motion_grad[0].abs().sum() > 0 and torch.count_nonzero(motion_grad[1]) == 0
    assert adapter.gate.weight.grad.abs().sum() > 0
    assert adapter.corrector[0].weight.grad.abs().sum() > 0
    changed = motion.detach().clone()
    changed[1, mm[1]] += 100
    altered, _ = adapter(spatial, sm, changed, mm)
    handle.remove()
    torch.testing.assert_close(altered[0], original[0], rtol=0, atol=0)
    assert not torch.allclose(altered[1], original[1])


def test_zero_init_first_step_gradient_and_strict_checkpoint(fusion):
    adapter = fusion.enable_aligned_fusion(_base(), "aligned_learned", 0)
    tokens, mask = adapter(*_inputs())
    tokens[mask].square().mean().backward()
    assert adapter.corrector[-1].weight.grad.abs().sum() > 0
    assert torch.count_nonzero(adapter.corrector[0].weight.grad) == 0
    assert torch.count_nonzero(adapter.gate.weight.grad) == 0
    restored = fusion.enable_aligned_fusion(_base(), "aligned_learned", 5)
    restored.load_state_dict(adapter.state_dict(), strict=True)
    inputs = _inputs()
    torch.testing.assert_close(restored(*inputs)[0], adapter(*inputs)[0], rtol=0, atol=0)


@pytest.mark.parametrize("case", ["hole_spatial", "hole_motion", "empty_spatial",
                                 "empty_motion", "wrong_count", "mask_type", "mask_shape",
                                 "feature_type", "feature_width", "nan", "empty_batch"])
def test_invalid_masks_and_empty_sequences_rejected(fusion, case):
    adapter = fusion.enable_aligned_fusion(_base(), "aligned_fixed", 0)
    spatial, sm, motion, mm = _inputs()
    if case == "hole_spatial":
        sm[0, 1] = False
    elif case == "hole_motion":
        mm[0] = torch.tensor([False, True])
    elif case == "empty_spatial":
        sm[0] = False
    elif case == "empty_motion":
        mm[0] = False
    elif case == "wrong_count":
        mm[0, 1] = False
    elif case == "mask_type":
        sm = sm.float()
    elif case == "mask_shape":
        sm = sm[:, :, None]
    elif case == "feature_type":
        motion = motion.long()
    elif case == "feature_width":
        spatial = spatial[:, :, :-1]
    elif case == "nan":
        spatial[0, 0, 0] = float("nan")
    elif case == "empty_batch":
        spatial, sm, motion, mm = spatial[:0], sm[:0], motion[:0], mm[:0]
    with pytest.raises((TypeError, ValueError)):
        adapter(spatial, sm, motion, mm)


def test_short_runtime_keeps_real_frames_and_original_tcn_minimum(fusion):
    adapter = fusion.enable_aligned_fusion(_base(), "aligned_fixed", 0)
    # 15 real frames + one motion token is legal; 10 + one cannot enter original TCN.
    output, mask = adapter(torch.randn(1, 15, 2048), torch.ones(1, 15, dtype=torch.bool),
                           torch.randn(1, 1, 1024), torch.ones(1, 1, dtype=torch.bool))
    assert output.shape == (1, 1, 2048) and mask.all()
    with pytest.raises(ValueError, match="at least 16 tokens"):
        adapter(torch.randn(1, 10, 2048), torch.ones(1, 10, dtype=torch.bool),
                torch.randn(1, 1, 1024), torch.ones(1, 1, dtype=torch.bool))


def test_attachment_rejects_wrong_backbone_arm_or_seed(fusion):
    base = _base()
    for arm, seed in [("unknown", 0), ("aligned_fixed", -1), ("reference", True)]:
        with pytest.raises(ValueError):
            fusion.enable_aligned_fusion(base, arm, seed)
    with pytest.raises(ValueError):
        fusion.enable_aligned_fusion(SpaMoVisualAdapter(2048, 1024, 768, 2048),
                                    "aligned_fixed", 0)
    attached = fusion.enable_aligned_fusion(base, "aligned_fixed", 0)
    with pytest.raises(ValueError):
        fusion.enable_aligned_fusion(attached, "aligned_learned", 0)
    assert set(fusion.FUSION_ARMS) == {"reference", "aligned_fixed", "aligned_learned"}
    json.dumps(fusion.FUSION_CONFIGS)


def test_attachment_rejects_altered_temporal_backbone(fusion):
    base = _base()
    base.temporal_encoder.temporal_conv[0] = nn.Conv1d(768, 768, kernel_size=3)
    for arm in ("reference", "aligned_fixed"):
        with pytest.raises(ValueError):
            fusion.enable_aligned_fusion(base, arm, 0)


@pytest.mark.parametrize("frames,windows", [(16, 1), (23, 1), (24, 2), (31, 2), (32, 3)])
def test_runtime_window_count_stride_boundaries(fusion, frames, windows):
    adapter = fusion.enable_aligned_fusion(_base(), "aligned_fixed", 0)
    inputs = (torch.randn(1, frames, 2048), torch.ones(1, frames, dtype=torch.bool),
              torch.randn(1, windows, 1024), torch.ones(1, windows, dtype=torch.bool))
    expected, expected_mask = _base()(*inputs)
    output, mask = adapter(*inputs)
    assert output.shape == expected.shape and torch.equal(mask, expected_mask)
    with pytest.raises(ValueError, match="motion counts"):
        adapter(inputs[0], inputs[1], torch.randn(1, windows + 1, 1024),
                torch.ones(1, windows + 1, dtype=torch.bool))


def test_fusion_frame_stats_excludes_invalid_and_padding(fusion):
    supported = torch.tensor([[True, True, False], [True, False, False]])
    gates = torch.tensor([[0.0, 0.5, float("nan")], [1.0, float("inf"), -1]])
    correction = torch.tensor([[[3.0, 4], [0, 0], [float("nan"), 0]],
                               [[0, 2], [float("inf"), 0], [99, 99]]])
    stats = fusion.fusion_frame_stats(supported, gates, correction, arm="aligned_learned")
    assert stats == {"supported_frame_count": 3, "gate_count": 3, "gate_sum": 1.5,
                     "gate_sumsq": 1.25, "gate_min": 0.0, "gate_max": 1.0,
                     "gate_below_0_05": 1, "gate_above_0_95": 1, "correction_norm": 7.0}
    empty = fusion.fusion_frame_stats(torch.zeros_like(supported), gates, correction,
                                     arm="aligned_fixed")
    assert empty["gate_count"] == 0 and empty["gate_sum"] == 0
    assert empty["gate_min"] is None and empty["gate_max"] is None
    assert empty["correction_norm"] == 0
    reference = fusion.fusion_frame_stats(supported, None, None, arm="reference")
    assert reference["supported_frame_count"] == 3  # Coverage remains defined for reference.
    assert reference["gate_count"] == 0 and reference["gate_sum"] is None
    assert reference["correction_norm"] is None
    with pytest.raises(ValueError):
        fusion.fusion_frame_stats(supported, gates[:1], correction, arm="aligned_fixed")
    gates[0, 0] = float("nan")
    with pytest.raises(ValueError):
        fusion.fusion_frame_stats(supported, gates, correction, arm="aligned_learned")


def test_stats_are_pure_and_thresholds_are_strict(fusion):
    supported = torch.tensor([True, True, True, True, False])
    gates = torch.tensor([0.05, 0.95, 0.04, 0.96, float("nan")],
                         dtype=torch.float64, requires_grad=True)
    correction = torch.zeros(5, 768, requires_grad=True)
    rng = torch.get_rng_state().clone()
    before = gates.detach().clone()
    stats = fusion.fusion_frame_stats(supported, gates, correction, arm="aligned_learned")
    assert stats["gate_below_0_05"] == 1 and stats["gate_above_0_95"] == 1
    assert stats["gate_count"] == 4 and stats["correction_norm"] == 0
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(gates, before, rtol=0, atol=0, equal_nan=True)
    assert gates.grad is None and correction.grad is None
    assert stats == fusion.fusion_frame_stats(supported, gates, correction, arm="aligned_learned")
    assert all(value is None or isinstance(value, (float, int)) for value in stats.values())


def test_direct_double_attachment_and_padding_gradients(fusion):
    adapter = fusion.enable_aligned_fusion(_base().double(), "aligned_learned", 0)
    assert adapter.corrector[0].weight.dtype == torch.float64
    assert adapter.gate.weight.dtype == torch.float64
    spatial, sm, motion, mm = _inputs()
    spatial, motion = spatial.double().requires_grad_(), motion.double().requires_grad_()
    with torch.no_grad():
        spatial[~sm] = float("nan")
        motion[~mm] = float("inf")
    tokens, mask = adapter(spatial, sm, motion, mm)
    tokens[mask].square().mean().backward()
    assert tokens.dtype == torch.float64
    spatial_grad, motion_grad = spatial.grad, motion.grad
    assert spatial_grad is not None and motion_grad is not None
    assert torch.count_nonzero(spatial_grad[~sm]) == 0
    assert torch.count_nonzero(motion_grad[~mm]) == 0
    for parameter in adapter.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_bfloat16_interpolation_and_fusion(fusion):
    adapter = fusion.enable_aligned_fusion(_base().cuda(), "aligned_learned", 0)
    assert adapter.corrector[0].weight.device.type == "cuda"
    inputs = tuple(value.cuda() for value in _inputs())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        tokens, valid = adapter(*inputs)
        loss = tokens[valid].float().square().mean()
    loss.backward()
    assert tokens.dtype == torch.bfloat16 and torch.isfinite(tokens).all()
    assert adapter.corrector[-1].weight.grad.abs().sum() > 0
    motion = torch.tensor([[0.0], [8.0]], device="cuda", dtype=torch.bfloat16,
                          requires_grad=True)
    values, support = fusion.interpolate_motion(
        motion, torch.tensor([10, 24], device="cuda"),
        torch.tensor([0, 8], device="cuda"), torch.tensor([15, 23], device="cuda"),
    )
    assert values.device.type == "cuda" and support.tolist() == [True, False]
    assert values[:, 0].tolist() == [2.5, 0]
    values.sum().backward()
    motion_grad = motion.grad
    assert motion_grad is not None
    assert motion_grad[:, 0].tolist() == [0.6875, 0.3125]

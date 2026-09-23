import pytest
import torch
from torch import nn
from torch.nn import functional as F

from despamo.losses.vt_align import VTAlignLoss, pooled_sequence


def test_masked_mean_ignores_padding() -> None:
    features = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    mask = torch.tensor([[True, True, False]])

    pooled = pooled_sequence(features, mask, "masked_mean")

    torch.testing.assert_close(pooled, torch.tensor([[2.0, 0.0]]))


@pytest.mark.parametrize("invalid_value", [float("nan"), float("inf"), float("-inf")])
def test_masked_mean_ignores_nonfinite_padding_and_gradients(invalid_value: float) -> None:
    features = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [invalid_value, invalid_value]]])
    features.requires_grad_()
    mask = torch.tensor([[True, True, False]])

    pooled = pooled_sequence(features, mask, "masked_mean")
    torch.testing.assert_close(pooled, torch.tensor([[2.0, 3.0]]))
    pooled.sum().backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()
    assert torch.count_nonzero(features.grad[~mask]) == 0


def test_legacy_mean_includes_padding_regardless_of_mask() -> None:
    features = torch.tensor([[[1.0, 0.0], [3.0, 0.0], [100.0, 0.0]]])
    mask = torch.tensor([[False, False, False]])

    torch.testing.assert_close(pooled_sequence(features, mask, "legacy_mean"), features.mean(1))


def test_masked_mean_rejects_empty_sequence_in_any_sample() -> None:
    features = torch.ones(2, 3, 4)
    mask = torch.tensor([[True, False, False], [False, False, False]])

    with pytest.raises(ValueError, match="no valid tokens"):
        pooled_sequence(features, mask, "masked_mean")


@pytest.mark.parametrize(
    "mask",
    [
        torch.ones(2, dtype=torch.bool),
        torch.ones(2, 3, 1, dtype=torch.bool),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(2, 2, dtype=torch.bool),
    ],
)
def test_pooling_rejects_invalid_mask_dimensions(mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="mask shape"):
        pooled_sequence(torch.ones(2, 3, 4), mask, "legacy_mean")


@pytest.mark.parametrize("mode", ["legacy_mean", "masked_mean"])
@pytest.mark.parametrize("mask", [torch.tensor([[1, -1]]), torch.tensor([[1.0, 0.0]])])
def test_pooling_rejects_nonboolean_mask(mode: str, mask: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="boolean"):
        pooled_sequence(torch.ones(1, 2, 3), mask, mode)


@pytest.mark.parametrize("mode", ["legacy_mean", "masked_mean"])
@pytest.mark.parametrize("batch,time", [(0, 2), (2, 0)])
def test_pooling_rejects_empty_batch_or_time(mode: str, batch: int, time: int) -> None:
    with pytest.raises(ValueError, match="non-empty batch and time"):
        pooled_sequence(torch.ones(batch, time, 3), torch.ones(batch, time, dtype=torch.bool), mode)


def test_pooling_rejects_invalid_mode() -> None:
    with pytest.raises(ValueError, match="unsupported pooling mode"):
        pooled_sequence(torch.ones(1, 2, 3), torch.ones(1, 2, dtype=torch.bool), "sum")


def test_vt_align_is_symmetric() -> None:
    loss_fn = VTAlignLoss(initial_logit_scale=0.0)
    visual = torch.eye(2).unsqueeze(1)
    text = torch.eye(2).unsqueeze(1)
    mask = torch.ones(2, 1, dtype=torch.bool)

    loss = loss_fn(visual, mask, text, mask, "masked_mean")

    expected = F.cross_entropy(torch.eye(2), torch.arange(2))
    torch.testing.assert_close(loss, expected)


def test_legacy_loss_matches_spamo_global_mean_and_clip_formula() -> None:
    visual = torch.tensor(
        [[[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]], [[0.0, 1.0], [1.0, 2.0], [2.0, 0.0]]]
    )
    text = torch.tensor(
        [[[2.0, 1.0], [1.0, 3.0], [4.0, 0.0]], [[1.0, 3.0], [2.0, 0.0], [1.0, 1.0]]]
    )
    visual_mask = torch.tensor([[True, True, False], [True, False, False]])
    text_mask = torch.tensor([[True, False, False], [True, True, False]])
    loss_fn = VTAlignLoss(initial_logit_scale=0.7)

    logits = (
        F.normalize(text.mean(1), dim=-1)
        @ F.normalize(visual.mean(1), dim=-1).T
        * loss_fn.logit_scale.exp()
    )
    labels = torch.arange(2)
    reference_loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2

    torch.testing.assert_close(
        loss_fn(visual, visual_mask, text, text_mask, "legacy_mean"), reference_loss
    )


def test_masked_loss_ignores_padding_values() -> None:
    loss_fn = VTAlignLoss(initial_logit_scale=0.0)
    visual = torch.tensor([[[1.0, 0.0], [50.0, 50.0]], [[0.0, 1.0], [60.0, 60.0]]])
    text = torch.tensor([[[1.0, 0.0], [70.0, 70.0]], [[0.0, 1.0], [80.0, 80.0]]])
    mask = torch.tensor([[True, False], [True, False]])

    loss = loss_fn(visual, mask, text, mask, "masked_mean")

    torch.testing.assert_close(loss, F.cross_entropy(torch.eye(2), torch.arange(2)))


def test_masked_loss_ignores_nonfinite_padding() -> None:
    loss_fn = VTAlignLoss(initial_logit_scale=0.0)
    visual = torch.tensor([[[1.0, 0.0], [float("nan"), float("inf")]], [[0.0, 1.0], [0.0, 0.0]]])
    text = torch.tensor([[[1.0, 0.0], [float("-inf"), float("nan")]], [[0.0, 1.0], [0.0, 0.0]]])
    mask = torch.tensor([[True, False], [True, False]])

    loss = loss_fn(visual, mask, text, mask, "masked_mean")

    torch.testing.assert_close(loss, F.cross_entropy(torch.eye(2), torch.arange(2)))


@pytest.mark.parametrize("mode", ["legacy_mean", "masked_mean"])
def test_vt_align_has_finite_gradients(mode: str) -> None:
    loss_fn = VTAlignLoss(initial_logit_scale=0.0)
    visual = torch.tensor([[[1.0, 1.0], [2.0, 1.0]], [[1.0, 2.0], [2.0, 2.0]]], requires_grad=True)
    text = torch.tensor([[[2.0, 1.0], [3.0, 1.0]], [[1.0, 2.0], [1.0, 3.0]]], requires_grad=True)
    mask = torch.tensor([[True, False], [True, True]])

    loss_fn(visual, mask, text, mask, mode).backward()

    for grad in (visual.grad, text.grad, loss_fn.logit_scale.grad):
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0
    if mode == "masked_mean":
        assert torch.count_nonzero(visual.grad[~mask]) == 0
        assert torch.count_nonzero(text.grad[~mask]) == 0


def test_vt_align_checkpoint_key_is_vt_align_logit_scale() -> None:
    model = nn.ModuleDict({"vt_align": VTAlignLoss()})

    assert set(model.state_dict()) == {"vt_align.logit_scale"}
    torch.testing.assert_close(model.state_dict()["vt_align.logit_scale"], torch.tensor(2.6592))

from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.schema import FACTORS, STABLE
from despamo.models.factor_heads import FactorHeads, reverse, select_representations


@pytest.mark.parametrize("factor", FACTORS)
def test_gradient_reversal_routes_only_stable_factors(factor):
    x = torch.tensor([[1.0, 0.2], [0.3, 1.0]], requires_grad=True)
    weight = torch.eye(2, requires_grad=True)
    target = torch.tensor([[0.2, 1.0], [1.0, 0.3]])
    ordinary = ((x @ weight) - target).square().sum()
    gx, gw = torch.autograd.grad(ordinary, (x, weight))
    routed = reverse(x, 0.5) if factor in STABLE else x
    actual = ((routed @ weight) - target).square().sum()
    actual_x, actual_w = torch.autograd.grad(actual, (x, weight))
    torch.testing.assert_close(actual_x, gx * (-0.5 if factor in STABLE else 1))
    torch.testing.assert_close(actual_w, gw)


def test_seven_independent_heads_ignore_failed_clip_gradients():
    torch.manual_seed(3)
    projected = torch.randn(3, 11, 8, requires_grad=True)
    spatial_mask = torch.ones(3, 11, dtype=torch.bool)
    rows = torch.tensor([[1, 3, 5, 7, 9]] * 3)
    masks, vectors, labels = {}, {}, {}
    for factor in FACTORS:
        shape = (3,) if factor in STABLE else (3, 5)
        masks[factor] = torch.ones(shape, dtype=torch.bool)
        masks[factor][-1] = False
        vectors[factor] = torch.randn(*shape, 4)
        vectors[factor][-1] = 0
        labels[factor] = tuple(f"{i % 3}-{factor}" for i in range(int(masks[factor].numel())))
    batch = SimpleNamespace(rows=rows, masks=masks, vectors=vectors, labels=labels)
    heads = FactorHeads(8, 4, list(FACTORS))
    losses, counts = heads(projected, spatial_mask, batch, alpha=0.5)
    assert set(losses) == set(FACTORS)
    assert counts == {factor: (2 if factor in STABLE else 10) for factor in FACTORS}
    assert all(torch.isfinite(item) for item in losses.values())
    sum(losses.values()).backward()
    assert projected.grad is not None and projected.grad[:2].abs().sum() > 0
    assert projected.grad[2].abs().sum() == 0
    assert all(heads.heads[f].weight.grad is not None for f in FACTORS)
    assert len({id(heads.heads[f].weight) for f in FACTORS}) == 7


def test_skip_all_masked_and_non_distinct_target_labels():
    visual = torch.randn(3, 6, 8, requires_grad=True)
    mask = torch.ones(3, 6, dtype=torch.bool)
    batch = SimpleNamespace(
        rows=torch.tensor([[0, 1, 2, 3, 4]] * 3),
        masks={f: torch.zeros((3,) if f in STABLE else (3, 5), dtype=torch.bool) for f in FACTORS},
        vectors={f: torch.zeros((3, 4) if f in STABLE else (3, 5, 4)) for f in FACTORS},
        labels={f: tuple("same" for _ in range(3 if f in STABLE else 15)) for f in FACTORS},
    )
    losses, counts = FactorHeads(8, 4, list(FACTORS))(visual, mask, batch, alpha=0.5)
    assert all(loss.item() == 0 and torch.isfinite(loss) for loss in losses.values())
    assert all(count == 0 for count in counts.values())
    sum(losses.values()).backward()
    assert visual.grad is not None and not visual.grad.any()


def test_row_mapping_rejects_padding_or_empty_spatial():
    projected = torch.randn(2, 8, 4)
    mask = torch.tensor([[True] * 8, [True] * 3 + [False] * 5])
    with pytest.raises(ValueError, match="padding"):
        select_representations(projected, mask, torch.tensor([[0, 1], [2, 3]]))
    with pytest.raises(ValueError, match="empty"):
        select_representations(
            projected, torch.zeros_like(mask), torch.zeros((2, 2), dtype=torch.long)
        )

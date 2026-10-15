import pytest
import torch

from tools.spamo_repro.model import build_optimizer
from tools.spamo_repro.train import accumulated_update


def test_original_adamw_groups_exclude_frozen_base():
    model = torch.nn.Linear(2, 2)
    model.bias.requires_grad_(False)
    optimizer = build_optimizer(model)
    assert optimizer.param_groups[0]["params"] == [model.weight]
    assert optimizer.defaults["betas"] == (0.9, 0.98)
    assert optimizer.defaults["eps"] == 1e-8
    assert optimizer.defaults["weight_decay"] == 0.01
    assert optimizer.defaults["lr"] == 0.0


@pytest.mark.parametrize("update,expected", [(1, 1e-8), (10000, 1e-4), (17760, 5e-5)])
def test_actual_optimizer_update_uses_declared_schedule_endpoint(update, expected):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = build_optimizer(model)

    def losses(model, batch, joint):
        loss = model.weight.square().sum()
        return loss, loss if joint else None

    result = accumulated_update(model, optimizer, [None], update, 17760, losses)
    assert result["lr"] == pytest.approx(expected, rel=1e-10, abs=0.0)


def test_vt_stage_rejects_unexpected_translation_loss_before_step():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = build_optimizer(model)
    before = model.weight.detach().clone()

    def losses(model, batch, joint):
        loss = model.weight.square().sum()
        return loss, loss

    with pytest.raises(ValueError, match="phase"):
        accumulated_update(model, optimizer, [None], 4000, 17760, losses)
    assert torch.equal(model.weight, before)
    assert not optimizer.state


def test_joint_stage_requires_translation_loss_before_step():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = build_optimizer(model)

    def losses(model, batch, joint):
        return model.weight.square().sum(), None

    with pytest.raises(ValueError, match="phase"):
        accumulated_update(model, optimizer, [None], 4001, 17760, losses)
    assert not optimizer.state

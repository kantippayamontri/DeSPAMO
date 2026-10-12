import pytest
import torch
from torch import nn

from tools.benchmark.compact_checkpoint import frozen_fingerprint
from tools.clip_s2.guards import RecoverableFault, assert_base_intact, check_ce_collapse


def frozen_model():
    model = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 1))
    model[0].requires_grad_(False)
    return model


def test_unchanged_frozen_weights_pass():
    model = frozen_model()
    assert_base_intact(model, frozen_fingerprint(model))


def test_silent_in_memory_change_of_frozen_weight_is_recoverable_fault():
    model = frozen_model()
    base = frozen_fingerprint(model)
    with torch.no_grad():
        model[0].weight[0, 0] += 1e-3
    with pytest.raises(RecoverableFault, match="frozen base"):
        assert_base_intact(model, base)


def test_trainable_weight_changes_are_ignored():
    model = frozen_model()
    base = frozen_fingerprint(model)
    with torch.no_grad():
        model[1].weight += 1
    assert_base_intact(model, base)


def test_ce_collapse_detected_only_after_enough_joint_updates():
    recipe = dict(vt_warmup_steps=4000)
    healthy = [3.5] * 200
    check_ce_collapse(healthy, 5000, recipe)
    check_ce_collapse([9.7] * 50, 4050, recipe)
    with pytest.raises(RecoverableFault, match="collapse"):
        check_ce_collapse([9.7] * 200, 4300, recipe)


def test_ce_collapse_uses_recent_window_mean():
    recipe = dict(vt_warmup_steps=4000)
    check_ce_collapse([9.7] * 100 + [3.0] * 200, 5000, recipe)
    with pytest.raises(RecoverableFault):
        check_ce_collapse([3.0] * 100 + [9.7] * 200, 5000, recipe)

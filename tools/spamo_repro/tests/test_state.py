import random
from typing import Any, cast

import numpy as np
import pytest
import torch

from tools.benchmark.compact_checkpoint import frozen_fingerprint
from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.spamo_repro.model import build_optimizer
from tools.spamo_repro.state import capture_state, publish_state, restore_state


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)
        self.linear = torch.nn.Linear(3, 3)
        self.dropout = torch.nn.Dropout(0.2)
        self.rng = random.Random(0)


def make():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    model = Model()
    return (
        model,
        build_optimizer(model),
        OfficialBatchSampler(tuple(map(str, range(24))), batch_size=8),
    )


def update(model, optimizer, sampler):
    sampler.next_ids()
    optimizer.param_groups[0]["lr"] = 1e-4
    optimizer.zero_grad(set_to_none=True)
    model.rng.random()
    random.random()
    np.random.random()
    loss = model.dropout(model.linear(torch.randn(8, 3))).square().mean()
    loss.backward()
    optimizer.step()


def test_complete_resume_replays_dropout_optimizer_sampler_and_all_rng(tmp_path):
    model, optimizer, sampler = make()
    update(model, optimizer, sampler)
    state = capture_state(model, optimizer, {"experiment": "paper"}, 1, sampler.state_dict())
    receipt = publish_state(tmp_path / "control", tmp_path / "payload/state.pt", state)
    update(model, optimizer, sampler)
    expected_weights = {k: v.clone() for k, v in model.state_dict().items()}
    expected_rng: Any = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state(),
        model.rng.getstate(),
    )
    expected_sampler = sampler.state_dict()
    expected_optimizer = optimizer.state_dict()
    resumed, resumed_opt, resumed_sampler = make()
    saved = restore_state(
        receipt, resumed, resumed_opt, {"experiment": "paper"}, frozen_fingerprint(resumed)
    )
    resumed_sampler.load_state_dict(saved["sampler"])
    update(resumed, resumed_opt, resumed_sampler)
    for name, value in resumed.state_dict().items():
        assert torch.equal(value, expected_weights[name])
    for key, params in resumed_opt.state_dict()["state"].items():
        for name, value in params.items():
            assert torch.equal(value, expected_optimizer["state"][key][name])
    assert resumed_sampler.state_dict() == expected_sampler
    assert random.getstate() == expected_rng[0]
    assert np.array_equal(cast(Any, np.random.get_state())[1], expected_rng[1][1])
    assert torch.equal(torch.get_rng_state(), expected_rng[2])
    assert resumed.rng.getstate() == expected_rng[3]


def test_checkpoint_capture_detaches_optimizer_state_from_later_updates():
    model, optimizer, sampler = make()
    update(model, optimizer, sampler)
    state = capture_state(model, optimizer, {"experiment": "paper"}, 1, sampler.state_dict())
    update(model, optimizer, sampler)
    assert all(v["step"].item() == 1 for v in state["optimizer"]["state"].values())


def test_publication_is_immutable_and_receipt_hash_drift_rejected(tmp_path):
    model, optimizer, sampler = make()
    update(model, optimizer, sampler)
    state = capture_state(model, optimizer, {"experiment": "paper"}, 1, sampler.state_dict())
    path = tmp_path / "payload/state.pt"
    receipt = publish_state(tmp_path / "control", path, state)
    with pytest.raises(ValueError, match="exists"):
        publish_state(tmp_path / "control", path, state)
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="receipt"):
        restore_state(receipt, model, optimizer, {"experiment": "paper"}, frozen_fingerprint(model))


def test_restore_rejects_other_experiment_before_mutating_model(tmp_path):
    model, optimizer, sampler = make()
    state = capture_state(model, optimizer, {"experiment": "paper"}, 0, sampler.state_dict())
    receipt = publish_state(tmp_path / "control", tmp_path / "payload/state.pt", state)
    before = model.linear.weight.detach().clone()
    with pytest.raises(ValueError, match="identity"):
        restore_state(receipt, model, optimizer, {"experiment": "old"}, frozen_fingerprint(model))
    assert torch.equal(before, model.linear.weight)


def test_nonfinite_optimizer_state_never_published(tmp_path):
    model, optimizer, sampler = make()
    update(model, optimizer, sampler)
    state = capture_state(model, optimizer, {"experiment": "paper"}, 1, sampler.state_dict())
    next(iter(state["optimizer"]["state"].values()))["exp_avg"].fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        publish_state(tmp_path / "control", tmp_path / "payload/state.pt", state)
    assert not (tmp_path / "payload/state.pt").exists()

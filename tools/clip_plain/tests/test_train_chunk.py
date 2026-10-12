import random

import pytest
import torch


def setup_loop(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.clip_plain import train as t
    from tools.clip_plain.policy import POLICY

    class Sampler:
        def __init__(self):
            self.batches_yielded = 0

        def next_ids(self):
            self.batches_yielded += 1
            return ("train0",)

        def state_dict(self):
            return dict(batches_yielded=self.batches_yielded)

        def load_state_dict(self, x):
            self.batches_yielded = x["batches_yielded"]

    torch.manual_seed(5)
    m = torch.nn.Linear(2, 1)
    m.rng = random.Random(7)
    o = torch.optim.AdamW(m.parameters())
    batch = SimpleNamespace(
        spatial=torch.ones(1, 2),
        spatial_mask=torch.ones(1, 2, dtype=torch.bool),
        motion=torch.ones(1, 2),
        motion_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    monkeypatch.setattr(t, "batch_for", lambda *a: batch)

    def update(model, opt, batch, k, recipe, pool):
        opt.zero_grad()
        (model(batch.spatial).sum() * model.rng.random()).backward()
        opt.step()
        return dict(translation_loss=None)

    monkeypatch.setattr(t, "update", update)
    monkeypatch.setattr(t, "epoch_guard", lambda *a: None)
    recipe = dict(steps_per_epoch=1, total_steps=4, dev_epochs=[2, 4], vt_warmup_steps=4000)
    identity = dict(policy=POLICY, recipe=recipe, seed=0)
    return t, m, o, Sampler(), identity


def test_boundary_epochs():
    from tools.clip_plain.train import next_dev_epoch

    recipe = dict(steps_per_epoch=887, dev_epochs=[2, 4, 6])
    assert next_dev_epoch(0, recipe) == 2 and next_dev_epoch(887, recipe) == 2
    assert next_dev_epoch(1774, recipe) == 4


def test_train_exits_at_boundary_and_never_generates(tmp_path, monkeypatch):
    t, m, o, s, i = setup_loop(tmp_path, monkeypatch)
    r = t.train_to_boundary(
        m, o, s, {}, None, i, tmp_path, 2, "base", torch.device("cpu"), lambda e: None
    )
    assert r["completed_updates"] == 2 and r["status"] == "awaiting_dev"
    assert s.batches_yielded == 2 and (tmp_path / "resume-epoch-2.pt").exists()
    assert (tmp_path / "candidate-epoch-2.pt").exists()


def test_failed_update_does_not_save_or_cleanup(tmp_path, monkeypatch):
    t, m, o, s, i = setup_loop(tmp_path, monkeypatch)
    error = RuntimeError("device failure")

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(t, "update", fail)
    monkeypatch.setattr(t, "capture_state", lambda *a, **k: pytest.fail("save after fault"))
    with pytest.raises(RuntimeError) as got:
        t.train_to_boundary(
            m, o, s, {}, None, i, tmp_path, 2, "base", torch.device("cpu"), lambda e: None
        )
    assert got.value is error and not list(tmp_path.glob("*.pt"))


def test_missing_even_epoch_dev_blocks_next_update(tmp_path, monkeypatch):
    from tools.clip_plain.train import validate_training_boundary

    recipe = dict(steps_per_epoch=887, dev_epochs=[2, 4])
    with pytest.raises(ValueError, match="dev"):
        validate_training_boundary(tmp_path, 1774, recipe)


def test_worker_gate_before_model(tmp_path):
    from tools.clip_plain.tests.helpers import config_at
    from tools.clip_plain.train import run_train_chunk

    with pytest.raises(RuntimeError, match="approval"):
        run_train_chunk(config_at(tmp_path), dict(seed=0), None)

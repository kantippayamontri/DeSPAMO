import json
import random

import pytest
import torch
from torch import nn

from tools.benchmark.compact_checkpoint import capture_state, load_snapshot, save_snapshot
from tools.clip_s2.train import candidate_to_archive, load_config, verify_initialization


def test_initialization_mismatch_blocks_launch(tmp_path):
    p = tmp_path / "seed0.json"
    p.write_text(
        json.dumps(dict(seed=0, common_mutable_fingerprint="common", frozen_fingerprint="base"))
    )
    assert verify_initialization(p, 0, "common", "base")
    with pytest.raises(ValueError):
        verify_initialization(p, 0, "changed", "base")


def test_training_config_cannot_enable_test_scoring(tmp_path):
    p = tmp_path / "inputs.json"
    p.write_text(json.dumps(dict(test_scoring=True)))
    with pytest.raises(ValueError):
        load_config(p)


def test_candidate_cold_copy_and_receipt_are_verified(tmp_path, monkeypatch):
    import tools.clip_s2.train as t

    monkeypatch.setattr(t, "require_archive_storage", lambda *a: None)
    root, archive = tmp_path / "run", tmp_path / "archive"
    root.mkdir()
    state = dict(
        identity={"policy": "fixture"},
        step=1774,
        frozen_fingerprint="base",
        weights={"value": torch.ones(1)},
    )
    target = candidate_to_archive(root, archive, state, 2)
    assert target.is_file() and target.with_suffix(".json").is_file()
    assert candidate_to_archive(root, archive, state, 2) == target
    target.write_bytes(b"broken")
    with pytest.raises(ValueError):
        candidate_to_archive(root, archive, state, 2)


def test_full_snapshot_replay_preserves_next_update_and_prompt_rng(tmp_path):
    def make():
        torch.manual_seed(7)
        model = nn.Sequential(nn.Linear(2, 3), nn.BatchNorm1d(3), nn.Linear(3, 1))
        model.rng = random.Random(9)
        return model, torch.optim.AdamW(model.parameters(), lr=0.01)

    model, optimizer = make()

    def step(m, opt):
        ids = [m.rng.randrange(10) for _ in range(4)]
        x = torch.randn(4, 2)
        opt.zero_grad()
        m(x).square().mean().backward()
        opt.step()
        return ids

    step(model, optimizer)
    identity = {"policy": "fixture"}
    state = capture_state(model, optimizer, identity, 1, {"batches_yielded": 1})
    path = tmp_path / "resume.pt"
    save_snapshot(path, state)
    expected_ids = step(model, optimizer)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    fresh, fresh_optimizer = make()
    load_snapshot(path, fresh, fresh_optimizer, identity)
    assert step(fresh, fresh_optimizer) == expected_ids
    for key, value in fresh.state_dict().items():
        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)

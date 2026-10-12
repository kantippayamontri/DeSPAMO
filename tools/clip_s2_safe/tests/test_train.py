import json
from pathlib import Path

import pytest
import torch

from despamo.data.batch import PhoenixBatch
from tools.clip_s2_safe.tests.test_state import binding_fixture


def setup(tmp_path, monkeypatch):
    from tools.clip_s2_safe import train as t

    s, c, b, m, o, sampler = binding_fixture(tmp_path)
    state = s.load_weights(Path(c["anchor"]), m, b, base_hash="frozen")
    s.restore_training(state, m, o, sampler, device=torch.device("cpu"))
    batch = PhoenixBatch(
        ("train0",),
        ("Signer01",),
        ("text",),
        ("G",),
        ("en",),
        ("es",),
        ("fr",),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
        torch.ones(1, 2),
        torch.ones(1, 2, dtype=torch.bool),
    )
    monkeypatch.setattr(t, "batch_for", lambda *a: batch)
    monkeypatch.setattr(t, "check_guard", lambda *a: None)

    def update(model, opt, batch, k, recipe, pool):
        opt.zero_grad()
        (model(batch.spatial).sum() * model.rng.random()).backward()
        opt.step()
        return dict(translation_loss=0.1 if k > recipe["vt_warmup_steps"] else None)

    monkeypatch.setattr(t, "update", update)
    job = dict(
        mode="production",
        stage="train",
        seed=0,
        start_step=2,
        target_step=4,
        binding_hash=t.digest(b),
    )
    return t, c, b, m, o, sampler, state["safe_guard"], job


def test_original_absolute_lr_and_boundary():
    from tools.clip_s2_safe.train import next_boundary
    from tools.followup.recipe import group_lr, schedule

    r = schedule(8)
    assert next_boundary(1774, r) == 3548
    assert group_lr(1775, r, "adapter") == pytest.approx(3e-4 * 1775 / 10000)
    assert group_lr(4001, r, "flan") == pytest.approx(5e-5 * 4001 / 10000)


def test_chunk_keeps_original_identity_and_publishes_handoff(tmp_path, monkeypatch):
    t, c, b, m, o, s, g, j = setup(tmp_path, monkeypatch)
    events = []
    r = t.train_chunk(
        m, o, s, {}, None, b, g, j, device=torch.device("cpu"), on_progress=events.append
    )
    assert (
        r["status"] == "awaiting_dev" and r["completed_updates"] == 4 and r["durable_updates"] == 4
    )
    receipt = json.loads(Path(r["checkpoint"]).with_suffix(".json").read_text())
    assert receipt["identity"] == b["original_identity"]
    assert s.batches_yielded == 4 and events[-1]["completed_updates"] == 4


def test_primary_update_fault_never_saves_after(tmp_path, monkeypatch):
    t, c, b, m, o, s, g, j = setup(tmp_path, monkeypatch)
    error = RuntimeError("primary")

    def fail(*a, **k):
        raise error

    monkeypatch.setattr(t, "update", fail)
    monkeypatch.setattr(t, "capture_state", lambda *a, **k: pytest.fail("capture after fault"))
    with pytest.raises(RuntimeError) as got:
        t.train_chunk(
            m, o, s, {}, None, b, g, j, device=torch.device("cpu"), on_progress=lambda e: None
        )
    assert got.value is error and not list(Path(c["runtime_root"]).glob("checkpoints/**/*.pt"))


def test_ce_guard_history_restored_across_chunks(tmp_path, monkeypatch):
    t, c, b, m, o, s, g, j = setup(tmp_path, monkeypatch)
    r = t.train_chunk(
        m, o, s, {}, None, b, g, j, device=torch.device("cpu"), on_progress=lambda e: None
    )
    from tools.clip_s2_safe.state import load_weights

    state = load_weights(Path(r["checkpoint"]), m, b, base_hash="frozen")
    j.update(start_step=4, target_step=6)
    r = t.train_chunk(
        m,
        o,
        s,
        {},
        None,
        b,
        state["safe_guard"],
        j,
        device=torch.device("cpu"),
        on_progress=lambda e: None,
    )
    guard = json.loads(Path(r["checkpoint"]).with_suffix(".guard.json").read_text())
    assert guard["last_update"] == 6 and guard["ce_values"] == [0.1, 0.1]


def test_probe_target_is_bounded_and_provisional(tmp_path, monkeypatch):
    t, c, b, m, o, s, g, j = setup(tmp_path, monkeypatch)
    j.update(mode="handoff_probe", target_step=3)
    r = t.train_chunk(
        m, o, s, {}, None, b, g, j, device=torch.device("cpu"), on_progress=lambda e: None
    )
    assert r["diagnostic_only"] and "/handoff_probe/" in r["checkpoint"]


def test_scope_before_constructor(tmp_path):
    from tools.clip_s2_safe.tests.helpers import original_fixture
    from tools.clip_s2_safe.train import run_train_job

    c, *_ = original_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="scope"):
        run_train_job(c, {}, None)

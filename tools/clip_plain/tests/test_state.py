import json
import random
from pathlib import Path

import pytest
import torch


def tiny():
    model = torch.nn.Linear(2, 1)
    model.rng = random.Random(5)
    model.register_buffer("persistent", torch.tensor([3.0]))
    opt = torch.optim.AdamW(model.parameters())
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()
    return model, opt


def identity():
    from tools.clip_plain.policy import POLICY

    return dict(
        policy=POLICY, seed=0, recipe=dict(steps_per_epoch=1), evaluation_policy_hash="eval"
    )


def test_plain_roundtrip_restores_all(tmp_path):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_plain import state as s

    model, opt = tiny()
    i = identity()
    sampler = {"batches_yielded": 1}
    snapshot = capture_state(model, opt, i, 1, sampler, base_hash="frozen")
    s.publish_epoch(tmp_path, snapshot, 1, dev_candidate=True)
    fresh, other = tiny()
    loaded = s.load_weights(tmp_path / "resume-epoch-1.pt", fresh, i, "frozen")

    class Sampler:
        def load_state_dict(self, value):
            self.value = value

    st = Sampler()
    s.restore_training(loaded, fresh, other, st)
    assert st.value == sampler and fresh.rng.getstate() == model.rng.getstate()
    assert torch.equal(fresh.persistent, model.persistent)
    for k, v in model.state_dict().items():
        assert torch.equal(fresh.state_dict()[k], v)
    for k, v in opt.state_dict()["state"].items():
        for name, value in v.items():
            assert torch.equal(value, other.state_dict()["state"][k][name])


def test_s2_epoch2_checkpoint_rejected_without_mutation(tmp_path):
    from tools.benchmark.compact_checkpoint import capture_state, save_snapshot
    from tools.clip_plain import state as s

    m, o = tiny()
    i = identity()
    i["policy"] = "old-s2"
    snap = capture_state(m, o, i, 1, {}, base_hash="frozen")
    p = tmp_path / "old.pt"
    save_snapshot(p, snap)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    with pytest.raises(ValueError):
        s.load_weights(p, m, i, "frozen")
    for k, v in before.items():
        assert torch.equal(m.state_dict()[k], v)


def test_phase_reconciliation_charges_once(tmp_path, monkeypatch):
    from tools.clip_plain import state as s

    now = [100.0]
    monkeypatch.setattr(s.time, "time", lambda: now[0])
    i = identity()
    for index, duration in enumerate([10, 3, 7]):
        attempt = dict(token=f"a{index}", stage="train")
        s.begin_stage(tmp_path, i, attempt)
        now[0] += duration
        s.record_stage(
            tmp_path,
            attempt,
            dict(completed_updates=index + 1, durable_updates=index + 1),
            final_status="complete",
        )
        s.record_stage(
            tmp_path,
            attempt,
            dict(completed_updates=index + 1, durable_updates=index + 1),
            final_status="complete",
        )
    b = json.loads((tmp_path / "budget.json").read_text())
    assert b["charged_seconds"] == 20


def test_abandoned_stage_needs_review_and_keeps_elapsed_floor(tmp_path, monkeypatch):
    from tools.clip_plain import state as s

    now = [100.0]
    monkeypatch.setattr(s.time, "time", lambda: now[0])
    s.begin_stage(tmp_path, identity(), dict(token="dead", stage="dev"))
    b = json.loads((tmp_path / "budget.json").read_text())
    b["owner"]["boot_id"] = "dead"
    s.atomic_json(tmp_path / "budget.json", b)
    now[0] = 107.0
    with pytest.raises(RuntimeError):
        s.reconcile_abandoned_stage(tmp_path, reviewed=False)
    s.reconcile_abandoned_stage(tmp_path, reviewed=True)
    assert json.loads((tmp_path / "budget.json").read_text())["charged_seconds"] == 7


def test_backwards_wall_clock_preserves_budget_and_blocks_reconciliation(tmp_path, monkeypatch):
    from tools.clip_plain import state as s

    now = [100.0]
    monkeypatch.setattr(s.time, "time", lambda: now[0])
    s.begin_stage(tmp_path, identity(), dict(token="x", stage="train"))
    before = (tmp_path / "budget.json").read_bytes()
    now[0] = 90
    with pytest.raises(ValueError):
        s.record_stage(tmp_path, dict(token="x"), dict(completed_updates=0, durable_updates=0))
    assert (tmp_path / "budget.json").read_bytes() == before


def test_owned_pruning_requires_permission_and_keeps_latest_two(tmp_path):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_plain import state as s

    m, o = tiny()
    i = identity()
    for n in [1, 2, 3]:
        s.publish_epoch(
            tmp_path, capture_state(m, o, i, n, {}, base_hash="f"), n, dev_candidate=True
        )
    assert s.prune_owned_resumes(tmp_path, i, allowed=False) == []
    assert s.prune_owned_resumes(tmp_path, i, allowed=True) == ["resume-epoch-1.pt"]
    assert len(list(tmp_path.glob("candidate-epoch-*.pt"))) == 3


def test_reviewed_replay_keeps_attempt_highwater_and_charges(tmp_path, monkeypatch):
    from tools.clip_plain import state as s

    now = [100.0]
    monkeypatch.setattr(s.time, "time", lambda: now[0])
    a = dict(token="partial", stage="train")
    s.begin_stage(tmp_path, identity(), a)
    now[0] = 110
    s.record_stage(tmp_path, a, dict(completed_updates=3, durable_updates=2), final_status="failed")
    b = json.loads((tmp_path / "budget.json").read_text())
    b["owner"]["boot_id"] = "old"
    s.atomic_json(tmp_path / "budget.json", b)
    fixed = s.reconcile_abandoned_stage(tmp_path, reviewed=True)
    assert fixed["completed_updates"] == 2 and fixed["charged_seconds"] == 10
    assert fixed["replay_history"][-1]["completed_before"] == 3
    s.begin_stage(tmp_path, identity(), dict(token="manual", stage="train"))
    s.record_stage(
        tmp_path,
        dict(token="manual"),
        dict(completed_updates=3, durable_updates=3),
        final_status="complete",
    )


def test_historical_cost_included_once(tmp_path):
    from tools.clip_plain.state import cost_summary
    from tools.clip_plain.tests.helpers import config_at
    from tools.dinov3.storage import atomic_json

    c = config_at(tmp_path)
    root = Path(c["study_root"])
    atomic_json(
        root / "lineage.json",
        dict(historical_charge_seconds=497.5533489369991, historical_updates=1774),
    )
    atomic_json(root / "runs/seed0/budget.json", dict(charged_seconds=20.0))
    atomic_json(Path(c["feature_base"]) / "preparation-budget.json", dict(charged_seconds=3.0))
    result = cost_summary(c)
    assert result["total_charged_seconds"] == 520.5533489369991
    assert result == cost_summary(c)


def test_identity_binds_actual_train_only_exemplar(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from tools.clip_plain import state as s
    from tools.clip_plain.tests.helpers import config_at
    from tools.followup.recipe import PromptPool

    c = config_at(tmp_path)
    m, _ = tiny()
    rows = [
        dict(fileid=f"train{i}", text=f"german{i}", en_text="en", fr_text="fr", es_text="es")
        for i in range(4)
    ]
    a = dict(identity={"data": "hash"}, datasets={"train": SimpleNamespace(records=rows)})
    monkeypatch.setattr(s, "base_files", lambda *a: {})
    result = s.make_identity(c, a, 0, m, "frozen")
    assert result["decoder"]["exemplar_id"] == PromptPool(rows, 0).eval_id

import random
from types import SimpleNamespace

import pytest
from torch import nn

from tools.clip_s2.guards import RecoverableFault


def test_collapsed_dev_report_is_never_published(tmp_path, monkeypatch):
    import tools.clip_s2.train as t

    monkeypatch.setattr(t, "capture_state", lambda *a, **k: {})
    monkeypatch.setattr(t, "candidate_to_archive", lambda *a, **k: tmp_path / "candidate.pt")
    monkeypatch.setattr(t, "eval_batches", lambda *a, **k: [])
    monkeypatch.setattr(
        t,
        "score_batches",
        lambda *a, **k: dict(items=[dict(prediction="sss")] * 5, metrics=dict(bleu4=0.0)),
    )
    model = nn.Linear(1, 1)
    model.rng = random.Random(0)
    with pytest.raises(RecoverableFault):
        t.score_dev(
            model,
            None,
            {"ids": {"dev": ()}},
            None,
            {"data": {"ordered_ids": {"dev": "h"}}},
            tmp_path,
            6,
            "base",
            SimpleNamespace(state_dict=lambda: {}),
        )
    assert not (tmp_path / "dev-epoch-6.json").exists()


def test_epoch_guard_checks_base_and_ce_history():
    import torch

    from tools.benchmark.compact_checkpoint import frozen_fingerprint
    from tools.clip_s2.train import epoch_guard

    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 1))
    model[0].requires_grad_(False)
    base = frozen_fingerprint(model)
    recipe = dict(vt_warmup_steps=4000)
    epoch_guard(model, base, [3.0] * 300, 5000, recipe)
    with pytest.raises(RecoverableFault):
        epoch_guard(model, base, [9.6] * 300, 5000, recipe)
    with torch.no_grad():
        model[0].bias += 1
    with pytest.raises(RecoverableFault):
        epoch_guard(model, base, [3.0] * 300, 5000, recipe)


def test_new_line_accepts_predecessor_stop_superseded_by_first_clip_attempt():
    import tools.clip_s2.train as t
    from tools.clip_s2.study import FIRST_ATTEMPT, supersedes

    assert supersedes(str(FIRST_ATTEMPT))
    assert supersedes(str(t.BASE))
    assert not supersedes("/somewhere/else")
    assert t.BASE != FIRST_ATTEMPT

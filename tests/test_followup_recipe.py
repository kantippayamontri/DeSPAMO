import random

import pytest

from tools.followup.recipe import PromptPool, group_lr, schedule, stop_state


def report(epoch, score):
    return {"epoch": epoch, "metrics": {"bleu4": score}}


def test_schedule_and_lr_boundaries():
    r = schedule(8)
    assert (r["epochs"], r["steps_per_epoch"], r["total_steps"]) == (200, 887, 177400)
    assert (r["vt_warmup_steps"], r["lr_warmup_steps"]) == (4000, 10000)
    assert group_lr(1, r, "adapter") == pytest.approx(3e-8)
    assert group_lr(10000, r, "flan") == pytest.approx(5e-5)
    assert group_lr(177400, r, "adapter") == pytest.approx(5e-5)
    assert group_lr(177400, r, "flan") == pytest.approx(1e-5)
    with pytest.raises(ValueError):
        group_lr(177401, r, "adapter")


def test_stop_ignores_vt_checks_and_keeps_actual_best():
    rows = [report(2, 100), report(4, 100)]
    rows += [report(e, 10 + (e - 6) * 0.002) for e in range(6, 41, 2)]
    s = stop_state(rows, schedule(8))
    assert s["stop"] and s["reason"] == "patience"
    assert s["anchor_epoch"] == 6 and s["bad_checks"] == 17
    assert s["best_epoch"] == 2
    assert not stop_state(rows[:-1], schedule(8))["stop"]


def test_cumulative_small_gains_reset_anchor_and_tenth_check_gain_prevents_stop():
    rows = [report(e, 0 if e < 6 else 10) for e in range(2, 61, 2)]
    # Significant improvements keep early stopping from triggering before60.
    for r in rows:
        if r["epoch"] >= 6:
            r["metrics"]["bleu4"] = 10 + r["epoch"] * 0.1
    rows += [report(e, 16 + 0.01 * (e - 60) / 2) for e in range(62, 81, 2)]
    s = stop_state(rows, schedule(8))
    assert not s["stop"] and s["anchor_epoch"] == 80 and s["bad_checks"] == 0
    assert s["best_epoch"] == 80


def test_reports_must_be_contiguous_and_finite():
    with pytest.raises(ValueError):
        stop_state([report(4, 1)], schedule(8))
    with pytest.raises(ValueError):
        stop_state([report(2, float("nan"))], schedule(8))


def records():
    return [
        dict(
            fileid=str(i),
            text=f"german{i}",
            en_text=f"english{i}",
            fr_text=f"french{i}",
            es_text=f"spanish{i}",
        )
        for i in range(8)
    ]


def test_prompt_pool_reference_disjoint_and_rng_replay():
    pool = PromptPool(records(), 0)
    rng = random.Random(9)
    state = rng.getstate()
    prompts, ids = pool.training(("0", "1"), rng)
    assert all(i not in ("0", "1") for i in ids)
    assert all(p.count("=") == 3 for p in prompts)
    rng.setstate(state)
    assert pool.training(("0", "1"), rng) == (prompts, ids)


def test_inference_is_fixed_and_overlap_rejected_not_reselected():
    pool = PromptPool(records(), 1)
    prompts, ids = pool.inference(("dev1", "test1"))
    assert prompts[0] == prompts[1] and ids[0] == ids[1]
    assert pool.inference(("other",))[0][0] == prompts[0]
    with pytest.raises(ValueError, match="overlap"):
        pool.guard_references((pool.by_id[ids[0]]["text"],))


def test_duplicate_training_references_are_excluded():
    rows = records()
    rows[2]["text"] = "  GERMAN0 "
    pool = PromptPool(rows, 0)
    for seed in range(30):
        assert pool.training(("0",), random.Random(seed))[1][0] not in ("0", "2")


def test_update_switches_at4001_with_legacy_pooling_and_train_icl():
    from types import SimpleNamespace

    import torch
    from torch import nn

    from despamo.losses.vt_align import VTAlignLoss
    from tools.benchmark.translation_model import optimizer_for
    from tools.followup.model import update

    class Visual(nn.Linear):
        def forward(self, spatial, mask, motion, motion_mask):
            return super().forward(spatial), mask

    class Language(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(4))
            self.prompts = None

        def target_embeddings(self, texts):
            return torch.arange(24).reshape(2, 3, 4).float(), torch.ones(2, 3, dtype=torch.bool)

        def translation_loss(self, visual, mask, prompts, targets):
            self.prompts = prompts
            return (visual.mean(1) * self.weight).square().mean()

    class Alignment(VTAlignLoss):
        def forward(self, *args):
            self.mode = args[-1]
            return super().forward(*args)

    model = nn.Module()
    model.visual_adapter = Visual(3, 4)
    model.language_model = Language()
    model.vt_align = Alignment()
    model.rng = random.Random(0)
    batch = SimpleNamespace(
        spatial=torch.randn(2, 5, 3),
        spatial_mask=torch.ones(2, 5, dtype=torch.bool),
        motion=torch.randn(2, 1, 2),
        motion_mask=torch.ones(2, 1, dtype=torch.bool),
        texts=("german0", "german1"),
        clip_ids=("0", "1"),
    )
    optimizer = optimizer_for(model)
    before = model.language_model.weight.detach().clone()
    assert (
        update(model, optimizer, batch, 4000, schedule(8), PromptPool(records(), 0))[
            "translation_loss"
        ]
        is None
    )
    assert torch.equal(before, model.language_model.weight)
    result = update(model, optimizer, batch, 4001, schedule(8), PromptPool(records(), 0))
    assert result["translation_loss"] is not None and model.vt_align.mode == "legacy_mean"
    assert all(p.count("=") == 3 for p in model.language_model.prompts)
    assert not torch.equal(before, model.language_model.weight)

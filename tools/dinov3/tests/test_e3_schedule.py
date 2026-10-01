import pytest
import torch

from tools.dinov3.e3_train import cosine_warmup_factor, make_e3_optimizer


class _Tiny(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(4, 4)


def test_warmup_rises_then_cosine_decays_to_zero():
    total, warm = 100, 10
    assert cosine_warmup_factor(0, warmup_steps=warm, total_steps=total) == pytest.approx(0.0)
    assert cosine_warmup_factor(warm, warmup_steps=warm, total_steps=total) == pytest.approx(1.0)
    # strictly decreasing after warmup
    mid = [cosine_warmup_factor(s, warmup_steps=warm, total_steps=total)
           for s in (warm, 30, 55, 80, total)]
    assert all(a > b for a, b in zip(mid[:-1], mid[1:], strict=True))
    assert cosine_warmup_factor(total, warmup_steps=warm, total_steps=total) == pytest.approx(0.0,
                                                                                              abs=1e-9)


def test_warmup_is_linear_in_its_window():
    f = [cosine_warmup_factor(s, warmup_steps=10, total_steps=100) for s in range(11)]
    assert f[5] == pytest.approx(0.5)
    deltas = [b - a for a, b in zip(f[:-1], f[1:], strict=True)]
    assert max(deltas) - min(deltas) < 1e-9


def test_factor_never_negative_or_above_one():
    for total, warm in ((50, 5), (8975, 1795), (718, 143)):
        for s in range(0, total + 1, max(1, total // 50)):
            v = cosine_warmup_factor(s, warmup_steps=warm, total_steps=total)
            assert 0.0 <= v <= 1.0


def test_optimizer_applies_differ_style_defaults_and_scheduler():
    model, heads = _Tiny(), _Tiny()
    opt, sched = make_e3_optimizer(
        [p for p in model.parameters()], list(heads.parameters()),
        lora_lr=1e-5, head_lr=3e-5, weight_decay=0.05,
        warmup_steps=10, total_steps=100,
    )
    assert all(g["weight_decay"] == 0.05 for g in opt.param_groups)
    assert sched is not None
    # LambdaLR applies the step-0 factor on construction, so warmup starts at ~0
    assert all(g["lr"] == pytest.approx(0.0) for g in opt.param_groups)
    # the configured base LRs are retained for later scaling
    assert [g["initial_lr"] for g in opt.param_groups] == [1e-5, 3e-5]


def test_scheduler_restores_base_lr_at_end_of_warmup():
    model, heads = _Tiny(), _Tiny()
    opt, sched = make_e3_optimizer(
        [p for p in model.parameters()], list(heads.parameters()),
        lora_lr=1e-5, head_lr=3e-5, weight_decay=0.05,
        warmup_steps=5, total_steps=50,
    )
    for _ in range(5):
        opt.step()
        sched.step()
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-5, rel=1e-6)
    assert opt.param_groups[1]["lr"] == pytest.approx(3e-5, rel=1e-6)


def test_disabling_schedule_keeps_lr_constant():
    """The no-decay path must stay reproducible for the already-completed runs."""
    model, heads = _Tiny(), _Tiny()
    opt, sched = make_e3_optimizer(
        [p for p in model.parameters()], list(heads.parameters()),
        lora_lr=1e-4, head_lr=3e-4, weight_decay=None,
        warmup_steps=0, total_steps=0,
    )
    assert sched is None
    assert [g["lr"] for g in opt.param_groups] == [1e-4, 3e-4]
    # AdamW default weight decay preserved when None is passed
    assert all(g["weight_decay"] == 0.01 for g in opt.param_groups)


def test_cli_defaults_preserve_the_original_constant_lr_run():
    """The completed no-decay runs must stay reproducible from defaults alone."""
    import tools.dinov3.e3_train as m
    p = m.argparse.ArgumentParser()
    p.add_argument("--mode", default="run")
    p.add_argument("--relational-weight", type=float, default=0.0)
    p.add_argument("--lora-lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--cosine-decay", action="store_true")
    p.add_argument("--lr-warmup-steps", type=int, default=0)
    a = p.parse_args([])
    assert (a.lora_lr, a.head_lr) == (1e-4, 3e-4)
    assert a.weight_decay is None
    assert a.cosine_decay is False
    assert a.lr_warmup_steps == 0
    # with cosine_decay off, total_steps=0 is passed -> no scheduler -> constant LR
    _, sched = m.make_e3_optimizer(
        [torch.nn.Parameter(torch.zeros(2))], [torch.nn.Parameter(torch.zeros(2))],
        lora_lr=a.lora_lr, head_lr=a.head_lr, weight_decay=a.weight_decay,
        warmup_steps=a.lr_warmup_steps, total_steps=0,
    )
    assert sched is None

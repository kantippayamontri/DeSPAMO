import pytest
import torch

from tools.dinov3.e3_model import relational_structure_loss


def _teacher(rows: int = 12, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(rows, 2048)


def test_both_terms_vanish_when_student_matches_teacher():
    t = _teacher()
    out = relational_structure_loss(t.clone().requires_grad_(True), t)
    assert set(out) == {"pattern", "spread"}
    assert out["pattern"].item() < 1e-4
    assert out["spread"].item() < 1e-4


def test_pattern_term_detects_scrambled_structure():
    t = _teacher()
    scrambled = t[torch.randperm(len(t), generator=torch.Generator().manual_seed(3))].clone()
    out = relational_structure_loss(scrambled.requires_grad_(True), t)
    assert out["pattern"].item() > 0.05


def test_spread_term_catches_uniform_shrink_that_pattern_ignores():
    """The measured E3 failure: cone collapse preserves pattern, destroys spread."""
    t = _teacher()
    collapsed = t.mean(0, keepdim=True) + 0.02 * (t - t.mean(0, keepdim=True))
    out = relational_structure_loss(collapsed.requires_grad_(True), t)
    assert out["spread"].item() > 1.0, "Term B must punish uniform shrink"
    assert out["pattern"].item() < out["spread"].item(), "Term A is scale-invariant by design"


def test_teacher_is_detached_and_student_receives_gradient():
    t = _teacher().requires_grad_(True)
    student = (_teacher(seed=1) * 0.5).requires_grad_(True)
    out = relational_structure_loss(student, t)
    (out["pattern"] + out["spread"]).backward()
    assert t.grad is None
    assert student.grad is not None and torch.isfinite(student.grad).all()


@pytest.mark.parametrize("rows", (2, 3, 5))
def test_small_batches_stay_finite(rows):
    t = _teacher(rows=rows, seed=rows)
    out = relational_structure_loss((t * 0.3).requires_grad_(True), t)
    assert all(torch.isfinite(v).all() for v in out.values())


def test_each_scale_half_is_compared_separately():
    """A change confined to the 448 half must still register."""
    t = _teacher()
    student = t.clone()
    student[:, 1024:] = student[:, 1024:].mean(0, keepdim=True) + 0.02 * (
        student[:, 1024:] - student[:, 1024:].mean(0, keepdim=True)
    )
    out = relational_structure_loss(student.requires_grad_(True), t)
    assert out["spread"].item() > 0.5

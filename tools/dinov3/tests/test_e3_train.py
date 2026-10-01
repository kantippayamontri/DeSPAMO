import json
from types import SimpleNamespace

import pytest
import torch

from tools.dinov3.e3_model import DINOFactorHeads, LoRALinear, scale_reference_loss
from tools.dinov3.e3_train import (
    E3BatchSampler,
    e3_run_path,
    grl_alpha,
    require_e3_storage,
    resume_spend,
    step_e3,
    validate_e3_resume,
    validate_run_args,
)
from tools.dinov3.identity import sha256_file


def _batch():
    images = torch.ones(2, 5, 3, 4, 4)
    targets = torch.randn(2, 19, 8)
    masks = torch.ones(2, 19, dtype=torch.bool)
    masks[1] = False
    labels = tuple(tuple(f"{i}-{k}" for k in range(19)) for i in range(2))
    return {"images224": images, "images448": images + 1,
            "teacher": torch.randn(2, 5, 8, requires_grad=True),
            "vectors": targets, "masks": masks, "labels": labels}


class _FakeDino(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(3, 4)

    def forward(self, pixel_values):
        pooled = pixel_values.mean(dim=(-1, -2))
        return SimpleNamespace(last_hidden_state=self.projection(pooled)[:, None, :])


def test_e3_lora_up_receives_gradient_and_base_is_frozen():
    layer = LoRALinear(torch.nn.Linear(16, 16), rank=8, alpha=8)
    layer(torch.ones(2, 16)).square().mean().backward()
    assert layer.up.weight.grad.abs().sum() > 0
    assert layer.base.weight.grad is None


def test_e3_head_mask_and_grl_reverse_only_nuisance():
    torch.manual_seed(0)
    heads = DINOFactorHeads(visual_width=8, text_width=8)
    batch = _batch()
    features = torch.randn(2, 5, 8, requires_grad=True)
    loss = heads(features, batch["vectors"], batch["masks"], batch["labels"], 1.0)
    assert all(torch.isfinite(value) for value in loss.values())
    assert all(loss[name] == 0 for name in ("biometric", "clothing", "hair", "background"))
    assert all(loss[name] > 0 for name in ("left_hand", "right_hand", "mouth"))


def test_e3_calibration_freezes_dino_then_joint_steps_update_it():
    torch.manual_seed(0)
    model = _FakeDino()
    heads = DINOFactorHeads(visual_width=8, text_width=8)
    batch = _batch()
    batch["masks"][1] = True
    calibration = step_e3(model, heads, batch, step=1, calibration_steps=2, adapt_steps=12)
    calibration["combined_loss"].backward()
    assert model.projection.weight.grad is None
    model.zero_grad(set_to_none=True)
    heads.zero_grad(set_to_none=True)
    joint = step_e3(model, heads, batch, step=4, calibration_steps=2, adapt_steps=12)
    joint["combined_loss"].backward()
    assert model.projection.weight.grad is not None
    assert torch.isfinite(model.projection.weight.grad).all()
    assert batch["teacher"].grad is None
    assert torch.isfinite(joint["combined_loss"])
    assert grl_alpha(2, 2, 12) == 0
    assert grl_alpha(4, 2, 12) == 1


def test_e3_cosine_teacher_is_detached():
    prediction = torch.randn(3, 8, requires_grad=True)
    teacher = torch.randn(3, 8, requires_grad=True)
    scale_reference_loss(prediction, teacher).backward()
    assert teacher.grad is None
    assert prediction.grad is not None


def test_e3_resume_rejects_unowned_and_stale_checkpoints(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    checkpoint = run / "resume-2.pt"
    identity = {"protocol_hash": "frozen", "run_policy": "e3-v1", "run_key": "run"}
    torch.save({"identity": identity, "step": 2, "gpu_seconds": 1.5,
                "optimizer": {}, "model": {}, "heads": {}, "sampler": {"batches_yielded": 2},
                "rng": {}}, checkpoint)
    (run / "resume-2.json").write_text(json.dumps({"sha256": sha256_file(checkpoint)}))
    assert validate_e3_resume(checkpoint, identity, 1.5)["step"] == 2
    with pytest.raises(ValueError, match="identity"):
        validate_e3_resume(checkpoint, {**identity, "protocol_hash": "wrong"}, 1.5)
    with pytest.raises(ValueError, match="spend"):
        validate_e3_resume(checkpoint, identity, 0.2)
    (run / "resume-3.pt").write_bytes(b"later")
    with pytest.raises(ValueError, match="latest"):
        validate_e3_resume(checkpoint, identity, 1.5)


def test_e3_sampler_resume_keeps_next_physical_batch():
    original = E3BatchSampler(tuple(str(i) for i in range(7)), seed=0, batch_size=4)
    assert len(original.next_ids()) == 4
    snapshot = original.state_dict()
    expected = original.next_ids()
    restored = E3BatchSampler(tuple(str(i) for i in range(7)), seed=0, batch_size=4)
    restored.load_state_dict(snapshot)
    assert restored.next_ids() == expected
    assert restored.state_dict()["batches_yielded"] == 2


def test_e3_sampler_never_repeats_clip_at_epoch_boundary():
    sampler = E3BatchSampler(tuple(str(i) for i in range(7)), seed=0, batch_size=4)
    for _ in range(30):
        batch = sampler.next_ids()
        assert len(batch) == len(set(batch)) == 4


@pytest.mark.parametrize("size", (4, 8, 16))
def test_e3_sampler_supports_adaptation_batch_sizes(size):
    sampler = E3BatchSampler(tuple(str(i) for i in range(40)), seed=0, batch_size=size)
    for _ in range(25):
        batch = sampler.next_ids()
        assert len(batch) == len(set(batch)) == size
    assert sampler.state_dict()["batches_yielded"] == 25


def test_e3_sampler_rejects_unsupported_batch_size():
    with pytest.raises(ValueError, match="batch"):
        E3BatchSampler(tuple(str(i) for i in range(40)), seed=0, batch_size=6)


def test_e3_sampler_resume_at_batch16_keeps_next_batch_and_step_link():
    ids = tuple(str(i) for i in range(40))
    original = E3BatchSampler(ids, seed=0, batch_size=16)
    for _ in range(3):
        original.next_ids()
    snapshot = original.state_dict()
    expected = original.next_ids()
    restored = E3BatchSampler(ids, seed=0, batch_size=16)
    restored.load_state_dict(snapshot)
    assert restored.next_ids() == expected
    assert restored.state_dict()["batches_yielded"] == 4


def test_step_e3_adds_relational_terms_and_scales_with_weight():
    torch.manual_seed(0)
    model = _FakeDino()
    heads = DINOFactorHeads(visual_width=8, text_width=8)
    batch = _batch()
    batch["masks"][1] = True
    off = step_e3(model, heads, batch, step=4, calibration_steps=2, adapt_steps=12,
                  relational_weight=0.0)
    on = step_e3(model, heads, batch, step=4, calibration_steps=2, adapt_steps=12,
                 relational_weight=0.2)
    assert {"relational_pattern", "relational_spread"} <= set(on)
    assert on["combined_loss"].item() != off["combined_loss"].item()
    assert torch.isfinite(on["combined_loss"])


def test_e3_run_requires_explicit_budget_and_authorization(tmp_path):
    args = SimpleNamespace(adapt_steps=100, calibration_steps=10, gpu_cap_seconds=60,
                           authorize_e3_run=False, output=tmp_path / "new-run", resume=None)
    with pytest.raises(ValueError, match="authorization"):
        validate_run_args(args)
    args.authorize_e3_run = True
    validate_run_args(args)
    args.calibration_steps = 100
    with pytest.raises(ValueError, match="steps"):
        validate_run_args(args)
    args.calibration_steps = 10
    args.output.mkdir()
    with pytest.raises(ValueError, match="output"):
        validate_run_args(args)


def test_e3_run_root_uses_identity_under_selected_output_base(tmp_path):
    path = e3_run_path(tmp_path, {"run_key": "e3-adapt-pinned"})
    assert path == tmp_path / "e3-adapt-pinned"


def test_e3_resume_spend_cannot_rewind_or_hide_unclean_gap():
    budget = {"gpu_seconds": 4., "wall_timestamp": 100., "status": "paused", "pid": 777}
    assert resume_spend(budget, checkpoint_seconds=5., now=200., cap=500.) == 5.
    budget["status"] = "running"
    assert resume_spend(budget, checkpoint_seconds=5., now=200., cap=500.) == 105.
    with pytest.raises(ValueError, match="cap"):
        resume_spend(budget, checkpoint_seconds=5., now=200., cap=100.)
    with pytest.raises(ValueError, match="clock"):
        resume_spend(budget, checkpoint_seconds=5., now=90., cap=500.)


def test_e3_run_checks_snapshot_storage_before_gpu(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.dinov3.e3_train.shutil.disk_usage",
                        lambda path: SimpleNamespace(free=1024))
    with pytest.raises(ValueError, match="storage"):
        require_e3_storage(tmp_path)

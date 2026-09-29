"""Signer-pilot reproducibility gates; no pretrained model/GPU needed."""

import random

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
from torch import nn

from despamo.training.pilot_factory import (
    PilotBaselineModule,
    PilotDataModule,
    PilotFactorModule,
    PilotRunState,
    make_variant,
    shared_tensor_hash,
)
from despamo.training.pilot_replay import (
    PilotBatchSampler,
    capture_rng,
    freeze_tier,
    restore_rng,
    verify_resume,
)
from tests.unit.training.test_factor_module import base_model, mixed_batch


class ToyBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual_adapter = nn.Linear(3, 4)
        self.language_model = nn.Linear(4, 2)
        self.vt_align = nn.ParameterDict({"logit_scale": nn.Parameter(torch.tensor(0.0))})


class ToyE2(nn.Module):
    def __init__(self, base, width):
        super().__init__()
        self.visual_adapter = base.visual_adapter
        self.language_model = base.language_model
        self.vt_align = base.vt_align
        self.auxiliary = nn.Linear(4, width)


def test_all_variants_start_from_same_fresh_shared_tensors_despite_e2_head_rng():
    def new_model():
        return ToyBase()

    e1 = make_variant("E1_frozen", new_model, lambda base: base, seed=0)
    e2 = make_variant("E2_projector", new_model, lambda base: ToyE2(base, 7), seed=0)
    e3 = make_variant("E3_dino_lora", new_model, lambda base: base, seed=0)
    assert shared_tensor_hash(e1) == shared_tensor_hash(e2) == shared_tensor_hash(e3)
    assert "auxiliary.weight" not in dict(e1.named_parameters())
    assert "auxiliary.weight" in dict(e2.named_parameters())
    assert e1 is not e3 and e1.language_model is not e3.language_model
    changed = ToyBase()
    with torch.no_grad():
        changed.language_model.bias.add_(1)
    assert shared_tensor_hash(changed) != shared_tensor_hash(e1)
    with pytest.raises(ValueError, match="variant"):
        make_variant("old_checkpoint", new_model, lambda base: base, seed=0)


def test_common_sampler_order_resumes_at_next_physical_batch():
    e1 = PilotBatchSampler(13, batch_size=4, seed=0, total_batches=15)
    e2 = PilotBatchSampler(13, batch_size=4, seed=0, total_batches=15)
    assert list(e1) == list(e2)
    first = PilotBatchSampler(13, batch_size=4, seed=0, total_batches=15)
    consumed = [next(iter(first)) for _ in range(5)]
    assert all(len(batch) == 4 for batch in consumed)
    state = first.state_dict()
    resumed = PilotBatchSampler(13, batch_size=4, seed=0, total_batches=15)
    resumed.load_state_dict(state)
    assert consumed + list(resumed) == list(PilotBatchSampler(13, 4, 0, 15))
    with pytest.raises(ValueError, match="sampler"):
        PilotBatchSampler(13, 2, 0, 15).load_state_dict(state)


def test_rng_snapshot_replays_python_numpy_torch_streams():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)
    state = capture_rng()
    expected = (random.random(), np.random.random(), torch.rand(3))
    restore_rng(state)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_resume_checks_full_identity_step_and_optimizer_scheduler():
    identity = {"protocol_hash": "frozen", "seed": 0, "variant": "E1_frozen", "steps": 4000}
    checkpoint = {
        "pilot_identity": identity,
        "global_step": 7,
        "optimizer_states": [{"state": {1: {"step": 7}}}],
        "lr_schedulers": [{"last_epoch": 7}],
        "pilot_rng": capture_rng(),
        "pilot_sampler": PilotBatchSampler(13, 2, 0, 20).state_dict(),
        "gpu_seconds_cumulative": 3.0,
    }
    verify_resume(checkpoint, identity, expected_step=7, spent_gpu_seconds=3.0)
    with pytest.raises(ValueError, match="identity"):
        verify_resume(checkpoint, {**identity, "protocol_hash": "other"}, 7, 3.0)
    with pytest.raises(ValueError, match="budget"):
        verify_resume(checkpoint, identity, 7, 2.0)
    with pytest.raises(ValueError, match="optimizer"):
        verify_resume({**checkpoint, "optimizer_states": []}, identity, 7, 3.0)


def test_freeze_tier_counts_three_runs_dino_dev_probe_and_reserve():
    profile = {
        "e1_seconds_per_step": 1.0,
        "e2_seconds_per_step": 1.5,
        "e3_spamo_seconds_per_step": 1.0,
        "dino_seconds_per_step": 2.0,
        "calibration_seconds_per_step": 2.0,
        "extract_seconds_per_frame": 0.01,
        "dev_seconds_per_full_split": 120.0,
        "final_and_probe_seconds": 1000.0,
    }
    tier = freeze_tier(profile, spent_gpu_seconds=100.0)
    assert tier["spaMo_steps"] == 8000 and tier["dino_steps"] == 2000
    assert tier["estimated_gpu_seconds"] <= 20 * 3600
    for key in profile:
        with pytest.raises(ValueError, match="profile"):
            freeze_tier({k: v for k, v in profile.items() if k != key}, 0)
    with pytest.raises(ValueError, match="tier"):
        freeze_tier({k: v * 100 for k, v in profile.items()}, 0)


def test_pilot_factor_objective_respects_warmup_exact_masks_and_seven_weights():
    torch.manual_seed(4)
    baseline = base_model()
    baseline.warm_up_steps = 2
    model = PilotFactorModule(
        baseline,
        text_width=4,
        total_steps=40,
        provenance={
            "supervision_policy": "qwen-schema98-unreviewed-v1",
            "human_review_status": "not_assessed",
        },
    )
    batch = mixed_batch()
    warmup = model.compute_losses_at_step(batch, 2)
    assert not any(key.startswith("factor/") for key in warmup)
    for name in batch.masks:
        batch.masks[name][:] = False
        batch.vectors[name][:] = 0
    losses = model.compute_losses_at_step(batch, 4)
    assert all(losses[f"valid/{name}"].item() == 0 for name in batch.masks)
    assert torch.isfinite(losses["combined_loss"])
    torch.testing.assert_close(
        losses["combined_loss"],
        losses["loss"] + model.vt_weight * losses["contra_loss"],
    )
    losses["combined_loss"].backward()
    assert model.visual_adapter.spatial_projector.weight.grad is not None
    assert model.auxiliary.heads["biometric"].weight.grad is not None
    assert torch.count_nonzero(model.auxiliary.heads["biometric"].weight.grad) == 0
    assert model.use_in_context is False


def test_pilot_optimizer_separates_shared_lora_adapter_and_factor_head_rates():
    model = PilotFactorModule(
        base_model(),
        4,
        40,
        {
            "supervision_policy": "qwen-schema98-unreviewed-v1",
            "human_review_status": "not_assessed",
        },
    )
    optimizer = model.configure_optimizers()
    assert isinstance(optimizer, torch.optim.AdamW)
    assert {g["lr"] for g in optimizer.param_groups} == {3e-4, 5e-5}
    assert sorted(id(p) for g in optimizer.param_groups for p in g["params"]) == sorted(
        id(p) for p in model.parameters() if p.requires_grad
    )
    baseline = PilotBaselineModule(base_model(), total_steps=40)
    optimizer = baseline.configure_optimizers()
    assert {g["lr"] for g in optimizer.param_groups} == {3e-4, 5e-5}
    assert not hasattr(baseline, "auxiliary")


def test_pilot_data_module_shares_physical_batch_order_and_restores_cursor():
    dataset = list(range(13))
    first = PilotDataModule(dataset, batch_size=4, seed=0, total_batches=15, collate_fn=list)
    second = PilotDataModule(dataset, batch_size=4, seed=0, total_batches=15, collate_fn=list)
    first_order = list(first.train_dataloader())
    assert first_order == list(second.train_dataloader())
    third = PilotDataModule(dataset, batch_size=4, seed=0, total_batches=15, collate_fn=list)
    it = iter(third.train_dataloader())
    consumed = [next(it) for _ in range(5)]
    resume = PilotDataModule(dataset, batch_size=4, seed=0, total_batches=15, collate_fn=list)
    resume.load_state_dict(third.state_dict())
    assert consumed + list(resume.train_dataloader()) == first_order


def test_pilot_checkpoint_callback_requires_nonreset_ledger_and_restores_sampler_rng():
    data = PilotDataModule(list(range(13)), 4, 0, 15, list)
    callback = PilotRunState({"protocol_hash": "frozen", "variant": "E1_frozen"}, 11.0)

    class Trainer:
        datamodule = data
        global_step = 2

    checkpoint = {"global_step": 2}
    next(iter(data.train_dataloader()))
    sampler = data.state_dict()
    callback.on_save_checkpoint(Trainer(), None, checkpoint)
    assert checkpoint["pilot_sampler"]["batches_yielded"] == 1
    assert checkpoint["gpu_seconds_cumulative"] >= 11
    resumed = PilotDataModule(list(range(13)), 4, 0, 15, list)

    class ResumeTrainer:
        datamodule = resumed
        global_step = 2

    saved = PilotRunState(
        {"protocol_hash": "frozen", "variant": "E1_frozen"},
        checkpoint["gpu_seconds_cumulative"],
    )
    saved.on_load_checkpoint(ResumeTrainer(), None, checkpoint)
    assert resumed.state_dict()["order"] == sampler["order"]
    assert resumed.state_dict()["cursor"] == sampler["cursor"]
    with pytest.raises(ValueError, match="budget"):
        PilotRunState({"protocol_hash": "frozen", "variant": "E1_frozen"}, 10.0).on_load_checkpoint(
            ResumeTrainer(), None, checkpoint
        )


def test_lightning_resume_keeps_physical_batches_optimizer_scheduler_and_rng(tmp_path):
    identity = {"protocol_hash": "frozen", "variant": "E1_frozen", "steps": 5}

    class TinyPilot(pl.LightningModule):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(1, 1)
            self.seen = []

        def training_step(self, batch, batch_idx):
            self.seen.append(tuple(int(x) for x in batch.flatten().tolist()))
            return self.linear(batch).square().mean() + torch.rand(())

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
            return {
                "optimizer": optimizer,
                "lr_scheduler": torch.optim.lr_scheduler.StepLR(optimizer, 2),
            }

    def train(steps, *, checkpoint=None, spent=0.0):
        torch.manual_seed(0)
        data = PilotDataModule([torch.tensor([float(i)]) for i in range(13)], 2, 0, 5, torch.stack)
        model = TinyPilot()
        trainer = pl.Trainer(
            max_steps=steps,
            max_epochs=-1,
            callbacks=[PilotRunState(identity, spent)],
            enable_checkpointing=False,
            enable_progress_bar=False,
            logger=False,
            enable_model_summary=False,
        )
        trainer.fit(model, datamodule=data, ckpt_path=str(checkpoint) if checkpoint else None)
        return trainer, model, data

    full, full_model, full_data = train(5)
    partial, part_model, part_data = train(2)
    path = tmp_path / "step-2.ckpt"
    partial.save_checkpoint(path)
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    verify_resume(
        checkpoint,
        identity,
        expected_step=2,
        spent_gpu_seconds=checkpoint["gpu_seconds_cumulative"],
    )
    resumed, resumed_model, resumed_data = train(
        5, checkpoint=path, spent=checkpoint["gpu_seconds_cumulative"]
    )
    assert partial.global_step == 2 and resumed.global_step == full.global_step == 5
    assert part_model.seen + resumed_model.seen == full_model.seen
    assert resumed_data.sampler.batches_yielded == full_data.sampler.batches_yielded == 5
    for name, value in full_model.state_dict().items():
        torch.testing.assert_close(resumed_model.state_dict()[name], value, rtol=0, atol=0)

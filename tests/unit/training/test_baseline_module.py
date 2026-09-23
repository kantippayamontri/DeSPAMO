import random
from dataclasses import FrozenInstanceError, is_dataclass
from math import cos, pi
from types import SimpleNamespace

import pytest
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from despamo.data.batch import PhoenixBatch, PhoenixSample, collate_phoenix
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.prompts import build_prompts
from despamo.training.baseline_module import (
    SpaMoBaselineModule,
    combine_losses,
    ensure_finite_losses,
)


class FakeAdapter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.inputs = None

    def forward(self, spatial, spatial_mask, motion, motion_mask):
        self.inputs = (spatial, spatial_mask, motion, motion_mask)
        return spatial * self.scale, spatial_mask


class FakeLanguageModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(2.0))
        self.targets = None
        self.prompts = None

    def target_embeddings(self, texts):
        embeddings = torch.tensor(
            [[1.0, 0.0] if i % 2 == 0 else [0.0, 1.0] for i in range(len(texts))]
        ).unsqueeze(1)
        return embeddings, torch.ones(len(texts), 1, dtype=torch.bool)

    def translation_loss(self, visual, visual_mask, prompts, targets):
        self.prompts = prompts
        self.targets = targets
        return visual.mean() + self.bias


def make_batch(size: int = 2) -> PhoenixBatch:
    return PhoenixBatch(
        clip_ids=tuple(f"clip-{i}" for i in range(size)),
        signers=("signer",) * size,
        texts=tuple(f"text-{i}" for i in range(size)),
        glosses=("GLOSS",) * size,
        en_texts=tuple(f"english-{i}" for i in range(size)),
        es_texts=tuple(f"spanish-{i}" for i in range(size)),
        fr_texts=tuple(f"french-{i}" for i in range(size)),
        spatial=torch.tensor([[[1.0 + i, 2.0], [3.0, 1.0]] for i in range(size)]),
        spatial_mask=torch.tensor([[True, i % 2 == 0] for i in range(size)]),
        motion=torch.ones(size, 1, 2),
        motion_mask=torch.ones(size, 1, dtype=torch.bool),
    )


def make_phoenix_samples() -> list[PhoenixSample]:
    return [
        PhoenixSample(
            clip_id=f"clip-{i}",
            signer=f"signer-{i}",
            text=f"text-{i}",
            gloss=f"gloss-{i}",
            en_text=f"english-{i}",
            es_text=f"spanish-{i}",
            fr_text=f"french-{i}",
            spatial=torch.tensor([[1.0 + i, 2.0], [3.0, 1.0]]),
            motion=torch.ones(1, 2),
        )
        for i in range(2)
    ]


def make_module(*, warm_up_steps: int | None = None, use_in_context: bool = False):
    return SpaMoBaselineModule(
        visual_adapter=FakeAdapter(),
        language_model=FakeLanguageModel(),
        vt_align=VTAlignLoss(initial_logit_scale=0.0),
        prompt_template="Translate into {}.",
        use_in_context=use_in_context,
        num_in_context=1,
        vt_pooling="masked_mean",
        vt_weight=0.5,
        warm_up_steps=warm_up_steps,
        learning_rate=0.01,
        weight_decay=0.03,
        seed=13,
    )


def test_trainer_fit_and_validation_transfer_collated_frozen_phoenix_batch() -> None:
    module = make_module()
    loader = DataLoader(make_phoenix_samples(), batch_size=2, collate_fn=collate_phoenix)
    trainer = pl.Trainer(
        accelerator="cpu",
        devices=1,
        max_steps=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )

    trainer.fit(module, train_dataloaders=loader)

    assert trainer.global_step == 1
    assert module.language_model.targets == ("text-0", "text-1")
    results = trainer.validate(module, dataloaders=loader, verbose=False)
    assert len(results) == 1
    assert set(results[0]) == {"val/loss", "val/contra_loss", "val/combined_loss"}
    assert module.language_model.targets == ("text-0", "text-1")


def test_transfer_keeps_frozen_batch_metadata_and_moves_only_tensors() -> None:
    module = make_module()
    batch = collate_phoenix(make_phoenix_samples())

    moved = module.transfer_batch_to_device(batch, torch.device("cpu"), 0)

    assert moved is not batch and isinstance(moved, PhoenixBatch) and is_dataclass(moved)
    for name in ("clip_ids", "signers", "texts", "glosses", "en_texts", "es_texts", "fr_texts"):
        assert getattr(moved, name) is getattr(batch, name)
    for name in ("spatial", "spatial_mask", "motion", "motion_mask"):
        assert getattr(moved, name).device == torch.device("cpu")
        torch.testing.assert_close(getattr(moved, name), getattr(batch, name))
    with pytest.raises(FrozenInstanceError):
        moved.texts = ("changed",)
    assert set(module.state_dict()) == {
        "visual_adapter.scale",
        "language_model.bias",
        "vt_align.logit_scale",
    }


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_transfer_collated_phoenix_batch_to_gpu() -> None:
    batch = collate_phoenix(make_phoenix_samples())

    moved = make_module().transfer_batch_to_device(batch, torch.device("cuda:0"), 0)

    assert moved.clip_ids is batch.clip_ids and moved.texts is batch.texts
    assert all(
        getattr(moved, name).device.type == "cuda"
        for name in ("spatial", "spatial_mask", "motion", "motion_mask")
    )
    assert batch.spatial.device.type == "cpu"


def test_transfer_other_batch_types_uses_lightning_default() -> None:
    other = (torch.ones(1), {"ids": ("clip-0",)})

    moved = make_module().transfer_batch_to_device(other, torch.device("cpu"), 0)

    assert moved[0].device.type == "cpu"
    assert moved[1]["ids"] == other[1]["ids"]


@pytest.mark.parametrize("step, expected", [(0, 2.0), (10, 2.0), (11, 5.0)])
def test_combine_losses_uses_vt_through_warmup_boundary(step: int, expected: float) -> None:
    result = combine_losses(torch.tensor(4.0), torch.tensor(2.0), step, 10, 0.5)

    torch.testing.assert_close(result, torch.tensor(expected))


def test_combine_losses_without_warmup_uses_weighted_sum() -> None:
    result = combine_losses(torch.tensor(4.0), torch.tensor(2.0), 0, None, 0.5)

    torch.testing.assert_close(result, torch.tensor(5.0))


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_loss_reports_clip_ids_and_loss_name(invalid: float) -> None:
    with pytest.raises(FloatingPointError, match="combined_loss.*clip-a.*clip-b"):
        ensure_finite_losses({"combined_loss": torch.tensor(invalid)}, ("clip-a", "clip-b"))


def test_non_finite_loss_reports_all_component_values_and_clip_ids() -> None:
    losses = {
        "loss": torch.tensor(1.25, requires_grad=True),
        "contra_loss": torch.tensor(float("inf")),
        "combined_loss": torch.tensor(float("nan")),
    }

    with pytest.raises(FloatingPointError) as exc:
        ensure_finite_losses(losses, ("clip-a", "clip-b"))

    message = str(exc.value)
    for part in ("loss=1.25", "contra_loss=inf", "combined_loss=nan", "clip-a", "clip-b"):
        assert part in message


def test_finite_losses_are_accepted() -> None:
    ensure_finite_losses({"loss": torch.tensor(1.0), "contra_loss": torch.tensor(2.0)}, ("a",))


def test_compute_losses_wires_batch_prompts_and_warmup() -> None:
    module = make_module(warm_up_steps=10)
    batch = make_batch()

    losses = module.compute_losses(batch)

    assert set(losses) == {"loss", "contra_loss", "combined_loss"}
    torch.testing.assert_close(losses["loss"], batch.spatial.mean() + module.language_model.bias)
    embeddings, mask = module.language_model.target_embeddings(batch.texts)
    torch.testing.assert_close(
        losses["contra_loss"],
        module.vt_align(batch.spatial, batch.spatial_mask, embeddings, mask, "masked_mean"),
    )
    torch.testing.assert_close(losses["combined_loss"], losses["contra_loss"])
    assert all(
        actual is expected
        for actual, expected in zip(
            module.visual_adapter.inputs,
            (batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask),
            strict=True,
        )
    )
    assert module.language_model.targets == batch.texts
    assert module.language_model.prompts == ["Translate into German."] * len(batch.texts)


def test_bf16_target_embeddings_align_with_fp32_visual_without_autocast() -> None:
    class BF16LanguageModel(FakeLanguageModel):
        def __init__(self) -> None:
            super().__init__()
            self.embeddings = nn.Parameter(torch.eye(2).to(dtype=torch.bfloat16))
            self.visual_dtype = None

        def target_embeddings(self, texts):
            return self.embeddings.unsqueeze(1), torch.ones(2, 1, dtype=torch.bool)

        def translation_loss(self, visual, visual_mask, prompts, targets):
            self.visual_dtype = visual.dtype
            return super().translation_loss(visual, visual_mask, prompts, targets)

    module = make_module()
    module.language_model = BF16LanguageModel()
    batch = make_batch()

    losses = module.compute_losses(batch)

    assert module.language_model.visual_dtype == torch.float32
    assert losses["contra_loss"].dtype == torch.float32
    assert torch.isfinite(losses["combined_loss"])
    losses["combined_loss"].backward()
    for grad in (
        module.visual_adapter.scale.grad,
        module.language_model.embeddings.grad,
        module.language_model.bias.grad,
        module.vt_align.logit_scale.grad,
    ):
        assert grad is not None and torch.isfinite(grad).all()


def test_train_and_validation_log_expected_keys_and_batch_size(monkeypatch) -> None:
    module = make_module()
    batch = make_batch()
    logged = []
    monkeypatch.setattr(
        module, "log_dict", lambda values, **kwargs: logged.append((values, kwargs))
    )

    loss = module.training_step(batch, 0)
    assert set(logged[0][0]) == {"train/loss", "train/contra_loss", "train/combined_loss"}
    assert logged[0][1] == {"batch_size": 2}
    torch.testing.assert_close(loss, logged[0][0]["train/combined_loss"])
    torch.testing.assert_close(
        loss, logged[0][0]["train/loss"] + 0.5 * logged[0][0]["train/contra_loss"]
    )

    assert module.validation_step(batch, 0) is None
    assert set(logged[1][0]) == {"val/loss", "val/contra_loss", "val/combined_loss"}
    assert logged[1][1] == {"batch_size": 2}


def test_singleton_validation_uses_plain_prompt_with_in_context_enabled(monkeypatch) -> None:
    module = make_module(use_in_context=True)
    batch = make_batch(1)
    logged = []
    monkeypatch.setattr(
        module, "log_dict", lambda values, **kwargs: logged.append((values, kwargs))
    )
    initial_state = module.rng.getstate()

    assert module.validation_step(batch, 0) is None

    assert module.language_model.prompts == ["Translate into German."]
    assert module.language_model.targets == batch.texts
    assert set(logged[0][0]) == {"val/loss", "val/contra_loss", "val/combined_loss"}
    assert logged[0][1] == {"batch_size": 1}
    assert module.rng.getstate() == initial_state


def test_adamw_step_updates_cpu_fake_and_preserves_checkpoint_prefixes(monkeypatch) -> None:
    module = make_module()
    monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)
    batch = make_batch()
    optimizer = module.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 0.01
    assert optimizer.param_groups[0]["weight_decay"] == 0.03
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert set(optimizer.param_groups[0]["params"]) == set(module.parameters())
    assert set(module.state_dict()) == {
        "visual_adapter.scale",
        "language_model.bias",
        "vt_align.logit_scale",
    }

    before = (module.visual_adapter.scale.item(), module.language_model.bias.item())
    module.training_step(batch, 0).backward()
    optimizer.step()

    assert module.visual_adapter.scale.item() != before[0]
    assert module.language_model.bias.item() != before[1]
    assert module.vt_align.logit_scale.grad is not None


def test_cosine_scheduler_uses_estimated_optimizer_steps_after_accumulation() -> None:
    module = make_module()
    module._trainer = SimpleNamespace(
        estimated_stepping_batches=20,
        num_training_batches=80,
        accumulate_grad_batches=4,
    )

    configured = module.configure_optimizers()

    assert set(configured) == {"optimizer", "lr_scheduler"}
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.defaults["lr"] == 0.01
    assert optimizer.param_groups[0]["betas"] == (0.9, 0.98)
    assert optimizer.param_groups[0]["eps"] == 1e-8
    assert optimizer.param_groups[0]["weight_decay"] == 0.03
    scheduler_config = configured["lr_scheduler"]
    assert scheduler_config["interval"] == "step"
    assert scheduler_config["frequency"] == 1
    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.LambdaLR)
    base_lr = optimizer.defaults["lr"]
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)
    for step in range(1, 22):
        optimizer.step()
        scheduler.step()
        if step in (1, 2, 11, 20, 21):
            expected = step / 2 if step < 2 else max(0.0, 0.5 * (1 + cos(pi * (step - 2) / 18)))
            assert optimizer.param_groups[0]["lr"] == pytest.approx(base_lr * expected)
    assert scheduler.last_epoch == 21


def test_short_schedule_has_no_warmup_when_ten_percent_truncates_to_zero() -> None:
    module = make_module()
    module._trainer = SimpleNamespace(estimated_stepping_batches=5)

    configured = module.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]

    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    for _ in range(5):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_cosine_scheduler_resumes_after_optimizer_and_scheduler_state_load() -> None:
    original = make_module()
    original._trainer = SimpleNamespace(estimated_stepping_batches=20)
    configured = original.configure_optimizers()
    optimizer = configured["optimizer"]
    scheduler = configured["lr_scheduler"]["scheduler"]
    for _ in range(6):
        optimizer.step()
        scheduler.step()

    restored = make_module()
    restored._trainer = SimpleNamespace(estimated_stepping_batches=20)
    restored_config = restored.configure_optimizers()
    restored_optimizer = restored_config["optimizer"]
    restored_scheduler = restored_config["lr_scheduler"]["scheduler"]
    restored_optimizer.load_state_dict(optimizer.state_dict())
    restored_scheduler.load_state_dict(scheduler.state_dict())

    assert restored_scheduler.last_epoch == scheduler.last_epoch == 6
    assert restored_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]
    for _ in range(7, 21):
        optimizer.step()
        scheduler.step()
        restored_optimizer.step()
        restored_scheduler.step()
        assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(
            optimizer.param_groups[0]["lr"]
        )
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(0.0)


def test_prompt_rng_reproducible_across_modules_and_advances_per_batch() -> None:
    batch = make_batch(4)
    first = make_module(use_in_context=True)
    second = make_module(use_in_context=True)
    expected_rng = random.Random(13)

    for _ in range(2):
        expected = build_prompts(batch, "Translate into {}.", True, 1, expected_rng)
        first.compute_losses(batch)
        second.compute_losses(batch)
        assert first.language_model.prompts == second.language_model.prompts == expected


def test_checkpoint_restores_prompt_rng_after_train_and_validation(monkeypatch) -> None:
    batch = make_batch(4)
    source = make_module(use_in_context=True)
    restored = make_module(use_in_context=True)
    for module in (source, restored):
        monkeypatch.setattr(module, "log_dict", lambda *args, **kwargs: None)

    source.training_step(batch, 0)
    source.validation_step(batch, 0)
    checkpoint = {"state_dict": source.state_dict()}
    source.on_save_checkpoint(checkpoint)

    assert checkpoint["prompt_rng_state"] == source.rng.getstate()
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint["state_dict"])
    assert restored.rng.getstate() == source.rng.getstate()
    source.training_step(batch, 1)
    restored.training_step(batch, 1)
    assert restored.language_model.prompts == source.language_model.prompts
    source.validation_step(batch, 1)
    restored.validation_step(batch, 1)
    assert restored.language_model.prompts == source.language_model.prompts


def test_old_checkpoint_without_rng_state_uses_module_seed() -> None:
    module = make_module(use_in_context=True)
    checkpoint = {"state_dict": module.state_dict()}

    module.on_load_checkpoint(checkpoint)
    module.compute_losses(make_batch(4))

    expected = build_prompts(make_batch(4), "Translate into {}.", True, 1, random.Random(13))
    assert module.language_model.prompts == expected


def test_checkpoint_provenance_round_trips_with_prompt_rng() -> None:
    metadata = {
        "config": {"model": {"name": "google/flan-t5-xl", "vt_pooling": "masked_mean"}},
        "seed": 13,
        "package_lock_sha256": "lock",
        "spatial_manifest_sha256": "spatial",
        "motion_manifest_sha256": "motion",
        "model_source": {"identifier": "google/flan-t5-xl", "tuning_type": "lora"},
    }
    source = make_module(use_in_context=True)
    source.run_metadata = metadata
    source.compute_losses(make_batch(4))
    checkpoint = {}

    source.on_save_checkpoint(checkpoint)

    assert checkpoint["run_metadata"] is metadata
    assert checkpoint["prompt_rng_state"] == source.rng.getstate()
    restored = make_module(use_in_context=True)
    restored.run_metadata = metadata.copy()
    restored.on_load_checkpoint(checkpoint)
    assert restored.rng.getstate() == source.rng.getstate()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("config", {"model": {"name": "other", "vt_pooling": "masked_mean"}}),
        ("seed", 99),
        ("package_lock_sha256", "other-lock"),
        ("spatial_manifest_sha256", "other-spatial"),
        ("motion_manifest_sha256", "other-motion"),
        ("model_source", {"identifier": "other", "tuning_type": "freeze"}),
    ],
)
def test_checkpoint_rejects_mismatched_resume_provenance(field, replacement) -> None:
    module = make_module()
    module.run_metadata = {
        "config": {"model": {"name": "google/flan-t5-xl", "vt_pooling": "masked_mean"}},
        "seed": 13,
        "package_lock_sha256": "lock",
        "spatial_manifest_sha256": "spatial",
        "motion_manifest_sha256": "motion",
        "model_source": {"identifier": "google/flan-t5-xl", "tuning_type": "lora"},
    }
    checkpoint = {"run_metadata": {**module.run_metadata, field: replacement}}

    with pytest.raises(ValueError, match=field):
        module.on_load_checkpoint(checkpoint)


def test_checkpoint_rejects_missing_cuda_version_even_when_runtime_is_cpu_only() -> None:
    module = make_module()
    module.run_metadata = {"cuda_version": None}

    with pytest.raises(ValueError, match="cuda_version"):
        module.on_load_checkpoint({"run_metadata": {}})


def test_old_converted_checkpoint_without_provenance_loads_with_configured_metadata() -> None:
    module = make_module()
    module.run_metadata = {"config": {"model": {"name": "google/flan-t5-xl"}}}

    module.on_load_checkpoint({"metadata": {"source_sha256": "converted-weights"}})

    assert module.rng.getstate() == random.Random(13).getstate()


@pytest.mark.parametrize(
    ("indicator", "value"),
    [
        ("pytorch-lightning_version", "1.9.5"),
        ("global_step", 2),
        ("optimizer_states", []),
    ],
)
def test_native_checkpoint_without_provenance_is_rejected(indicator, value) -> None:
    module = make_module()
    module.run_metadata = {"seed": 13}
    initial_rng = module.rng.getstate()

    with pytest.raises(ValueError, match="run_metadata"):
        module.on_load_checkpoint(
            {indicator: value, "prompt_rng_state": random.Random(7).getstate()}
        )

    assert module.rng.getstate() == initial_rng

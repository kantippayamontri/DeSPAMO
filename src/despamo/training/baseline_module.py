import random
from dataclasses import replace
from functools import partial
from math import cos, pi
from typing import Any

import pytorch_lightning as pl
import torch

from despamo.data.batch import PhoenixBatch
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.prompts import build_prompts
from despamo.models.visual_adapter import SpaMoVisualAdapter


def combine_losses(
    translation_loss: torch.Tensor,
    vt_loss: torch.Tensor,
    global_step: int,
    warm_up_steps: int | None,
    vt_weight: float,
) -> torch.Tensor:
    if warm_up_steps is not None and global_step <= warm_up_steps:
        return vt_loss
    return translation_loss + vt_weight * vt_loss


def _cosine_warmup_factor(current_step: int, *, warmup_steps: int, total_steps: int) -> float:
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return max(0.0, 0.5 * (1.0 + cos(pi * progress)))


def ensure_finite_losses(losses: dict[str, torch.Tensor], clip_ids: tuple[str, ...]) -> None:
    invalid = [name for name, value in losses.items() if not torch.isfinite(value).all()]
    if invalid:
        values = ", ".join(
            f"{name}={value.detach().cpu().tolist()}" for name, value in losses.items()
        )
        raise FloatingPointError(
            f"non-finite losses {invalid} for clips {list(clip_ids)}; values: {values}"
        )


def validate_run_metadata(
    current: dict[str, Any], saved: Any, *, check_revision: bool = True
) -> None:
    if not isinstance(saved, dict):
        raise ValueError("resume checkpoint missing run_metadata")
    for key in (
        "config",
        "seed",
        "package_lock_sha256",
        "spatial_manifest_sha256",
        "motion_manifest_sha256",
        "torch_version",
        "cuda_version",
    ):
        if key in current and (key not in saved or saved[key] != current[key]):
            raise ValueError(f"resume checkpoint run_metadata mismatch: {key}")
    if "model_source" in current:
        source = current["model_source"]
        previous = saved.get("model_source")
        if not isinstance(previous, dict):
            raise ValueError("resume checkpoint run_metadata mismatch: model_source")
        fields = ("identifier", "tuning_type", "revision", "revision_status")
        for key in fields if check_revision else fields[:2]:
            if key in source and (key not in previous or source[key] != previous[key]):
                raise ValueError(f"resume checkpoint run_metadata mismatch: model_source.{key}")


class SpaMoBaselineModule(pl.LightningModule):
    def __init__(
        self,
        visual_adapter: SpaMoVisualAdapter,
        language_model: FlanT5Backbone,
        vt_align: VTAlignLoss,
        prompt_template: str,
        use_in_context: bool,
        num_in_context: int,
        vt_pooling: str,
        vt_weight: float,
        warm_up_steps: int | None,
        learning_rate: float,
        weight_decay: float,
        seed: int,
    ) -> None:
        super().__init__()
        self.visual_adapter = visual_adapter
        self.language_model = language_model
        self.vt_align = vt_align
        self.prompt_template = prompt_template
        self.use_in_context = use_in_context
        self.num_in_context = num_in_context
        self.vt_pooling = vt_pooling
        self.vt_weight = vt_weight
        self.warm_up_steps = warm_up_steps
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.rng = random.Random(seed)
        self.run_metadata: dict[str, Any] | None = None

    def transfer_batch_to_device(
        self, batch: Any, device: torch.device, dataloader_idx: int
    ) -> Any:
        if isinstance(batch, PhoenixBatch):
            return replace(
                batch,
                spatial=batch.spatial.to(device),
                spatial_mask=batch.spatial_mask.to(device),
                motion=batch.motion.to(device),
                motion_mask=batch.motion_mask.to(device),
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def compute_losses(self, batch: PhoenixBatch) -> dict[str, torch.Tensor]:
        visual, visual_mask = self.visual_adapter(
            batch.spatial.to(self.device),
            batch.spatial_mask.to(self.device),
            batch.motion.to(self.device),
            batch.motion_mask.to(self.device),
        )
        text_embeddings, text_mask = self.language_model.target_embeddings(batch.texts)
        vt_loss = self.vt_align(
            visual.float(), visual_mask, text_embeddings.float(), text_mask, self.vt_pooling
        )
        prompts = build_prompts(
            batch,
            self.prompt_template,
            self.use_in_context,
            self.num_in_context,
            self.rng,
        )
        translation_loss = self.language_model.translation_loss(
            visual, visual_mask, prompts, batch.texts
        )
        combined_loss = combine_losses(
            translation_loss, vt_loss, self.global_step, self.warm_up_steps, self.vt_weight
        )
        losses = {
            "loss": translation_loss,
            "contra_loss": vt_loss,
            "combined_loss": combined_loss,
        }
        ensure_finite_losses(losses, batch.clip_ids)
        return losses

    def training_step(self, batch: PhoenixBatch, batch_idx: int) -> torch.Tensor:
        losses = self.compute_losses(batch)
        self.log_dict(
            {f"train/{key}": value for key, value in losses.items()},
            batch_size=len(batch.texts),
        )
        return losses["combined_loss"]

    def validation_step(self, batch: PhoenixBatch, batch_idx: int) -> None:
        losses = self.compute_losses(batch)
        self.log_dict(
            {f"val/{key}": value for key, value in losses.items()},
            batch_size=len(batch.texts),
        )

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint["prompt_rng_state"] = self.rng.getstate()
        if self.run_metadata is not None:
            checkpoint["run_metadata"] = self.run_metadata

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        if "run_metadata" not in checkpoint and any(
            key in checkpoint
            for key in ("pytorch-lightning_version", "global_step", "optimizer_states")
        ):
            raise ValueError("Lightning resume checkpoint missing run_metadata")
        if self.run_metadata is not None and "run_metadata" in checkpoint:
            validate_run_metadata(self.run_metadata, checkpoint["run_metadata"])
        # Older and converted checkpoints lack this key; keep the seeded RNG state.
        if "prompt_rng_state" in checkpoint:
            self.rng.setstate(checkpoint["prompt_rng_state"])

    def configure_optimizers(self) -> torch.optim.AdamW | dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-8,
            weight_decay=self.weight_decay,
        )
        if self._trainer is None:
            return optimizer
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * 0.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            partial(_cosine_warmup_factor, warmup_steps=warmup_steps, total_steps=total_steps),
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

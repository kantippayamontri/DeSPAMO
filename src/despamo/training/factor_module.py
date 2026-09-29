"""Normal Lightning training with Qwen-derived, optionally masked factor losses."""

import math
from dataclasses import replace

import torch

from despamo.appearance.schema import ARTICULATORS, STABLE
from despamo.data.factors import FactorBatch
from despamo.models.factor_adapter import FactorVisualAdapter
from despamo.models.factor_heads import FactorHeads
from despamo.training.baseline_module import (
    SpaMoBaselineModule,
    combine_losses,
    ensure_finite_losses,
)


def ramp(step: int, warm: int | None, total: int) -> float | None:
    first_joint = 0 if warm is None else warm + 1
    if total <= first_joint or step < 0:
        raise ValueError("factor schedule requires positive joint optimizer steps")
    if step < first_joint:
        return None
    duration = max(1, math.ceil(0.1 * (total - first_joint)))
    return min(1.0, (step - first_joint) / duration)


class DualPathModule(SpaMoBaselineModule):
    def __init__(
        self,
        base: SpaMoBaselineModule,
        enabled: list[str],
        text_width: int,
        total_steps: int,
        provenance: dict,
    ) -> None:
        old = base.visual_adapter
        adapter = FactorVisualAdapter(
            old.spatial_projector.in_features,
            old.motion_projector.in_features,
            old.spatial_projector.out_features,
            old.multimodal_projector[-1].out_features,
        )
        adapter.load_state_dict(old.state_dict(), strict=True)
        super().__init__(
            adapter,
            base.language_model,
            base.vt_align,
            base.prompt_template,
            base.use_in_context,
            base.num_in_context,
            base.vt_pooling,
            base.vt_weight,
            base.warm_up_steps,
            base.learning_rate,
            base.weight_decay,
            seed=0,
        )
        self.rng = base.rng
        self.run_metadata = base.run_metadata
        self.auxiliary = FactorHeads(old.spatial_projector.out_features, text_width, enabled)
        self.total_steps = total_steps
        ramp(0, self.warm_up_steps, total_steps)
        if provenance.get("supervision_policy") != "qwen-schema98-unreviewed-v1" or provenance.get(
            "human_review_status"
        ) != "not_assessed":
            raise ValueError("unreviewed factor provenance required")
        self.factor_provenance = {
            **provenance,
            "enabled": list(enabled),
            "total_steps": total_steps,
            "nuisance_factors": list(STABLE),
            "articulator_factors": list(ARTICULATORS),
        }

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if isinstance(batch, FactorBatch):
            return replace(
                batch,
                base=super().transfer_batch_to_device(batch.base, device, dataloader_idx),
                rows=batch.rows.to(device),
                vectors={factor: value.to(device) for factor, value in batch.vectors.items()},
                masks={factor: value.to(device) for factor, value in batch.masks.items()},
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def compute_losses_at_step(self, batch, step: int) -> dict[str, torch.Tensor]:
        if not isinstance(batch, FactorBatch):
            return super().compute_losses(batch)
        base = batch.base
        losses = super().compute_losses(base)
        total = combine_losses(
            losses["loss"], losses["contra_loss"], step, self.warm_up_steps, self.vt_weight
        )
        alpha = ramp(step, self.warm_up_steps, self.total_steps)
        if alpha is not None:
            projected = self.visual_adapter.spatial_projector(base.spatial.to(self.device))
            factor_losses, counts = self.auxiliary(
                projected, base.spatial_mask.to(self.device), batch, alpha
            )
            for factor, value in factor_losses.items():
                weight = 1.0 if factor in STABLE else alpha
                total = total + weight * value
                losses[f"factor/{factor}"] = value
                losses[f"valid/{factor}"] = value.new_tensor(counts[factor])
        losses["combined_loss"] = total
        ensure_finite_losses(losses, base.clip_ids)
        return losses

    def compute_losses(self, batch):
        return self.compute_losses_at_step(batch, self.global_step)

    def training_step(self, batch, batch_idx):
        losses = self.compute_losses(batch)
        self.log_dict(
            {f"train/{name}": value for name, value in losses.items()},
            batch_size=len(batch.base.texts),
        )
        return losses["combined_loss"]

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        checkpoint["factor_provenance"] = self.factor_provenance

    def on_load_checkpoint(self, checkpoint):
        if checkpoint.get("factor_provenance") != self.factor_provenance:
            raise ValueError("factor provenance mismatch on resume")
        super().on_load_checkpoint(checkpoint)

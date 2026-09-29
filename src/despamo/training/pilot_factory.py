"""Seed and attest fresh shared SpaMo tensors for matched pilot variants."""

import hashlib
import random
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader

from despamo.appearance.schema import FACTORS, STABLE
from despamo.data.factors import FactorBatch
from despamo.training.baseline_module import (
    SpaMoBaselineModule,
    _cosine_warmup_factor,
    combine_losses,
    ensure_finite_losses,
)
from despamo.training.factor_module import DualPathModule, ramp
from despamo.training.pilot_replay import PilotBatchSampler, capture_rng, restore_rng

VARIANTS = ("E1_frozen", "E2_projector", "E3_dino_lora")
SHARED_PREFIXES = ("visual_adapter.", "language_model.", "vt_align.")


def shared_tensor_hash(model: nn.Module) -> str:
    """Hash trainable shared tensors, excluding independent auxiliary heads."""
    sha = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.named_parameters()):
        if not tensor.requires_grad or not name.startswith(SHARED_PREFIXES):
            continue
        count += 1
        sha.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}\n".encode())
        sha.update(
            tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        )
    if not count:
        raise ValueError("missing trainable shared SpaMo tensors")
    return sha.hexdigest()


def make_variant(
    variant: str,
    build_base: Callable[[], nn.Module],
    wrap: Callable[[nn.Module], nn.Module],
    *,
    seed: int,
) -> nn.Module:
    """Reinitialize fresh baseline per run; E2 head RNG never changes shared weights."""
    if variant not in VARIANTS or seed != 0:
        raise ValueError("unknown pilot variant/seed")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    base = build_base()
    initial_hash = shared_tensor_hash(base)
    model = wrap(base)
    if shared_tensor_hash(model) != initial_hash:
        raise ValueError("pilot wrapper altered fresh shared tensors")
    return model


def _pilot_optimizer(model: SpaMoBaselineModule):
    groups: dict[str, list[nn.Parameter]] = {"adapter": [], "flan": [], "factor": []}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.startswith("language_model."):
            groups["flan"].append(param)
        elif name.startswith("auxiliary."):
            groups["factor"].append(param)
        elif name.startswith(("visual_adapter.", "vt_align.")):
            groups["adapter"].append(param)
        else:
            raise ValueError(f"unexpected pilot trainable parameter: {name}")
    if not groups["adapter"] or not groups["flan"]:
        raise ValueError("missing trainable pilot adapter or FLAN LoRA")
    optimizer = torch.optim.AdamW(
        [
            {"params": params, "lr": 5e-5 if group == "flan" else 3e-4}
            for group, params in groups.items()
            if params
        ],
        betas=(0.9, 0.98),
        eps=1e-8,
        weight_decay=model.weight_decay,
    )
    if model._trainer is None:
        return optimizer
    total_steps = model.total_steps
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        partial(
            _cosine_warmup_factor,
            warmup_steps=int(total_steps * 0.1),
            total_steps=total_steps,
        ),
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
    }


class PilotBaselineModule(SpaMoBaselineModule):
    """Fresh E1/E3 SpaMo with pilot-only optimizer/prompt policy."""

    def __init__(self, base: SpaMoBaselineModule, total_steps: int) -> None:
        if base.use_in_context or total_steps < 1:
            raise ValueError("pilot requires reference-free prompts and fixed steps")
        super().__init__(
            base.visual_adapter,
            base.language_model,
            base.vt_align,
            base.prompt_template,
            False,
            0,
            base.vt_pooling,
            base.vt_weight,
            base.warm_up_steps,
            base.learning_rate,
            base.weight_decay,
            seed=0,
        )
        self.total_steps = total_steps
        self.run_metadata = base.run_metadata

    def configure_optimizers(self):
        return _pilot_optimizer(self)


class PilotFactorModule(DualPathModule):
    """E2: four GRL nuisance heads, three positive articulator heads."""

    def __init__(
        self, base: SpaMoBaselineModule, text_width: int, total_steps: int, provenance: dict
    ) -> None:
        if base.use_in_context or total_steps < 1:
            raise ValueError("pilot requires reference-free prompts and fixed steps")
        super().__init__(base, list(FACTORS), text_width, total_steps, provenance)

    def compute_losses_at_step(self, batch, step: int) -> dict[str, torch.Tensor]:
        if not isinstance(batch, FactorBatch):
            return SpaMoBaselineModule.compute_losses(self, batch)
        base = batch.base
        losses = SpaMoBaselineModule.compute_losses(self, base)
        losses["combined_loss"] = combine_losses(
            losses["loss"], losses["contra_loss"], step, self.warm_up_steps, self.vt_weight
        )
        alpha = ramp(step, self.warm_up_steps, self.total_steps)
        if alpha is not None:
            projected = self.visual_adapter.spatial_projector(base.spatial.to(self.device))
            factor_losses, counts = self.auxiliary(
                projected, base.spatial_mask.to(self.device), batch, alpha
            )
            for factor, value in factor_losses.items():
                weight = 0.05 if factor in STABLE else 0.10
                losses["combined_loss"] = losses["combined_loss"] + weight * value
                losses[f"factor/{factor}"] = value
                losses[f"valid/{factor}"] = value.new_tensor(counts[factor])
        ensure_finite_losses(losses, base.clip_ids)
        return losses

    def configure_optimizers(self):
        return _pilot_optimizer(self)


class PilotDataModule(pl.LightningDataModule):
    """Physical batch sampling independent of E1/E2 dataset and global torch RNG."""

    def __init__(self, train, batch_size: int, seed: int, total_batches: int, collate_fn) -> None:
        super().__init__()
        self.train = train
        self.sampler = PilotBatchSampler(len(train), batch_size, seed, total_batches)
        self.collate_fn = collate_fn

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_sampler=self.sampler,
            num_workers=0,  # prefetch would advance sampler past the checkpointed optimizer step
            collate_fn=self.collate_fn,
            generator=torch.Generator().manual_seed(self.sampler.seed),
        )

    def state_dict(self):
        return self.sampler.state_dict()

    def load_state_dict(self, state_dict):
        self.sampler.load_state_dict(state_dict)


class PilotRunState(pl.Callback):
    """Attest exact sampler/RNG/identity and non-resetting elapsed GPU-time on resume."""

    def __init__(self, identity: dict, spent_gpu_seconds: float) -> None:
        super().__init__()
        if not identity.get("protocol_hash") or not 0 <= spent_gpu_seconds < 24 * 3600:
            raise ValueError("invalid pilot identity/budget")
        self.identity = dict(identity)
        self.spent_gpu_seconds = spent_gpu_seconds
        self.started = time.monotonic()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        elapsed = self.spent_gpu_seconds + time.monotonic() - self.started
        if elapsed >= 24 * 3600:
            raise ValueError("pilot GPU-hour ceiling exceeded")
        checkpoint["pilot_identity"] = self.identity
        checkpoint["pilot_sampler"] = trainer.datamodule.state_dict()
        checkpoint["pilot_rng"] = capture_rng()
        checkpoint["gpu_seconds_cumulative"] = elapsed

    def on_load_checkpoint(self, trainer, pl_module, checkpoint) -> None:
        if checkpoint.get("pilot_identity") != self.identity:
            raise ValueError("pilot resume identity mismatch")
        previous = checkpoint.get("gpu_seconds_cumulative")
        if (
            not isinstance(previous, (int, float))
            or not 0 <= previous <= self.spent_gpu_seconds < 24 * 3600
        ):
            raise ValueError("pilot resume budget reset/exceeded")
        trainer.datamodule.load_state_dict(checkpoint["pilot_sampler"])
        restore_rng(checkpoint["pilot_rng"])

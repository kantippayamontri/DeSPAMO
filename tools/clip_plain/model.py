"""Fresh 1024-input reference model; no S² checkpoint surgery."""

from typing import Any, cast

import torch
from torch import nn

from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule
from tools.benchmark.compact_checkpoint import frozen_fingerprint


def build_model(seed: int, cache: str) -> tuple[SpaMoBaselineModule, str]:
    if seed not in (0, 1):
        raise ValueError("Reference seeds0/1 only")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        language = FlanT5Backbone.from_pretrained(
            "google/flan-t5-xl",
            cache,
            1024,
            "lora",
            16,
            32,
            0.1,
            revision="7d6315df2c2fb742f0f5b556879d730926ca9001",
        )
        visual = SpaMoVisualAdapter(1024, 1024, 768, 2048)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(100 + seed)
            cast(Any, visual).spatial_projector = nn.Sequential(
                nn.Linear(1024, 1024), nn.GELU(), nn.Linear(1024, 768)
            )
        model = SpaMoBaselineModule(
            visual,
            language,
            VTAlignLoss(),
            "Translate the given sentence into {}.",
            True,
            3,
            "legacy_mean",
            1.0,
            4000,
            3e-4,
            0.01,
            seed,
        )
        transformer = cast(Any, language.model)
        transformer.gradient_checkpointing_enable()
        transformer.enable_input_require_grads()
        transformer.config.use_cache = False
        language.max_text_length = 1024
        return model, frozen_fingerprint(model)

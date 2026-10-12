"""Check the new safe generation core also accepts original S² adapter output."""

import pytest
import torch
from torch import nn
from transformers import T5Config, T5ForConditionalGeneration

from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.visual_adapter import SpaMoVisualAdapter
from tools.clip_plain.generation import generate_cached_text
from tools.clip_s2_inference.tests.helpers import Tokenizer


def language2048():
    config = T5Config(
        vocab_size=32,
        d_model=2048,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=2,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=None,
        use_cache=False,
        dropout_rate=0.0,
    )
    return FlanT5Backbone(T5ForConditionalGeneration(config), Tokenizer(), 1024)


def adapter():
    model = SpaMoVisualAdapter(2048, 1024, 768, 2048).eval()
    model.spatial_projector = nn.Sequential(nn.Linear(2048, 1024), nn.GELU(), nn.Linear(1024, 768))
    return model


def test_s2_input_uses_both_1024_halves():
    torch.manual_seed(0)
    model = adapter()
    left_only = torch.zeros(1, 20, 2048)
    with_tiles = left_only.clone()
    with_tiles[:, :, 1024:] = 1.0
    with torch.inference_mode():
        a = model.spatial_projector(left_only)
        b = model.spatial_projector(with_tiles)
    assert model.spatial_projector[0].weight.shape == (1024, 2048)
    assert not torch.equal(a, b)


def test_s2_adapter_and_real_peft_t5_cached_generation_on_cpu():
    from peft import LoraConfig, TaskType, get_peft_model

    torch.manual_seed(0)
    spatial = torch.randn(1, 20, 2048)
    motion = torch.randn(1, 1, 1024)
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        visual, valid = adapter()(
            spatial,
            torch.ones(1, 20, dtype=torch.bool),
            motion,
            torch.ones(1, 1, dtype=torch.bool),
        )
        lang = language2048()
        lang.model = get_peft_model(
            lang.model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                target_modules=["q", "v"],
                lora_dropout=0.1,
                task_type=TaskType.SEQ_2_SEQ_LM,
            ),
        )
        events = []
        result = generate_cached_text(lang, visual, valid, ["translate"], on_event=events.append)
    assert visual.shape[-1] == 2048 and len(result["texts"]) == 1
    assert lang.model.config.use_cache is False
    assert result["decoding"]["use_cache"] and result["decoding"]["beam_size"] == 5
    steps = [e for e in events if e["phase"] == "decoder_start"]
    assert steps and all(e["decoder_length"] == 1 and e["beam_rows"] == 5 for e in steps)
    assert all(e["finite"] for e in events if e["phase"] == "decoder_logits")


def test_s2_generation_preserves_primary_failure_without_cuda_cleanup(monkeypatch):
    lang = language2048()
    error = RuntimeError("simulated S² generation fault")

    def fail(**kwargs):
        raise error

    def forbidden(*args, **kwargs):
        pytest.fail("Must not perform CUDA cleanup")

    monkeypatch.setattr(lang.model, "generate", fail)
    for name in ["set_rng_state_all", "memory_allocated", "synchronize", "empty_cache"]:
        monkeypatch.setattr(torch.cuda, name, forbidden)
    with pytest.raises(RuntimeError) as caught:
        generate_cached_text(
            lang,
            torch.ones(1, 3, 2048),
            torch.ones(1, 3, dtype=torch.bool),
            ["translate"],
            on_event=lambda e: None,
        )
    assert caught.value is error
    assert not lang.model._forward_hooks and not lang.model._forward_pre_hooks

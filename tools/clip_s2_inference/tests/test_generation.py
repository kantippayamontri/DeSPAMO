import copy

import pytest
import torch

from tools.clip_s2_inference.generation import generate_cached_text
from tools.clip_s2_inference.tests.helpers import language


def inputs():
    return torch.randn(4, 3, 32), torch.ones(4, 3, dtype=torch.bool), ["translate to german"] * 4


def test_real_t5_uses_single_token_cached_decoder_and_matches_uncached_output():
    lang = language().eval()
    visual, valid, prompts = inputs()
    expected = lang.generate_text(visual, valid, prompts, "deterministic", 5, 12)
    shapes = []
    handle = lang.model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    try:
        actual = generate_cached_text(lang, visual, valid, prompts, beam_size=5, max_length=12)
    finally:
        handle.remove()
    assert actual == expected
    assert shapes == [(20, 1)] * 11


def test_training_config_checkpointing_and_mixed_module_modes_are_unchanged():
    lang = language().train()
    lang.model.encoder.block[0].eval()
    config = copy.deepcopy(lang.model.config.to_dict())
    flags = [m.training for m in lang.modules()]
    result = generate_cached_text(lang, *inputs(), max_length=12)
    assert len(result) == 4
    assert lang.model.config.to_dict() == config
    assert lang.model.config.use_cache is False and lang.model.is_gradient_checkpointing
    assert [m.training for m in lang.modules()] == flags


def test_generation_failure_restores_modes_without_cuda_rng_cleanup(monkeypatch):
    lang = language().train()
    flags = [m.training for m in lang.modules()]

    def fail(**kwargs):
        assert not lang.model.training
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(lang.model, "generate", fail)
    with pytest.raises(RuntimeError, match="simulated device failure"):
        generate_cached_text(lang, *inputs(), max_length=12)
    assert [m.training for m in lang.modules()] == flags
    assert lang.model.config.use_cache is False


def test_observer_receives_joint_lengths_cache_and_effective_beams():
    lang = language().eval()
    records = []
    generate_cached_text(lang, *inputs(), max_length=12, on_start=records.append)
    assert records[0]["joint_shape"] == [4, 6, 32]
    assert records[0]["joint_lengths"] == [6] * 4
    assert records[0]["effective_beam_sequences"] == 20
    assert records[0]["use_cache"] is True and records[0]["do_sample"] is False


def test_invalid_valid_token_is_rejected_but_masked_padding_is_ignored():
    lang = language().eval()
    visual, valid, prompts = inputs()
    visual[0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        generate_cached_text(lang, visual, valid, prompts, max_length=12)
    valid[0, 0] = False
    assert len(generate_cached_text(lang, visual, valid, prompts, max_length=12)) == 4


def test_gpu_hold_requires_explicit_review_and_receipt_release_without_gpu_access(
    tmp_path, monkeypatch
):
    import json

    import tools.clip_s2_inference.generation as g

    hold = tmp_path / "hold.json"
    hold.write_text(json.dumps({"automatic_recovery_disabled": True}))
    monkeypatch.setattr(g, "GPU_HOLD", hold)
    g.guard_device(torch.device("cpu"))
    with pytest.raises(RuntimeError, match="review"):
        g.guard_device(torch.device("cuda"))
    with pytest.raises(RuntimeError, match="hold"):
        g.guard_device(torch.device("cuda"), gpu_reviewed=True)
    hold.write_text(
        json.dumps({"automatic_recovery_disabled": True, "gpu_validation_approved": True})
    )
    g.guard_device(torch.device("cuda"), gpu_reviewed=True)


def test_real_peft_lora_wrapper_preserves_cached_decoder_behavior():
    from peft import LoraConfig, TaskType, get_peft_model

    lang = language().eval()
    lang.model = get_peft_model(
        lang.model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q", "v"],
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.SEQ_2_SEQ_LM,
        ),
    )
    lang.eval()
    visual, valid, prompts = inputs()
    expected = lang.generate_text(visual, valid, prompts, "deterministic", 5, 12)
    observed = []
    handle = lang.model.base_model.model.decoder.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    try:
        assert generate_cached_text(lang, visual, valid, prompts, max_length=12) == expected
    finally:
        handle.remove()
    assert observed == [(20, 1)] * 11
    assert lang.model.config.use_cache is False

import copy

import pytest
import torch

from tools.clip_s2_inference.tests.helpers import language


def inputs():
    return torch.randn(2, 3, 32), torch.ones(2, 3, dtype=torch.bool), ["translate"] * 2


def test_each_cached_decoder_step_has_one_token():
    from tools.clip_plain.generation import generate_cached_text

    lang = language().train()
    config = copy.deepcopy(lang.model.config.to_dict())
    flags = [m.training for m in lang.modules()]
    events = []
    result = generate_cached_text(lang, *inputs(), on_event=events.append)
    steps = [e for e in events if e["phase"] == "decoder_start"]
    assert steps and all(e["decoder_length"] == 1 and e["beam_rows"] == 10 for e in steps)
    assert len(result["token_ids"]) == 2 and result["decoding"]["use_cache"]
    assert result["decoding"]["max_length"] == 64 and result["decoding"]["beam_size"] == 5
    assert lang.model.config.to_dict() == config and [m.training for m in lang.modules()] == flags


def test_real_peft_cached_decoder(monkeypatch):
    from peft import LoraConfig, TaskType, get_peft_model

    from tools.clip_plain.generation import generate_cached_text

    lang = language()
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
    r = generate_cached_text(lang, *inputs(), on_event=events.append)
    assert len(r["texts"]) == 2 and any(e["phase"] == "decoder_logits" for e in events)


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_nonfinite_raw_logits_abort_with_last_step_record(bad):
    from tools.clip_plain.generation import generate_cached_text

    lang = language()
    events = []

    def poison(module, args, out):
        out.logits[:, -1, 0] = bad

    handle = lang.model.register_forward_hook(poison)
    with pytest.raises(FloatingPointError):
        generate_cached_text(lang, *inputs(), on_event=events.append)
    handle.remove()
    assert any(e["phase"] == "decoder_logits" and not e["finite"] for e in events)
    assert len(lang.model._forward_pre_hooks) == 0 and len(lang.model._forward_hooks) == 0


def test_primary_fault_survives_failing_error_observer(monkeypatch):
    from tools.clip_plain.generation import generate_cached_text

    lang = language()
    error = RuntimeError("primary")
    calls = []

    def fail(**kwargs):
        calls.append(1)
        raise error

    def observer(event):
        if event["phase"] == "generation_error":
            raise OSError("logger")

    monkeypatch.setattr(lang.model, "generate", fail)
    with pytest.raises(RuntimeError) as got:
        generate_cached_text(lang, *inputs(), on_event=observer)
    assert got.value is error and len(calls) == 1 and any("logger" in n for n in error.__notes__)
    assert not lang.model._forward_hooks and not lang.model._forward_pre_hooks


def test_invalid_valid_token_but_masked_padding_allowed():
    from tools.clip_plain.generation import generate_cached_text

    lang = language()
    v, m, p = inputs()
    v[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        generate_cached_text(lang, v, m, p, on_event=lambda e: None)
    m[0, 0] = False
    assert len(generate_cached_text(lang, v, m, p, on_event=lambda e: None)["texts"]) == 2


def test_processed_negative_infinity_is_not_raw_logit_failure():
    from tools.clip_plain.generation import generate_cached_text

    lang = language()
    events = []
    assert len(generate_cached_text(lang, *inputs(), on_event=events.append)["texts"]) == 2
    assert all(e["finite"] for e in events if e["phase"] == "decoder_logits")

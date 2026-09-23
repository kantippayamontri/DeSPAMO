import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from despamo.models.flan_t5 import FlanT5Backbone


class FakeTokenizer:
    pad_token_id = 0

    def __init__(self, max_length: int = 2) -> None:
        self.max_length = max_length
        self.calls: list[dict] = []
        self.max_lengths: list[int | None] = []

    def __call__(self, texts, *, padding, return_tensors, truncation, max_length=None):
        self.max_lengths.append(max_length)
        self.calls.append(
            {
                "texts": texts,
                "padding": padding,
                "return_tensors": return_tensors,
                "truncation": truncation,
            }
        )
        limit = self.max_length if max_length is None else max_length
        lengths = [
            min(len(text.split()), limit) if truncation else len(text.split()) for text in texts
        ]
        ids = torch.zeros(len(texts), max(lengths), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for row, size in enumerate(lengths):
            ids[row, :size] = torch.arange(1, size + 1)
            mask[row, :size] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, ids, *, skip_special_tokens):
        assert skip_special_tokens is True
        return ["HELLO", "WORLD"][: len(ids)]


class FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.arange(64).view(16, 4).float())
        self.encoder = SimpleNamespace(embed_tokens=self.embedding)
        self.forward_kwargs = None
        self.generate_kwargs = None
        self.generate_grad_enabled = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        return SimpleNamespace(loss=kwargs["inputs_embeds"].sum())

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        self.generate_grad_enabled = torch.is_grad_enabled()
        return torch.tensor([[1, 2], [2, 0]])


def test_target_embeddings_return_token_mask_without_truncating_targets() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    backbone = FlanT5Backbone(model, tokenizer, max_text_length=2)

    embeddings, mask = backbone.target_embeddings(["one two three", "four"])

    assert embeddings.shape == (2, 3, 4)
    assert mask.dtype == torch.bool
    assert mask.tolist() == [[True, True, True], [True, False, False]]
    torch.testing.assert_close(embeddings[0, 0], model.embedding.weight[1])
    assert tokenizer.calls[0] == {
        "texts": ["one two three", "four"],
        "padding": "longest",
        "return_tensors": "pt",
        "truncation": False,
    }


def test_prompt_tokens_respect_backbone_cap_without_truncating_targets() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer(max_length=8)
    backbone = FlanT5Backbone(model, tokenizer, max_text_length=2)

    backbone.translation_loss(
        torch.ones(1, 1, 4),
        torch.ones(1, 1, dtype=torch.bool),
        ["one two three four"],
        ("one two three four",),
    )

    assert tokenizer.max_lengths == [2, None]
    assert model.forward_kwargs["inputs_embeds"].shape == (1, 3, 4)
    assert model.forward_kwargs["labels"].tolist() == [[1, 2, 3, 4]]


def test_translation_loss_joins_unmasked_visual_and_prompt_tokens_and_ignores_padding() -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    backbone = FlanT5Backbone(model, tokenizer, max_text_length=2)
    visual = torch.tensor(
        [
            [[101.0] * 4, [999.0] * 4, [102.0] * 4],
            [[201.0] * 4, [999.0] * 4, [999.0] * 4],
        ],
        requires_grad=True,
    )
    visual_mask = torch.tensor([[True, False, True], [True, False, False]])

    loss = backbone.translation_loss(
        visual,
        visual_mask,
        ["prompt has three words", "short"],
        ("target has three", "one"),
    )

    kwargs = model.forward_kwargs
    assert kwargs is not None
    torch.testing.assert_close(kwargs["inputs_embeds"][0, :2], visual[0, visual_mask[0]])
    torch.testing.assert_close(kwargs["inputs_embeds"][0, 2:4], model.embedding.weight[1:3])
    torch.testing.assert_close(kwargs["inputs_embeds"][1, 1], model.embedding.weight[1])
    assert kwargs["inputs_embeds"].shape == (2, 4, 4)
    assert kwargs["attention_mask"].dtype == torch.bool
    assert kwargs["attention_mask"].tolist() == [
        [True, True, True, True],
        [True, True, False, False],
    ]
    assert kwargs["labels"].tolist() == [[1, 2, 3], [1, -100, -100]]
    assert kwargs["decoder_attention_mask"].tolist() == [[True, True, True], [True, False, False]]
    assert kwargs["return_dict"] is True
    torch.testing.assert_close(loss, kwargs["inputs_embeds"].sum())
    loss.backward()
    assert visual.grad is not None
    assert torch.count_nonzero(visual.grad[~visual_mask]) == 0
    assert tokenizer.calls[0]["truncation"] is True
    assert tokenizer.calls[1]["truncation"] is False


def test_mixed_dtype_visual_tokens_match_t5_embeddings_and_keep_gradients() -> None:
    model = FakeModel().to(torch.bfloat16)
    backbone = FlanT5Backbone(model, FakeTokenizer(), max_text_length=2)
    visual = torch.ones(2, 2, 4, dtype=torch.float32, requires_grad=True)
    visual_mask = torch.tensor([[True, False], [True, True]])

    loss = backbone.translation_loss(visual, visual_mask, ["one", "two"], ("one", "two"))

    joint = model.forward_kwargs["inputs_embeds"]
    assert joint.dtype == torch.bfloat16
    torch.testing.assert_close(joint[0, 0], visual[0, 0].to(torch.bfloat16))
    loss.backward()
    assert visual.grad is not None and visual.grad.dtype == torch.float32
    assert torch.count_nonzero(visual.grad[visual_mask]) > 0
    assert torch.count_nonzero(visual.grad[~visual_mask]) == 0

    backbone.generate_text(visual, visual_mask, ["one", "two"], "deterministic", 5, 30)

    assert model.generate_kwargs["inputs_embeds"].dtype == torch.bfloat16


@pytest.mark.parametrize(
    "mode, do_sample, top_p",
    [
        ("upstream", True, 0.9),
        ("deterministic", False, None),
    ],
)
def test_generate_text_uses_requested_decode_settings(mode, do_sample, top_p) -> None:
    model = FakeModel()
    backbone = FlanT5Backbone(model, FakeTokenizer(), max_text_length=2)
    visual = torch.ones(2, 2, 4)
    visual_mask = torch.tensor([[True, False], [True, False]])

    result = backbone.generate_text(visual, visual_mask, ["one two three", "one"], mode, 5, 30)

    assert result == ["hello", "world"]
    kwargs = model.generate_kwargs
    assert kwargs is not None
    assert kwargs["inputs_embeds"].shape == (2, 3, 4)
    assert kwargs["attention_mask"].tolist() == [[True, True, True], [True, True, False]]
    assert kwargs["num_beams"] == 5
    assert kwargs["max_length"] == 30
    assert kwargs["do_sample"] is do_sample
    assert kwargs.get("top_p") == top_p
    assert model.generate_grad_enabled is False


def test_generate_text_rejects_unknown_mode() -> None:
    backbone = FlanT5Backbone(FakeModel(), FakeTokenizer(), max_text_length=2)

    with pytest.raises(ValueError, match="unsupported generation mode"):
        backbone.generate_text(
            torch.ones(1, 1, 4), torch.ones(1, 1, dtype=torch.bool), ["one"], "unknown", 5, 30
        )


def test_force_eval_keeps_frozen_model_in_eval_mode() -> None:
    backbone = FlanT5Backbone(FakeModel(), FakeTokenizer(), max_text_length=2, force_eval=True)

    backbone.train()

    assert backbone.training
    assert not backbone.model.training


@pytest.mark.parametrize("tuning_type", ["freeze", "lora"])
def test_from_pretrained_uses_expected_tuning_with_lazy_library_imports(
    monkeypatch, tuning_type
) -> None:
    model = FakeModel()
    tokenizer = FakeTokenizer()
    seen = {}
    transformers = ModuleType("transformers")
    peft = ModuleType("peft")

    class FakeT5:
        @staticmethod
        def from_pretrained(name, **kwargs):
            seen["model"] = (name, kwargs)
            return model

    class FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(name, **kwargs):
            seen["tokenizer"] = (name, kwargs)
            return tokenizer

    def lora_config(**kwargs):
        seen["lora"] = kwargs
        return kwargs

    def get_peft_model(base, config):
        seen["wrapped"] = (base, config)
        return base

    transformers.T5ForConditionalGeneration = FakeT5
    transformers.AutoTokenizer = FakeAutoTokenizer
    peft.LoraConfig = lora_config
    peft.TaskType = SimpleNamespace(SEQ_2_SEQ_LM="seq2seq")
    peft.get_peft_model = get_peft_model
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "peft", peft)

    backbone = FlanT5Backbone.from_pretrained("flan", "/cache", 32, tuning_type, 16, 32, 0.1)

    assert seen["model"] == ("flan", {"cache_dir": "/cache", "torch_dtype": torch.bfloat16})
    assert seen["tokenizer"] == ("flan", {"cache_dir": "/cache", "max_length": 32})
    assert backbone.model is model and backbone.tokenizer is tokenizer
    if tuning_type == "freeze":
        assert backbone.force_eval and not model.training
        assert all(not parameter.requires_grad for parameter in model.parameters())
        backbone.train()
        assert not model.training
    else:
        assert not backbone.force_eval and model.training
        assert seen["lora"] == {
            "r": 16,
            "lora_alpha": 32,
            "target_modules": ["q", "v"],
            "lora_dropout": 0.1,
            "bias": "none",
            "task_type": "seq2seq",
        }
        assert seen["wrapped"] == (model, seen["lora"])


def test_from_pretrained_rejects_unknown_tuning_before_loading() -> None:
    backbone = FlanT5Backbone

    with pytest.raises(ValueError, match="unsupported tuning_type"):
        backbone.from_pretrained("flan", "/cache", 32, "bad", 16, 32, 0.1)

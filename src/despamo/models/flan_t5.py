from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence


@dataclass(frozen=True)
class TokenBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor


class FlanT5Backbone(nn.Module):
    def __init__(
        self, model: nn.Module, tokenizer, max_text_length: int, force_eval: bool = False
    ) -> None:
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.max_text_length = max_text_length
        self.force_eval = force_eval

    @classmethod
    def from_pretrained(
        cls,
        model_name: str,
        cache_dir: str,
        max_text_length: int,
        tuning_type: str,
        lora_rank: int,
        lora_alpha: int,
        lora_dropout: float,
    ) -> FlanT5Backbone:
        if tuning_type not in {"lora", "freeze"}:
            raise ValueError(f"unsupported tuning_type: {tuning_type}")

        from transformers import AutoTokenizer, T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(
            model_name, cache_dir=cache_dir, torch_dtype=torch.bfloat16
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, cache_dir=cache_dir, max_length=max_text_length
        )
        if tuning_type == "lora":
            from peft import LoraConfig, TaskType, get_peft_model

            model = get_peft_model(
                model,
                LoraConfig(
                    r=lora_rank,
                    lora_alpha=lora_alpha,
                    target_modules=["q", "v"],
                    lora_dropout=lora_dropout,
                    bias="none",
                    task_type=TaskType.SEQ_2_SEQ_LM,
                ),
            )
        else:
            model.requires_grad_(False)
            model.eval()
        return cls(model, tokenizer, max_text_length, force_eval=tuning_type == "freeze")

    def train(self, mode: bool = True) -> FlanT5Backbone:
        super().train(mode)
        if self.force_eval:
            self.model.eval()
        return self

    def _tokenize(self, texts: list[str] | tuple[str, ...], truncate: bool) -> TokenBatch:
        encoded = self.tokenizer(
            list(texts),
            padding="longest",
            truncation=truncate,
            max_length=self.max_text_length if truncate else None,
            return_tensors="pt",
        )
        return TokenBatch(encoded["input_ids"], encoded["attention_mask"].bool())

    def target_embeddings(
        self, texts: list[str] | tuple[str, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self._tokenize(texts, truncate=False)
        device = next(self.model.parameters()).device
        input_ids = tokens.input_ids.to(device)
        mask = tokens.attention_mask.to(device)
        return self.model.encoder.embed_tokens(input_ids), mask

    def _joint_inputs(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prompt_tokens = self._tokenize(prompts, truncate=True)
        prompt_ids = prompt_tokens.input_ids.to(visual.device)
        prompt_mask = prompt_tokens.attention_mask.to(visual.device)
        prompt_embeddings = self.model.encoder.embed_tokens(prompt_ids)
        visual = visual.to(dtype=prompt_embeddings.dtype)
        samples = [
            torch.cat((visual[i, visual_mask[i]], prompt_embeddings[i, prompt_mask[i]]), dim=0)
            for i in range(visual.shape[0])
        ]
        lengths = torch.tensor([sample.shape[0] for sample in samples], device=visual.device)
        joint = pad_sequence(samples, batch_first=True)
        mask = torch.arange(joint.shape[1], device=visual.device)[None, :] < lengths[:, None]
        return joint, mask

    def translation_loss(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
        targets: tuple[str, ...],
    ) -> torch.Tensor:
        joint, joint_mask = self._joint_inputs(visual, visual_mask, prompts)
        target_tokens = self._tokenize(targets, truncate=False)
        labels = target_tokens.input_ids.to(visual.device)
        labels = labels.masked_fill(labels == self.tokenizer.pad_token_id, -100)
        output = self.model(
            inputs_embeds=joint,
            attention_mask=joint_mask,
            decoder_attention_mask=target_tokens.attention_mask.to(visual.device),
            labels=labels,
            return_dict=True,
        )
        return output.loss

    @torch.no_grad()
    def generate_text(
        self,
        visual: torch.Tensor,
        visual_mask: torch.Tensor,
        prompts: list[str],
        mode: str,
        beam_size: int,
        max_length: int,
    ) -> list[str]:
        if mode not in {"upstream", "deterministic"}:
            raise ValueError(f"unsupported generation mode: {mode}")
        joint, joint_mask = self._joint_inputs(visual, visual_mask, prompts)
        generation_kwargs = {
            "inputs_embeds": joint,
            "attention_mask": joint_mask,
            "num_beams": beam_size,
            "max_length": max_length,
            "do_sample": mode == "upstream",
        }
        if mode == "upstream":
            generation_kwargs["top_p"] = 0.9
        generated = self.model.generate(**generation_kwargs)
        return [
            text.lower()
            for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
        ]

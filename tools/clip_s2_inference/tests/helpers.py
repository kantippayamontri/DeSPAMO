import torch
from transformers import T5Config, T5ForConditionalGeneration

from despamo.models.flan_t5 import FlanT5Backbone


class Tokenizer:
    pad_token_id = 0

    def __call__(self, texts, *, padding, return_tensors, truncation, max_length=None):
        rows = [[3 + len(word) % 20 for word in text.split()] or [3] for text in texts]
        if truncation and max_length is not None:
            rows = [row[:max_length] for row in rows]
        ids = torch.zeros(len(rows), max(map(len, rows)), dtype=torch.long)
        mask = torch.zeros_like(ids)
        for n, row in enumerate(rows):
            ids[n, : len(row)] = torch.tensor(row)
            mask[n, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, ids, *, skip_special_tokens):
        return [" ".join(f"TOKEN{int(i)}" for i in row if int(i) != 0) for row in ids]


def language():
    torch.set_num_threads(2)
    torch.manual_seed(3)
    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        decoder_start_token_id=0,
        pad_token_id=0,
        eos_token_id=None,
        use_cache=False,
        dropout_rate=0.0,
    )
    model = T5ForConditionalGeneration(config)
    model.gradient_checkpointing_enable()
    return FlanT5Backbone(model, Tokenizer(), 1024)

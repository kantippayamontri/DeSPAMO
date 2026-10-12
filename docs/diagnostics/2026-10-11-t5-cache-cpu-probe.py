"""CPU-only audit of the installed T5 generation cache behavior.

This is a small diagnostic, not a production model or a GPU stability test.
It reproduces the training config's use_cache=False and compares generation
against an explicit inference-only cache override without changing that config.
"""

import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from transformers import T5Config, T5ForConditionalGeneration  # noqa: E402


def main():
    torch.set_num_threads(2)
    torch.manual_seed(0)
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
    )
    model = T5ForConditionalGeneration(config).eval()
    model.gradient_checkpointing_enable()
    assert next(model.parameters()).device.type == "cpu"
    ids = torch.tensor([[3, 4, 5], [6, 7, 8], [9, 10, 11], [12, 13, 14]])
    observed = []

    def hook(module, args, kwargs):
        observed.append(tuple(kwargs["input_ids"].shape))

    handle = model.decoder.register_forward_pre_hook(hook, with_kwargs=True)
    results, outputs = {}, {}
    try:
        for label, extra in (
            ("current_generation_default", {}),
            ("inference_cache_enabled", {"use_cache": True}),
        ):
            observed.clear()
            with torch.inference_mode():
                outputs[label] = model.generate(
                    inputs_embeds=model.get_input_embeddings()(ids),
                    attention_mask=torch.ones_like(ids),
                    num_beams=5,
                    max_length=12,
                    do_sample=False,
                    **extra,
                )
            results[label] = dict(
                decoder_shapes=list(observed),
                total_decoder_token_positions=sum(b * length for b, length in observed),
                training_config_use_cache=model.config.use_cache,
            )
    finally:
        handle.remove()
    equal = torch.equal(outputs["current_generation_default"], outputs["inference_cache_enabled"])
    assert equal and model.config.use_cache is False
    print(json.dumps(dict(results=results, same_generated_ids=equal, cpu_only=True), indent=2))


if __name__ == "__main__":
    main()

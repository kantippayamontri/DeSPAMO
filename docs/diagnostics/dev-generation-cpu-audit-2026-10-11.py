"""CPU-only checks of original dev input/adapter and installed generation behavior."""

import argparse
import json
import random
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tools.benchmark import translation_data as dino_data
from tools.benchmark.translation_model import plain_adapter
from tools.clip_s2 import data as clip_data
from tools.clip_s2 import train as clip_train
from tools.followup.recipe import PromptPool
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

from despamo.models.visual_adapter import SpaMoVisualAdapter
from tools.dinov3.storage import atomic_json

ROOT = clip_train.BASE / "diagnostics/dev-generation-code-audit-2026-10-11"
CLIP = (
    clip_train.BASE
    / "runs"
    / "translation-reference-seed0-fe5c22c66305d9e3bea3074792c0f64955b2c01c4b61bf92656448c0f7ae5671"
    / "resume-epoch-2.pt"
)
DINO = (
    Path("/mnt/b/DeSpaMo/archives/official-phoenix14t-translation-spamo200-v1")
    / "translation-reference-seed0-90105640d6d36be42fa5f5a3742b296b1fcb8b75d519aaee47b6833a63404cf5"
    / "model-epoch-2.pt"
)
SNAPSHOT = (
    Path(clip_train.CACHE)
    / "models--google--flan-t5-xl/snapshots"
    / "7d6315df2c2fb742f0f5b556879d730926ca9001"
)


def features() -> dict:
    config = clip_train.load_config(clip_train.BASE / "study-inputs.json")
    admitted = {
        "clip": clip_data.admit_data(
            Path(config["spatial_root"]), config["spatial_manifest_sha256"], verify_arrays=False
        ),
        "dino": dino_data.admit_data(verify_arrays=False),
    }
    assert admitted["clip"]["ids"] == admitted["dino"]["ids"]
    tokenizer = AutoTokenizer.from_pretrained(str(SNAPSHOT), local_files_only=True)
    result = {}
    for name, checkpoint, module in [("clip", CLIP, clip_data), ("dino", DINO, dino_data)]:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = plain_adapter(SpaMoVisualAdapter(2048, 1024, 768, 2048), 0).eval()
        weights = {
            key.removeprefix("visual_adapter."): value
            for key, value in state["weights"].items()
            if key.startswith("visual_adapter.")
        }
        model.load_state_dict(weights, strict=True)
        pool = PromptPool(admitted[name]["datasets"]["train"].records, 0)
        prompts, _ = pool.inference(admitted[name]["ids"]["dev"][:1])
        tokens = tokenizer(prompts, padding=True, return_tensors="pt")
        prompt_length = int(tokens["attention_mask"].sum())
        info = dict(
            checkpoint_step=state["step"],
            prompt_tokens=prompt_length,
            prompt_token_id_max=int(tokens["input_ids"].max()),
            clips=0,
            batches=0,
            spatial_absmax=0.0,
            visual_absmax=0.0,
            min_visual_tokens=10**9,
            max_visual_tokens=0,
            max_spatial_frames=0,
            max_joint_tokens=0,
            nonfinite_inputs=0,
            nonfinite_visual_outputs=0,
            invalid_masks=0,
            max_joint_batch_ids=[],
        )
        with torch.inference_mode():
            for batch in module.eval_batches(admitted[name], "dev", batch_size=4):
                info["clips"] += len(batch.clip_ids)
                info["batches"] += 1
                info["nonfinite_inputs"] += int(not torch.isfinite(batch.spatial).all())
                info["nonfinite_inputs"] += int(not torch.isfinite(batch.motion).all())
                info["spatial_absmax"] = max(
                    info["spatial_absmax"], float(batch.spatial.abs().max())
                )
                info["max_spatial_frames"] = max(
                    info["max_spatial_frames"], int(batch.spatial_mask.sum(1).max())
                )
                with torch.autocast("cpu", dtype=torch.bfloat16):
                    visual, valid = model(
                        batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
                    )
                info["nonfinite_visual_outputs"] += int(not torch.isfinite(visual[valid]).all())
                lengths = valid.sum(1)
                info["invalid_masks"] += int(
                    valid.dtype != torch.bool
                    or valid.shape != visual.shape[:2]
                    or not (lengths > 0).all()
                )
                info["visual_absmax"] = max(info["visual_absmax"], float(visual[valid].abs().max()))
                info["min_visual_tokens"] = min(info["min_visual_tokens"], int(lengths.min()))
                info["max_visual_tokens"] = max(info["max_visual_tokens"], int(lengths.max()))
                joint = int(lengths.max()) + prompt_length
                if joint > info["max_joint_tokens"]:
                    info["max_joint_tokens"] = joint
                    info["max_joint_batch_ids"] = list(batch.clip_ids)
        assert info["clips"] == 519
        result[name] = info
        print(json.dumps({name: info}), flush=True)
    return result


def decoder() -> dict:
    torch.manual_seed(0)
    model = T5ForConditionalGeneration(
        T5Config(
            vocab_size=32128,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0,
            decoder_start_token_id=0,
            eos_token_id=1,
            pad_token_id=0,
        )
    ).to(torch.bfloat16)
    model.gradient_checkpointing_enable()
    model.eval()
    inputs = torch.randn(4, 24, 16, dtype=torch.bfloat16)
    mask = torch.ones(4, 24, dtype=torch.bool)
    results = {}
    for cache in [False, True]:
        decoder_lengths, topk, logits = [], [], []
        hook = model.decoder.register_forward_pre_hook(
            lambda module, args, kwargs, record=decoder_lengths: record.append(
                kwargs["input_ids"].shape[1]
            ),
            with_kwargs=True,
        )
        original_topk, original_softmax = torch.topk, torch.nn.functional.log_softmax

        def observed_topk(value, k, *args, record=topk, original=original_topk, **kwargs):
            record.append(dict(shape=list(value.shape), dtype=str(value.dtype), k=k))
            return original(value, k, *args, **kwargs)

        def observed_softmax(value, *args, record=logits, original=original_softmax, **kwargs):
            record.append(
                dict(
                    shape=list(value.shape),
                    dtype=str(value.dtype),
                    contiguous=value.is_contiguous(),
                    finite=bool(torch.isfinite(value).all()),
                )
            )
            return original(value, *args, **kwargs)

        with (
            torch.inference_mode(),
            patch("torch.topk", observed_topk),
            patch("torch.nn.functional.log_softmax", observed_softmax),
        ):
            generated = model.generate(
                inputs_embeds=inputs,
                attention_mask=mask,
                num_beams=5,
                max_length=8,
                min_length=8,
                do_sample=False,
                use_cache=cache,
            )
        hook.remove()
        results[str(cache)] = dict(
            decoder_lengths=decoder_lengths,
            decoder_token_positions=sum(decoder_lengths) * 20,
            topk=topk,
            logits=logits,
            ids=generated.tolist(),
        )
    with torch.inference_mode():
        long = model.encoder(
            inputs_embeds=torch.randn(1, 600, 16, dtype=torch.bfloat16),
            attention_mask=torch.ones(1, 600, dtype=torch.bool),
        )
    results["relative_position_600_tokens_finite"] = bool(
        torch.isfinite(long.last_hidden_state).all()
    )
    results["generated_ids_equal"] = results["False"]["ids"] == results["True"]["ids"]
    return results


def cleanup() -> dict:
    model = torch.nn.Linear(1, 1)
    model.rng = random.Random(0)
    with (
        patch.object(clip_train, "capture_state", return_value={}),
        patch.object(
            clip_train, "candidate_to_archive", return_value=Path("/tmp/opencode/dummy.pt")
        ),
        patch.object(clip_train, "capture_rng", return_value={}),
        patch.object(
            clip_train,
            "score_batches",
            side_effect=RuntimeError("PRIMARY simulated generation fault"),
        ),
        patch.object(
            clip_train,
            "restore_rng",
            side_effect=RuntimeError("SECONDARY simulated CUDA RNG fault"),
        ),
    ):
        try:
            clip_train.score_dev(
                model,
                None,
                {"ids": {"dev": ("dummy",)}},
                None,
                {"data": {"ordered_ids": {"dev": "dummy"}}},
                Path("/tmp/opencode/no-dev-report-for-cpu-audit"),
                2,
                "dummy",
                SimpleNamespace(state_dict=lambda: {}),
            )
        except RuntimeError as error:
            return dict(propagated=str(error), primary_context=str(error.__context__))
    raise AssertionError("Expected injected failure did not propagate")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("features", "decoder", "cleanup"))
    args = parser.parse_args()
    assert not torch.cuda.is_initialized()
    result = {"features": features, "decoder": decoder, "cleanup": cleanup}[args.phase]()
    assert not torch.cuda.is_initialized()
    atomic_json(ROOT / f"{args.phase}.json", result)
    print(json.dumps(result), flush=True)

"""Measure upstream batch-ICL reference exposure on official test.

Only the prompt differs between modes. Decoding is identical: one clip per
generate call, beam5, max64, deterministic, cached.

- clean: fixed training exemplar (paper Appendix A.2).
- leaky: upstream `derangement` within the annotation-ordered test batch of 2
  (DataLoader shuffle=False, batch_size=2), i.e. each clip's prompt contains its
  batch partner's German test reference.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from despamo.data.batch import collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.evaluation.metrics import evaluate_translations
from despamo.factory import build_model
from tools.clip_plain.generation import generate_cached_text
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json
from tools.followup.recipe import PromptPool
from tools.spamo_repro.policy import REVISION

ANNOTATIONS = Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T")
RELEASED = Path("/home/kan/datasets/spamo/ckpt/despamo-spamo-baseline.pt")
RELEASED_SHA = "44ace3e8536817691f6c6b3104f8881fd65e6e1ac8ff92364a4af16efd93f45f"


def leaky_partners(count: int) -> list[int]:
    """Upstream batch2 derangement: swap within each consecutive pair."""
    if count % 2:
        raise ValueError("Even test count required for batch2 pairing")
    return [i + 1 if i % 2 == 0 else i - 1 for i in range(count)]


def model_config() -> dict:
    return {
        "seed": 0,
        "comparison": {"enabled": True},
        "optimizer": {"learning_rate": 1e-4, "weight_decay": 0.01},
        "model": {
            "name": "google/flan-t5-xl",
            "revision": REVISION,
            "cache_dir": "/home/kan/datasets/spamo/hf_cache",
            "max_text_length": 512,
            "spatial_dim": 2048,
            "motion_dim": 1024,
            "adapter_dim": 768,
            "language_dim": 2048,
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "prompt": "Translate the given sentence into {}.",
            "use_in_context": True,
            "num_in_context": 3,
            "vt_pooling": "legacy_mean",
            "vt_weight": 1.0,
            "warm_up_steps": 4000,
        },
    }


def load_model(checkpoint: Path):
    model = build_model(OmegaConf.create(model_config()))
    if checkpoint == RELEASED:
        if sha256_file(checkpoint) != RELEASED_SHA:
            raise ValueError("Released checkpoint hash mismatch")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state["state_dict"], strict=True)
    else:
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        result = model.load_state_dict(state["weights"], strict=False)
        if result.unexpected_keys or any(k in result.missing_keys for k in state["weights"]):
            raise ValueError("Reproduction checkpoint key mismatch")
    del state
    model.requires_grad_(False).eval().to("cuda")
    return model


def records(split: str) -> list[dict]:
    raw = np.load(ANNOTATIONS / f"{split}_info_ml.npy", allow_pickle=True).item()
    return [raw[k] for k in sorted(k for k in raw if type(k) is int)]


def run(checkpoint: Path, label: str, output: Path, limit: int | None) -> None:
    root = Path("/home/kan/datasets/spamo/features")
    dataset = Phoenix14T(
        ANNOTATIONS / "test_info_ml.npy",
        "test",
        root / "vit_feat_Phoenix14T",
        root / "mae_feat_Phoenix14T",
        FeatureManifest.load(root / "manifests/phoenix14t_spatial.json"),
        FeatureManifest.load(root / "manifests/phoenix14t_motion.json"),
        max_frame_len=512,
        spatial_crop_mode="full",
    )
    test_rows = dataset.records
    if len(test_rows) != 642:
        raise ValueError("Official 642 test clips required")
    pool = PromptPool(records("train"), 0)
    pool.guard_references(tuple(r["text"] for r in test_rows))
    partners = leaky_partners(len(test_rows))
    model = load_model(checkpoint)
    scope = {
        "approved": True,
        "consumed": False,
        "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
    }
    count = len(test_rows) if limit is None else limit
    results = {}
    for mode in ("clean", "leaky"):
        items, start = [], time.time()
        for i in range(count):
            batch = collate_phoenix([dataset[i]])
            batch = replace(
                batch,
                spatial=batch.spatial.cuda(),
                spatial_mask=batch.spatial_mask.cuda(),
                motion=batch.motion.cuda(),
                motion_mask=batch.motion_mask.cuda(),
            )
            if mode == "clean":
                prompt, source = pool.inference(batch.clip_ids)[0][0], pool.eval_id
            else:
                partner = test_rows[partners[i]]
                prompt, source = pool.render(partner), partner["fileid"]
                if partner["fileid"] == batch.clip_ids[0]:
                    raise ValueError("Derangement must not use own reference")
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                visual, mask = model.visual_adapter(
                    batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
                )
                out = generate_cached_text(
                    model.language_model,
                    visual,
                    mask,
                    [prompt],
                    on_event=lambda e: None,
                    scope=scope,
                )
            items.append(
                {
                    "clip_id": batch.clip_ids[0],
                    "prompt_source": source,
                    "prediction": out["texts"][0],
                    "reference": batch.texts[0].lower(),
                }
            )
            if (i + 1) % 100 == 0:
                print(
                    f"{label} {mode} {i + 1}/{count} {(i + 1) / (time.time() - start):.2f} clip/s",
                    flush=True,
                )
        metrics = evaluate_translations(
            [x["prediction"] for x in items], [x["reference"] for x in items]
        )
        copied = sum(
            x["prediction"].strip(" .") == test_rows[partners[j]]["text"].lower().strip(" .")
            for j, x in enumerate(items)
        )
        results[mode] = {
            "metrics": metrics,
            "clips": len(items),
            "predictions_equal_partner_reference": copied,
            "items": items,
        }
        print(
            f"{label} {mode} BLEU4 {metrics['bleu4']:.4f} ROUGE-L {metrics['rougeL_f1']:.4f} "
            f"partner-copies {copied}",
            flush=True,
        )
    report = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "label": label,
        "split": "test",
        "complete": limit is None,
        "decoding": {
            "batch_size": 1,
            "beam_size": 5,
            "max_length": 64,
            "do_sample": False,
            "use_cache": True,
            "prompt_truncation": 512,
        },
        "leaky_policy": "upstream batch2 annotation-order derangement (pair swap)",
        "clean_exemplar": pool.eval_id,
        "delta_bleu4_leaky_minus_clean": results["leaky"]["metrics"]["bleu4"]
        - results["clean"]["metrics"]["bleu4"],
        **results,
    }
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(output / f"icl-leak-{label}{'' if limit is None else '-smoke'}.json", report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run(args.checkpoint, args.label, args.output, args.limit)


if __name__ == "__main__":
    main()

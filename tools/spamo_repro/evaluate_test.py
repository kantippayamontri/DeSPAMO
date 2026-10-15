"""Evaluate a trained reproduction checkpoint on the official 642 Test clips.

Strict train-only in-context learning exemplar, batch_size=1, beam_size=5, deterministic, cached.
Zero test ground-truth leakage into prompts.
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from despamo.data.batch import collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.evaluation.metrics import evaluate_translations
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from tools.benchmark.compact_checkpoint import frozen_fingerprint, mutable_keys
from tools.clip_plain.generation import generate_cached_text
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json
from tools.followup.recipe import PromptPool
from tools.spamo_repro.model import build_visual_adapter
from tools.spamo_repro.policy import REVISION, validate_config

PER_EPOCH = 444


class PaperModel(torch.nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        torch.manual_seed(config["seed"])
        self.visual_adapter = build_visual_adapter(config)
        self.language_model = FlanT5Backbone.from_pretrained(
            "google/flan-t5-xl",
            config["model"]["cache_dir"],
            512,
            "lora",
            16,
            32,
            0.1,
            revision=REVISION,
        )
        self.vt_align = VTAlignLoss()


def load_test_dataset(config: dict) -> tuple[Phoenix14T, tuple[str, ...]]:
    data_cfg = config["data"]
    annotation = Path(data_cfg["annotation_root"]) / "test_info_ml.npy"
    spat_root = Path(data_cfg["spatial_root"])
    mot_root = Path(data_cfg["motion_root"])
    spat_manifest = FeatureManifest.load(Path(data_cfg["spatial_manifest"]))
    mot_manifest = FeatureManifest.load(Path(data_cfg["motion_manifest"]))
    dataset = Phoenix14T(
        annotation,
        "test",
        spat_root,
        mot_root,
        spat_manifest,
        mot_manifest,
        max_frame_len=512,
        spatial_crop_mode="full",
    )
    test_ids = tuple(r["fileid"] for r in dataset.records)
    if len(test_ids) != 642:
        raise ValueError(f"Expected 642 test clips, got {len(test_ids)}")
    return dataset, test_ids


def load_train_pool(config: dict) -> PromptPool:
    data_cfg = config["data"]
    annotation = Path(data_cfg["annotation_root"]) / "train_info_ml.npy"
    raw = np_load_dict(annotation)
    records = [raw[k] for k in sorted(k for k in raw if isinstance(k, int))]
    return PromptPool(records, seed=0)


def np_load_dict(path: Path) -> dict:
    import numpy as np

    return np.load(path, allow_pickle=True).item()


def evaluate_test(
    config: dict,
    checkpoint_path: Path,
    output_dir: Path,
    epoch: int,
    limit: int | None = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on {device}...")
    model = PaperModel(config)
    model.to(device)
    model.eval()
    base_hash = frozen_fingerprint(model)

    print(f"Loading checkpoint {checkpoint_path}...")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if state["frozen_fingerprint"] != base_hash or set(state["weights"]) != mutable_keys(model):
        raise ValueError("Checkpoint base fingerprint or mutable keys mismatch")
    model.load_state_dict(state["weights"], strict=False)
    del state

    print("Loading test data and train prompt pool...")
    test_dataset, test_ids = load_test_dataset(config)
    lookup = {r["fileid"]: idx for idx, r in enumerate(test_dataset.records)}
    pool = load_train_pool(config)
    pool.guard_references(tuple(r["text"] for r in test_dataset.records))

    eval_ids = test_ids if limit is None else test_ids[:limit]
    print(f"Evaluating {len(eval_ids)} clips (exemplar: {pool.eval_id})...")

    events = {"starts": 0, "logits": 0}

    def observe(event: dict) -> None:
        if event["phase"] == "decoder_start":
            if event["decoder_length"] != 1 or event["beam_rows"] != 5:
                raise ValueError("Cached decoder evidence mismatch")
            events["starts"] += 1
        elif event["phase"] == "decoder_logits":
            if event["finite"] is not True or event["shape"][0] != 5:
                raise ValueError("Invalid raw decoder logits")
            events["logits"] += 1

    boot_id_str = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    scope = {"approved": True, "boot_id": boot_id_str, "consumed": False}

    items = []
    start_time = time.time()
    for idx, clip in enumerate(eval_ids):
        sample = test_dataset[lookup[clip]]
        batch = collate_phoenix([sample])
        batch = replace(
            batch,
            spatial=batch.spatial.to(device),
            spatial_mask=batch.spatial_mask.to(device),
            motion=batch.motion.to(device),
            motion_mask=batch.motion_mask.to(device),
        )
        prompts, exemplars = pool.inference(batch.clip_ids)
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16):
            visual, mask = model.visual_adapter(
                batch.spatial, batch.spatial_mask, batch.motion, batch.motion_mask
            )
            result = generate_cached_text(
                model.language_model, visual, mask, prompts, on_event=observe, scope=scope
            )
        pred = result["texts"][0]
        ref = batch.texts[0].lower()
        items.append(
            {
                "clip_id": clip,
                "prediction": pred,
                "reference": ref,
                "token_ids": result["token_ids"][0],
                "exemplar_id": exemplars[0],
            }
        )
        if (idx + 1) % 50 == 0 or (idx + 1) == len(eval_ids):
            elapsed = time.time() - start_time
            rate = (idx + 1) / elapsed
            print(f"[{idx + 1}/{len(eval_ids)}] clips completed ({rate:.2f} clips/s)")

    if events["starts"] != events["logits"] or not events["starts"]:
        raise ValueError("Decoder evidence incomplete")

    metrics = evaluate_translations(
        [i["prediction"] for i in items], [i["reference"] for i in items]
    )
    print("\n--- Test Metrics ---")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    report = {
        "split": "test",
        "epoch": epoch,
        "complete": limit is None,
        "clips": len(items),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "metrics": metrics,
        "decoder_steps": events["starts"],
        "exemplar_id": pool.eval_id,
        "decoding": {
            "batch_size": 1,
            "beam_size": 5,
            "max_length": 64,
            "do_sample": False,
            "use_cache": True,
        },
        "items": items,
        "elapsed_seconds": time.time() - start_time,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"test-epoch-{epoch}.json"
    atomic_json(out_file, report)
    print(f"Saved test evaluation report to {out_file}")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epoch", type=int, default=36)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())
    validate_config(config)
    evaluate_test(config, args.checkpoint, args.output_dir, args.epoch, limit=args.limit)


if __name__ == "__main__":
    main()

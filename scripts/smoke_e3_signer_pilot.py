"""Bounded E3 data-alignment and one-step DINO LoRA gradient smoke."""

import argparse
import json
import signal
import time
from pathlib import Path

from tools.dinov3.identity import digest
from tools.dinov3.identity import sha256_file as file_hash
from tools.dinov3.storage import atomic_json

PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
DATASET_KEY = "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"
BASE_SHA = "ea8dc2863c51be0a264bab82070e3e8836b02d51"


def choose_smoke_clips(protocol: dict, source: dict, text: dict) -> tuple[str, str, str]:
    """Accept nullable targets on valid clips; mask only wholly failed clips."""
    if (protocol.get("protocol_hash") != digest({k: v for k, v in protocol.items()
                                                if k != "protocol_hash"})
            or protocol.get("source_hash") != digest(source)):
        raise ValueError("E3 protocol/source content hash mismatch")
    groups = protocol["split"]["groups"]
    train, dev, test = (tuple(groups[name]["clip_ids"]) for name in ("train", "dev", "test"))
    if len(set(train + dev + test)) != len(train + dev + test):
        raise ValueError("E3 train/dev/test clip overlap")
    clips = {clip["clip_id"]: clip for clip in source["clips"]}
    records = {record["clip_id"]: record for record in text["records"]}
    if (len(clips) != len(source["clips"]) or len(records) != len(text["records"])
            or not set(train) <= clips.keys() or not set(train) <= records.keys()):
        raise ValueError("E3 missing or duplicate source/target clips")
    valid, failed = [], []
    for clip_id in train:
        clip, row = clips[clip_id], records[clip_id]
        if len(clip["frames"]) != 5 or len(row["targets"]) != 19:
            raise ValueError("E3 source frames/target count mismatch")
        values = [target.get("text") for target in row["targets"]]
        if any(value is not None and (not isinstance(value, str) or not value) for value in values):
            raise ValueError("E3 invalid factor target text")
        (valid if any(value is not None for value in values) else failed).append(clip_id)
    if len(valid) != groups["train"]["valid"] or len(failed) != groups["train"]["failed"]:
        raise ValueError("E3 factor valid/failed count mismatch")
    if len(valid) < 2 or not failed:
        raise ValueError("E3 smoke needs two valid and one fully failed clip")
    return valid[0], valid[1], failed[0]


def build_smoke_summary(protocol: dict, selected: tuple[str, ...], counts: list[int]) -> dict:
    """Use JSON-native fields so durable receipt round-trips unchanged."""
    return {"protocol_hash": PROTOCOL_HASH, "source_hash": protocol["source_hash"],
            "text_manifest_hash": protocol["text_manifest_hash"],
            "frame_rows_hash": protocol["frame_rows_hash"],
            "selected_train_clip_ids": list(selected), "target_counts": counts}


def smoke(args) -> dict:
    """Verify frozen sources, then run one real pinned DINO LoRA optimizer step."""
    import numpy as np

    if args.output.exists() or args.output.is_symlink():
        raise ValueError("E3 smoke output exists; preserving prior artifacts")
    if args.dataset.name != DATASET_KEY or args.protocol.parent.name != PROTOCOL_HASH:
        raise ValueError("E3 smoke frozen dataset/protocol path mismatch")
    protocol = json.loads(args.protocol.read_text())
    source = json.loads((args.dataset / "source.json").read_text())
    text = json.loads(args.text_manifest.read_text())
    rows_path = args.dino_root / "complete/frame_rows.json"
    if (protocol.get("protocol_hash") != PROTOCOL_HASH
            or file_hash(args.text_manifest) != protocol.get("text_manifest_hash")
            or file_hash(rows_path) != protocol.get("frame_rows_hash")
            or file_hash(args.dino_root / "complete/manifest.json")
            != protocol.get("spatial_manifest_hash")):
        raise ValueError("E3 frozen text/DINO identity mismatch")
    selected = choose_smoke_clips(protocol, source, text)
    rows = json.loads(rows_path.read_text())["clips"]
    clips = {clip["clip_id"]: clip for clip in source["clips"]}
    entries = {entry["clip_id"]: entry for entry in text["records"]}
    teacher, vectors, masks, labels, paths = [], [], [], [], []
    for clip_id in selected:
        clip, entry, row = clips[clip_id], entries[clip_id], rows[clip_id]
        spatial = args.dino_root / "train" / f"{clip_id}.npy"
        if (file_hash(spatial) != row["feature_hash"]
                or row["source_indices"] != list(range(clip["frame_count"]))):
            raise ValueError(f"E3 original DINO frame alignment failed: {clip_id}")
        original = np.load(spatial, allow_pickle=False)
        if original.shape != (clip["frame_count"], 2048) or not np.isfinite(original).all():
            raise ValueError(f"E3 original DINO feature shape/values invalid: {clip_id}")
        teacher.append(original[[frame["index"] for frame in clip["frames"]]])
        for frame in clip["frames"]:
            image = (args.frames / frame["path"]).resolve()
            if args.frames.resolve() not in image.parents or file_hash(image) != frame["sha256"]:
                raise ValueError(f"E3 source PNG changed: {clip_id}")
            paths.append(image)
        target = (args.text_manifest.parent / entry["path"]).resolve()
        if (target.parent != args.text_manifest.parent.resolve()
                or file_hash(target) != entry["file_hash"]):
            raise ValueError(f"E3 factor NPZ changed: {clip_id}")
        with np.load(target, allow_pickle=False) as saved:
            values, valid = saved["vectors"].copy(), saved["valid"].copy()
        expected = np.array([item["text"] is not None for item in entry["targets"]])
        if (values.shape != (19, 768) or valid.shape != (19,) or valid.dtype != np.bool_
                or not np.array_equal(valid, expected) or np.any(values[~valid])):
            raise ValueError(f"E3 factor masks/values not aligned: {clip_id}")
        vectors.append(values)
        masks.append(valid)
        labels.append(tuple(item["text"] or "" for item in entry["targets"]))
    summary = build_smoke_summary(
        protocol, selected, [int(np.count_nonzero(mask)) for mask in masks]
    )
    if args.check_inputs:
        return {"status": "inputs_valid", **summary}

    import torch
    from tools.dinov3.adapt_model import attach_late_lora, scale_reference_loss
    from tools.dinov3.adapt_train import DINOFactorHeads
    from transformers import AutoModel

    from tools.dinov3.frames import preprocess
    from tools.dinov3.identity import MODEL

    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E3 smoke requires 12 GiB free CUDA memory")
    torch.manual_seed(0)
    started = time.monotonic()
    model = AutoModel.from_pretrained(
        MODEL, revision=BASE_SHA, cache_dir=args.cache, local_files_only=True,
        use_safetensors=True, trust_remote_code=False,
    ).eval().to("cuda")
    if (model.config.hidden_size != 1024 or model.config.patch_size != 16
            or model.config.num_register_tokens != 4):
        raise ValueError("E3 pinned ViT-L model incompatible")
    targets = attach_late_lora(model)
    heads = DINOFactorHeads().to("cuda")
    images224 = torch.stack([preprocess(path, 224) for path in paths]).to("cuda")
    images448 = torch.stack([preprocess(path, 448) for path in paths]).to("cuda")
    original = torch.from_numpy(np.stack(teacher)).float().to("cuda")
    text_vectors = torch.from_numpy(np.stack(vectors)).float().to("cuda")
    valid_mask = torch.from_numpy(np.stack(masks)).bool().to("cuda")
    optimizer = torch.optim.AdamW([
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": 1e-4},
        {"params": list(heads.parameters()), "lr": 3e-4},
    ])
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        adapted = torch.cat((model(pixel_values=images224).last_hidden_state[:, 0].float(),
                             model(pixel_values=images448).last_hidden_state[:, 0].float()), dim=-1)
        factors = heads(adapted.reshape(3, 5, 2048), text_vectors, valid_mask, tuple(labels), 1.0)
        reference = scale_reference_loss(adapted, original.reshape(-1, 2048))
        loss = sum((0.05 if name in ("biometric", "clothing", "hair", "background") else 0.10)
                   * value for name, value in factors.items()) + 0.10 * reference
    if not torch.isfinite(loss):
        raise FloatingPointError("E3 smoke loss non-finite")
    loss.backward()
    lora_grads = {name: float(param.grad.float().norm()) for name, param in model.named_parameters()
                  if name.endswith("up.weight") and param.grad is not None}
    head_grads = {
        name: float(module.weight.grad.float().norm())
        for name, module in heads.heads.items() if module.weight.grad is not None
    }
    if (len(lora_grads) != len(targets) or not any(value > 0 for value in lora_grads.values())
            or not any(value > 0 for value in head_grads.values())
            or not all(np.isfinite(value) for value in
                       (*lora_grads.values(), *head_grads.values()))):
        raise FloatingPointError("E3 LoRA/factor gradients missing or non-finite")
    optimizer.step()
    torch.cuda.synchronize()
    recovery = args.recovery_root / "tools/dinov3"
    result = {
        "status": "complete", "purpose": "one_step_gradient_smoke", **summary,
        "supervision_policy": "qwen-schema98-unreviewed-v1", "human_review_status": "not_assessed",
        "base_revision": BASE_SHA, "loss": float(loss.detach()),
        "reference_loss": float(reference.detach()),
        "factor_losses": {name: float(value.detach()) for name, value in factors.items()},
        "nonzero_lora_up_gradients": sum(value > 0 for value in lora_grads.values()),
        "lora_targets": len(targets),
        "nonzero_factor_head_gradients": sum(value > 0 for value in head_grads.values()),
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "elapsed_seconds": time.monotonic() - started,
        "code_hashes": {"smoke_e3_signer_pilot.py": file_hash(Path(__file__)),
                        **{name: file_hash(recovery / name) for name in
                           ("adapt_model.py", "adapt_train.py", "frames.py", "identity.py")}},
        "lock_hash": file_hash(recovery / "uv.lock"),
    }
    atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "frames", "cache",
                 "recovery-root", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--max-wall-seconds", type=int, default=300)
    args = parser.parse_args()
    if not 1 <= args.max_wall_seconds <= 600:
        parser.error("E3 smoke wall cap must be 1–600 seconds")

    def timeout(signum, frame):
        raise TimeoutError("E3 smoke cap")

    if not args.check_inputs:
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.max_wall_seconds)
    try:
        print(json.dumps(smoke(args), sort_keys=True))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

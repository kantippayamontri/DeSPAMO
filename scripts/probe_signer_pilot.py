"""Compare signer information in selected E1/E2 spatial projectors."""

import argparse
import gc
import time
from pathlib import Path

import torch

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.evaluation.pilot_probe import (
    pool_spatial,
    project_pooled,
    score_probe,
    split_probe_clips,
)
from despamo.training.pilot_runtime import load_pilot_views

PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
SHARED_HASH = "9e3a02201ca44b7ee1a1d62da3eba1308a91ee914cab983036dc6505aa8a4d88"
CHECKPOINT_HASHES = {
    "e1": "04ce223fd1ae00362a7ceaa1e78c1a814c825a8b9551483d8ac2a332631c7755",
    "e2": "7d73af1b8c3a190dd6b9f9725f7ebb01e63e3258837bb322e73494fb1b65575e",
}
POLICIES = {"e1": ("E1_frozen", "signer-pilot-e1-fresh-retry-v3"),
            "e2": ("E2_projector", "signer-pilot-e2-projector-bounded-v1")}


def _validate_run(root: Path, variant: str, protocol: dict) -> dict:
    if not root.is_dir():
        raise ValueError("pilot run root missing")
    status = read_json(root / "run-status.json")
    budget = read_json(root / "budget.json")
    selected = read_json(root / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    expected_variant, expected_policy = POLICIES[variant]
    if (status.get("status") != "complete" or status.get("global_step") != 4000
            or budget.get("status") != "closed"
            or identity.get("run_policy") != expected_policy
            or identity.get("variant") != expected_variant
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("initial_shared_hash") != SHARED_HASH
            or (variant == "e2" and (
                identity.get("matched_e1_checkpoint_hash") != CHECKPOINT_HASHES["e1"]
                or identity.get("factor_provenance", {}).get("supervision_policy")
                != "qwen-schema98-unreviewed-v1"
                or identity.get("factor_provenance", {}).get("human_review_status")
                != "not_assessed"))):
        raise ValueError(f"{variant} run identity mismatch")
    checkpoint = root / "step-4000.ckpt"
    if (selected.get("step") != 4000
            or selected.get("checkpoint_hash") != CHECKPOINT_HASHES[variant]
            or not checkpoint.is_file()
            or file_hash(checkpoint) != CHECKPOINT_HASHES[variant]):
        raise ValueError(f"{variant} selected checkpoint mismatch")
    return {"checkpoint": checkpoint, "checkpoint_hash": CHECKPOINT_HASHES[variant],
            "run_policy": expected_policy, "run_key": root.name}


def _load_projector(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)  # own locked pilot run
    state = checkpoint["state_dict"]
    prefix = "visual_adapter.spatial_projector."
    weight, bias = state[prefix + "weight"], state[prefix + "bias"]
    if (not isinstance(weight, torch.Tensor) or not isinstance(bias, torch.Tensor)
            or weight.ndim != 2 or weight.shape[1] != 2048 or weight.shape[0] < 1
            or bias.shape != (weight.shape[0],)
            or not torch.isfinite(weight).all() or not torch.isfinite(bias).all()):
        raise ValueError("pilot checkpoint spatial projector invalid")
    result = weight.detach().float().clone(), bias.detach().float().clone()
    del checkpoint, state, weight, bias
    gc.collect()
    return result


def probe(args: argparse.Namespace) -> dict:
    """Preflight or run aggregate diagnostic against immutable pilot runs."""
    if args.mode not in {"preflight", "run"}:
        raise ValueError("unsupported probe mode")
    output = args.output.resolve(strict=False)
    if (args.output.is_symlink() or args.output.exists()
            or any(output.is_relative_to(root.resolve()) for root in
                   (args.e1_run, args.e2_run, args.dataset, args.dino_root, args.motion_root))):
        raise ValueError("probe output exists or is inside frozen source/run")
    protocol, views = load_pilot_views(
        args.protocol, args.dataset, args.text_manifest, args.dino_root,
        args.motion_root, args.motion_manifest, args.annotation,
    )
    if protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("probe protocol mismatch")
    train = protocol["split"]["groups"]["train"]
    ids = tuple(train["clip_ids"])
    view = views["train"]
    signers = {record["fileid"]: record["signer"] for record in view.records}
    if (len(ids) != 5746 or len(signers) != len(view.records)
            or set(signers) != set(ids) or len(set(signers.values())) != 7
            or sorted(set(signers.values())) != train["signers"]):
        raise ValueError("pilot logical train clip/signer inventory mismatch")
    runs = {variant: _validate_run(root, variant, protocol)
            for variant, root in (("e1", args.e1_run), ("e2", args.e2_run))}
    summary = {"split_hash": protocol["split_hash"], "protocol_hash": PROTOCOL_HASH,
               "train_clips": len(ids), "runs": runs}
    if args.mode == "preflight":
        return {**summary, "runs": {key: {k: str(v) if isinstance(v, Path) else v
                                       for k, v in value.items()} for key, value in runs.items()}}
    start = time.monotonic()
    partition = split_probe_clips([(clip_id, signers[clip_id]) for clip_id in ids])
    pooled = pool_spatial(view, ids)
    scores = {}
    for variant, item in runs.items():
        weight, bias = _load_projector(item["checkpoint"])
        projected = project_pooled(pooled, weight, bias)
        del weight, bias
        scores[variant] = score_probe(projected, [signers[clip_id] for clip_id in ids],
                                      partition, ids)
        del projected
    result = {
        "status": "complete", "purpose": "diagnostic_only",
        "human_review_status": "not_assessed", "representation": "spatial_projector_mean",
        "protocol_hash": PROTOCOL_HASH, "split_hash": protocol["split_hash"],
        "source_hashes": {name: protocol[name] for name in
                          ("source_hash", "version_hash", "records_hash",
                           "annotation_hash", "frame_rows_hash", "spatial_manifest_hash",
                           "motion_manifest_hash", "text_manifest_hash")},
        "checkpoint_hashes": {name: runs[name]["checkpoint_hash"] for name in runs},
        "run_keys": {name: runs[name]["run_key"] for name in runs},
        "partition_hash": digest(partition),
        "partition_counts": {name: len(values) for name, values in partition.items()},
        "probe": {"seed": 0, "optimizer": "AdamW", "epochs": 200,
                  "learning_rate": 0.01, "weight_decay_grid": [0, 0.0001, 0.01]},
        "scores": scores,
        "interpretation": "probe test clips were seen during original E1/E2 translation training; "
                          "scores do not measure unseen-signer generalization",
        "e2_minus_e1": {key: scores["e2"][key] - scores["e1"][key]
                         for key in ("accuracy", "balanced_accuracy")},
        "elapsed_seconds": time.monotonic() - start,
    }
    atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "e1-run", "e2-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(probe(args))


if __name__ == "__main__":
    main()

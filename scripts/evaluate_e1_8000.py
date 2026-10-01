"""Once-only Signer07 evaluation for dev-selected E1@8000 run."""

import argparse
import gc
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from despamo.appearance.provenance import atomic_json
from despamo.data.batch import collate_phoenix
from despamo.evaluation.signer_pilot import score_pilot_batches
from despamo.training.pilot_runtime import load_pilot_views
from scripts.evaluate_signer_pilot import DECODER, PROTOCOL_HASH
from scripts.train_e3_pilot import _validate_matched_e1
from scripts.train_signer_pilot import _model


def evaluate(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("output exists")
    protocol, views = load_pilot_views(
        args.protocol, args.dataset, args.text_manifest, args.dino_root,
        args.motion_root, args.motion_manifest, args.annotation,
    )
    if protocol.get("protocol_hash") != PROTOCOL_HASH or protocol.get("decoder") != DECODER:
        raise ValueError("protocol mismatch")
    run = _validate_matched_e1(args.e1_run, protocol, args.steps)
    checkpoint = run["checkpoint"]
    identity = run["identity"]
    config = OmegaConf.create(identity["config"])
    group = protocol["split"]["groups"]["test"]
    ids = tuple(group["clip_ids"])
    view = views["test"]
    if len(ids) != 768 or group.get("signers") != ["Signer07"]:
        raise ValueError("test split mismatch")
    if args.mode == "preflight":
        return {"status": "ready", "step": run["step"], "checkpoint": str(checkpoint),
                "test_clips": len(ids)}
    model = _model(config, "E1_frozen", 768, args.steps,
                   {"supervision_policy": "qwen-schema98-unreviewed-v1",
                    "human_review_status": "not_assessed"})
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if state.get("run_metadata") != identity or state.get("global_step") != run["step"]:
        raise ValueError("checkpoint state mismatch")
    model.load_state_dict(state["state_dict"], strict=True)
    del state
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    model.to("cuda").eval()
    loader = DataLoader(view, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_phoenix,
                        generator=torch.Generator().manual_seed(0))
    try:
        report = score_pilot_batches(model, loader, ids, "test", protocol["split_hash"],
                                     beam_size=5, max_length=64)
        report.update({"model_variant": "E1_frozen", "checkpoint_step": run["step"],
                       "checkpoint_hash": run["checkpoint_hash"],
                       "protocol_hash": PROTOCOL_HASH, "run_key": run["run_key"],
                       "steps": args.steps})
        atomic_json(args.output, report)
        return report
    finally:
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("preflight", "run"), required=True)
    p.add_argument("--steps", type=int, default=8000)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "e1-run", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(evaluate(args), indent=2))


if __name__ == "__main__":
    main()

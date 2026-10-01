"""Once-only Signer07 translation evaluation for dev-selected E3 pilot."""

import argparse
import gc
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from despamo.appearance.provenance import atomic_json, digest, file_hash, read_json
from despamo.data.batch import collate_phoenix
from despamo.evaluation.signer_pilot import score_pilot_batches
from despamo.training.e3_runtime import load_e3_views
from scripts.train_e3_pilot import POLICY, WARMUP, select_e3_checkpoint
from scripts.train_signer_pilot import _model, checkpoint_steps

PROTOCOL_HASH = "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7"
DECODER = {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}


def _validate_e3_selection(run: Path, protocol: dict, dev_ids: tuple[str, ...]) -> dict:
    status = read_json(run / "run-status.json")
    budget = read_json(run / "budget.json")
    selected = read_json(run / "selected-checkpoint.json")
    identity = budget.get("identity", {})
    steps = identity.get("steps", 4000)
    if steps not in WARMUP or status.get("global_step") != steps:
        raise ValueError("E3 held-out run step tier mismatch")
    reports = [read_json(run / f"dev-{step}.json")
               for step in checkpoint_steps(steps, WARMUP[steps])]
    best = select_e3_checkpoint(reports, dev_ids, steps)
    checkpoint = run / f"step-{selected.get('step')}.ckpt"
    if (status.get("status") != "complete"
            or budget.get("status") != "closed"
            or identity.get("run_policy") != "signer-pilot-e3-adapted-bounded-v1"
            or identity.get("variant") != "E3_dino_lora"
            or identity.get("protocol_hash") != protocol["protocol_hash"]
            or identity.get("split_hash") != protocol["split_hash"]
            or identity.get("supervision_policy") != POLICY
            or identity.get("human_review_status") != "not_assessed"
            or selected.get("model_variant") != "E3_dino_lora"
            or selected.get("step") != best["checkpoint_step"]
            or selected.get("checkpoint_hash") != best["checkpoint_hash"]
            or selected.get("adaptation_checkpoint_hash")
            != identity.get("adaptation_checkpoint_hash")
            or not checkpoint.is_file() or file_hash(checkpoint) != selected["checkpoint_hash"]):
        raise ValueError("E3 selected checkpoint/dev/protocol identity mismatch")
    return {"checkpoint": checkpoint, "checkpoint_hash": selected["checkpoint_hash"],
            "step": selected["step"], "identity": identity}


def _score_e3(run: dict, view, ids: tuple[str, ...], split_hash: str) -> dict:
    identity = run["identity"]
    config = OmegaConf.create(identity["config"])
    if config.data.num_workers != 0 or config.model.use_in_context:
        raise ValueError("E3 held-out model/loader policy changed")
    model = _model(config, "E3_dino_lora", 768, 4000,
                   {"supervision_policy": POLICY, "human_review_status": "not_assessed"})
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if (checkpoint.get("run_metadata") != identity
            or checkpoint.get("global_step") != run["step"]):
        raise ValueError("E3 checkpoint full training metadata mismatch")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("E3 Signer07 scoring requires CUDA")
    model.to("cuda").eval()
    loader = DataLoader(view, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_phoenix,
                        generator=torch.Generator().manual_seed(0))
    try:
        return score_pilot_batches(model, loader, ids, "test", split_hash,
                                   beam_size=5, max_length=64)
    finally:
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()


def evaluate(args) -> dict:
    if args.mode not in ("preflight", "run"):
        raise ValueError("E3 held-out mode invalid")
    output = args.output.resolve(strict=False)
    if (args.output.is_symlink() or args.output.exists()
            or any(output.is_relative_to(path.resolve()) for path in
                   (args.e3_run, args.adapted_root, args.dataset, args.dino_root))):
        raise ValueError("E3 held-out output already exists or lies in source/run")
    original = {name: getattr(args, name) for name in
                ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
                 "motion_manifest", "annotation")}
    adapted = read_json(args.adapted_root / "complete/identity.json")
    protocol, views = load_e3_views(original, args.adapted_root, adapted)
    if protocol.get("protocol_hash") != PROTOCOL_HASH or protocol.get("decoder") != DECODER:
        raise ValueError("E3 held-out frozen protocol/decoder mismatch")
    group = protocol["split"]["groups"]["test"]
    ids = tuple(group["clip_ids"])
    view = views["test"]
    if (len(ids) != 768 or len(set(ids)) != 768 or group.get("signers") != ["Signer07"]
            or tuple(row["fileid"] for row in view.records) != ids
            or any(row["signer"] != "Signer07" for row in view.records)):
        raise ValueError("E3 held-out requires exact ordered 768 Signer07 clips")
    dev_ids = tuple(protocol["split"]["groups"]["dev"]["clip_ids"])
    run = _validate_e3_selection(args.e3_run, protocol, dev_ids)
    if run["identity"]["adaptation_checkpoint_hash"] != adapted["checkpoint_hash"]:
        raise ValueError("E3 translation/adapted spatial checkpoint mismatch")
    summary = {"protocol_hash": PROTOCOL_HASH, "split_hash": protocol["split_hash"],
               "test_clips": 768, "clip_ids_hash": digest(ids),
               "checkpoint_hash": run["checkpoint_hash"],
               "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
               "adapted_encoder_key": digest(adapted), "decoding": DECODER,
               "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    if args.mode == "preflight":
        return {"status": "ready", **summary}
    if not torch.cuda.is_available():
        raise RuntimeError("E3 Signer07 scoring requires CUDA")
    report = _score_e3(run, view, ids, protocol["split_hash"])
    if (report.get("split") != "test" or report.get("split_hash") != protocol["split_hash"]
            or report.get("decoding") != DECODER
            or tuple(item["clip_id"] for item in report.get("items", ())) != ids):
        raise ValueError("E3 held-out result incomplete or mismatched")
    result = {**report, **summary, "status": "complete", "checkpoint_step": run["step"],
              "model_variant": "E3_dino_lora"}
    atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "e3-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    result = evaluate(parser.parse_args())
    if result.get("status") == "ready":
        print(result)
    else:
        print({key: result[key] for key in ("status", "test_clips", "checkpoint_hash", "metrics")})


if __name__ == "__main__":
    main()

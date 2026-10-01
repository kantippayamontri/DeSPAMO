"""Score selected E1/E2 checkpoints once on frozen Signer07 logical test split."""

import argparse
import gc
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from despamo.appearance.provenance import atomic_json, digest, read_json
from despamo.data.batch import collate_phoenix
from despamo.evaluation.signer_pilot import score_pilot_batches
from despamo.training.pilot_runtime import load_pilot_views
from scripts.probe_signer_pilot import PROTOCOL_HASH, _validate_run
from scripts.train_signer_pilot import _model

DECODER = {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}


def require_factor_provenance(checkpoint: dict, model, identity: dict) -> None:
    """E2 checkpoint stores expanded factor metadata, unlike run identity."""
    expected = model.factor_provenance
    if (any(expected.get(key) != value for key, value in identity["factor_provenance"].items())
            or checkpoint.get("factor_provenance") != expected):
        raise ValueError("E2 saved factor provenance mismatch")


def _score(variant: str, run: dict, view, expected_ids: tuple[str, ...], split_hash: str) -> dict:
    """Rebuild pilot module, strictly load its selected snapshot, score all clips."""
    identity = read_json(run["checkpoint"].parent / "budget.json")["identity"]
    config = OmegaConf.create(identity["config"])
    if (config.data.num_workers != 0 or config.model.use_in_context
            or config.model.spatial_crop_mode != "full" or config.seed != 0):
        raise ValueError("pilot evaluation config mismatch")
    provenance = identity.get("factor_provenance", {
        "supervision_policy": "qwen-schema98-unreviewed-v1",
        "human_review_status": "not_assessed",
    })
    model = _model(config, "E1_frozen" if variant == "e1" else "E2_projector",
                   768, 4000, provenance)
    checkpoint = torch.load(run["checkpoint"], map_location="cpu", weights_only=False)
    if checkpoint.get("run_metadata") != identity or checkpoint.get("global_step") != 4000:
        raise ValueError(f"{variant} saved model/selection identity mismatch")
    if variant == "e2":
        require_factor_provenance(checkpoint, model, identity)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    del checkpoint
    if not torch.cuda.is_available():
        raise RuntimeError("Signer07 evaluation requires CUDA")
    model.to("cuda").eval()
    loader = DataLoader(view, batch_size=4, shuffle=False, num_workers=0,
                        collate_fn=collate_phoenix,
                        generator=torch.Generator().manual_seed(0))
    try:
        report = score_pilot_batches(model, loader, expected_ids, "test", split_hash,
                                     beam_size=5, max_length=64)
        report.update({"model_variant": "E1_frozen" if variant == "e1" else "E2_projector",
                       "checkpoint_step": 4000, "checkpoint_hash": run["checkpoint_hash"],
                       "protocol_hash": PROTOCOL_HASH, "run_key": run["run_key"],
                       "supervision_policy": "qwen-schema98-unreviewed-v1",
                       "human_review_status": "not_assessed"})
        return report
    finally:
        del model, loader
        gc.collect()
        torch.cuda.empty_cache()


def _require_report(report: dict, variant: str, run: dict, ids: tuple[str, ...], split_hash: str):
    items = report.get("items", ())
    metrics = report.get("metrics", {})
    if (report.get("split") != "test" or report.get("split_hash") != split_hash
            or report.get("checkpoint_hash") != run["checkpoint_hash"]
            or report.get("model_variant") != ("E1_frozen" if variant == "e1" else "E2_projector")
            or report.get("decoding") != DECODER
            or tuple(item.get("clip_id") for item in items) != ids
            or not isinstance(metrics.get("bleu4"), (int, float))
            or not math.isfinite(metrics["bleu4"])):
        raise ValueError("incomplete or mismatched Signer07 result")


def evaluate(args: argparse.Namespace) -> dict:
    """Validate held-out inputs; score each selected checkpoint at most once."""
    if args.mode not in {"preflight", "run"}:
        raise ValueError("unknown pilot evaluation mode")
    output = args.output.resolve(strict=False)
    if (args.output.is_symlink()
            or any(output.is_relative_to(root.resolve()) for root in
                   (args.e1_run, args.e2_run, args.dataset, args.dino_root, args.motion_root))):
        raise ValueError("pilot held-out output exists or is inside frozen source/run")
    if args.output.exists() and (
        args.mode == "preflight" or not args.output.is_dir()
        or not (args.output / "e1-test.json").is_file()
        or (args.output / "e2-test.json").exists()
        or (args.output / "paired.json").exists()
        or set(path.name for path in args.output.iterdir()) != {"e1-test.json"}
    ):
        raise ValueError("pilot held-out output exists without resumable E1 result")
    protocol, views = load_pilot_views(
        args.protocol, args.dataset, args.text_manifest, args.dino_root, args.motion_root,
        args.motion_manifest, args.annotation,
    )
    if protocol.get("protocol_hash") != PROTOCOL_HASH or protocol.get("decoder") != DECODER:
        raise ValueError("pilot held-out protocol/decoder mismatch")
    group = protocol["split"]["groups"]["test"]
    ids = tuple(group["clip_ids"])
    view = views["test"]
    if (len(ids) != 768 or len(set(ids)) != 768 or group.get("signers") != ["Signer07"]
            or tuple(record["fileid"] for record in view.records) != ids
            or any(record["signer"] != "Signer07" for record in view.records)):
        raise ValueError("pilot held-out test requires exactly 768 ordered Signer07 clips")
    runs = {name: _validate_run(root, name, protocol) for name, root in
            (("e1", args.e1_run), ("e2", args.e2_run))}
    summary = {"protocol_hash": PROTOCOL_HASH, "split_hash": protocol["split_hash"],
               "test_clips": len(ids), "clip_ids_hash": digest(ids),
               "checkpoint_hashes": {key: run["checkpoint_hash"] for key, run in runs.items()},
               "decoding": DECODER}
    if args.mode == "preflight":
        return summary
    if not torch.cuda.is_available():
        raise RuntimeError("Signer07 evaluation requires CUDA")
    if not args.output.exists():
        args.output.mkdir(parents=True, exist_ok=False)
    reports = {}
    for name in ("e1", "e2"):
        previous = args.output / f"{name}-test.json"
        report = read_json(previous) if previous.is_file() and name == "e1" else _score(
            name, runs[name], view, ids, protocol["split_hash"]
        )
        _require_report(report, name, runs[name], ids, protocol["split_hash"])
        if not previous.exists():
            atomic_json(previous, report)
        reports[name] = report
    if ([item["reference"] for item in reports["e1"]["items"]]
            != [item["reference"] for item in reports["e2"]["items"]]):
        raise ValueError("E1/E2 held-out references differ")
    scores = {name: reports[name]["metrics"] for name in ("e1", "e2")}
    paired = {**summary, "status": "complete", "scores": scores,
              "e2_minus_e1_bleu4": scores["e2"]["bleu4"] - scores["e1"]["bleu4"],
              "human_review_status": "not_assessed",
              "supervision_policy": "qwen-schema98-unreviewed-v1"}
    atomic_json(args.output / "paired.json", paired)
    return paired


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "e1-run", "e2-run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    print(evaluate(parser.parse_args()))


if __name__ == "__main__":
    main()

"""Optional matched closed-set signer diagnostic for selected E3 projector."""

import argparse
from pathlib import Path

from despamo.appearance.provenance import atomic_json, digest, read_json
from despamo.evaluation.pilot_probe import (
    pool_spatial,
    project_pooled,
    score_probe,
    split_probe_clips,
)
from despamo.training.e3_runtime import load_e3_views
from scripts.evaluate_e3_pilot import PROTOCOL_HASH, _validate_e3_selection
from scripts.probe_signer_pilot import _load_projector


def probe(args) -> dict:
    if args.mode not in ("preflight", "run"):
        raise ValueError("E3 probe mode invalid")
    output = args.output.resolve(strict=False)
    if (args.output.is_symlink() or args.output.exists()
            or any(output.is_relative_to(root.resolve()) for root in
                   (args.e3_run, args.adapted_root, args.dataset, args.dino_root))):
        raise ValueError("E3 probe output exists or is inside frozen source/run")
    original = {name: getattr(args, name) for name in
                ("protocol", "dataset", "text_manifest", "dino_root", "motion_root",
                 "motion_manifest", "annotation")}
    adapted = read_json(args.adapted_root / "complete/identity.json")
    protocol, views = load_e3_views(original, args.adapted_root, adapted)
    if protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("E3 probe protocol mismatch")
    records = views["train"].records
    ids = tuple(protocol["split"]["groups"]["train"]["clip_ids"])
    signers = {row["fileid"]: row["signer"] for row in records}
    if (len(ids) != len(signers) or set(ids) != set(signers)
            or any(name in {"Signer03", "Signer07"} for name in signers.values())):
        raise ValueError("E3 probe must use only seven logical training signers")
    split = split_probe_clips([(clip_id, signers[clip_id]) for clip_id in ids])
    matched = read_json(args.split_manifest)
    if (matched.get("protocol_hash") != PROTOCOL_HASH
            or matched.get("split_hash") != protocol["split_hash"]
            or matched.get("partition_hash") != digest(split)):
        raise ValueError("E3 signer-probe partition differs from matched E1/E2")
    run = _validate_e3_selection(args.e3_run, protocol,
                                 tuple(protocol["split"]["groups"]["dev"]["clip_ids"]))
    if run["identity"]["adaptation_checkpoint_hash"] != adapted["checkpoint_hash"]:
        raise ValueError("E3 probe checkpoint/adapted DINO mismatch")
    summary = {"purpose": "diagnostic_only", "protocol_hash": PROTOCOL_HASH,
               "split_hash": protocol["split_hash"], "partition_hash": digest(split),
               "partition_counts": {key: len(value) for key, value in split.items()},
               "checkpoint_hash": run["checkpoint_hash"],
               "adaptation_checkpoint_hash": adapted["checkpoint_hash"],
               "supervision_policy": "qwen-schema98-unreviewed-v1",
               "human_review_status": "not_assessed"}
    if args.mode == "preflight":
        return {"status": "ready", **summary}
    pooled = pool_spatial(views["train"], ids)
    weight, bias = _load_projector(run["checkpoint"])
    projected = project_pooled(pooled, weight, bias)
    scores = score_probe(projected, [signers[clip_id] for clip_id in ids], split, ids)
    result = {**summary, "status": "complete", "representation": "spatial_projector_mean",
              "scores": scores,
              "interpretation": "probe test clips were seen during E3 translation training"}
    atomic_json(args.output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "run"), required=True)
    for name in ("protocol", "dataset", "text-manifest", "dino-root", "motion-root",
                 "motion-manifest", "annotation", "adapted-root", "e3-run", "split-manifest",
                 "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    print(probe(parser.parse_args()))


if __name__ == "__main__":
    main()

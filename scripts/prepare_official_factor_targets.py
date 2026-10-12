"""Prepare merged appearance targets CPU-only, without changing live training inputs."""

import argparse
import json
from pathlib import Path

import numpy as np

from despamo.appearance.merged_targets import import_crop_export, merge_targets, merged_target_key
from despamo.appearance.merged_text_features import (
    encode_merged_targets,
    load_pinned_encoder,
    read_stable_vectors,
)
from despamo.appearance.provenance import file_hash, read_json

CROP_SHA = "73948b9784e23acac61f51cf46d0aec4259f8402faf10be97df40aa8a7e606dc"


def official_ids(raw: dict) -> tuple[str, ...]:
    ids = tuple(raw[k]["fileid"] for k in sorted(k for k in raw if type(k) is int))
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("empty or duplicate official annotation clip IDs")
    return ids


def validate_annotation_join(raw: dict, source: dict) -> None:
    expected = {raw[k]["fileid"]: raw[k]["signer"] for k in raw if type(k) is int}
    actual = {r["clip_id"]: r["signer"] for r in source["clips"]}
    if actual != expected or len(actual) != len(source["clips"]):
        raise ValueError("official annotation/source signer join mismatch")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preflight", "encode"), required=True)
    for name in ("crop-export", "old-manifest", "source", "teacher-root", "train-annotation",
                 "clip-cache", "output-base"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    crop = import_crop_export(args.crop_export, CROP_SHA)
    parents = {name: dict(path=str(p.resolve()), sha256=file_hash(p))
               for name, p in (("crop_export", args.crop_export),
                               ("old_manifest", args.old_manifest), ("source", args.source))}
    old, source = read_json(args.old_manifest), read_json(args.source)
    row_path = args.teacher_root / "complete/frame_rows.json"
    rows = read_json(row_path)
    if source.get("frame_rows_hash") != file_hash(row_path):
        raise ValueError("teacher row-map/source hash mismatch")
    raw = np.load(args.train_annotation, allow_pickle=True).item()
    ids = official_ids(raw)
    if len(ids) != 7096 or len({raw[k]["signer"] for k in raw if type(k) is int}) != 9:
        raise ValueError("official train requires 7096 clips and nine signers")
    validate_annotation_join(raw, source)
    document = merge_targets(old, crop, source, rows, ids, parents)
    read_stable_vectors(document, args.old_manifest.parent)
    for parent in parents.values():
        if file_hash(Path(parent["path"])) != parent["sha256"]:
            raise ValueError("source changed during admission")
    report = dict(status="ready", clips=len(ids), merged_key=merged_target_key(document),
                  valid_counts=document["valid_counts"], masked_counts=document["masked_counts"],
                  human_review_status="not_assessed")
    if args.mode == "encode":
        encoder = load_pinned_encoder(args.clip_cache)
        path = encode_merged_targets(document, args.old_manifest.parent, encoder, args.output_base)
        report.update(status="complete", manifest=str(path), manifest_sha256=file_hash(path))
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()

"""Freeze a read-only signer-disjoint protocol before any pilot training."""

import argparse
from pathlib import Path

import numpy as np

from despamo.appearance.canonical import targets
from despamo.appearance.generation import load_records
from despamo.appearance.provenance import atomic_json, digest, file_hash, load_dataset, read_json
from despamo.appearance.schema import Record
from despamo.data.factors import masked_targets
from despamo.data.manifest import FeatureManifest
from despamo.data.signer_split import build_split
from despamo.pilot import bind_protocol, validate_live_counts
from tools.dinov3.storage import verify_location

ROOT = Path(__file__).resolve().parents[1]
DATASET_KEY = "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"


def publish_protocol(protocol: dict, root: Path) -> Path:
    if protocol.get("protocol_hash") != digest(
        {k: v for k, v in protocol.items() if k != "protocol_hash"}
    ):
        raise ValueError("protocol hash mismatch")
    path = root / protocol["protocol_hash"] / "protocol.json"
    if path.parent.exists():
        raise ValueError("protocol key already exists; preserve frozen split")
    atomic_json(path, protocol)
    return path


def freeze(args) -> Path:
    if args.dataset.name != DATASET_KEY:
        raise ValueError("wrong original Qwen dataset key")
    verify_location(args.output_base, ROOT)
    output = args.output_base.resolve()
    dataset = args.dataset.resolve()
    if output == dataset or dataset in output.parents or output in dataset.parents:
        raise ValueError("pilot protocol must be outside the source dataset")
    source, version = load_dataset(dataset)
    journals = load_records(dataset)
    split = build_split(source, journals)
    validate_live_counts(split)
    if version["model"] != "qwen3-vl:8b" or source.get("frame_rows_hash") != file_hash(
        args.frame_rows
    ):
        raise ValueError("Qwen model or original DINO frame-row identity changed")
    by_id = {row["clip_id"]: row for row in journals}
    clips = {clip["clip_id"]: clip for clip in source["clips"]}
    text = read_json(args.text_manifest)
    if text["dataset_key"] != dataset.name or text["encoder_key"] != digest(text["metadata"]):
        raise ValueError("frozen CLIP manifest belongs to another caption version")
    text_rows = {row["clip_id"]: row for row in text["records"]}
    if len(text_rows) != len(text["records"]) or set(text_rows) != set(clips):
        raise ValueError("CLIP clip inventory mismatch")
    for clip_id, row in text_rows.items():
        journal = by_id[clip_id]
        expected = (
            targets(Record.model_validate(journal["record"]))
            if journal["status"] == "valid" else masked_targets(clips[clip_id])
        )
        path = (args.text_manifest.parent / row["path"]).resolve()
        if (
            row["record_hash"] != digest(journal)
            or row["targets"] != expected
            or path.parent != args.text_manifest.parent.resolve()
            or file_hash(path) != row["file_hash"]
        ):
            raise ValueError(f"CLIP target/record/file identity changed: {clip_id}")
    frame_data = read_json(args.frame_rows)
    spatial = FeatureManifest.load(args.spatial_manifest)
    motion = FeatureManifest.load(args.motion_manifest)
    if (
        frame_data["spatial_manifest_hash"] != file_hash(args.spatial_manifest)
        or spatial.encoder != "dinov3:" + frame_data["encoder_key"]
        or spatial.expected_dim != 2048
        or motion.expected_dim != 1024
    ):
        raise ValueError("original DINO/motion feature manifest mismatch")
    for clip_id, clip in clips.items():
        row = frame_data["clips"][clip_id]
        feature = spatial.require("train", clip_id)
        motion.require("train", clip_id)
        if (
            row["source_indices"] != list(range(clip["frame_count"]))
            or feature.length != clip["frame_count"]
            or row["feature_hash"] != file_hash(args.spatial_root / feature.path)
        ):
            raise ValueError(f"DINO feature row/source identity changed: {clip_id}")
        for frame in clip["frames"]:
            if (
                row["paths"][frame["index"]] != frame["path"]
                or row["sampled_images"].get(str(frame["index"])) != frame["sha256"]
            ):
                raise ValueError(f"DINO/Qwen sampled-frame identity changed: {clip_id}")
    # Trusted local PHOENIX annotations, never arbitrary pickled input.
    annotation = np.load(args.annotation, allow_pickle=True).item()
    annotated = {
        item["fileid"]: item["signer"] for key, item in annotation.items() if type(key) is int
    }
    if len(annotated) != len(clips) or annotated != {
        clip_id: clip["signer"] for clip_id, clip in clips.items()
    }:
        raise ValueError("annotated source signer/ID inventory changed")
    protocol = bind_protocol(
        split,
        dataset.name,
        digest(source),
        digest(version),
        digest(journals),
        file_hash(args.text_manifest),
        text["metadata"],
        file_hash(args.frame_rows),
        file_hash(args.spatial_manifest),
        file_hash(args.motion_manifest),
        file_hash(args.annotation),
    )
    return publish_protocol(protocol, args.output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in (
        "dataset", "text-manifest", "frame-rows", "spatial-manifest",
        "motion-manifest", "spatial-root", "annotation", "output-base",
    ):
        parser.add_argument("--" + key, required=True, type=Path)
    print(freeze(parser.parse_args()))


if __name__ == "__main__":
    main()

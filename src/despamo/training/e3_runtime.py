"""Bind adapted E3 spatial features without relaxing original pilot protocol."""

from pathlib import Path

import numpy as np

from despamo.appearance.provenance import digest, file_hash, read_json
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.data.signer_split import PhoenixTrainView
from despamo.training.pilot_runtime import load_pilot_views


def validate_adapted_manifest(
    root: Path, identity: dict, expected_ids: tuple[str, ...]
) -> dict:
    complete = root / "complete"
    if (not complete.is_dir() or read_json(complete / "identity.json") != identity
            or not identity.get("checkpoint_hash") or not identity.get("source_hash")):
        raise ValueError("E3 adapted checkpoint/identity mismatch")
    manifest_path = complete / "manifest.json"
    manifest = FeatureManifest.load(manifest_path)
    row_path = complete / "frame_rows.json"
    rows = read_json(row_path)
    if (manifest.encoder != digest(identity) or manifest.expected_dim != 2048
            or tuple(record.clip_id for record in manifest.records) != expected_ids
            or any(record.split != "train" for record in manifest.records)
            or rows.get("encoder_key") != manifest.encoder
            or set(rows.get("clips", {})) != set(expected_ids)
            or set(path.stem for path in (root / "train").glob("*.npy")) != set(expected_ids)):
        raise ValueError("E3 adapted feature inventory mismatch")
    for record in manifest.records:
        clip_id = record.clip_id
        row = rows["clips"][clip_id]
        path = root / record.path
        receipt = read_json(root / "receipts/train" / f"{clip_id}.json")
        if (record.path != f"train/{clip_id}.npy" or record.width != 2048
                or record.dtype != "float32" or row.get("frame_count") != record.length
                or row.get("feature_hash") != file_hash(path)
                or receipt.get("feature_hash") != row["feature_hash"]
                or receipt.get("source_hash") != row.get("source_hash")
                or receipt.get("checkpoint_hash") != identity["checkpoint_hash"]):
            raise ValueError(f"E3 adapted feature/receipt hash mismatch: {clip_id}")
        values = np.load(path, mmap_mode="r", allow_pickle=False)
        if values.shape != (record.length, 2048) or not np.isfinite(values).all():
            raise ValueError(f"E3 adapted feature shape/values invalid: {clip_id}")
    return {"encoder_key": manifest.encoder, "manifest_hash": file_hash(manifest_path),
            "frame_rows_hash": file_hash(row_path),
            "checkpoint_hash": identity["checkpoint_hash"], "clips": len(expected_ids)}


def load_e3_views(
    original_args: dict[str, Path], adapted_root: Path, adaptation: dict
) -> tuple[dict, dict[str, PhoenixTrainView]]:
    """Validate immutable original source, then bind separately verified E3 output."""
    protocol, original = load_pilot_views(
        original_args["protocol"], original_args["dataset"], original_args["text_manifest"],
        original_args["dino_root"], original_args["motion_root"],
        original_args["motion_manifest"], original_args["annotation"],
    )
    if (adaptation.get("protocol_hash") != protocol["protocol_hash"]
            or adaptation.get("source_hash") != protocol["source_hash"]
            or adaptation.get("frame_rows_hash") != protocol["frame_rows_hash"]):
        raise ValueError("E3 adaptation protocol/source identity mismatch")
    groups = protocol["split"]["groups"]
    expected_ids = tuple(sorted(clip_id for group in ("train", "dev", "test")
                                for clip_id in groups[group]["clip_ids"]))
    validate_adapted_manifest(adapted_root, adaptation, expected_ids)
    spatial = FeatureManifest.load(adapted_root / "complete/manifest.json")
    motion = FeatureManifest.load(original_args["motion_manifest"])
    parent = Phoenix14T(
        original_args["annotation"], "train", adapted_root, original_args["motion_root"],
        spatial, motion, spatial_crop_mode="full",
    )
    if ({row["fileid"]: row["signer"] for row in parent.records}
            != {row["fileid"]: row["signer"] for view in original.values()
                for row in view.records}):
        raise ValueError("E3 adapted annotations differ from frozen pilot")
    views = {name: PhoenixTrainView(parent, tuple(groups[name]["clip_ids"]))
             for name in ("train", "dev", "test")}
    return protocol, views

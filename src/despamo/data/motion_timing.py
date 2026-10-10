"""Versioned motion timing, conditional on explicit external chronology evidence.

The pure builder preserves sparse original spatial indices. Runtime validation
additionally requires full source rows and full crop mode. Evidence claims are
attestations, not facts inferred from feature lengths; the CLI verifies their
file identities. Neither entry point invents historical extraction provenance.
"""

from __future__ import annotations

import copy
import re
from dataclasses import dataclass

from despamo.appearance.provenance import digest
from despamo.data.manifest import FeatureManifest, FeatureRecord

FILE_KEYS = {"frame_rows", "motion_manifest", "annotation", "extractor", "helpers"}
CLAIMS = {"historical_window_parameters", "motion_row_order", "source_frame_order"}
CONVENTION = {
    "window_size": 16,
    "stride": 8,
    "tail_policy": "full_windows_only",
    "short_clip_policy": "repeat_last_to_16",
    "frame_order": "lexicographic",
    "motion_row_order": "ascending_window_start",
}


@dataclass(frozen=True)
class ClipTiming:
    clip_id: str
    source_indices: tuple[int, ...]
    frame_count: int
    windows: tuple[tuple[int, int], ...]


def _object(value: object, name: str) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _hash(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _version(value: dict, name: str) -> None:
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError(f"unsupported {name} schema_version")


def _manifest(value: dict) -> FeatureManifest:
    _version(value, "motion manifest")
    records = value.get("records")
    if not isinstance(records, list) or not all(isinstance(row, dict) for row in records):
        raise ValueError("motion manifest records must be a list of objects")
    try:
        manifest = FeatureManifest(
            records=tuple(FeatureRecord(**row) for row in records),
            **{key: item for key, item in value.items() if key != "records"},
        )
    except TypeError as exc:
        raise ValueError(f"invalid motion manifest: {exc}") from exc
    if manifest.expected_dim != 1024 or manifest.encoder != "MCG-NJU/videomae-large":
        raise ValueError("expected frozen 1024-wide VideoMAE motion manifest")
    return manifest


def _provenance(value: dict, hashes: dict) -> None:
    _version(value, "extraction provenance")
    if value.get("status") not in ("original_receipt", "reconstructed_source_convention"):
        raise ValueError("unknown extraction chronology: unsupported provenance status")
    for key, expected in CONVENTION.items():
        if type(value.get(key)) is not type(expected) or value[key] != expected:
            raise ValueError(f"unknown or unsupported extraction convention: {key}")
    if value.get("physical_split") != "train":
        raise ValueError("this study requires physical train motion rows")
    if not isinstance(value.get("spatial_crop_mode"), str):
        raise ValueError("spatial_crop_mode is required")
    if value.get("input_sha256") != hashes:
        raise ValueError("input content hash mismatch")
    files = _object(value.get("file_sha256"), "file_sha256")
    if set(files) != FILE_KEYS:
        raise ValueError("exact input/extractor/helper file hashes required")
    for key, item in files.items():
        _hash(item, key)
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("chronology evidence required; status alone is insufficient")
    claims: set[str] = set()
    paths: set[str] = set()
    for item in evidence:
        item = _object(item, "evidence entry")
        path = item.get("path")
        if not isinstance(path, str) or not path.startswith("/") or path in paths:
            raise ValueError("unique absolute evidence path required")
        paths.add(path)
        _hash(item.get("sha256"), "evidence hash")
        asserted = item.get("claims")
        if (
            not isinstance(asserted, list)
            or not asserted
            or not all(isinstance(claim, str) and claim in CLAIMS for claim in asserted)
            or len(set(asserted)) != len(asserted)
        ):
            raise ValueError("invalid chronology evidence claims")
        claims.update(asserted)
    if claims != CLAIMS:
        raise ValueError("evidence must establish historical parameters and both frame/row order")


def build_motion_timing(
    frame_rows: dict,
    motion_manifest: dict,
    extraction_provenance: dict,
) -> dict:
    """Build deterministic real-frame support; never infer missing chronology.

    Hashes in input_sha256 use canonical JSON (appearance.provenance.digest).
    file_sha256 binds raw bytes, verified by the CPU CLI. Evidence must explicitly
    cover historical window parameters, source frame order and motion row order.
    """
    frame_rows = _object(frame_rows, "frame_rows")
    motion_manifest = _object(motion_manifest, "motion_manifest")
    provenance = _object(extraction_provenance, "extraction_provenance")
    hashes = {"frame_rows": digest(frame_rows), "motion_manifest": digest(motion_manifest)}
    _provenance(provenance, hashes)
    manifest = _manifest(motion_manifest)
    clips = _object(frame_rows.get("clips"), "frame_rows clips")
    if not clips or not isinstance(frame_rows.get("encoder_key"), str):
        raise ValueError("nonempty frame_rows clips and encoder_key required")
    records = {row.clip_id: row for row in manifest.records if row.split == "train"}
    receipts = _object(provenance.get("clips"), "provenance clips")
    if set(clips) != set(records) or set(clips) != set(receipts):
        raise ValueError("frame/motion/provenance clip identities must match physical train")
    result = {}
    real_frames = covered_frames = 0
    for clip_id, row in clips.items():
        row = _object(row, f"frame rows {clip_id}")
        count = row.get("frame_count")
        if type(count) is not int or count <= 0:
            raise ValueError(f"invalid real frame_count: {clip_id}")
        indices = row.get("source_indices")
        if (
            not isinstance(indices, list)
            or not indices
            or any(type(i) is not int or not 0 <= i < count for i in indices)
            or any(a >= b for a, b in zip(indices, indices[1:], strict=False))
        ):
            raise ValueError(f"source indices must be unique, monotonic real ordinals: {clip_id}")
        paths = row.get("paths")
        if (
            not isinstance(paths, list)
            or len(paths) != len(indices)
            or not all(isinstance(path, str) and path for path in paths)
            or paths != sorted(paths)
            or len(set(paths)) != len(paths)
        ):
            raise ValueError(f"source paths must match lexicographic source order: {clip_id}")
        _hash(row.get("source_hash"), "source_hash")
        _hash(row.get("feature_hash"), "feature_hash")
        record = records[clip_id]
        raw_record = {
            "clip_id": record.clip_id,
            "split": record.split,
            "path": record.path,
            "length": record.length,
            "width": record.width,
            "dtype": record.dtype,
        }
        receipt = _object(receipts[clip_id], f"provenance clip {clip_id}")
        if (
            type(receipt.get("frame_count")) is not int
            or receipt["frame_count"] != count
            or receipt.get("source_hash") != row["source_hash"]
            or receipt.get("motion_record_sha256") != digest(raw_record)
        ):
            raise ValueError(f"clip source/count/motion identity mismatch: {clip_id}")
        windows = (
            [[0, count - 1]]
            if count < 16
            else [[start, start + 15] for start in range(0, count - 15, 8)]
        )
        if len(windows) != record.length:
            raise ValueError(f"motion row count disagrees with proven windows: {clip_id}")
        real_frames += count
        covered_frames += windows[-1][1] + 1
        result[clip_id] = {
            "source_indices": list(indices),
            "frame_count": count,
            "windows": windows,
            "source_hash": row["source_hash"],
            "motion_record": raw_record,
        }
    return {
        "schema_version": 1,
        "kind": "despamo-motion-timing",
        "provenance_status": provenance["status"],
        "input_sha256": hashes,
        "provenance": copy.deepcopy(provenance),
        "provenance_payload_sha256": digest(provenance),
        "clips": result,
        "coverage": {
            "real_frame_count": real_frames,
            "covered_frame_count": covered_frames,
            "uncovered_frame_count": real_frames - covered_frames,
            "fraction": covered_frames / real_frames,
        },
    }


def validate_motion_timing(
    timing: dict,
    frame_rows: dict,
    motion_manifest: dict,
) -> dict[str, ClipTiming]:
    """Fail-closed runtime admission of an externally evidenced timing map.

    This is a pure consistency check, not a verifier of historical truth or
    current filesystem bytes. Use the CLI for file binding. For short clips,
    windows describe real support, not padded extraction extent; extraction
    centres remain 8*j+7.5, including the single short-clip window.
    """
    timing = _object(timing, "timing")
    _version(timing, "timing")
    provenance = _object(timing.get("provenance"), "extraction provenance")
    expected = build_motion_timing(frame_rows, motion_manifest, provenance)
    core = {key: value for key, value in timing.items() if key != "provenance_sha256"}
    if digest(core) != digest(expected):
        raise ValueError("timing schema/content/provenance mismatch")
    if "provenance_sha256" in timing:
        _hash(timing["provenance_sha256"], "provenance_sha256")
    if timing["provenance"]["spatial_crop_mode"] != "full":
        raise ValueError("runtime requires full spatial crop mode")
    admitted = {}
    for clip_id, clip in timing["clips"].items():
        if clip["source_indices"] != list(range(clip["frame_count"])):
            raise ValueError(f"runtime requires full source rows 0..N-1: {clip_id}")
        admitted[clip_id] = ClipTiming(
            clip_id,
            tuple(clip["source_indices"]),
            clip["frame_count"],
            tuple(tuple(window) for window in clip["windows"]),
        )
    return admitted

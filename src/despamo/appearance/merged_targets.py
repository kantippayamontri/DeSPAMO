"""Versioned v1-stable/v2-articulator joins; never modify parent datasets."""

from collections import Counter
from copy import deepcopy
from pathlib import Path

from despamo.appearance.provenance import digest, file_hash, read_json, sample_indices
from despamo.appearance.schema import ARTICULATORS, POSITIONS, STABLE

POLICY = "qwen-stable-v1-crop-articulators-v2-unreviewed-v1"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


def _check_export(export: dict) -> None:
    meta, records = export.get("metadata", {}), export.get("records", [])
    if (meta.get("format_version") != "qwen-crop-text-targets-v1"
            or meta.get("canonical_version") != "crop-closed-v1"
            or meta.get("split") != "train" or meta.get("target_slots") != 19
            or meta.get("human_review_status") != "not_assessed"
            or meta.get("clip_count") != len(records)
            or meta.get("records_sha256") != digest(records)):
        raise ValueError("crop export schema/count/records digest mismatch")


def import_crop_export(path: Path, expected_sha256: str) -> dict:
    if file_hash(path) != expected_sha256:
        raise ValueError("crop export raw digest mismatch")
    export = read_json(path)
    _check_export(export)
    if file_hash(path) != expected_sha256:
        raise ValueError("crop export changed during import")
    return export


def _index(records: list[dict], ids: tuple[str, ...], label: str) -> dict:
    result = {r["clip_id"]: r for r in records}
    if len(result) != len(records) or set(result) != set(ids):
        raise ValueError(f"{label} clip inventory missing/extra/duplicate")
    return result


def _targets(record: dict, indices: list[int]) -> list[dict]:
    targets = record.get("targets", [])
    expected = [(f, -1) for f in STABLE] + [(f, i) for i in indices for f in ARTICULATORS]
    if [(t.get("factor"), t.get("frame_index")) for t in targets] != expected:
        raise ValueError("factor/frame target slot order mismatch")
    for t in targets:
        text = t.get("text")
        if text is not None and (not isinstance(text, str) or not text.strip()):
            raise ValueError("invalid target text")
    return targets


def merge_targets(old_manifest: dict, crop_export: dict, source: dict,
                  teacher_rows: dict, official_ids: tuple[str, ...], parents: dict) -> dict:
    _check_export(crop_export)
    if (not official_ids or len(set(official_ids)) != len(official_ids)
            or any(not isinstance(i, str) or not i or "/" in i or "\\" in i
                   for i in official_ids)):
        raise ValueError("official clip IDs invalid")
    meta = old_manifest.get("metadata", {})
    if (meta.get("model") != "openai/clip-vit-large-patch14"
            or meta.get("revision") != REVISION or meta.get("tokenizer_revision") != REVISION
            or meta.get("width") != 768 or meta.get("dtype") != "float32"):
        raise ValueError("previous encoder contract mismatch")
    old = _index(old_manifest["records"], official_ids, "previous")
    new = _index(crop_export["records"], official_ids, "crop")
    sources = _index(source["clips"], official_ids, "source")
    rows = teacher_rows["clips"]
    if not set(official_ids) <= rows.keys():
        raise ValueError("teacher clip inventory missing")
    result, valid, masked = [], Counter(), Counter()
    for clip_id in official_ids:
        prior, fresh, clip, teacher = old[clip_id], new[clip_id], sources[clip_id], rows[clip_id]
        count = clip["frame_count"]
        indices = sample_indices(count)
        frames = clip["frames"]
        paths = [f["path"] for f in frames]
        if (fresh.get("split") != "train" or fresh.get("signer") != clip["signer"]
                or fresh.get("num_frames") != count or fresh.get("frame_indices") != indices
                or fresh.get("normalized_positions") != list(POSITIONS)
                or [f["index"] for f in frames] != indices
                or fresh.get("frame_paths") != paths
                or teacher.get("frame_count") != count
                or teacher.get("source_indices") != list(range(count))
                or any(teacher.get("sampled_images", {}).get(str(f["index"])) != f["sha256"]
                       for f in frames)
                or any(not p.startswith(f"train/{clip_id}/") or ".." in Path(p).parts
                       for p in paths)
                or [teacher["paths"][i] for i in indices] != paths):
            raise ValueError(f"source/teacher/crop frame alignment mismatch: {clip_id}")
        prior_targets, fresh_targets = _targets(prior, indices), _targets(fresh, indices)
        targets = deepcopy(prior_targets[:4]) + deepcopy(fresh_targets[4:])
        for i, target in enumerate(targets):
            target["source_version"] = "stable-v1" if i < 4 else "crop-articulators-v2"
            if i >= 4:
                if (target["text"] is None and not target.get("mask_reason")) or (
                        target["text"] is not None and target.get("mask_reason") is not None):
                    raise ValueError("crop text/mask reason mismatch")
            (masked if target["text"] is None else valid)[target["factor"]] += 1
        result.append(dict(clip_id=clip_id, signer=clip["signer"], frame_count=count,
                           frame_indices=indices, frame_paths=paths, targets=targets,
                           old_vector={k: prior[k] for k in ("path", "file_hash", "record_hash")},
                           source_frame_hashes=[f["sha256"] for f in frames]))
    return dict(schema_version=1, supervision_policy=POLICY, human_review_status="not_assessed",
                parents=deepcopy(parents), previous_encoder=deepcopy(meta),
                crop_metadata=deepcopy(crop_export["metadata"]), records=result,
                valid_counts=dict(valid), masked_counts=dict(masked), target_slots=19,
                provenance_level="imported-export-bound")


def merged_target_key(document: dict) -> str:
    return digest(document)

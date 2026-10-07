"""Portable 19-slot canonical text handoff; no text or visual features are computed.

Target order matches despamo.appearance.canonical.targets. The crop-label schema
is versioned separately: this is not the historical schema-2 Record dataset, and
encode_dataset cannot consume it unchanged. Its target texts can be passed to a
text encoder in slot order, with masked slots kept zero-valued.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from how2sign_probe.completion import validate_completion
from how2sign_probe.frames import POSITIONS, sample_indices
from how2sign_probe.labeling import atomic_json, journal_path, make_job
from how2sign_probe.labels import parse_label
from how2sign_probe.provenance import package_code_sha256, sha256_file
from how2sign_probe.schemas import HAND_FIELDS, MOUTH_FIELDS

STABLE = ("biometric", "clothing", "hair", "background")
ARTICULATORS = ("left_hand", "right_hand", "mouth")
FACTORS = {"left_hand": ("hand", "LEFT"), "right_hand": ("hand", "RIGHT"),
           "mouth": ("mouth", "MOUTH")}
FORMAT_VERSION = "qwen-crop-text-targets-v1"


def _clean(text):
    return " ".join(text.replace("_", " ").lower().split())


def canonical_text(factor, label):
    if label["visibility"] not in ("clear", "partial"):
        return None
    fields = MOUTH_FIELDS if factor == "mouth" else HAND_FIELDS
    parts = [f"{_clean(field)} {_clean(label[field])}" for field in fields
             if field != "visibility" and label[field] != "uncertain"]
    return f"{_clean(factor)}: " + "; ".join(parts) if parts else None


def clip_record(root, entry, regions, split, frame_names=None, frame_indices=None):
    indices = (
        sample_indices(entry["num_frames"])
        if frame_indices is None else list(frame_indices)
    )
    if len(indices) != 5:
        raise ValueError("expected five frame positions")
    target_list = [dict(factor=f, frame_index=-1, text=None, label=None,
                        mask_reason="stable_not_extracted", job_key=None) for f in STABLE]
    for index in indices:
        best = {}
        for kind, side, box, score in regions.get(f"{entry['clip']}|{index}", []):
            slot = kind, side
            if slot not in best or score > best[slot][1]:
                best[slot] = box, score
        for factor in ARTICULATORS:
            kind, side = FACTORS[factor]
            target = dict(factor=factor, frame_index=index, text=None, label=None,
                          mask_reason="not_detected", job_key=None)
            if (kind, side) in best:
                box, _score = best[kind, side]
                job = make_job(kind, side, entry["clip"], index, json.dumps(box))
                path = journal_path(root, job)
                if not path.exists():
                    raise ValueError(f"unfinished job {job.key}")
                state = json.loads(path.read_text())
                if json.loads(state["identity"]) != list(job.identity):
                    raise ValueError(f"journal identity mismatch: {job.key}")
                target.update(job_key=job.key, journal_sha256=sha256_file(path),
                              bbox=box, status=state["status"])
                if state["status"] == "done":
                    label = parse_label(kind, json.dumps(state["label"]))
                    text = canonical_text(factor, label)
                    target.update(label=label, text=text, mask_reason=None if text else
                                  "not_visible_or_uncertain")
                elif state["status"] == "failed":
                    target["mask_reason"] = "label_failed"
                else:
                    raise ValueError(f"unfinished job {job.key}")
            target_list.append(target)
    return dict(
        clip_id=entry["clip"], split=split, signer=entry["stratum"],
        num_frames=entry["num_frames"], frame_indices=indices,
        normalized_positions=list(POSITIONS),
        frame_paths=[f"{split}/{entry['clip']}/{frame_names[i]}" for i in indices]
        if frame_names is not None else [None] * 5,
        targets=target_list, valid_mask=[t["text"] is not None for t in target_list],
    )


def export_file(root, output, completion=None):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    sample = json.loads((root / "sample.json").read_text())
    if sample.get("source") != "phoenix-dir":
        raise ValueError("this export expects the directory-backed PHOENIX source")
    provenance = json.loads((root / "provenance.json").read_text())
    regions = json.loads((root / "regions.json").read_text())
    records = []
    for entry in sample["clips"]:
        directory = Path(sample["frames_root"]) / sample["split"] / entry["clip"]
        names = [p.name for p in sorted(directory.glob("*.png"))]
        if len(names) != entry["num_frames"]:
            raise ValueError(f"source-frame count changed: {entry['clip']}")
        records.append(clip_record(root, entry, regions, sample["split"], names))
    clips = [r["clip_id"] for r in records]
    if len(clips) != len(set(clips)):
        raise ValueError("duplicate clip ids")
    reasons = Counter(t["mask_reason"] for r in records for t in r["targets"]
                      if t["text"] is None)
    metadata = dict(
        format_version=FORMAT_VERSION, canonical_version="crop-closed-v1",
        clip_count=len(records), target_slots=19, stable_factors=list(STABLE),
        articulators=list(ARTICULATORS), slot_order="4 stable, then 5 frames x left/right/mouth",
        frame_index_convention="zero-based position in sorted source PNG filenames",
        frame_path_base="PHOENIX-2014-T/features/fullFrame-210x260px",
        split=sample["split"], model=provenance.get("model"),
        model_digest=provenance.get("model_digest"),
        supervision_policy="qwen-schema98-unreviewed-v1", human_review_status="not_assessed",
        source_run=root.name, source_sample_sha256=sha256_file(root / "sample.json"),
        source_regions_sha256=sha256_file(root / "regions.json"),
        source_provenance_sha256=sha256_file(root / "provenance.json"),
        export_code_sha256=package_code_sha256(),
        masked_slot_counts=dict(reasons),
        valid_text_count=sum(sum(r["valid_mask"]) for r in records),
        completion=completion,
        encoder_instructions="Encode non-null targets[].text; preserve slot order; "
        "leave masked slots zero-valued. No features are included.",
    )
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    metadata["records_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    atomic_json(output, dict(metadata=metadata, records=records))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    completion = validate_completion(args.root)
    metadata = export_file(args.root, args.output, completion)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

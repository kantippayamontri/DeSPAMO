"""Join whole-frame stable labels and corrected crop labels by clip and source frame.

Portable text-only export. Retains all 31,165 train clips, even unreadable videos.
Never regenerates or overwrites a completed source extraction.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from how2sign_probe.completion import validate_completion
from how2sign_probe.export_targets import ARTICULATORS, STABLE, clip_record
from how2sign_probe.labeling import atomic_json
from how2sign_probe.provenance import sha256_file
from how2sign_probe.stable import record_path, target_list, validate_response, verify_complete


def export_file(crop_root, stable_root, output):
    crop_root, stable_root, output = Path(crop_root), Path(stable_root), Path(output)
    if output.exists():
        raise FileExistsError(output)
    sample = json.loads((crop_root / "sample.json").read_text())
    stable_sample = json.loads((stable_root / "sample.json").read_text())
    if sample != stable_sample:
        raise ValueError("stable and crop clip census differ")
    regions = json.loads((crop_root / "regions.json").read_text())
    version = json.loads((stable_root / "version.json").read_text())
    crop_provenance = json.loads((crop_root / "provenance.json").read_text())
    records = []
    for entry in sample["clips"]:
        clip = entry["clip"]
        path = record_path(stable_root, clip)
        state = json.loads(path.read_text())
        if state.get("dataset_key") != version["dataset_key"] or state["clip_id"] != clip:
            raise ValueError("stable version or clip identity mismatch")
        receipt = json.loads((crop_root / "repair_clips" / f"{clip}.json").read_text())
        indices = [f["index"] for f in receipt["frames"]]
        if indices and len(indices) != 5:
            raise ValueError(f"incomplete repaired-frame inventory: {clip}")
        if state.get("frame_indices") and state["frame_indices"] != indices:
            raise ValueError(f"whole-frame and crop frame alignment differs: {clip}")
        if state.get("source_frame_sha256") and state["source_frame_sha256"] != [
            f["source_frame_sha256"] for f in receipt["frames"]
        ]:
            raise ValueError(f"whole-frame and crop source pixels differ: {clip}")
        row = clip_record(crop_root, {**entry, "num_frames": state.get("frame_count")},
                          regions, "train", frame_indices=indices or [None] * 5)
        # VIDEO_ID strata are not signer IDs in How2Sign.
        row.update(signer=None, video_stratum=entry["stratum"],
                   video_path=f"train_sentence_videos/{clip}.mp4",
                   source_errors=receipt.get("errors", []))
        if state["status"] == "valid":
            record = validate_response(json.dumps(state["record"]), clip, indices)
            texts = target_list(record)
            for n, factor in enumerate(STABLE):
                text = texts[n]["text"]
                row["targets"][n].update(text=text, label=record.stable.model_dump()[factor],
                                          mask_reason=None if text else "not_visible_or_uncertain",
                                          status="valid", journal_sha256=sha256_file(path))
        elif state["status"] == "failed":
            for target in row["targets"][:4]:
                target.update(mask_reason="stable_label_failed", status="failed",
                              journal_sha256=sha256_file(path))
        else:
            raise ValueError(f"unfinished stable clip: {clip}")
        if not indices:
            for target in row["targets"][4:]:
                target["mask_reason"] = "source_unavailable"
        row["valid_mask"] = [t["text"] is not None for t in row["targets"]]
        records.append(row)
    reasons = Counter(t["mask_reason"] for row in records for t in row["targets"]
                      if t["text"] is None)
    metadata = dict(
        format_version="how2sign-qwen-seven-factor-targets-v1", clip_count=len(records),
        target_slots=19, stable_factors=list(STABLE), articulators=list(ARTICULATORS),
        slot_order="4 stable, then 5 frames x left/right/mouth", split="train",
        frame_index_convention="zero-based decoded video frame index",
        source_crop_run=crop_root.name, source_stable_run=stable_root.name,
        source_regions_sha256=sha256_file(crop_root / "regions.json"),
        stable_version=version,
        crop_model=crop_provenance.get("model"),
        crop_model_digest=crop_provenance.get("model_digest"),
        human_review_status="not_assessed", supervision_policy="qwen-schema98-unreviewed-v1",
        masked_slot_counts=dict(reasons),
        valid_text_count=sum(sum(r["valid_mask"]) for r in records),
        records_sha256=hashlib.sha256(json.dumps(
            records, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest(),
        encoder_instructions="Encode non-null targets[].text, retain 19-slot order and "
        "keep masked vectors zero. No embeddings are included.",
    )
    atomic_json(output, dict(metadata=metadata, records=records))
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--stable-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate_completion(args.crop_root)
    verify_complete(args.stable_root)
    print(json.dumps(export_file(args.crop_root, args.stable_root, args.output), indent=2))


if __name__ == "__main__":
    main()

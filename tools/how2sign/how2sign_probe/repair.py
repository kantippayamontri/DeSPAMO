"""Versioned mouth-only recovery. Old artifacts remain immutable.

Every sampled frame is revisited, including those with no old mouth detection.
Only schema-valid completed hand journals are copied; no mouth journal is reused.
Per-clip receipts provide resumable tracking of old and corrected mouth jobs.
"""

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import cv2

from how2sign_probe.detector import Region, mouth_region
from how2sign_probe.frames import frame_count, read_frame, sample_indices
from how2sign_probe.labeling import atomic_json, journal_path, make_job
from how2sign_probe.labels import parse_label
from how2sign_probe.phoenix import index_frames_dir
from how2sign_probe.probe import detector_gate, region_stats, write_crop
from how2sign_probe.provenance import package_code_sha256, sha256_file

REPAIR_VERSION = "mouth-source-coordinates-v2"


def frames_for_clip(sample, entry, video_root):
    if sample.get("source") == "phoenix-dir":
        fi = index_frames_dir([entry], Path(sample["frames_root"]), sample.get("split", "train"))
        frames = fi[entry["clip"]]
        return sample_indices(entry["num_frames"]), lambda i: cv2.imread(
            str(frames.abs_path(frames[i]))
        )
    if video_root is None:
        raise ValueError("MP4 repair requires --video-root")
    path = Path(video_root) / f"{entry['clip']}.mp4"
    return sample_indices(frame_count(path)), lambda i: read_frame(path, i)


def _link_crop(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(source) != sha256_file(destination):
            raise ValueError(f"existing crop differs: {destination}")
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def _reuse_hand(source, output, job):
    _link_crop(source / "crops" / f"{job.key}.png", output / "crops" / f"{job.key}.png")
    src = journal_path(source, job)
    if not src.exists():
        return 0
    state = json.loads(src.read_text())
    if state.get("status") != "done":
        return 0
    if json.loads(state["identity"]) != list(job.identity):
        raise ValueError(f"hand identity mismatch: {job.key}")
    parse_label("hand", json.dumps(state["label"]))
    dst = journal_path(output, job)
    if not dst.exists():
        atomic_json(dst, {**state, "reused_from": str(source)})
    return 1


def prepare(source, output, video_root=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output:
        raise ValueError("repair requires a new output root")
    sample = json.loads((source / "sample.json").read_text())
    old_regions = json.loads((source / "regions.json").read_text())
    identity = dict(version=REPAIR_VERSION, source_root=str(source),
                    sample_sha256=sha256_file(source / "sample.json"),
                    regions_sha256=sha256_file(source / "regions.json"))
    binding = output / "repair_identity.json"
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("repair source changed")
    atomic_json(binding, identity)
    atomic_json(output / "sample.json", sample)
    provenance = json.loads((source / "provenance.json").read_text())
    atomic_json(output / "provenance.json", {**provenance, "repair": identity,
                "repair_code_sha256": package_code_sha256(),
                "supervision_policy": "qwen-schema98-unreviewed-v1",
                "human_review_status": "not_assessed"})
    receipts = []
    for number, entry in enumerate(sample["clips"], 1):
        clip = entry["clip"]
        if Path(clip).name != clip or clip in (".", ".."):
            raise ValueError("invalid clip id")
        path = output / "repair_clips" / f"{clip}.json"
        if path.exists():
            receipt = json.loads(path.read_text())
        else:
            receipt = dict(clip=clip, frames=[], regions={}, hands_imported=0, errors=[])
            try:
                indices, reader = frames_for_clip(sample, entry, video_root)
            except ValueError as error:
                if sample.get("source") == "phoenix-dir":
                    raise
                indices, reader = [], None
                receipt["errors"].append(str(error))
            for index in indices:
                key = f"{clip}|{index}"
                old = old_regions.get(key, [])
                hands = [r for r in old if r[0] == "hand"]
                tracking = dict(index=index, old_mouths=[], new_mouth_key=None,
                                action="recrop_and_relabel")
                for kind, side, box, _score in old:
                    job = make_job(kind, side, clip, index, json.dumps(box))
                    if kind == "hand":
                        receipt["hands_imported"] += _reuse_hand(source, output, job)
                    elif kind == "mouth":
                        journal = journal_path(source, job)
                        status = (
                            json.loads(journal.read_text())["status"]
                            if journal.exists() else "unlabeled"
                        )
                        tracking["old_mouths"].append(dict(key=job.key, status=status, box=box))
                frame = reader(index)
                if frame is None:
                    raise ValueError(f"cannot decode {clip} frame {index}")
                # These hashes bind the repaired crop to the decoded source frame.
                tracking["source_frame_sha256"] = hashlib.sha256(frame.tobytes()).hexdigest()
                height, width = frame.shape[:2]
                region = mouth_region(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), height, width)
                corrected = list(hands)
                if region is not None:
                    job = make_job("mouth", "MOUTH", clip, index, json.dumps(list(region.box)))
                    crop = write_crop(frame, region, output / "crops", job.key)
                    tracking.update(new_mouth_key=job.key, box=list(region.box),
                                    crop_sha256=sha256_file(crop))
                    corrected.append(["mouth", "MOUTH", list(region.box), region.score])
                else:
                    tracking["action"] = "no_mouth_detected_after_repair"
                receipt["regions"][key] = corrected
                receipt["frames"].append(tracking)
            atomic_json(path, receipt)
        receipts.append(receipt)
        if number % 100 == 0 or number == len(sample["clips"]):
            print(f"mouth repair {number}/{len(sample['clips'])} clips", flush=True)
    regions = {key: items for receipt in receipts for key, items in receipt["regions"].items()}
    atomic_json(output / "regions.json", regions)
    atomic_json(output / "repair_manifest.json", dict(
        **identity, clips=len(receipts), frames=sum(len(r["frames"]) for r in receipts),
        hands_imported=sum(r["hands_imported"] for r in receipts),
        mouth_labels_invalidated=sum(len(f["old_mouths"]) for r in receipts for f in r["frames"]),
        corrected_mouth_jobs=sum(
            f["new_mouth_key"] is not None for r in receipts for f in r["frames"]
        ),
        clips_with_source_errors=[r["clip"] for r in receipts if r["errors"]],
        tracking_directory="repair_clips",
    ))
    stats = region_stats({k: [Region(a, b, tuple(c), d) for a, b, c, d in v]
                          for k, v in regions.items()})
    gate = detector_gate(stats)
    atomic_json(
        output / "detector_report.json", dict(stats=stats, gate=gate, repair=REPAIR_VERSION)
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--video-root", type=Path)
    args = parser.parse_args()
    prepare(args.source_root, args.output_root, args.video_root)


if __name__ == "__main__":
    main()

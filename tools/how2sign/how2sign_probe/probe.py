"""Stage orchestration for the How2Sign crop probe (design 2026-10-01, Stage 1)."""

import json

import cv2
import numpy as np

from how2sign_probe.crops import resize_to_224
from how2sign_probe.detector import DETECTION_FLOOR_PX, Region
from how2sign_probe.labeling import make_job

GATE = {
    "min_hand_rate": 0.65,
    "min_two_hand_rate": 0.50,
    "min_mouth_rate": 0.70,
    "max_below_floor_share": 0.20,
}


def enumerate_jobs(clips, regions):
    jobs = []
    for clip in clips:
        for index in clip["frames"]:
            found = regions.get((clip["clip"], index), [])
            best = {}
            for region in found:
                slot = (region.kind, region.side)
                if slot not in best or region.score > best[slot].score:
                    best[slot] = region
            for (kind, side), region in sorted(best.items()):
                jobs.append(make_job(kind, side, clip["clip"], index, json.dumps(list(region.box))))
    return jobs


def region_stats(regions_by_frame):
    frames = list(regions_by_frame)
    hand_rate = sum(
        min(len([r for r in regions_by_frame[k] if r.kind == "hand"]), 2) / 2.0 for k in frames
    ) / len(frames)
    two_hand_rate = sum(
        len([r for r in regions_by_frame[k] if r.kind == "hand"]) >= 2 for k in frames
    ) / len(frames)
    mouth_rate = sum(any(r.kind == "mouth" for r in regions_by_frame[k]) for k in frames) / len(
        frames
    )
    sides = [
        max(r.box[2] - r.box[0] + 1, r.box[3] - r.box[1] + 1)
        for k in frames
        for r in regions_by_frame[k]
    ]
    below_floor = sum(side < DETECTION_FLOOR_PX for side in sides)
    return {
        "frames": len(frames),
        "hand_rate": round(hand_rate, 4),
        "two_hand_rate": round(two_hand_rate, 4),
        "mouth_rate": round(mouth_rate, 4),
        "median_source_crop_px": float(np.median(sides)) if sides else None,
        "crop_px_quantiles": [float(v) for v in np.percentile(sides, [5, 25, 75, 95])]
        if sides
        else [],
        "crop_count": len(sides),
        "below_floor_share": round(below_floor / len(sides), 4) if sides else None,
    }


def detector_gate(stats, thresholds=None) -> dict:
    thresholds = thresholds or GATE
    reasons = []
    if stats["hand_rate"] < thresholds["min_hand_rate"]:
        reasons.append(f"hand_rate {stats['hand_rate']} < {thresholds['min_hand_rate']}")
    if stats["two_hand_rate"] < thresholds["min_two_hand_rate"]:
        reasons.append(
            f"two_hand_rate {stats['two_hand_rate']} < {thresholds['min_two_hand_rate']}"
        )
    if stats["mouth_rate"] < thresholds["min_mouth_rate"]:
        reasons.append(f"mouth_rate {stats['mouth_rate']} < {thresholds['min_mouth_rate']}")
    if (
        stats["below_floor_share"] is not None
        and stats["below_floor_share"] > thresholds["max_below_floor_share"]
    ):
        reasons.append(
            f"below_floor_share {stats['below_floor_share']} "
            f"> {thresholds['max_below_floor_share']}"
        )
    return {"pass": not reasons, "reasons": reasons, "thresholds": dict(thresholds)}


def write_crop(frame, region: Region, crops_dir, key: str):
    crop, _factor = resize_to_224(frame, region.box)
    crops_dir.mkdir(parents=True, exist_ok=True)
    path = crops_dir / f"{key}.png"
    cv2.imwrite(str(path), crop)
    return path

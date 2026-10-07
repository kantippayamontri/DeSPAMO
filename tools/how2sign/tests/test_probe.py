import numpy as np
from how2sign_probe.detector import Region
from how2sign_probe.probe import (
    detector_gate,
    enumerate_jobs,
    region_stats,
    write_crop,
)


def _clip(clip_id, frames=(6, 18)):
    return {"clip": clip_id, "stratum": "v-1", "frames": list(frames)}


def test_enumerate_jobs_one_per_detected_region():
    regions = {
        ("A", 6): [
            Region("hand", "LEFT", (0, 0, 50, 50), 0.9),
            Region("mouth", "MOUTH", (60, 60, 100, 100), 0.7),
        ],
        ("A", 18): [Region("hand", "RIGHT", (0, 0, 50, 50), 0.9)],
    }
    jobs = enumerate_jobs([_clip("A")], regions)
    keys = sorted((j.clip, j.frame_index, j.side) for j in jobs)
    assert keys == [("A", 6, "LEFT"), ("A", 6, "MOUTH"), ("A", 18, "RIGHT")]


def test_enumerate_jobs_drops_duplicate_sides():
    regions = {
        ("A", 6): [
            Region("hand", "LEFT", (0, 0, 50, 50), 0.9),
            Region("hand", "LEFT", (0, 0, 40, 40), 0.4),
        ]
    }
    jobs = enumerate_jobs([_clip("A")], regions)
    assert len(jobs) == 1


def _stats(regions):
    return region_stats({"c": regions})


def _hand(side="LEFT", w=100, score=0.9):
    return Region("hand", side, (0, 0, w, w), score)


def test_detector_gate_passes_good_sample():
    per_frame = {("c", i): [_hand("LEFT"), _hand("RIGHT"), _hand_mouth()] for i in range(100)}
    stats = region_stats(per_frame)
    assert stats["hand_rate"] == 1.0
    gate = detector_gate(stats)
    assert gate["pass"] is True, gate


def _hand_mouth():
    return Region("mouth", "MOUTH", (0, 0, 80, 40), 0.8)


def test_detector_gate_fails_low_hand_rate():
    per_frame = {("c", i): [_hand(), _hand_mouth()] for i in range(10)}
    gate = detector_gate(region_stats(per_frame))
    assert gate["pass"] is False
    assert gate["reasons"], gate


def test_detector_gate_fails_low_mouth_rate():
    per_frame = {("c", i): [_hand("LEFT"), _hand("RIGHT")] for i in range(10)}
    gate = detector_gate(region_stats(per_frame))
    assert gate["pass"] is False


def test_detector_gate_fails_tiny_crops():
    per_frame = {
        ("c", i): [_hand("LEFT", w=10), _hand("RIGHT", w=10), _hand_mouth()] for i in range(10)
    }
    stats = region_stats(per_frame)
    assert stats["below_floor_share"] > 0.5
    assert detector_gate(stats)["pass"] is False


def test_region_stats_shape():
    stats = region_stats({("c", i): [_hand(), _hand_mouth()] for i in range(4)})
    assert stats["frames"] == 4
    assert stats["hand_rate"] == 0.5
    assert stats["mouth_rate"] == 1.0
    assert stats["two_hand_rate"] == 0.0
    assert stats["median_source_crop_px"] == 91


def test_write_crop_png_roundtrip(tmp_path):
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[10:60, 10:60] = 7
    import cv2

    region = Region("hand", "LEFT", (10, 10, 60, 60), 0.9)
    path = write_crop(frame, region, tmp_path, "key1")
    crop = cv2.imread(str(path))
    assert crop.shape == (224, 224, 3)
    assert crop.mean() > 0

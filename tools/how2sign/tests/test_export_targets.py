import json

import pytest
from how2sign_probe import export_targets
from how2sign_probe.labeling import atomic_json, journal_path, make_job

HAND = {
    "handshape_class": "fist", "finger_spread": "not_applicable",
    "palm_direction": "uncertain", "thumb_position": "tucked", "visibility": "partial",
}


def test_clip_export_orders_19_slots_and_masks_failures_without_dropping_clip(tmp_path):
    entry = dict(clip="A", stratum="Signer01", num_frames=21)
    box = [10, 10, 50, 50]
    # Two candidates for LEFT: only the higher-score region was selected by labeling.
    raw = {"A|2": [
        ["hand", "LEFT", [0, 0, 40, 40], .5], ["hand", "LEFT", box, .9],
        ["mouth", "MOUTH", box, .9],
    ]}
    hand = make_job("hand", "LEFT", "A", 2, json.dumps(box))
    mouth = make_job("mouth", "MOUTH", "A", 2, json.dumps(box))
    atomic_json(journal_path(tmp_path, hand), dict(
        identity=json.dumps(hand.identity), status="done", label=HAND, attempts=[],
    ))
    atomic_json(journal_path(tmp_path, mouth), dict(
        identity=json.dumps(mouth.identity), status="failed", attempts=[],
    ))
    row = export_targets.clip_record(tmp_path, entry, raw, split="train")
    assert row["clip_id"] == "A"
    assert row["frame_indices"] == [2, 6, 10, 14, 18]
    assert len(row["targets"]) == len(row["valid_mask"]) == 19
    assert [t["factor"] for t in row["targets"][:7]] == [
        "biometric", "clothing", "hair", "background", "left_hand", "right_hand", "mouth",
    ]
    assert all(t["frame_index"] == -1 for t in row["targets"][:4])
    assert row["targets"][4]["text"] == (
        "left hand: handshape class fist; finger spread not applicable; thumb position tucked"
    )
    assert row["targets"][4]["label"] == HAND
    assert row["targets"][6]["mask_reason"] == "label_failed"
    assert row["targets"][5]["mask_reason"] == "not_detected"
    assert sum(row["valid_mask"]) == 1


def test_canonical_text_masks_nonvisible_and_all_uncertain():
    assert export_targets.canonical_text("left_hand", {**HAND, "visibility": "uncertain"}) is None
    all_uncertain = {k: "uncertain" for k in HAND}
    all_uncertain["visibility"] = "clear"
    assert export_targets.canonical_text("left_hand", all_uncertain) is None
    assert "uncertain" not in export_targets.canonical_text("left_hand", HAND)


def test_export_refuses_pending_or_identity_mismatch(tmp_path):
    entry = dict(clip="A", stratum="Signer01", num_frames=21)
    box = [0, 0, 40, 40]
    raw = {"A|2": [["hand", "LEFT", box, .9]]}
    job = make_job("hand", "LEFT", "A", 2, json.dumps(box))
    atomic_json(journal_path(tmp_path, job), dict(
        identity=json.dumps(job.identity), status="pending", attempts=[],
    ))
    with pytest.raises(ValueError, match="unfinished"):
        export_targets.clip_record(tmp_path, entry, raw, split="train")
    atomic_json(journal_path(tmp_path, job), dict(
        identity=json.dumps(["wrong"]), status="done", label=HAND, attempts=[],
    ))
    with pytest.raises(ValueError, match="identity"):
        export_targets.clip_record(tmp_path, entry, raw, split="train")


def test_export_creates_portable_single_file_and_refuses_overwrite(tmp_path):
    entry = dict(clip="A", stratum="Signer01", num_frames=21)
    frames = tmp_path / "dataset"
    directory = frames / "train" / "A"
    directory.mkdir(parents=True)
    for i in range(101, 122):
        (directory / f"images{i:04d}.png").touch()
    atomic_json(tmp_path / "sample.json", dict(
        source="phoenix-dir", split="train", frames_root=str(frames), clips=[entry],
    ))
    atomic_json(tmp_path / "regions.json", {})
    atomic_json(tmp_path / "provenance.json", dict(
        model="qwen3-vl:8b", model_digest="frozen-digest",
        annotations_sha256="annotation-hash", human_review_status="not_assessed",
    ))
    output = tmp_path / "handoff.json"
    export_targets.export_file(tmp_path, output)
    doc = json.loads(output.read_text())
    assert doc["metadata"]["clip_count"] == 1
    assert doc["metadata"]["target_slots"] == 19
    assert doc["metadata"]["human_review_status"] == "not_assessed"
    assert doc["records"][0]["clip_id"] == "A"
    assert str(tmp_path) not in output.read_text()
    assert doc["records"][0]["frame_paths"][0] == "train/A/images0103.png"
    with pytest.raises(FileExistsError):
        export_targets.export_file(tmp_path, output)

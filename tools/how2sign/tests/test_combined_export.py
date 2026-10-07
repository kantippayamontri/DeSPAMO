import json

import pytest
from how2sign_probe import combined_export, stable
from how2sign_probe.labeling import atomic_json, journal_path, make_job
from test_stable import LABELS, response


def test_combines_four_stable_and_fifteen_articulator_slots(tmp_path):
    crop, whole = tmp_path / "crop", tmp_path / "whole"
    entry = dict(clip="A", stratum="video")
    sample = dict(clips=[entry])
    atomic_json(crop / "sample.json", sample)
    atomic_json(whole / "sample.json", sample)
    indices = [2, 6, 10, 14, 18]
    atomic_json(crop / "repair_clips/A.json", dict(
        clip="A", frames=[dict(index=i, source_frame_sha256=f"hash{i}") for i in indices],
        errors=[],
    ))
    atomic_json(crop / "provenance.json", dict(model="qwen3-vl:8b", model_digest="digest"))
    atomic_json(whole / "version.json", dict(dataset_key="key", model="qwen3-vl:8b",
                                           model_digest="digest"))
    record = stable.validate_response(response(), "A", indices)
    atomic_json(stable.record_path(whole, "A"), dict(
        clip_id="A", dataset_key="key", status="valid", record=record.model_dump(),
        frame_count=21, frame_indices=indices, targets=stable.target_list(record),
        source_frame_sha256=[f"hash{i}" for i in indices], attempts=[],
    ))
    box = [0, 0, 40, 40]
    atomic_json(crop / "regions.json", {"A|2": [["hand", "LEFT", box, .9]]})
    hand = make_job("hand", "LEFT", "A", 2, json.dumps(box))
    atomic_json(journal_path(crop, hand), dict(
        identity=json.dumps(hand.identity), status="done", attempts=[], label=dict(
            handshape_class="fist", finger_spread="together", palm_direction="up",
            thumb_position="tucked", visibility="clear",
        ),
    ))
    output = tmp_path / "combined.json"
    combined_export.export_file(crop, whole, output)
    doc = json.loads(output.read_text())
    row = doc["records"][0]
    assert len(row["targets"]) == len(row["valid_mask"]) == 19
    assert row["valid_mask"][:5] == [True] * 5
    assert row["targets"][0]["text"] == "biometric: apparent build average"
    assert row["targets"][1]["label"] == LABELS["clothing"]
    assert row["targets"][4]["factor"] == "left_hand"
    assert row["frame_indices"] == indices
    assert row["video_path"] == "train_sentence_videos/A.mp4"
    assert str(tmp_path) not in output.read_text()
    # Same clip but different source pixels must never be silently joined.
    state = json.loads(stable.record_path(whole, "A").read_text())
    state["source_frame_sha256"][0] = "wrong"
    atomic_json(stable.record_path(whole, "A"), state)
    with pytest.raises(ValueError, match="pixels"):
        combined_export.export_file(crop, whole, tmp_path / "bad.json")


def test_unreadable_clip_keeps_19_masked_slots(tmp_path):
    crop, whole = tmp_path / "crop", tmp_path / "whole"
    sample = dict(clips=[dict(clip="unreadable", stratum="video")])
    for root in [crop, whole]:
        atomic_json(root / "sample.json", sample)
    atomic_json(crop / "repair_clips/unreadable.json", dict(
        clip="unreadable", frames=[], errors=["cannot open video"],
    ))
    atomic_json(crop / "regions.json", {})
    atomic_json(crop / "provenance.json", dict(model="qwen3-vl:8b", model_digest="digest"))
    atomic_json(whole / "version.json", dict(dataset_key="key"))
    atomic_json(stable.record_path(whole, "unreadable"), dict(
        clip_id="unreadable", dataset_key="key", status="failed", targets=stable.masked_targets(),
        error="source unavailable", attempts=[],
    ))
    output = tmp_path / "combined.json"
    combined_export.export_file(crop, whole, output)
    row = json.loads(output.read_text())["records"][0]
    assert row["clip_id"] == "unreadable"
    assert len(row["targets"]) == 19
    assert row["valid_mask"] == [False] * 19
    assert row["frame_indices"] == [None] * 5

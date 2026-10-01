import json

import pytest

from scripts.smoke_e3_signer_pilot import build_smoke_summary, choose_smoke_clips
from tools.dinov3.identity import digest


def _inputs():
    train = ["a", "b", "c"]
    source = {"split": "train", "clips": [
        {"clip_id": clip, "frames": [1] * 5} for clip in train
    ]}
    text = {"records": [
        {"clip_id": "a", "targets": [{"text": "visible"}] * 18 + [{"text": None}]},
        {"clip_id": "b", "targets": [{"text": "visible"}] * 19},
        {"clip_id": "c", "targets": [{"text": None}] * 19},
    ]}
    protocol = {"source_hash": digest(source), "split": {"groups": {
        "train": {"clip_ids": train, "valid": 2, "failed": 1},
        "dev": {"clip_ids": ["d"]}, "test": {"clip_ids": ["e"]},
    }}}
    protocol["protocol_hash"] = digest(protocol)
    return protocol, source, text


def test_e3_smoke_accepts_valid_partially_masked_and_fully_failed_clips():
    protocol, source, text = _inputs()
    assert choose_smoke_clips(protocol, source, text) == ("a", "b", "c")


def test_e3_smoke_rejects_empty_target_and_failed_count_mismatch():
    protocol, source, text = _inputs()
    text["records"][0]["targets"][0] = {"text": ""}
    with pytest.raises(ValueError, match="target"):
        choose_smoke_clips(protocol, source, text)
    text["records"][0]["targets"][0] = {"text": "visible"}
    protocol["split"]["groups"]["train"]["failed"] = 2
    protocol["protocol_hash"] = digest({k: v for k, v in protocol.items() if k != "protocol_hash"})
    with pytest.raises(ValueError, match="count"):
        choose_smoke_clips(protocol, source, text)


def test_e3_smoke_rejects_overlapping_dev_or_test_clip():
    protocol, source, text = _inputs()
    protocol["split"]["groups"]["test"]["clip_ids"] = ["a"]
    protocol["protocol_hash"] = digest({k: v for k, v in protocol.items() if k != "protocol_hash"})
    with pytest.raises(ValueError, match="overlap"):
        choose_smoke_clips(protocol, source, text)


def test_e3_smoke_receipt_keeps_clip_ids_after_json_round_trip():
    protocol, _, _ = _inputs()
    protocol["text_manifest_hash"] = "text-hash"
    protocol["frame_rows_hash"] = "rows-hash"
    summary = build_smoke_summary(protocol, ("a", "b", "c"), [19, 18, 0])
    assert json.loads(json.dumps(summary)) == summary
    assert summary["selected_train_clip_ids"] == ["a", "b", "c"]

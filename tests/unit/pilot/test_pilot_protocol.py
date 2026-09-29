from copy import deepcopy

import pytest

from despamo.pilot import bind_protocol, validate_live_counts
from tests.unit.pilot.test_signer_split import fixture
from despamo.data.signer_split import build_split
from despamo.appearance.provenance import digest, read_json
from scripts.freeze_signer_pilot import publish_protocol


def test_protocol_binds_split_and_unmodified_text_encoder():
    source, journals = fixture()
    split = build_split(source, journals)
    metadata = {
        "model": "openai/clip-vit-large-patch14",
        "revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "tokenizer_revision": "32bd64288804d66eefd0ccbe215aa642df71cc41",
        "schema_version": 2,
        "width": 768,
    }
    proto = bind_protocol(
        split, "qwen-version", "source-hash", "version-hash", "records-hash",
        "text-hash", metadata, "frame-rows-hash", "spatial-hash", "motion-hash",
        "annotation-hash",
    )
    assert proto["split_hash"] == split["split_hash"]
    assert proto["decoder"] == {"mode": "deterministic", "beam_size": 5,
                                 "max_length": 64, "in_context": False}
    assert proto["supervision_policy"] == "qwen-schema98-unreviewed-v1"
    assert proto["human_review_status"] == "not_assessed"
    assert len(proto["protocol_hash"]) == 64
    changed = deepcopy(split)
    changed["split_hash"] = "different"
    assert bind_protocol(changed, "qwen-version", "source-hash", "version-hash",
                         "records-hash", "text-hash", metadata, "frame-rows-hash",
                         "spatial-hash", "motion-hash", "annotation-hash")["protocol_hash"] != proto["protocol_hash"]
    for changed_meta in ({**metadata, "revision": "other"},
                         {**metadata, "tokenizer_revision": "other"},
                         {**metadata, "model": "other"}):
        with pytest.raises(ValueError, match="CLIP"):
            bind_protocol(split, "qwen-version", "source-hash", "version-hash",
                          "records-hash", "text-hash", changed_meta, "frame-rows-hash",
                          "spatial-hash", "motion-hash", "annotation-hash")


def test_live_counts_require_exact_frozen_split_and_98_percent():
    split = {
        "groups": {
            "train": {"clip_ids": [str(i) for i in range(5746)], "valid": 5641, "failed": 105},
            "dev": {"clip_ids": [str(i) for i in range(582)], "valid": 571, "failed": 11},
            "test": {"clip_ids": [str(i) for i in range(768)], "valid": 765, "failed": 3},
        }
    }
    validate_live_counts(split)
    changed = deepcopy(split)
    changed["groups"]["train"]["valid"] = 5600
    changed["groups"]["train"]["failed"] = 146
    with pytest.raises(ValueError, match="count|98%"):
        validate_live_counts(changed)
    changed = deepcopy(split)
    changed["groups"]["dev"]["clip_ids"].pop()
    with pytest.raises(ValueError, match="count"):
        validate_live_counts(changed)


def test_protocol_published_once_without_overwriting(tmp_path):
    payload = {"version": "fixture", "split_hash": "fixed"}
    protocol = {**payload, "protocol_hash": digest(payload)}
    output = publish_protocol(protocol, tmp_path)
    assert read_json(output) == protocol
    assert output.name == "protocol.json" and output.parent.name == protocol["protocol_hash"]
    with pytest.raises(ValueError, match="exists"):
        publish_protocol(protocol, tmp_path)
    with pytest.raises(ValueError, match="hash"):
        publish_protocol({**protocol, "split_hash": "changed"}, tmp_path)

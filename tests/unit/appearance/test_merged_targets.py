import importlib
import json
from copy import deepcopy

import pytest

from despamo.appearance.provenance import digest, file_hash
from tests.unit.appearance.merge_fixtures import inputs, refreshed


def module():
    return importlib.import_module("despamo.appearance.merged_targets")


def merge(value=None):
    return module().merge_targets(*(value or inputs()), parents={})


def test_v1_stable_v2_articulator_sources():
    result = merge()
    targets = result["records"][0]["targets"]
    assert [t["text"] for t in targets[:4]] == [
        "old biometric", "old clothing", "old hair", "old background"]
    assert targets[4]["text"] == "new left_hand 5"
    assert targets[4]["bbox"] == [1, 2, 3, 4]
    assert result["human_review_status"] == "not_assessed"
    assert result["valid_counts"]["mouth"] == 10


def test_v2_mask_never_backfills():
    value = inputs()
    value[1]["records"][0]["targets"][4].update(text=None, mask_reason="not_detected")
    target = merge(refreshed(value))["records"][0]["targets"][4]
    assert target["text"] is None and target["mask_reason"] == "not_detected"


def test_old_failed_clip_keeps_valid_new_articulators():
    value = inputs()
    for target in value[0]["records"][0]["targets"]:
        target["text"] = None
    targets = merge(value)["records"][0]["targets"]
    assert all(t["text"] is None for t in targets[:4])
    assert all(t["text"] is not None for t in targets[4:])


def test_shuffled_records_join_by_clip_id():
    value = inputs()
    value[0]["records"].reverse()
    value[1]["records"].reverse()
    value[2]["clips"].reverse()
    result = merge(refreshed(value))
    assert [r["clip_id"] for r in result["records"]] == ["a", "b"]
    assert result["records"][0]["signer"] == "Signer03"


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "signer", "count",
                                    "index", "path", "factor", "empty", "teacher", "old_order",
                                    "frame_hash"])
def test_invalid_join_rejected(change):
    value = deepcopy(inputs())
    crop = value[1]["records"]
    if change == "duplicate":
        crop.append(deepcopy(crop[0]))
    elif change == "missing":
        crop.pop()
    elif change == "extra":
        crop.append({**deepcopy(crop[0]), "clip_id": "unknown"})
    elif change == "signer":
        crop[0]["signer"] = "Signer99"
    elif change == "count":
        crop[0]["num_frames"] = 54
    elif change == "index":
        crop[0]["frame_indices"][0] = 6
    elif change == "path":
        crop[0]["frame_paths"][0] = "../outside.png"
    elif change == "factor":
        crop[0]["targets"][4]["factor"] = "right_hand"
    elif change == "empty":
        crop[0]["targets"][4]["text"] = ""
    elif change == "teacher":
        value[3]["clips"]["a"]["source_indices"][0] = 1
    elif change == "old_order":
        value[0]["records"][0]["targets"][0]["factor"] = "hair"
    elif change == "frame_hash":
        value[3]["clips"]["a"]["sampled_images"]["5"] = "f"*64
    with pytest.raises(ValueError):
        merge(refreshed(value))


def test_import_checks_raw_and_records_digests(tmp_path):
    value = inputs()[1]
    path = tmp_path / "crop.json"
    path.write_text(json.dumps(value))
    assert module().import_crop_export(path, file_hash(path)) == value
    with pytest.raises(ValueError):
        module().import_crop_export(path, "f" * 64)
    value["metadata"]["records_sha256"] = "f" * 64
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        module().import_crop_export(path, file_hash(path))


def test_dataset_key_changes_with_targets_or_parents():
    first = merge()
    other = deepcopy(first)
    other["parents"] = {"crop_export": {"sha256": "f" * 64}}
    assert module().merged_target_key(first) != module().merged_target_key(other)
    other = deepcopy(first)
    other["records"][0]["targets"][4]["text"] = "different"
    assert module().merged_target_key(first) != module().merged_target_key(other)
    assert digest(first) == module().merged_target_key(first)

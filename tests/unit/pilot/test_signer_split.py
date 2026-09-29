from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from despamo.data.signer_split import PhoenixTrainView, build_split, verify_split


def fixture():
    source = {
        "split": "train",
        "clips": [
            {"clip_id": "t-2", "signer": "Signer01"},
            {"clip_id": "t-1", "signer": "Signer01"},
            {"clip_id": "u-1", "signer": "Signer05"},
            {"clip_id": "d-1", "signer": "Signer03"},
            {"clip_id": "e-1", "signer": "Signer07"},
        ],
    }
    journals = [
        {"clip_id": c["clip_id"], "status": "failed" if c["clip_id"] == "t-2" else "valid"}
        for c in source["clips"]
    ]
    return source, journals


def test_split_excludes_holdouts_from_train_and_binds_failures():
    source, journals = fixture()
    result = build_split(source, journals)
    assert result["groups"]["train"]["clip_ids"] == ["t-1", "t-2", "u-1"]
    assert result["groups"]["dev"]["clip_ids"] == ["d-1"]
    assert result["groups"]["test"]["clip_ids"] == ["e-1"]
    assert result["groups"]["train"]["valid"] == 2
    assert result["groups"]["train"]["failed"] == 1
    assert set(result["groups"]["train"]["signers"]) == {"Signer01", "Signer05"}
    assert result["groups"]["dev"]["signers"] == ["Signer03"]
    assert result["groups"]["test"]["signers"] == ["Signer07"]
    assert len(result["split_hash"]) == 64
    verify_split(result, source, journals)


def test_status_order_does_not_change_split_and_tampering_is_rejected():
    source, journals = fixture()
    result = build_split(source, journals)
    assert result == build_split(source, list(reversed(journals)))
    altered = deepcopy(result)
    altered["groups"]["train"]["clip_ids"][0] = "e-1"
    with pytest.raises(ValueError, match="split"):
        verify_split(altered, source, journals)
    changed = deepcopy(journals)
    changed[1]["status"] = "failed"
    with pytest.raises(ValueError, match="split"):
        verify_split(result, source, changed)


@pytest.mark.parametrize("change", ["pending", "missing", "duplicate", "duplicate_source"])
def test_incomplete_or_duplicate_source_and_journals_fail(change):
    source, journals = fixture()
    if change == "pending":
        journals[0]["status"] = "pending"
    elif change == "missing":
        journals.pop()
    elif change == "duplicate":
        journals.append(deepcopy(journals[0]))
    else:
        source["clips"].append(deepcopy(source["clips"][0]))
    with pytest.raises(ValueError):
        build_split(source, journals)


def test_logical_holdouts_keep_physical_train_feature_lookup():
    source, _ = fixture()
    calls = []

    class Parent:
        split = "train"
        spatial_root = Path("/sample/train-features")
        motion_root = Path("/sample/motion")
        spatial_manifest = SimpleNamespace(require=lambda split, clip_id: calls.append((split, clip_id)))

        def __init__(self):
            self.records = [{"fileid": c["clip_id"]} for c in source["clips"]]

        def __getitem__(self, index):
            clip_id = self.records[index]["fileid"]
            self.spatial_manifest.require(self.split, clip_id)
            return SimpleNamespace(clip_id=clip_id)

    parent = Parent()
    view = PhoenixTrainView(parent, ("d-1", "e-1"))
    assert view.split == "train" and len(view) == 2
    assert [record["fileid"] for record in view.records] == ["d-1", "e-1"]
    assert [view[i].clip_id for i in range(len(view))] == ["d-1", "e-1"]
    assert calls == [("train", "d-1"), ("train", "e-1")]
    assert view.spatial_root == parent.spatial_root
    with pytest.raises(ValueError, match="unknown"):
        PhoenixTrainView(parent, ("not-in-parent",))


def test_view_rejects_duplicate_and_nontrain_parent():
    class Parent:
        split = "train"
        records = [{"fileid": "a"}, {"fileid": "b"}]

    with pytest.raises(ValueError, match="duplicate"):
        PhoenixTrainView(Parent(), ("a", "a"))
    parent = Parent()
    parent.split = "dev"
    with pytest.raises(ValueError, match="train"):
        PhoenixTrainView(parent, ("a",))

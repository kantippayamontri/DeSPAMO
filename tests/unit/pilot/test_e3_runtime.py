import json
from types import SimpleNamespace

import numpy as np
import pytest

from despamo.appearance.provenance import digest, file_hash
from despamo.training import e3_runtime


def _fixture(tmp_path, monkeypatch):
    names = (("dev", "Signer03"), ("test", "Signer07"), ("train", "Signer01"))
    adapted = tmp_path / "adapted"
    (adapted / "train").mkdir(parents=True)
    (adapted / "complete").mkdir()
    (adapted / "receipts/train").mkdir(parents=True)
    identity = {"checkpoint_hash": "a" * 64, "source_hash": "b" * 64,
                "protocol_hash": "protocol", "frame_rows_hash": "original-rows"}
    key = digest(identity)
    records = []
    rows = {}
    annotation = {}
    for n, (clip_id, signer) in enumerate(names):
        annotation[n] = {"fileid": clip_id, "signer": signer, "gloss": "gloss",
                         "text": "text", "en_text": "english", "es_text": "espanol",
                         "fr_text": "francais"}
        path = adapted / "train" / f"{clip_id}.npy"
        np.save(path, np.full((5, 2048), n + 1, dtype=np.float32))
        entry = {"clip_id": clip_id, "split": "train", "path": f"train/{clip_id}.npy",
                 "length": 5, "width": 2048, "dtype": "float32"}
        records.append(entry)
        rows[clip_id] = {"feature_hash": file_hash(path), "frame_count": 5,
                         "source_hash": "source-" + clip_id}
        (adapted / "receipts/train" / f"{clip_id}.json").write_text(json.dumps({
            "feature_hash": rows[clip_id]["feature_hash"],
            "checkpoint_hash": identity["checkpoint_hash"],
            "source_hash": rows[clip_id]["source_hash"],
        }))
    (adapted / "complete/manifest.json").write_text(json.dumps({
        "schema_version": 1, "encoder": key, "expected_dim": 2048, "records": records,
    }))
    (adapted / "complete/frame_rows.json").write_text(json.dumps({
        "encoder_key": key, "clips": rows,
    }))
    (adapted / "complete/identity.json").write_text(json.dumps(identity))
    annotation_path = tmp_path / "train_info_ml.npy"
    np.save(annotation_path, annotation)
    motion_root = tmp_path / "motion"
    (motion_root / "train").mkdir(parents=True)
    for clip_id, _ in names:
        np.save(motion_root / "train" / f"{clip_id}.npy", np.ones((5, 1024)))
    motion_manifest = tmp_path / "motion.json"
    motion_manifest.write_text(json.dumps({
        "schema_version": 1, "encoder": "motion", "expected_dim": 1024,
        "records": [{"clip_id": clip_id, "split": "train", "path": f"train/{clip_id}.npy",
                     "length": 5, "width": 1024, "dtype": "float64"} for clip_id, _ in names],
    }))
    protocol = {"protocol_hash": "protocol", "source_hash": identity["source_hash"],
                "frame_rows_hash": "original-rows",
                "split": {"groups": {name: {"clip_ids": [name]} for name, _ in names}}}
    original = {name: SimpleNamespace(records=[{"fileid": name, "signer": signer}])
                for name, signer in names}
    monkeypatch.setattr(e3_runtime, "load_pilot_views", lambda *args: (protocol, original))
    paths = {"annotation": annotation_path, "motion_root": motion_root,
             "motion_manifest": motion_manifest, "protocol": tmp_path / "protocol",
             "dataset": tmp_path / "dataset", "text_manifest": tmp_path / "text",
             "dino_root": tmp_path / "original"}
    return paths, adapted, identity


def test_e3_views_bind_adapted_features_and_preserve_physical_train(tmp_path, monkeypatch):
    paths, adapted, identity = _fixture(tmp_path, monkeypatch)
    protocol, views = e3_runtime.load_e3_views(paths, adapted, identity)
    assert protocol["protocol_hash"] == "protocol"
    assert [views[name].records[0]["fileid"] for name in ("train", "dev", "test")] == [
        "train", "dev", "test"
    ]
    assert views["test"].split == "train"
    assert views["test"].spatial_root == adapted
    assert views["test"][0].signer == "Signer07"


def test_e3_views_reject_changed_adaptation_checkpoint(tmp_path, monkeypatch):
    paths, adapted, identity = _fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="identity|checkpoint"):
        e3_runtime.load_e3_views(paths, adapted, {**identity, "checkpoint_hash": "c" * 64})
    feature = adapted / "train/test.npy"
    np.save(feature, np.zeros((5, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match="hash|feature"):
        e3_runtime.load_e3_views(paths, adapted, identity)

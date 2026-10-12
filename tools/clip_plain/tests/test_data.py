import json
from pathlib import Path

import numpy as np
import pytest

from tools.clip_plain.tests.helpers import parent_fixture
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json


def admission_fixture(tmp_path, monkeypatch):
    from tools.clip_plain import data, features, policy

    c = parent_fixture(tmp_path)
    monkeypatch.setattr(features, "require_official_storage", lambda *a, **k: {})
    monkeypatch.setattr(policy, "COUNTS", dict(train=8, dev=3, test=2))
    features.derive_package(c)
    parent = Path(c["parent_root"])
    motion = Path(c["motion_root"])
    identity = dict(kind="motion", width=1024, policy="motion-fixture")
    records = []
    for row in json.loads((parent / "complete/manifest.json").read_text())["records"]:
        split, clip = row["split"], row["clip_id"]
        p = motion / split / f"{clip}.npy"
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(p, np.ones((1, 1024), np.float32))
        r = json.loads((parent / "receipts" / split / f"{clip}.json").read_text())
        r.update(
            identity_hash=digest(identity), feature_rows=1, width=1024, feature_hash=sha256_file(p)
        )
        atomic_json(motion / "receipts" / split / f"{clip}.json", r)
        records.append(
            dict(
                clip_id=clip,
                split=split,
                path=f"{split}/{clip}.npy",
                length=1,
                width=1024,
                dtype="float32",
            )
        )
    atomic_json(motion / "complete/identity.json", identity)
    atomic_json(
        motion / "complete/manifest.json",
        dict(schema_version=1, encoder=digest(identity), expected_dim=1024, records=records),
    )
    c["motion_manifest_sha256"] = sha256_file(motion / "complete/manifest.json")
    return c, data.admit_data(c)


def test_same_motion_and_frame_order(tmp_path, monkeypatch):
    from tools.clip_plain.data import batch_for

    c, a = admission_fixture(tmp_path, monkeypatch)
    b = batch_for(a, "dev", ("dev0", "dev1"))
    assert b.spatial.shape == (2, 20, 1024) and b.motion.shape == (2, 1, 1024)
    assert b.clip_ids == ("dev0", "dev1")
    np.testing.assert_array_equal(b.motion.numpy(), np.ones((2, 1, 1024), np.float32))


def test_changed_array_rejected(tmp_path, monkeypatch):
    from tools.clip_plain.data import batch_for

    _, a = admission_fixture(tmp_path, monkeypatch)
    np.save(a["roots"]["spatial"] / "dev/dev0.npy", np.zeros((20, 1024), np.float32))
    with pytest.raises(ValueError):
        batch_for(a, "dev", ("dev0",))


def test_bounded_loader_does_not_read_next_clip(tmp_path, monkeypatch):
    from tools.clip_plain import data

    _, a = admission_fixture(tmp_path, monkeypatch)
    (a["roots"]["spatial"] / "dev/dev1.npy").unlink()
    assert len(list(data.eval_batches(a, max_batches=1))) == 1


def test_only_plain_complete_package_admitted(tmp_path, monkeypatch):
    from tools.clip_plain import data

    c, a = admission_fixture(tmp_path, monkeypatch)
    p = a["roots"]["spatial"] / "complete/identity.json"
    x = json.loads(p.read_text())
    x["policy"] = "old-s2"
    atomic_json(p, x)
    with pytest.raises(ValueError):
        data.admit_data(c)

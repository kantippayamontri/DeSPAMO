import json

import numpy as np
import pytest

from tools.dinov3.e3_extract import (
    adapted_key,
    adapted_output_path,
    publish_adapted_version,
    verify_adapted_version,
    write_adapted_clip,
)
from tools.dinov3.identity import sha256_file


def _fixture(tmp_path):
    root = tmp_path / "e3"
    root.mkdir()
    frames = tmp_path / "frames"
    frames.mkdir()
    identity = {"checkpoint_hash": "a" * 64, "source_hash": "b" * 64,
                "protocol_hash": "c" * 64, "base_revision": "d" * 40}
    paths = []
    for i in range(5):
        path = frames / f"images{i:04d}.png"
        path.write_bytes(f"image{i}".encode())
        paths.append(path)
    return root, identity, paths


def test_adapted_clip_writes_exact_row_and_refuses_corrupt_reuse(tmp_path):
    root, identity, paths = _fixture(tmp_path)

    def fake_extract(paths, model, device, *, batch_size):
        assert len(paths) == 5
        return np.arange(5 * 2048, dtype=np.float32).reshape(5, 2048), [[20, 20]] * 5

    row = write_adapted_clip(root, "clip", paths, object(), identity, extract_fn=fake_extract)
    assert row["frame_count"] == 5
    assert row["source_indices"] == list(range(5))
    feature = root / "train/clip.npy"
    assert sha256_file(feature) == row["feature_hash"]
    assert write_adapted_clip(root, "clip", paths, object(), identity,
                              extract_fn=fake_extract) == row
    np.save(feature, np.ones((5, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match="receipt|hash|feature"):
        write_adapted_clip(root, "clip", paths, object(), identity,
                           extract_fn=fake_extract)


def test_complete_version_binds_checkpoint_and_exact_clip_inventory(tmp_path):
    root, identity, paths = _fixture(tmp_path)
    assert adapted_key(identity) == adapted_key(dict(reversed(list(identity.items()))))
    row = write_adapted_clip(root, "clip", paths, object(), identity,
                             extract_fn=lambda paths, model, device, **kwargs:
                             (np.ones((5, 2048), dtype=np.float32), [[20, 20]] * 5))
    (root / "complete").mkdir()
    manifest = {"schema_version": 1, "encoder": adapted_key(identity), "expected_dim": 2048,
                "records": [{"clip_id": "clip", "split": "train", "path": "train/clip.npy",
                             "length": 5, "width": 2048, "dtype": "float32"}]}
    (root / "complete/manifest.json").write_text(json.dumps(manifest))
    rows = {"encoder_key": adapted_key(identity), "clips": {"clip": row}}
    (root / "complete/frame_rows.json").write_text(json.dumps(rows))
    (root / "complete/identity.json").write_text(json.dumps(identity))
    result = verify_adapted_version(root, identity, ("clip",))
    assert result["manifest_hash"] == sha256_file(root / "complete/manifest.json")
    with pytest.raises(ValueError, match="identity|checkpoint"):
        verify_adapted_version(root, {**identity, "checkpoint_hash": "e" * 64}, ("clip",))
    with pytest.raises(ValueError, match="inventory"):
        verify_adapted_version(root, identity, ("clip", "missing"))


def test_e3_feature_version_root_is_derived_from_identity(tmp_path):
    identity = {"checkpoint_hash": "a" * 64, "source_hash": "b" * 64}
    expected = tmp_path / f"e3-features-{adapted_key(identity)}"
    assert adapted_output_path(tmp_path, identity) == expected


def test_e3_version_cannot_publish_before_integrity_passes(tmp_path):
    root, identity, paths = _fixture(tmp_path)
    row = write_adapted_clip(root, "clip", paths, object(), identity,
                             extract_fn=lambda paths, model, device, **kwargs:
                             (np.ones((5, 2048), dtype=np.float32), [[20, 20]] * 5))
    record = {"clip_id": "clip", "split": "train", "path": "train/clip.npy",
              "length": 5, "width": 2048, "dtype": "float32"}
    corrupt = {**row, "frame_count": 6}
    with pytest.raises(ValueError, match="integrity|shape|mismatch"):
        publish_adapted_version(root, identity, ("clip",), [record], {"clip": corrupt})
    assert not (root / "complete").exists()

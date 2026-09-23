import json
import pickle
import subprocess
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from despamo.data.manifest import FeatureManifest, index_feature_tree


def test_index_feature_tree_records_shape(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "clip-1.npy", np.zeros((7, 2048), dtype=np.float32))

    manifest = index_feature_tree(tmp_path, 2048, "openai/clip-vit-large-patch14")

    record = manifest.require("train", "clip-1")
    assert manifest.schema_version == 1
    assert manifest.encoder == "openai/clip-vit-large-patch14"
    assert manifest.expected_dim == 2048
    assert record.length == 7
    assert record.width == 2048
    assert record.dtype == "float32"
    assert record.path == "train/clip-1.npy"
    with pytest.raises(FrozenInstanceError):
        record.length = 8
    with pytest.raises(FrozenInstanceError):
        manifest.expected_dim = 1024


def test_manifest_round_trip_and_split_lookup(tmp_path: Path) -> None:
    features = tmp_path / "features"
    for split in ("train", "dev"):
        (features / split).mkdir(parents=True)
        np.save(features / split / "same.npy", np.ones((2, 4), dtype=np.float32))

    manifest = index_feature_tree(features, 4, "clip")
    output = tmp_path / "nested" / "manifest.json"
    manifest.save(output)

    loaded = FeatureManifest.load(output)
    assert loaded == manifest
    assert [record.split for record in loaded.records] == ["dev", "train"]
    assert loaded.require("train", "same").path == "train/same.npy"
    with pytest.raises(KeyError, match="expected one feature for dev/missing, found 0"):
        loaded.require("dev", "missing")


def test_manifest_lookup_is_immutable_and_excluded_from_json(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "clip.npy", np.zeros((2, 4), dtype=np.float32))
    manifest = index_feature_tree(tmp_path, 4, "clip")
    record = manifest.require("train", "clip")

    assert manifest._lookup[("train", "clip")] is record
    with pytest.raises(TypeError):
        manifest._lookup[("train", "clip")] = record

    output = tmp_path / "manifest.json"
    manifest.save(output)
    assert set(json.loads(output.read_text())) == {
        "schema_version",
        "encoder",
        "expected_dim",
        "records",
    }
    loaded = FeatureManifest.load(output)
    assert loaded._lookup[("train", "clip")] is loaded.require("train", "clip")


def test_manifest_pickle_round_trip_rebuilds_immutable_lookup(tmp_path: Path) -> None:
    for split in ("train", "dev"):
        directory = tmp_path / split
        directory.mkdir()
        np.save(directory / "clip.npy", np.zeros((2, 4), dtype=np.float32))
    manifest = index_feature_tree(tmp_path, 4, "clip")
    original_json = tmp_path / "before.json"
    manifest.save(original_json)

    restored = pickle.loads(pickle.dumps(manifest))

    assert restored == manifest
    assert restored.require("train", "clip") is restored._lookup[("train", "clip")]
    assert restored.require("dev", "clip").path == "dev/clip.npy"
    with pytest.raises(KeyError, match="expected one feature for test/clip, found 0"):
        restored.require("test", "clip")
    with pytest.raises(TypeError):
        restored._lookup[("train", "clip")] = restored.require("train", "clip")
    with pytest.raises(FrozenInstanceError):
        restored.require("train", "clip").length = 3
    after_json = tmp_path / "after.json"
    restored.save(after_json)
    assert after_json.read_bytes() == original_json.read_bytes()


@pytest.mark.parametrize("dtype", [np.float16, np.float64, np.dtype(">f4")])
def test_index_records_floating_dtype_without_casting(tmp_path: Path, dtype: object) -> None:
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "clip.npy", np.ones((2, 4), dtype=dtype))

    manifest = index_feature_tree(tmp_path, 4, "clip")
    output = tmp_path / "manifest.json"
    manifest.save(output)

    assert manifest.require("train", "clip").dtype == str(np.dtype(dtype))
    assert FeatureManifest.load(output) == manifest


@pytest.mark.parametrize(
    ("array", "error"),
    [
        (np.zeros((3, 1024), dtype=np.float32), "expected width 2048, got 1024"),
        (np.zeros((0, 2048), dtype=np.float32), "non-empty rank 2"),
        (np.zeros((3, 0), dtype=np.float32), "expected width 2048, got 0"),
        (np.zeros((2048,), dtype=np.float32), "non-empty rank 2"),
        (np.zeros((2, 3, 2048), dtype=np.float32), "non-empty rank 2"),
        (np.zeros((3, 2048), dtype=np.int32), "dtype"),
        (np.zeros((3, 2048), dtype=np.bool_), "dtype"),
        (np.full((3, 2048), "abc"), "dtype"),
        (np.full((3, 2048), None, dtype=object), "invalid feature file"),
    ],
)
def test_index_feature_tree_rejects_invalid_arrays(
    tmp_path: Path, array: np.ndarray, error: str
) -> None:
    split = tmp_path / "test"
    split.mkdir()
    np.save(split / "bad.npy", array)

    with pytest.raises(ValueError, match=error):
        index_feature_tree(tmp_path, 2048, "clip")


@pytest.mark.parametrize("missing_root", [False, True])
def test_index_feature_tree_rejects_missing_features(tmp_path: Path, missing_root: bool) -> None:
    root = tmp_path / "absent" if missing_root else tmp_path
    with pytest.raises(ValueError, match="no .npy feature files found"):
        index_feature_tree(root, 4, "clip")


def test_index_feature_tree_rejects_unreadable_file(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    (split / "broken.npy").write_bytes(b"")

    with pytest.raises(ValueError, match="broken.npy"):
        index_feature_tree(tmp_path, 4, "clip")


def test_index_feature_tree_rejects_invalid_metadata(tmp_path: Path) -> None:
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "a.npy", np.zeros((1, 4), dtype=np.float32))

    with pytest.raises(ValueError, match="expected_dim"):
        index_feature_tree(tmp_path, 0, "clip")
    with pytest.raises(ValueError, match="encoder"):
        index_feature_tree(tmp_path, 4, "")


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ({"records": []}, "records"),
        ({"records": "bad"}, "records"),
        ({"schema_version": 2}, "schema_version"),
        ({"expected_dim": 0}, "expected_dim"),
        ({"encoder": ""}, "encoder"),
        ({"records": [{"clip_id": "a"}]}, "record"),
        ({"records": [None]}, "record"),
    ],
)
def test_load_rejects_invalid_manifest(
    tmp_path: Path, change: dict[str, object], error: str
) -> None:
    record = dict(
        clip_id="a", split="train", path="train/a.npy", length=1, width=4, dtype="float32"
    )
    payload = dict(schema_version=1, encoder="clip", expected_dim=4, records=[record])
    payload.update(change)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))

    with pytest.raises(ValueError, match=error):
        FeatureManifest.load(path)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("length", 0, "length"),
        ("width", 5, "width"),
        ("dtype", "int32", "dtype"),
        ("dtype", "object", "dtype"),
        ("dtype", "float128", "dtype"),
        ("path", "../outside.npy", "path"),
        ("clip_id", "", "clip_id"),
    ],
)
def test_load_rejects_invalid_record_field(
    tmp_path: Path, field: str, value: object, error: str
) -> None:
    record = dict(
        clip_id="a", split="train", path="train/a.npy", length=1, width=4, dtype="float32"
    )
    record[field] = value
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=4, records=[record]))
    )

    with pytest.raises(ValueError, match=error):
        FeatureManifest.load(path)


def test_load_rejects_duplicate_clip_ids(tmp_path: Path) -> None:
    record = dict(
        clip_id="a", split="train", path="train/a.npy", length=1, width=4, dtype="float32"
    )
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=4, records=[record, record]))
    )

    with pytest.raises(ValueError, match="duplicate feature: train/a"):
        FeatureManifest.load(path)


def test_load_rejects_malformed_json(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text("{")

    with pytest.raises(ValueError):
        FeatureManifest.load(path)


def test_cli_writes_manifest(tmp_path: Path) -> None:
    root = tmp_path / "features"
    (root / "dev").mkdir(parents=True)
    np.save(root / "dev" / "clip.npy", np.zeros((2, 4), dtype=np.float32))
    output = tmp_path / "manifests" / "test.json"
    script = Path(__file__).resolve().parents[3] / "scripts" / "index_features.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--output",
            str(output),
            "--expected-dim",
            "4",
            "--encoder",
            "clip",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "indexed 1 features" in result.stdout
    assert FeatureManifest.load(output).require("dev", "clip").length == 2


@pytest.mark.parametrize(
    ("counts", "expected_error"),
    [
        (["train=2", "dev=1", "test=1"], "train: expected 2, got 1"),
        (["train=1", "dev=1", "test=1"], "test: expected 1, got 0"),
        (["train=1"], "dev: expected 0, got 1"),
    ],
)
def test_cli_rejects_incomplete_or_extra_splits_before_save(
    tmp_path: Path, counts: list[str], expected_error: str
) -> None:
    root = tmp_path / "features"
    for split in ("train", "dev"):
        (root / split).mkdir(parents=True)
        np.save(root / split / "clip.npy", np.zeros((2, 4), dtype=np.float32))
    output = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[3] / "scripts" / "index_features.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--output",
            str(output),
            "--expected-dim",
            "4",
            "--encoder",
            "clip",
            *(arg for count in counts for arg in ("--expected-count", count)),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert expected_error in result.stderr
    assert not output.exists()


def test_cli_accepts_complete_expected_counts(tmp_path: Path) -> None:
    root = tmp_path / "features"
    for split in ("train", "dev"):
        (root / split).mkdir(parents=True)
        np.save(root / split / "clip.npy", np.zeros((2, 4), dtype=np.float32))
    output = tmp_path / "manifest.json"
    script = Path(__file__).resolve().parents[3] / "scripts" / "index_features.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--root",
            str(root),
            "--output",
            str(output),
            "--expected-dim",
            "4",
            "--encoder",
            "clip",
            "--expected-count",
            "train=1",
            "--expected-count",
            "dev=1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "indexed 2 features" in result.stdout
    assert len(FeatureManifest.load(output).records) == 2

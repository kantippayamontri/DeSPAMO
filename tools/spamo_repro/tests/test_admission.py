import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from despamo.data.manifest import FeatureRecord
from tools.spamo_repro import data
from tools.spamo_repro.policy import validate_config


def approved_config():
    return json.loads((Path(__file__).parents[1] / "config.json").read_text())


def test_approved_recipe_is_admitted_without_loading_model():
    assert validate_config(approved_config()) == 17760


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("recipe", "epochs", 200),
        ("recipe", "accumulation", 1),
        ("recipe", "batch_size", 16),
        ("recipe", "vt_warmup_steps", 0),
        ("recipe", "min_lr", 0.0),
        ("recipe", "peak_lr", 0.0006),
        ("recipe", "dev_every", 1),
        ("data", "batch_size", 16),
        ("data", "num_workers", 4),
        ("model", "spatial_dim", 1024),
        ("model", "revision", "main"),
        ("model", "vt_pooling", "masked_mean"),
        ("model", "warm_up_steps", 0),
        ("model", "use_in_context", False),
        ("model", "max_text_length", 1024),
        ("optimizer", "weight_decay", 0.0),
        ("optimizer", "learning_rate", 0.0003),
        ("comparison", "enabled", False),
    ],
)
def test_recipe_drift_rejected(section, key, value):
    config = approved_config()
    config[section][key] = value
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


@pytest.mark.parametrize("key", ["checkpoint", "initial_checkpoint", "resume", "test_selection"])
def test_old_initialization_or_test_selection_rejected(key):
    config = approved_config()
    config[key] = "/old/weights.pt"
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


def test_boolean_is_not_seed_zero():
    config = approved_config()
    config["seed"] = False
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


def test_independent_storage_roots_required():
    config = approved_config()
    config["run_root"] = config["original_paused_run"]
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


def test_admission_does_not_change_config():
    config = approved_config()
    before = copy.deepcopy(config)
    validate_config(config)
    assert config == before


def test_test_ids_are_bound_without_reading_test_features(tmp_path):
    annotation = tmp_path / "test_info_ml.npy"
    np.save(annotation, np.array({i: {"fileid": f"test-{i}"} for i in range(642)}, dtype=object))
    names, binding = data.annotation_ids(annotation, 642)
    assert names == tuple(f"test-{i}" for i in range(642))
    assert binding == hashlib.sha256(annotation.read_bytes()).hexdigest()


def test_upstream_prefix_metadata_is_bound_but_not_counted_as_clip(tmp_path):
    annotation = tmp_path / "test_info_ml.npy"
    np.save(
        annotation,
        np.array(
            {
                "prefix": "./dataset/Phoenix14T/features/fullFrame-210x260px",
                0: {"fileid": "one"},
                1: {"fileid": "two"},
            },
            dtype=object,
        ),
    )
    names, binding = data.annotation_ids(annotation, 2)
    assert names == ("one", "two")
    assert binding == hashlib.sha256(annotation.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "payload",
    [
        {0: {"fileid": "same"}, 1: {"fileid": "same"}},
        {0: {"fileid": "../escape"}},
        {1: {"fileid": "valid"}},
        {0: {"fileid": 1}},
        {0: {"fileid": "valid"}, "hidden": {"fileid": "other"}},
    ],
)
def test_bad_annotation_ids_rejected(tmp_path, payload):
    annotation = tmp_path / "test_info_ml.npy"
    np.save(annotation, payload)
    with pytest.raises(ValueError, match="split IDs"):
        data.annotation_ids(annotation, 2)


def test_all_three_splits_must_be_disjoint():
    with pytest.raises(ValueError, match="overlap"):
        data.require_disjoint({"train": ("a",), "dev": ("b",), "test": ("a",)})


def test_feature_bytes_bound_and_finite(tmp_path):
    (tmp_path / "train").mkdir()
    path = tmp_path / "train/clip.npy"
    np.save(path, np.zeros((3, 2048), dtype=np.float32))
    record = FeatureRecord("clip", "train", "train/clip.npy", 3, 2048, "float32")
    assert (
        data.feature_hash(tmp_path, record, 2048) == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    np.save(path, np.full((3, 2048), np.nan, dtype=np.float32))
    with pytest.raises(ValueError, match="feature"):
        data.feature_hash(tmp_path, record, 2048)


@pytest.mark.parametrize("shape,dtype", [((3, 1024), np.float32), ((3, 2048), np.float64)])
def test_shape_and_dtype_drift_rejected(tmp_path, shape, dtype):
    (tmp_path / "train").mkdir()
    np.save(tmp_path / "train/clip.npy", np.zeros(shape, dtype=dtype))
    record = FeatureRecord("clip", "train", "train/clip.npy", 3, 2048, "float32")
    with pytest.raises(ValueError, match="feature"):
        data.feature_hash(tmp_path, record, 2048)


def test_feature_symlink_escape_rejected(tmp_path):
    root = tmp_path / "features"
    (root / "train").mkdir(parents=True)
    outside = tmp_path / "outside.npy"
    np.save(outside, np.zeros((3, 2048), dtype=np.float32))
    (root / "train/clip.npy").symlink_to(outside)
    record = FeatureRecord("clip", "train", "train/clip.npy", 3, 2048, "float32")
    with pytest.raises(ValueError, match="feature"):
        data.feature_hash(root, record, 2048)

import random
from pathlib import Path

import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T


def _record(clip_id: str = "clip-1") -> dict[str, str]:
    return {
        "fileid": clip_id,
        "signer": "Signer01",
        "gloss": "WIND",
        "text": "  Es bleibt windig  ",
        "en_text": "It remains windy.",
        "es_text": "Permanece ventoso.",
        "fr_text": "Il reste venteux.",
    }


def _dataset(tmp_path: Path) -> Phoenix14T:
    annotation = tmp_path / "train_info_ml.npy"
    np.save(annotation, {"prefix": "ignored", 0: _record()})
    spatial_root = tmp_path / "spatial"
    motion_root = tmp_path / "motion"
    (spatial_root / "train").mkdir(parents=True)
    (motion_root / "train").mkdir(parents=True)
    np.save(
        spatial_root / "train/clip-1.npy",
        np.ones((20, 2048), dtype=np.float32),
    )
    np.save(
        motion_root / "train/clip-1.npy",
        np.ones((4, 1024), dtype=np.float32),
    )
    spatial_manifest = FeatureManifest(
        1,
        "clip",
        2048,
        (FeatureRecord("clip-1", "train", "train/clip-1.npy", 20, 2048, "float32"),),
    )
    motion_manifest = FeatureManifest(
        1, "mae", 1024, (FeatureRecord("clip-1", "train", "train/clip-1.npy", 4, 1024, "float32"),)
    )
    return Phoenix14T(
        annotation, "train", spatial_root, motion_root, spatial_manifest, motion_manifest
    )


def _long_dataset(tmp_path: Path) -> Phoenix14T:
    dataset = _dataset(tmp_path)
    np.save(
        dataset.spatial_root / "train/clip-1.npy",
        np.arange(520, dtype=np.float32)[:, None].repeat(2048, axis=1),
    )
    np.save(dataset.motion_root / "train/clip-1.npy", np.ones((530, 1024), dtype=np.float32))
    dataset.spatial_manifest = FeatureManifest(
        1,
        "clip",
        2048,
        (FeatureRecord("clip-1", "train", "train/clip-1.npy", 520, 2048, "float32"),),
    )
    dataset.motion_manifest = FeatureManifest(
        1,
        "mae",
        1024,
        (FeatureRecord("clip-1", "train", "train/clip-1.npy", 530, 1024, "float32"),),
    )
    dataset.max_frame_len = 512
    return dataset


def test_center_crop_is_contiguous_repeatable_and_preserves_motion(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "center"

    first, second = dataset[0], dataset[0]

    assert first.spatial[:, 0].tolist() == list(range(4, 516))
    assert torch.equal(first.spatial, second.spatial)
    assert first.motion.shape == (530, 1024)


def test_upstream_random_crop_uses_inclusive_contiguous_start(tmp_path: Path, monkeypatch) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "upstream_random"
    calls = []

    def choose_start(low, high):
        calls.append((low, high))
        return high

    monkeypatch.setattr(random, "randint", choose_start)

    sample = dataset[0]

    assert calls == [(0, 8)]
    assert sample.spatial[:, 0].tolist() == list(range(8, 520))
    assert sample.motion.shape == (530, 1024)


def test_random_crop_repeats_after_run_seed(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "upstream_random"

    pl.seed_everything(17, workers=True)
    first = dataset[0].spatial
    pl.seed_everything(17, workers=True)
    second = dataset[0].spatial

    assert first.shape == (512, 2048)
    assert torch.equal(first, second)


def test_full_mode_retains_all_spatial_and_motion_frames(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "full"

    sample = dataset[0]

    assert sample.spatial[:, 0].tolist() == list(range(520))
    assert sample.motion.shape == (530, 1024)


def test_full_mode_allows_unbounded_spatial_features(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.max_frame_len = None

    assert dataset[0].spatial[:, 0].tolist() == list(range(520))


@pytest.mark.parametrize("mode", ["center", "upstream_random", "full"])
def test_short_spatial_sequences_are_untouched(tmp_path: Path, monkeypatch, mode: str) -> None:
    dataset = _dataset(tmp_path)
    dataset.max_frame_len = 512
    dataset.spatial_crop_mode = mode
    monkeypatch.setattr(random, "randint", lambda *_: pytest.fail("short sequence drew random"))

    assert dataset[0].spatial.shape == (20, 2048)


def test_cropped_spatial_feature_still_checks_full_manifest_shape(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "center"
    np.save(dataset.spatial_root / "train/clip-1.npy", np.ones((521, 2048), dtype=np.float32))

    with pytest.raises(ValueError, match="spatial shape mismatch.*clip-1"):
        dataset[0]


def test_cropped_spatial_feature_still_checks_full_manifest_dtype(tmp_path: Path) -> None:
    dataset = _long_dataset(tmp_path)
    dataset.spatial_crop_mode = "center"
    np.save(dataset.spatial_root / "train/clip-1.npy", np.ones((520, 2048), dtype=np.float64))

    with pytest.raises(ValueError, match="spatial dtype mismatch.*clip-1"):
        dataset[0]


def test_invalid_crop_mode_is_rejected_at_construction(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="spatial_crop_mode"):
        Phoenix14T(
            tmp_path / "train_info_ml.npy",
            "train",
            dataset.spatial_root,
            dataset.motion_root,
            dataset.spatial_manifest,
            dataset.motion_manifest,
            max_frame_len=512,
            spatial_crop_mode="random",
        )


@pytest.mark.parametrize("mode", ["center", "upstream_random"])
@pytest.mark.parametrize("max_frame_len", [None, 0])
def test_cropped_mode_requires_positive_limit(
    tmp_path: Path, mode: str, max_frame_len: int | None
) -> None:
    dataset = _dataset(tmp_path)

    with pytest.raises(ValueError, match="max_frame_len"):
        Phoenix14T(
            tmp_path / "train_info_ml.npy",
            "train",
            dataset.spatial_root,
            dataset.motion_root,
            dataset.spatial_manifest,
            dataset.motion_manifest,
            max_frame_len=max_frame_len,
            spatial_crop_mode=mode,
        )


def test_dataset_loads_integer_annotation_records(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)

    sample = dataset[0]

    assert len(dataset) == 1
    assert sample.clip_id == "clip-1"
    assert sample.signer == "Signer01"
    assert sample.gloss == "WIND"
    assert sample.text == "es bleibt windig."
    assert (sample.en_text, sample.es_text, sample.fr_text) == (
        "It remains windy.",
        "Permanece ventoso.",
        "Il reste venteux.",
    )
    assert sample.spatial.shape == (20, 2048)
    assert sample.motion.shape == (4, 1024)
    assert sample.spatial.dtype == sample.motion.dtype == torch.float32


def test_dataset_casts_supported_manifest_floats(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    np.save(dataset.spatial_root / "train/clip-1.npy", np.full((20, 2048), 2, dtype=np.float16))
    np.save(dataset.motion_root / "train/clip-1.npy", np.full((4, 1024), 3, dtype=np.float64))
    dataset.spatial_manifest = FeatureManifest(
        1,
        "clip",
        2048,
        (FeatureRecord("clip-1", "train", "train/clip-1.npy", 20, 2048, "float16"),),
    )
    dataset.motion_manifest = FeatureManifest(
        1, "mae", 1024, (FeatureRecord("clip-1", "train", "train/clip-1.npy", 4, 1024, "float64"),)
    )

    sample = dataset[0]

    assert sample.spatial.dtype == sample.motion.dtype == torch.float32
    assert sample.spatial[0, 0].item() == 2
    assert sample.motion[0, 0].item() == 3


@pytest.mark.parametrize("modality", ["spatial", "motion"])
@pytest.mark.parametrize("bad", ["length", "width", "rank", "dtype"])
def test_dataset_rejects_feature_manifest_mismatch(tmp_path: Path, modality: str, bad: str) -> None:
    dataset = _dataset(tmp_path)
    shape = (20, 2048) if modality == "spatial" else (4, 1024)
    if bad == "length":
        array = np.ones((shape[0] + 1, shape[1]), dtype=np.float32)
    elif bad == "width":
        array = np.ones((shape[0], shape[1] - 1), dtype=np.float32)
    elif bad == "rank":
        array = np.ones((shape[0] * shape[1],), dtype=np.float32)
    else:
        array = np.ones(shape, dtype=np.float64)
    root = dataset.spatial_root if modality == "spatial" else dataset.motion_root
    np.save(root / "train/clip-1.npy", array)

    with pytest.raises(ValueError, match=rf"{modality}.*(shape|dtype).*clip-1"):
        dataset[0]


def test_dataset_rejects_missing_feature_for_clip(tmp_path: Path) -> None:
    configured = _dataset(tmp_path)
    annotation = tmp_path / "train_info_ml.npy"
    np.save(annotation, {0: _record("missing")})
    dataset = Phoenix14T(
        annotation,
        "train",
        configured.spatial_root,
        configured.motion_root,
        configured.spatial_manifest,
        configured.motion_manifest,
    )

    with pytest.raises(KeyError, match="expected one feature for train/missing, found 0"):
        dataset[0]


def test_dataset_rejects_missing_file_with_manifest_record(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (dataset.spatial_root / "train/clip-1.npy").unlink()

    with pytest.raises(ValueError, match="invalid spatial feature for clip-1"):
        dataset[0]


def test_dataset_rejects_empty_time_after_manifest_creation(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    np.save(dataset.motion_root / "train/clip-1.npy", np.empty((0, 1024), dtype=np.float32))

    with pytest.raises(ValueError, match="motion shape mismatch for clip-1") as error:
        dataset[0]
    assert "expected (4, 1024), got (0, 1024)" in str(error.value)


def test_dataset_reports_corrupt_feature_with_clip_id(tmp_path: Path) -> None:
    dataset = _dataset(tmp_path)
    (dataset.motion_root / "train/clip-1.npy").write_bytes(b"")

    with pytest.raises(ValueError, match="motion.*clip-1"):
        dataset[0]


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        ([], "annotation"),
        ({"prefix": "ignored"}, "annotation"),
        ({0: "bad"}, "record"),
        ({0: {"fileid": "a"}}, "signer"),
        ({0: _record("../outside")}, "fileid"),
        ({0: {**_record(), "text": None}}, "text"),
        ({0: {**_record(), "text": " "}}, "text"),
        ({0: _record(), 1: _record()}, "duplicate.*clip-1"),
    ],
)
def test_dataset_rejects_malformed_annotation(tmp_path: Path, payload: object, error: str) -> None:
    dataset = _dataset(tmp_path)
    annotation = tmp_path / "train_info_ml.npy"
    np.save(annotation, payload)

    with pytest.raises(ValueError, match=error):
        Phoenix14T(
            annotation,
            "train",
            dataset.spatial_root,
            dataset.motion_root,
            dataset.spatial_manifest,
            dataset.motion_manifest,
        )

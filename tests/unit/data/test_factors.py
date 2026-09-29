from pathlib import Path

import numpy as np
import pytest
import torch

from despamo.appearance.canonical import targets
from despamo.appearance.provenance import atomic_json, digest, file_hash, load_dataset, read_json
from despamo.appearance.schema import ARTICULATORS, FACTORS, STABLE, Record
from despamo.data.batch import PhoenixSample
from despamo.data.factors import FactorDataset, collate_factors, load_factor_targets, map_rows
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.appearance.text_features import encode_dataset
from tests.unit.appearance.helpers import clip, payload, prepared_dataset


def test_source_ordinals_are_not_feature_row_numbers():
    assert map_rows([3, 7], [1, 3, 5, 7], 4) == [1, 3]
    for requested, source, length in (
        ([4], [1, 3, 5], 3),
        ([3], [1, 3, 3], 3),
        ([3], [1, 3], 3),
    ):
        with pytest.raises(ValueError):
            map_rows(requested, source, length)


def fixture_targets(tmp_path: Path, *, failed: bool):
    tmp_path.mkdir(exist_ok=True)
    image = clip()
    record = payload()
    journal = {
        "clip_id": image["clip_id"],
        "status": "failed" if failed else "valid",
        "record": None if failed else record,
    }
    items = (
        [{"factor": factor, "frame_index": -1, "text": None} for factor in STABLE]
        + [
            {"factor": factor, "frame_index": frame["index"], "text": None}
            for frame in image["frames"]
            for factor in ARTICULATORS
        ]
        if failed
        else targets(Record.model_validate(record))
    )
    valid = np.array([item["text"] is not None for item in items], dtype=np.bool_)
    vectors = np.zeros((19, 4), dtype=np.float32)
    vectors[valid] = 1
    name = digest(image["clip_id"]) + ".npz"
    path = tmp_path / name
    np.savez_compressed(path, vectors=vectors, valid=valid)
    entry = {"clip_id": image["clip_id"], "record_hash": digest(journal),
             "targets": items, "path": name, "file_hash": file_hash(path)}
    return image, journal, entry


@pytest.mark.parametrize("failed", [False, True])
def test_targets_keep_nineteen_rows_and_failed_masks(tmp_path, failed):
    image, journal, entry = fixture_targets(tmp_path, failed=failed)
    vectors, mask, labels = load_factor_targets(image, journal, entry, tmp_path, width=4)
    assert vectors.shape == (19, 4) and mask.shape == (19,)
    assert bool(mask.any()) is not failed
    if failed:
        assert not vectors.any() and labels == tuple("" for _ in range(19))


def test_target_hash_or_mask_drift_fails(tmp_path):
    image, journal, entry = fixture_targets(tmp_path, failed=True)
    with pytest.raises(ValueError, match="record hash"):
        load_factor_targets(image, journal, {**entry, "record_hash": "old"}, tmp_path, 4)
    path = tmp_path / entry["path"]
    with np.load(path, allow_pickle=False) as source:
        data = source["vectors"].copy()
        valid = source["valid"].copy()
    valid[0] = True
    np.savez_compressed(path, vectors=data, valid=valid)
    with pytest.raises(ValueError, match="mask"):
        load_factor_targets(image, journal, {**entry, "file_hash": file_hash(path)}, tmp_path, 4)


def test_mixed_batch_keeps_failed_translation_clip_but_no_factor_gradients(tmp_path):
    samples = []
    for index, failed in enumerate((False, False, True)):
        image, journal, entry = fixture_targets(tmp_path / str(index), failed=failed)
        record = PhoenixSample(
            f"clip-{index}", "s", "translation", "g", "en", "es", "fr",
            torch.randn(11, 4), torch.randn(3, 2),
        )
        vectors, mask, labels = load_factor_targets(image, journal, entry, tmp_path / str(index), 4)
        samples.append((record, torch.tensor([1, 3, 5, 7, 9]), vectors, mask, labels))
    batch = collate_factors(samples)
    assert len(batch.base.clip_ids) == 3
    assert batch.rows.shape == (3, 5)
    for factor in FACTORS:
        assert not batch.masks[factor][-1].any()
        assert not batch.vectors[factor][-1].any()
        assert batch.vectors[factor].shape == ((3, 4) if factor in STABLE else (3, 5, 4))


def fixture_dataset(tmp_path):
    dataset = prepared_dataset(tmp_path, count=51)
    source, _ = load_dataset(dataset)

    class FakeEncoder:
        width = 4
        revision = "32bd64288804d66eefd0ccbe215aa642df71cc41"

        def encode(self, texts):
            return np.ones((len(texts), 4), dtype=np.float32)

    text_manifest = encode_dataset(dataset, FakeEncoder())
    spatial_root_path = tmp_path / "spatial"
    rows, records = {}, []
    for item in source["clips"]:
        clip_id = item["clip_id"]
        feature = spatial_root_path / "train" / (clip_id + ".npy")
        feature.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature, np.ones((11, 4), dtype=np.float32))
        records.append(FeatureRecord(clip_id, "train", f"train/{clip_id}.npy", 11, 4, "float32"))
        rows[clip_id] = {
            "feature_hash": file_hash(feature),
            "source_indices": list(range(11)),
            "sampled_images": {str(f["index"]): f["sha256"] for f in item["frames"]},
        }
    spatial_manifest_path = tmp_path / "manifest.json"
    FeatureManifest(1, "dinov3:fixture", 4, tuple(records)).save(spatial_manifest_path)
    frame_rows = tmp_path / "rows.json"
    atomic_json(frame_rows, {"spatial_manifest_hash": file_hash(spatial_manifest_path),
                             "clips": rows, "encoder_key": "fixture"})

    class Base:
        split = "train"
        records = [{"fileid": item["clip_id"]} for item in source["clips"]]
        spatial_manifest = FeatureManifest.load(spatial_manifest_path)
        spatial_root = spatial_root_path

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            item = self.records[index]
            return PhoenixSample(item["fileid"], "s", "translation", "g", "en", "es", "fr",
                                 torch.ones(11, 4), torch.ones(3, 2))

    return Base(), dataset, text_manifest, frame_rows, spatial_manifest_path


def test_unreviewed_training_join_keeps_failed_clip_in_translation(tmp_path):
    base, dataset, text, rows, manifest = fixture_dataset(tmp_path)
    data = FactorDataset(base, dataset, text, rows, manifest, "qwen-schema98-unreviewed-v1")
    batch = collate_factors([data[0], data[1], data[50]])
    assert len(data) == 51 and batch.base.clip_ids == ("clip-0000", "clip-0001", "clip-0050")
    assert batch.rows[0].tolist() == [1, 3, 5, 7, 9]
    assert all(not batch.masks[f][-1].any() for f in FACTORS)
    assert data.provenance["human_review_status"] == "not_assessed"
    assert data.provenance["supervision_policy"] == "qwen-schema98-unreviewed-v1"
    assert data.provenance["policy_version"] == "qwen-schema98-unreviewed-v1"
    assert "human_requirements" not in data.provenance
    assert data.provenance["schema_eligibility"]["valid_count"] == 50
    assert data.provenance["schema_eligibility"]["failed_count"] == 1


def test_join_rejects_old_audit_requirement_or_stale_text_hash(tmp_path):
    base, dataset, text, rows, manifest = fixture_dataset(tmp_path)
    with pytest.raises(ValueError, match="policy"):
        FactorDataset(base, dataset, text, rows, manifest, "qwen-schema98-human-v2")
    source = read_json(text)
    source["records"][50]["record_hash"] = "changed"
    atomic_json(text, source)
    with pytest.raises(ValueError, match="record hash"):
        FactorDataset(base, dataset, text, rows, manifest, "qwen-schema98-unreviewed-v1")


def test_unselected_text_file_hash_drift_fails_before_training(tmp_path):
    base, dataset, text, rows, manifest = fixture_dataset(tmp_path)
    item = read_json(text)["records"][24]
    (text.parent / item["path"]).write_bytes(b"corrupted unselected clip")
    with pytest.raises(ValueError, match="text feature file hash"):
        FactorDataset(base, dataset, text, rows, manifest, "qwen-schema98-unreviewed-v1")

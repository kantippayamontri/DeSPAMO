from types import SimpleNamespace

import numpy as np
import pytest
import torch

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.evaluation.pilot_probe import (
    pool_spatial,
    project_pooled,
    score_probe,
    split_probe_clips,
)


def test_split_is_stratified_disjoint_and_repeatable():
    clips = [(f"s{signer}-{i}", f"Signer{signer:02d}") for signer in range(1, 8) for i in range(20)]
    result = split_probe_clips(clips)
    assert result == split_probe_clips(list(reversed(clips)))
    assert {part: len(ids) for part, ids in result.items()} == {
        "fit": 98, "validation": 21, "test": 21
    }
    assert len(set().union(*map(set, result.values()))) == 140
    assert all(sum(clip.startswith(f"s{signer}-") for clip in result[part]) == count
               for signer in range(1, 8)
               for part, count in (("fit", 14), ("validation", 3), ("test", 3)))


def test_split_rejects_duplicate_ids_and_missing_classes():
    with pytest.raises(ValueError):
        split_probe_clips([("same", "Signer01"), ("same", "Signer02")])
    with pytest.raises(ValueError):
        split_probe_clips([(f"id{i}", "Signer01") for i in range(20)])


def test_pool_spatial_uses_physical_train_rows_in_requested_order(tmp_path):
    records = tuple(FeatureRecord(name, "train", f"train/{name}.npy", 2, 2048, "float32")
                    for name in ("b", "a"))
    (tmp_path / "train").mkdir()
    for record, start in zip(records, (2., 6.), strict=True):
        np.save(tmp_path / record.path,
                np.stack([np.full(2048, start, dtype=np.float32),
                          np.full(2048, start + 2, dtype=np.float32)]))
    view = SimpleNamespace(records=[{"fileid": "b"}, {"fileid": "a"}],
                           spatial_root=tmp_path,
                           spatial_manifest=FeatureManifest(1, "test", 2048, records))
    pooled = pool_spatial(view, ("a", "b"))
    assert pooled.shape == (2, 2048)
    torch.testing.assert_close(pooled[:, 0], torch.tensor([7., 3.]))
    with pytest.raises(ValueError):
        pool_spatial(view, ("b", "b"))


def test_pool_spatial_rejects_nonfinite_frames(tmp_path):
    (tmp_path / "train").mkdir()
    record = FeatureRecord("a", "train", "train/a.npy", 1, 2048, "float32")
    values = np.ones((1, 2048), dtype=np.float32)
    values[0, 0] = np.nan
    np.save(tmp_path / record.path, values)
    view = SimpleNamespace(records=[{"fileid": "a"}], spatial_root=tmp_path,
                           spatial_manifest=FeatureManifest(1, "test", 2048, (record,)))
    with pytest.raises(ValueError, match="non-finite"):
        pool_spatial(view, ("a",))


def test_linear_projection_after_pool_matches_frame_projection():
    frames = torch.tensor([[1., 3.], [5., 7.]])
    weight = torch.tensor([[2., -1.], [1., 2.]])
    bias = torch.tensor([3., -2.])
    result = project_pooled(frames.mean(dim=0, keepdim=True), weight, bias)
    torch.testing.assert_close(result, (frames @ weight.T + bias).mean(dim=0, keepdim=True))
    with pytest.raises(ValueError):
        project_pooled(frames, weight[:, :1], bias)


def test_probe_scores_same_partition_deterministically_and_handles_constant_column():
    clips = [(f"s{signer}-{i}", f"Signer{signer:02d}") for signer in range(1, 8) for i in range(20)]
    ids = tuple(clip_id for clip_id, _ in clips)
    labels = [signer for _, signer in clips]
    features = torch.tensor([[float(int(signer[-2:])), 1.] for signer in labels])
    split = split_probe_clips(clips)
    first = score_probe(features, labels, split, ids)
    assert first == score_probe(features, labels, split, ids)
    assert first["counts"] == {f"Signer{i:02d}": 3 for i in range(1, 8)}
    assert 0 <= first["accuracy"] <= 1
    assert 0 <= first["balanced_accuracy"] <= 1
    assert first["majority_baseline"] == 1 / 7
    assert "predictions" not in first


def test_probe_rejects_missing_or_repeated_clip_labels():
    with pytest.raises(ValueError):
        score_probe(torch.ones(2, 3), ["Signer01", "Signer02"],
                    {"fit": ("a",), "validation": ("b",), "test": ("b",)},
                    ("a", "b"))

import importlib
from copy import deepcopy

import numpy as np
import pytest
from PIL import Image

from tools.dinov3.identity import sha256_file


def module():
    return importlib.import_module("tools.dinov3.official_data")


def annotations():
    return {split: {"metadata": "not a clip", **{
        i: dict(fileid=f"{split}-{i}", signer=f"Signer{i % 9 + 1:02}")
        for i in range(n)}} for split, n in (("train", 7096), ("dev", 519), ("test", 642))}


def fixture(tmp_path):
    frames, teacher, targets = (tmp_path / name for name in ("frames", "teacher", "targets"))
    clips, records, doc, rows, train = [], [], [], {}, {}
    indices = [5, 16, 26, 36, 47]
    for name, signer in (("a", "Signer03"), ("b", "Signer07")):
        train[name] = dict(fileid=name, signer=signer)
        paths = [f"train/{name}/images{i+1:04}.png" for i in range(53)]
        sampled = []
        for i in indices:
            path = frames / paths[i]
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (210, 260), (i, 20, 30)).save(path)
            sampled.append(dict(index=i, path=paths[i], sha256=sha256_file(path)))
        feature = teacher / "train" / f"{name}.npy"
        feature.parent.mkdir(parents=True, exist_ok=True)
        values = np.repeat(np.arange(53, dtype=np.float32)[:, None], 2048, axis=1)
        np.save(feature, values)
        labels = [dict(factor=f, frame_index=-1, text=f"stable {f}")
                  for f in ("biometric", "clothing", "hair", "background")]
        labels += [dict(factor=f, frame_index=i, text=f"{f} {i}")
                   for i in indices for f in ("left_hand", "right_hand", "mouth")]
        labels[4]["text"] = None
        if name == "b":
            for t in labels[:4]:
                t["text"] = None
        valid = np.array([t["text"] is not None for t in labels])
        vectors = np.ones((19, 768), np.float32)
        vectors[~valid] = 0
        target = targets / f"{name}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(target, vectors=vectors, valid=valid)
        records.append(dict(clip_id=name, path=target.name, file_hash=sha256_file(target),
                            targets=labels))
        clips.append(dict(clip_id=name, signer=signer, frame_count=53, frames=sampled))
        doc.append(dict(clip_id=name, signer=signer, frame_count=53, frame_indices=indices,
                        frame_paths=[paths[i] for i in indices],
                        source_frame_hashes=[f["sha256"] for f in sampled], targets=labels))
        rows[name] = dict(frame_count=53, paths=paths, source_indices=list(range(53)),
                          sampled_images={str(f["index"]): f["sha256"] for f in sampled},
                          feature_hash=sha256_file(feature))
    descriptors = module()._join_records(
        dict(clips=clips), dict(records=doc), dict(clips=rows), dict(records=records),
        train, teacher, targets)
    admission = module()._seal_admission(descriptors, frames, {})
    return admission


def test_official_inventory_ignores_metadata_and_includes_all_signers():
    split = module()._official_inventory(annotations())
    assert [len(split[s]) for s in ("train", "dev", "test")] == [7096, 519, 642]
    assert split["train"]["train-2"]["signer"] == "Signer03"
    assert split["train"]["train-6"]["signer"] == "Signer07"


@pytest.mark.parametrize("change", ["overlap", "missing", "duplicate", "signer", "path"])
def test_invalid_official_inventory_rejected(change):
    raw = annotations()
    if change == "overlap":
        raw["dev"][0]["fileid"] = "train-0"
    elif change == "missing":
        raw["train"].pop(0)
    elif change == "duplicate":
        raw["train"][1]["fileid"] = "train-0"
    elif change == "signer":
        raw["train"][0]["signer"] = "Signer99"
    else:
        raw["train"][0]["fileid"] = "../escape"
    with pytest.raises(ValueError):
        module()._official_inventory(raw)


def test_batch_admits_train_signer03_and07_with_independent_masks(tmp_path):
    admission = fixture(tmp_path)
    batch = module().load_official_batch(("a", "b"), admission)
    assert batch["images224"].shape == (2, 5, 3, 224, 224)
    assert batch["images448"].shape == (2, 5, 3, 448, 448)
    assert batch["teacher"].shape == (2, 5, 2048)
    assert batch["vectors"].shape == (2, 19, 768)
    assert batch["teacher"][0, :, 0].tolist() == [5, 16, 26, 36, 47]
    assert not batch["masks"][0, 4] and batch["masks"][0, :4].all()
    assert not batch["masks"][1, :4].any() and batch["masks"][1, 5:].all()
    assert batch["labels"][0][4] == ""
    assert batch["images224"].device.type == "cpu"
    with pytest.raises(ValueError):
        module().load_official_batch(("dev-0",), admission)
    with pytest.raises(ValueError):
        module().load_official_batch(("a", "a"), admission)


@pytest.mark.parametrize("change", ["png", "target", "teacher", "descriptor", "escape"])
def test_changed_batch_binding_rejected(tmp_path, change):
    admission = fixture(tmp_path)
    row = admission["clips"]["a"]
    if change == "png":
        (tmp_path / "frames" / row["frame_paths"][0]).write_bytes(b"changed")
    elif change == "target":
        row["target_path"].write_bytes(b"changed")
    elif change == "teacher":
        row["teacher_path"].write_bytes(b"changed")
    elif change == "descriptor":
        row["frame_indices"][0] = 6
    else:
        row["frame_paths"][0] = "../outside.png"
    with pytest.raises(ValueError):
        module().load_official_batch(("a",), admission)


@pytest.mark.parametrize("change", ["teacher_width", "target_dtype", "masked_nonzero"])
def test_hash_valid_malformed_arrays_rejected(tmp_path, change):
    admission = fixture(tmp_path)
    row = admission["clips"]["a"]
    if change == "teacher_width":
        np.save(row["teacher_path"], np.ones((53, 1024), np.float32))
        row["teacher_sha256"] = sha256_file(row["teacher_path"])
    else:
        valid = np.array([t["text"] is not None for t in row["targets"]])
        vectors = np.ones((19, 768), np.float64 if change == "target_dtype" else np.float32)
        vectors[~valid] = 0
        if change == "masked_nonzero":
            vectors[~valid] = 1
        np.savez_compressed(row["target_path"], vectors=vectors, valid=valid)
        row["target_sha256"] = sha256_file(row["target_path"])
    resealed = module()._seal_admission(deepcopy(admission["clips"]),
                                        tmp_path / "frames", {})
    with pytest.raises(ValueError):
        module().load_official_batch(("a",), resealed)


def test_changed_metadata_signature_rejected(tmp_path):
    admission = fixture(tmp_path)
    meta = tmp_path / "manifest.json"
    meta.write_text("original")
    admission = module()._seal_admission(admission["clips"], tmp_path / "frames",
                                         {meta: sha256_file(meta)})
    meta.write_text("changed")
    with pytest.raises(ValueError):
        module().load_official_batch(("a",), admission)


def test_metadata_change_while_sealing_rejected(tmp_path, monkeypatch):
    admission = fixture(tmp_path)
    meta = tmp_path / "manifest.json"
    meta.write_text("original")
    sha = sha256_file(meta)
    real_hash = module().sha256_file

    def changed_after_hash(path):
        value = real_hash(path)
        if path == meta:
            path.write_text("changed inside seal")
        return value

    monkeypatch.setattr(module(), "sha256_file", changed_after_hash)
    with pytest.raises(ValueError):
        module()._seal_admission(admission["clips"], tmp_path / "frames", {meta: sha})

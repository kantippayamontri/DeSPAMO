import hashlib
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.dinov3.e3_data import load_e3_batch, validate_e3_sources
from tools.dinov3.identity import digest


def _fixture(tmp_path: Path):
    ids = ("a", "b", "c")
    frames = tmp_path / "frames"
    dino = tmp_path / "dino"
    text_manifest = tmp_path / "text" / "manifest.json"
    text_manifest.parent.mkdir()
    source_clips, text_records, rows = [], [], {}
    for i, clip_id in enumerate(ids):
        paths = []
        for frame in range(5):
            path = frames / "train" / clip_id / f"images{frame:04d}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (20, 20), color=(frame * 30, i * 40, 0)).save(path)
            paths.append({"path": path.relative_to(frames).as_posix(), "index": frame,
                          "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        feature = dino / "train" / f"{clip_id}.npy"
        feature.parent.mkdir(parents=True, exist_ok=True)
        np.save(feature, np.full((5, 2048), i + 1, dtype=np.float32))
        rows[clip_id] = {"source_indices": list(range(5)),
                         "feature_hash": hashlib.sha256(feature.read_bytes()).hexdigest()}
        source_clips.append({"clip_id": clip_id, "signer": "Signer01", "frame_count": 5,
                             "frames": paths})
        targets = ["visible"] * 19 if i == 1 else (["visible"] * 18 + [None] if i == 0
                                                  else [None] * 19)
        valid = np.array([value is not None for value in targets], dtype=np.bool_)
        target_path = text_manifest.parent / f"{clip_id}.npz"
        np.savez(target_path, vectors=np.where(valid[:, None],
                 np.ones((19, 768), dtype=np.float32), 0), valid=valid)
        text_records.append({"clip_id": clip_id, "path": target_path.name,
                             "file_hash": hashlib.sha256(target_path.read_bytes()).hexdigest(),
                             "targets": [{"text": value} for value in targets]})
    source = {"split": "train", "clips": source_clips}
    text = {"records": text_records}
    protocol = {"source_hash": digest(source), "split": {"groups": {
        "train": {"clip_ids": list(ids), "valid": 2, "failed": 1},
        "dev": {"clip_ids": ["dev"]}, "test": {"clip_ids": ["test"]},
    }}}
    protocol["protocol_hash"] = digest(protocol)
    return protocol, source, text, rows, frames, dino, text_manifest


def test_e3_data_accepts_partial_valid_and_full_failed_mask(tmp_path):
    protocol, source, text, rows, frames, dino, manifest = _fixture(tmp_path)
    assert validate_e3_sources(protocol, source, text) == ("a", "b", "c")
    batch = load_e3_batch(("a", "b", "c"), source, text, rows, frames=frames,
                          dino_root=dino, text_manifest=manifest)
    assert batch["images224"].shape == (3, 5, 3, 224, 224)
    assert batch["images448"].shape == (3, 5, 3, 448, 448)
    assert batch["teacher"].shape == (3, 5, 2048)
    assert batch["vectors"].shape == (3, 19, 768)
    assert batch["masks"].sum(dim=1).tolist() == [18, 19, 0]


def test_e3_data_rejects_mask_or_dino_hash_drift(tmp_path):
    protocol, source, text, rows, frames, dino, manifest = _fixture(tmp_path)
    path = manifest.parent / "a.npz"
    np.savez(path, vectors=np.ones((19, 768), dtype=np.float32),
             valid=np.ones(19, dtype=np.bool_))
    text["records"][0]["file_hash"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="mask|target"):
        load_e3_batch(("a",), source, text, rows, frames=frames,
                      dino_root=dino, text_manifest=manifest)
    path = dino / "train/b.npy"
    np.save(path, np.zeros((5, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match="DINO|feature"):
        load_e3_batch(("b",), source, text, rows, frames=frames,
                      dino_root=dino, text_manifest=manifest)


def test_e3_data_rejects_heldout_clip_ids(tmp_path):
    protocol, source, text, rows, frames, dino, manifest = _fixture(tmp_path)
    with pytest.raises(ValueError, match="train"):
        load_e3_batch(("dev",), source, text, rows, frames=frames,
                      dino_root=dino, text_manifest=manifest)
    protocol["split"]["groups"]["test"]["clip_ids"] = ["a"]
    protocol["protocol_hash"] = digest({k: v for k, v in protocol.items() if k != "protocol_hash"})
    with pytest.raises(ValueError, match="overlap"):
        validate_e3_sources(protocol, source, text)

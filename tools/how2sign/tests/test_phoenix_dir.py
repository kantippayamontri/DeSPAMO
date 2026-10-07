import cv2
import numpy as np
import pytest
from how2sign_probe.phoenix import index_frames_dir, load_dir_clips


def _write_dir(tmp_path, clips):
    root = tmp_path / "fullFrame-210x260px"
    for split, names in clips.items():
        for clip, serials in names.items():
            directory = root / split / clip
            directory.mkdir(parents=True)
            frame = np.full((260, 210, 3), len(clip) % 200, np.uint8)
            for serial in serials:
                cv2.imwrite(str(directory / f"{serial}.png"), frame)
    return root


CLIPS = {
    "train": {
        "A-1": [f"images{i:04d}" for i in range(1, 10)],
        "B-2": [f"images{i:04d}" for i in range(1, 8)],
        "S3": [f"images{i:04d}" for i in range(1, 6)],
        "S7": [f"images{i:04d}" for i in range(1, 6)],
    },
    "dev": {"D-9": [f"images{i:04d}" for i in range(1, 7)]},
}


@pytest.fixture
def frames_root(tmp_path):
    return _write_dir(tmp_path, CLIPS)


def _annotations(tmp_path, frames_root, splits=("train",)):
    raw = {"prefix": str(frames_root) + "/"}
    position = 0
    for split in splits:
        for clip, serials in CLIPS[split].items():
            raw[position] = {
                "fileid": clip,
                "folder": f"{split}/{clip}/*.png",
                "signer": {
                    "A-1": "Signer01",
                    "B-2": "Signer02",
                    "S3": "Signer03",
                    "S7": "Signer07",
                    "D-9": "Signer04",
                }[clip],
                "num_frames": str(len(serials)),
                "text": "x",
            }
            position += 1
    path = tmp_path / "ann.npy"
    np.save(path, raw)
    return path


def test_load_dir_clips_checks_dirs_against_annotations(tmp_path, frames_root):
    ann = _annotations(tmp_path, frames_root)
    rows = load_dir_clips(ann, frames_root, split="train")
    assert [r["clip"] for r in rows] == ["A-1", "B-2"]  # S03/S07 excluded
    assert rows[0]["num_frames"] == 9
    assert rows[0]["num_frames"] == 9 == len(list((frames_root / "train" / "A-1").glob("*.png")))


def test_load_dir_clips_flags_count_mismatch(tmp_path, frames_root):
    ann = _annotations(tmp_path, frames_root)
    data = np.load(ann, allow_pickle=True).item()
    for value in data.values():
        if isinstance(value, dict) and value.get("fileid") == "A-1":
            value["num_frames"] = "12"
    np.save(ann, data)
    with pytest.raises(ValueError, match="A-1"):
        load_dir_clips(ann, frames_root, split="train")


def test_load_dir_clips_flags_missing_dir(tmp_path, frames_root):
    ann = _annotations(tmp_path, frames_root)
    (frames_root / "train" / "B-2").rename(frames_root / "train" / "renamed")
    with pytest.raises(ValueError, match="B-2"):
        load_dir_clips(ann, frames_root, split="train")


def test_load_dir_clips_accepts_other_split(tmp_path, frames_root):
    ann = _annotations(tmp_path, frames_root, splits=("dev",))
    rows = load_dir_clips(ann, frames_root, split="dev")
    assert [r["clip"] for r in rows] == ["D-9"]


def test_index_frames_dir_lists_sorted_serials(tmp_path, frames_root):
    ann = _annotations(tmp_path, frames_root)
    rows = load_dir_clips(ann, frames_root, split="train")
    frames = index_frames_dir(rows, frames_root, split="train")
    assert list(frames["A-1"][:3]) == ["images0001", "images0002", "images0003"]
    assert frames["A-1"].abs_path("images0001").name == "images0001.png"
    with pytest.raises(ValueError):
        frames["A-1"].abs_path("images9999")


def test_index_frames_dir_rejects_unsafe_clip(tmp_path, frames_root):
    rows = [{"clip": "../evil", "stratum": "Signer01", "num_frames": 5}]
    with pytest.raises(ValueError, match="invalid clip"):
        index_frames_dir(rows, frames_root, split="train")

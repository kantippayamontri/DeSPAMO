import tarfile

import cv2
import numpy as np
import pytest
from how2sign_probe.phoenix import (
    extract_frames,
    index_frames,
    load_clips,
    planned_indices,
    read_frame,
    verify_counts,
)

PREFIX = "PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px/train"


def _png(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (26, 21, 3), dtype=np.uint8)
    return cv2.imencode(".png", img)[1].tobytes()


def _tar(tmp_path, clips):
    path = tmp_path / "release.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for clip, serials in clips.items():
            for serial in serials:
                data = _png(int(serial[-4:]) % 7)
                info = tarfile.TarInfo(f"{PREFIX}/{clip}/{serial}.png")
                info.size = len(data)
                import io

                archive.addfile(info, io.BytesIO(data))
    return path


CLIPS = {
    "A-1": [f"images{i:04d}" for i in range(9)],
    "B-2": [f"images{i:04d}" for i in range(3, 10)],
    "C-3": [f"images{i:04d}" for i in range(5)],
}


def test_load_clips_excludes_pilot_signers_and_reads_counts(tmp_path):
    raw = {
        "prefix": "x",
        0: {"fileid": "A-1", "signer": "Signer08", "num_frames": "9"},
        1: {"fileid": "S3", "signer": "Signer03", "num_frames": "10"},
        2: {"fileid": "S7", "signer": "Signer07", "num_frames": "10"},
        3: {"fileid": "B-2", "signer": "Signer01", "num_frames": "7"},
    }
    path = tmp_path / "ann.npy"
    np.save(path, raw)
    rows = load_clips(path)
    assert [r["clip"] for r in rows] == ["A-1", "B-2"]
    assert rows[1]["num_frames"] == 7
    with pytest.raises(ValueError, match="duplicate"):
        np.save(
            path,
            {
                "0": {"fileid": "X", "signer": "Signer01", "num_frames": "5"},
                "1": {"fileid": "X", "signer": "Signer01", "num_frames": "5"},
            },
        )
        load_clips(path)


def test_index_frames_single_pass(tmp_path):
    tar = _tar(tmp_path, CLIPS)
    frames = index_frames(tar, {"A-1", "C-3"})
    assert sorted(frames["A-1"]) == sorted(CLIPS["A-1"])
    assert "B-2" not in frames
    with pytest.raises(ValueError, match="absent"):
        index_frames(tar, {"A-1", "nope"})


def test_verify_counts_flags_mismatch(tmp_path):
    frames = index_frames(_tar(tmp_path, CLIPS), set(CLIPS))
    verify_counts(frames, {"A-1": 9, "B-2": 7})
    with pytest.raises(ValueError, match="annotation says"):
        verify_counts(frames, {"A-1": 8})


def test_planned_indices_rejects_out_of_range(tmp_path):
    frames = index_frames(_tar(tmp_path, CLIPS), {"C-3"})
    with pytest.raises(ValueError, match="outside"):
        planned_indices(frames, {"C-3": [0, 99]})


def test_extract_and_read_roundtrip(tmp_path):
    tar = _tar(tmp_path, CLIPS)
    frames = index_frames(tar, {"A-1", "B-2"})
    plan = planned_indices(frames, {"A-1": [0, 2, 4, 6, 8], "B-2": [1, 2, 3, 4, 5]})
    out = tmp_path / "frames"
    extract_frames(tar, frames, plan, out)
    assert sorted(p.name for p in (out / "A-1").iterdir()) == [
        f"{s}.png" for s in ["images0000", "images0002", "images0004", "images0006", "images0008"]
    ]
    image = read_frame(out, "B-2", 3)
    assert image.shape == (26, 21, 3)
    first = read_frame(out, "A-1", 0)
    second = read_frame(out, "A-1", 2)
    assert not np.array_equal(first, second)
    with pytest.raises(ValueError, match="outside"):
        read_frame(out, "A-1", 5)


def test_extract_rejects_unsafe_names(tmp_path):
    with pytest.raises(ValueError, match="unsafe"):
        from how2sign_probe.phoenix import extract_members

        extract_members(tmp_path / "x.tar.gz", ["bad name;rm -rf /"], tmp_path)


def test_index_frames_reads_uncompressed_tar(tmp_path):
    path = tmp_path / "release.tar"
    import io

    with tarfile.open(path, "w") as archive:
        for serial in CLIPS["A-1"]:
            data = _png(1)
            info = tarfile.TarInfo(f"{PREFIX}/A-1/{serial}.png")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    frames = index_frames(path, {"A-1"})
    assert len(frames["A-1"]) == 9

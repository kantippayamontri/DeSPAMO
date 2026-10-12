import pytest

from tools.clip_s2.inventory import frame_paths


def test_absolute_completed_source_rows_are_supported_and_reorder_fails(tmp_path):
    root = tmp_path / "frames"
    folder = root / "train" / "clip"
    folder.mkdir(parents=True)
    paths = [folder / "1.png", folder / "2.png"]
    for p in paths:
        p.write_bytes(b"image")
    row = dict(paths=[str(p) for p in paths], source_indices=[0, 1], frame_count=2)
    assert frame_paths(row, root, "train", "clip") == paths
    row["paths"].reverse()
    with pytest.raises(ValueError):
        frame_paths(row, root, "train", "clip")


def test_wrong_source_directory_is_rejected(tmp_path):
    row = dict(paths=[str(tmp_path / "elsewhere.png")], source_indices=[0], frame_count=1)
    with pytest.raises(ValueError):
        frame_paths(row, tmp_path / "frames", "train", "clip")

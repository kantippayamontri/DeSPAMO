import json
import struct
import zlib

import numpy as np
import pytest
import torch
from PIL import Image

from tools.dinov3.frames import annotations, preprocess, sample_indices, scan
from tools.dinov3.identity import digest, sha256_file


def _annotations(tmp_path):
    ann = tmp_path / "ann"
    ann.mkdir()
    records = []
    for split in ("train", "dev", "test"):
        np.save(ann / f"{split}_info_ml.npy", {0: {"fileid": split, "num_frames": 5}})
        records.append(
            dict(
                clip_id=split,
                split=split,
                path=f"{split}/{split}.npy",
                length=5,
                width=2048,
                dtype="float32",
            )
        )
    manifest = tmp_path / "clip.json"
    manifest.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    return ann, manifest, records


def _idat_chunk(raw):
    offset = 8  # PNG signature
    while raw[offset + 4 : offset + 8] != b"IDAT":
        length = struct.unpack_from(">I", raw, offset)[0]
        offset += 12 + length
    return offset, struct.unpack_from(">I", raw, offset)[0]


def test_sorted_rows_hash_and_five_distinct_samples(tmp_path):
    folder = tmp_path / "train" / "c"
    folder.mkdir(parents=True)
    for serial in (14, 10, 12, 11, 13):
        Image.new("RGB", (210, 260), (serial, 0, 0)).save(folder / f"c-{serial:06}.png")
    source = scan(tmp_path, "train", "c", (5, 5))
    paths = [f"train/c/c-{serial:06}.png" for serial in range(10, 15)]
    items = [
        dict(path=path, size=(tmp_path / path).stat().st_size, sha256=sha256_file(tmp_path / path))
        for path in paths
    ]
    assert source == dict(
        paths=paths,
        source_hash=digest(items),
        source_indices=list(range(5)),
        sampled_images={str(i): items[i]["sha256"] for i in range(5)},
        resolutions=[],
    )
    before = source["source_hash"]
    Image.new("RGB", (210, 260), (255, 0, 0)).save(folder / "c-000010.png")
    assert scan(tmp_path, "train", "c", (5, 5))["source_hash"] != before
    with pytest.raises(ValueError, match="train/c.*count"):
        scan(tmp_path, "train", "c", (6, 6))
    with pytest.raises(ValueError, match="train/c.*count"):
        scan(tmp_path, "train", "c", (5, 6))


def test_frame_serials_must_be_unique_but_need_not_start_at_zero(tmp_path):
    folder = tmp_path / "dev" / "c"
    folder.mkdir(parents=True)
    for prefix, serial in (("a", 10), ("b", 11), ("c", 12), ("d", 13), ("e", 10)):
        Image.new("RGB", (2, 3)).save(folder / f"{prefix}-{serial:06}.png")
    with pytest.raises(ValueError, match="dev/c.*duplicate"):
        scan(tmp_path, "dev", "c", (5, 5))
    (folder / "e-000010.png").rename(folder / "e-invalid.png")
    with pytest.raises(ValueError, match="dev/c.*invalid"):
        scan(tmp_path, "dev", "c", (5, 5))


def test_missing_directory_and_corrupt_png_fail_with_clip_context(tmp_path):
    with pytest.raises(ValueError, match="test/c.*missing"):
        scan(tmp_path, "test", "c", (5, 5))
    folder = tmp_path / "test" / "c"
    folder.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (3, 2)).save(folder / f"{i:06}.png")
    bad = folder / "000002.png"
    bad.write_bytes(b"not a PNG")
    with pytest.raises(ValueError, match="test/c.*PNG"):
        scan(tmp_path, "test", "c", (5, 5))
    with pytest.raises(ValueError, match="PNG"):
        preprocess(bad, 224)


def test_bad_png_checksum_reports_clip_qualified_value_error(tmp_path):
    folder = tmp_path / "train" / "c"
    folder.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (3, 2), (i, 0, 0)).save(folder / f"{i:06}.png")
    bad = folder / "000002.png"
    raw = bytearray(bad.read_bytes())
    offset, length = _idat_chunk(raw)
    raw[offset + 8 + length] ^= 1  # Corrupt IDAT CRC, leaving PNG header and payload intact.
    bad.write_bytes(raw)

    with pytest.raises(ValueError, match=r"train/c: invalid PNG 000002\.png") as caught:
        scan(tmp_path, "train", "c", (5, 5))
    assert isinstance(caught.value.__cause__, SyntaxError)


def test_preprocess_bad_png_crc_is_path_qualified_value_error(tmp_path):
    image = tmp_path / "frame.png"
    Image.new("RGB", (3, 2), "white").save(image)
    raw = bytearray(image.read_bytes())
    offset, length = _idat_chunk(raw)
    raw[offset + 8 + length] ^= 1
    image.write_bytes(raw)

    with pytest.raises(ValueError, match=r"invalid PNG .*frame\.png") as caught:
        preprocess(image, 224)
    assert isinstance(caught.value.__cause__, SyntaxError)


def test_bad_idat_with_valid_checksum_fails_source_scan(tmp_path):
    folder = tmp_path / "dev" / "c"
    folder.mkdir(parents=True)
    for i in range(5):
        Image.new("RGB", (3, 2), (i, 0, 0)).save(folder / f"{i:06}.png")
    bad = folder / "000002.png"
    raw = bytearray(bad.read_bytes())
    offset, length = _idat_chunk(raw)
    raw[offset + 8] ^= 0xFF  # Invalidate zlib header without changing PNG chunk length.
    crc_offset = offset + 8 + length
    struct.pack_into(">I", raw, crc_offset, zlib.crc32(raw[offset + 4 : crc_offset]))
    bad.write_bytes(raw)

    with Image.open(bad) as image:
        image.verify()  # Checksum remains valid; verify() does not decode IDAT.
    with Image.open(bad) as image, pytest.raises(OSError):
        image.load()
    with pytest.raises(ValueError, match=r"dev/c: invalid PNG 000002\.png"):
        scan(tmp_path, "dev", "c", (5, 5))


def test_preprocess_square_rgb_and_imagenet(tmp_path):
    path = tmp_path / "a.png"
    Image.new("RGB", (210, 260), "white").save(path)
    for size in (224, 448):
        pixels = preprocess(path, size)
        assert pixels.shape == (3, size, size)
        assert pixels.dtype == torch.float32
        np.testing.assert_array_equal(
            pixels[:, 0, 0],
            (np.ones(3, dtype=np.float32) - np.array((0.485, 0.456, 0.406), dtype=np.float32))
            / np.array((0.229, 0.224, 0.225), dtype=np.float32),
        )
    with pytest.raises(ValueError, match="scale"):
        preprocess(path, 256)


def test_nonuniform_bicubic_matches_pillow_at_both_scales(tmp_path):
    path = tmp_path / "gradient.png"
    source = np.array(
        [
            [[0, 80, 255], [40, 160, 200], [255, 20, 10]],
            [[255, 240, 0], [100, 5, 100], [0, 255, 160]],
        ],
        dtype=np.uint8,
    )
    Image.fromarray(source).save(path)
    for size in (224, 448):
        with Image.open(path) as image:
            bicubic = np.asarray(
                image.resize((size, size), Image.Resampling.BICUBIC), dtype=np.float32
            ) / np.float32(255)
            bilinear = np.asarray(image.resize((size, size), Image.Resampling.BILINEAR))
        reference = (bicubic - np.array((0.485, 0.456, 0.406), dtype=np.float32)) / (
            np.array((0.229, 0.224, 0.225), dtype=np.float32)
        )
        actual = preprocess(path, size)
        assert actual.dtype == torch.float32
        np.testing.assert_array_equal(actual.numpy(), reference.transpose(2, 0, 1))
        assert not np.array_equal(bilinear, np.asarray(bicubic * np.float32(255)))


def test_preprocess_converts_grayscale_to_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (2, 3), 255).save(path)
    pixels = preprocess(path, 224)
    np.testing.assert_array_equal(
        pixels[:, 0, 0].numpy(),
        (np.ones(3, dtype=np.float32) - np.array((0.485, 0.456, 0.406), dtype=np.float32))
        / np.array((0.229, 0.224, 0.225), dtype=np.float32),
    )


def test_sample_indices_round_half_up_and_require_five():
    assert sample_indices(5) == [0, 1, 2, 3, 4]
    assert sample_indices(6) == [1, 2, 3, 4, 5]
    assert sample_indices(11) == [1, 3, 5, 7, 9]
    with pytest.raises(ValueError, match="five"):
        sample_indices(4)
    with pytest.raises(ValueError, match="five"):
        sample_indices(True)


def test_annotation_and_clip_parity(tmp_path):
    ann, manifest, records = _annotations(tmp_path)
    assert annotations(ann, manifest)["dev"]["dev"] == (5, 5)
    records[1]["length"] = 6
    manifest.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    with pytest.raises(ValueError, match="dev/dev.*CLIP"):
        annotations(ann, manifest)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda records: records.append(records[0].copy()), "duplicate CLIP"),
        (
            lambda records: records.append(
                {**records[0], "clip_id": "extra", "path": "train/extra.npy"}
            ),
            "CLIP.*extra",
        ),
        (lambda records: records[0].update(split="other"), "CLIP split"),
        (lambda records: records[0].update(path="train/other.npy"), "CLIP record"),
        (lambda records: records[0].update(width=1024), "CLIP record"),
    ],
)
def test_rejects_duplicate_unmatched_or_invalid_clip_records(tmp_path, change, message):
    ann, manifest, records = _annotations(tmp_path)
    change(records)
    manifest.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    with pytest.raises(ValueError, match=message):
        annotations(ann, manifest)


def test_rejects_missing_duplicate_and_cross_split_annotation_ids(tmp_path):
    ann, manifest, _ = _annotations(tmp_path)
    (ann / "train_info_ml.npy").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="train_info_ml"):
        annotations(ann, manifest)
    np.save(
        ann / "train_info_ml.npy",
        {0: {"fileid": "train", "num_frames": 5}, 1: {"fileid": "train", "num_frames": 5}},
    )
    with pytest.raises(ValueError, match="duplicate.*train/train"):
        annotations(ann, manifest)
    np.save(ann / "train_info_ml.npy", {0: {"fileid": "train", "num_frames": 5}})
    np.save(ann / "dev_info_ml.npy", {0: {"fileid": "train", "num_frames": 5}})
    with pytest.raises(ValueError, match="dev/train.*CLIP"):
        annotations(ann, manifest)


def test_rejects_wrong_annotation_count_and_duplicate_id_across_splits(tmp_path):
    ann, manifest, records = _annotations(tmp_path)
    np.save(ann / "train_info_ml.npy", {0: {"fileid": "train", "num_frames": 4}})
    with pytest.raises(ValueError, match="count: train/train"):
        annotations(ann, manifest)
    np.save(ann / "train_info_ml.npy", {0: {"fileid": "train", "num_frames": 5}})
    np.save(ann / "dev_info_ml.npy", {0: {"fileid": "train", "num_frames": 5}})
    records[1].update(clip_id="train", path="dev/train.npy")
    manifest.write_text(
        json.dumps(dict(schema_version=1, encoder="clip", expected_dim=2048, records=records))
    )
    with pytest.raises(ValueError, match="duplicate clip ID across splits"):
        annotations(ann, manifest)

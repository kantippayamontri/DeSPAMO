import json
import re
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from tools.dinov3.identity import MEAN, STD, digest, sha256_file

SPLITS = {
    "train": "train_info_ml.npy",
    "dev": "dev_info_ml.npy",
    "test": "test_info_ml.npy",
}
POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def annotations(root: Path, clip_manifest: Path) -> dict[str, dict[str, tuple[int, int]]]:
    manifest = json.loads(clip_manifest.read_text())
    if not isinstance(manifest, dict) or (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or type(manifest.get("expected_dim")) is not int
        or manifest["expected_dim"] != 2048
    ):
        raise ValueError("CLIP schema/width mismatch")
    if not isinstance(manifest.get("encoder"), str) or not manifest["encoder"].strip():
        raise ValueError("invalid CLIP encoder")
    if not isinstance(manifest.get("records"), list):
        raise ValueError("invalid CLIP records")
    records: dict[tuple[str, str], int] = {}
    for record in manifest["records"]:
        if not isinstance(record, dict):
            raise ValueError("invalid CLIP record")
        split, clip = record.get("split"), record.get("clip_id")
        if split not in SPLITS or (
            not isinstance(clip, str) or clip in {"", ".", ".."} or "/" in clip or "\\" in clip
        ):
            raise ValueError("invalid CLIP split/clip")
        if (
            set(record) != {"clip_id", "split", "path", "length", "width", "dtype"}
            or record["path"] != f"{split}/{clip}.npy"
            or type(record["width"]) is not int
            or record["width"] != 2048
            or record["dtype"] not in {"float16", "float32", "float64"}
            or type(record["length"]) is not int
            or record["length"] < 5
            or (split, clip) in records
        ):
            raise ValueError(f"invalid/duplicate CLIP record: {split}/{clip}")
        records[split, clip] = record["length"]
    result: dict[str, dict[str, tuple[int, int]]] = {}
    for split, filename in SPLITS.items():
        raw = np.load(root / filename, allow_pickle=True).item()  # trusted local annotation
        if not isinstance(raw, dict):
            raise ValueError(f"invalid annotation: {split}")
        entries: dict[str, tuple[int, int]] = {}
        for key in sorted(k for k in raw if type(k) is int):
            item = raw[key]
            if not isinstance(item, dict):
                raise ValueError(f"invalid annotation record: {split}/{key}")
            clip, count = item.get("fileid"), item.get("num_frames")
            if not isinstance(clip, str) or clip in {"", ".", ".."} or "/" in clip or "\\" in clip:
                raise ValueError(f"invalid annotation clip in {split}")
            if type(count) is not int or count < 5 or clip in entries:
                raise ValueError(f"invalid/duplicate annotation count: {split}/{clip}")
            if records.get((split, clip)) != count:
                raise ValueError(f"{split}/{clip}: annotation/CLIP row count mismatch")
            entries[clip] = (count, records[split, clip])
        result[split] = entries
    if set(records) != {(split, clip) for split, clips in result.items() for clip in clips}:
        raise ValueError("CLIP manifest extra/missing clips")
    if len({clip for clips in result.values() for clip in clips}) != len(records):
        raise ValueError("duplicate clip ID across splits; row-map keys would collide")
    return result


def sample_indices(count: int) -> list[int]:
    if type(count) is not int or count < 5:
        raise ValueError("at least five frames required")
    indices = [int((count - 1) * position + 0.5) for position in POSITIONS]
    if len(set(indices)) != 5:
        raise ValueError("five sampled source indices must be distinct")
    return indices


def scan(root: Path, split: str, clip_id: str, expected: tuple[int, int]) -> dict:
    if split not in SPLITS or (
        not isinstance(clip_id, str)
        or clip_id in {"", ".", ".."}
        or "/" in clip_id
        or "\\" in clip_id
    ):
        raise ValueError("invalid source clip")
    directory = root / split / clip_id
    if not directory.is_dir():
        raise ValueError(f"{split}/{clip_id}: missing source directory")
    frames = sorted(directory.glob("*.png"), key=lambda path: path.name)
    if len(frames) < 5 or len(frames) != expected[0] or len(frames) != expected[1]:
        raise ValueError(f"{split}/{clip_id}: PNG count/annotation/CLIP mismatch")
    serials = [re.search(r"(\d+)\.png\Z", path.name) for path in frames]
    if any(match is None for match in serials):
        raise ValueError(f"{split}/{clip_id}: invalid frame serial")
    if len({int(match.group(1)) for match in serials}) != len(frames):
        raise ValueError(f"{split}/{clip_id}: duplicate frame serial")
    items = []
    for path in frames:
        try:
            with Image.open(path) as image:
                if image.format != "PNG":
                    raise ValueError("not a PNG")
                image.verify()
            with Image.open(path) as image:
                image.load()
            items.append(
                dict(
                    path=path.relative_to(root).as_posix(),
                    size=path.stat().st_size,
                    sha256=sha256_file(path),
                )
            )
        except (OSError, ValueError, SyntaxError) as error:
            raise ValueError(f"{split}/{clip_id}: invalid PNG {path.name}: {error}") from error
    sampled = {str(i): items[i]["sha256"] for i in sample_indices(len(frames))}
    return dict(
        paths=[item["path"] for item in items],
        source_hash=digest(items),
        source_indices=list(range(len(items))),
        sampled_images=sampled,
        resolutions=[],
    )


def preprocess(path: Path, size: int) -> torch.Tensor:
    if size not in (224, 448):
        raise ValueError("unsupported DINOv3 scale")
    try:
        with Image.open(path) as source:
            if source.format != "PNG":
                raise ValueError("not a PNG")
            source.verify()
        with Image.open(path) as source:
            image = source.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
            values = np.asarray(image, dtype=np.float32).copy() / np.float32(255)
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"invalid PNG {path}: {error}") from error
    values = (values - np.asarray(MEAN, dtype=np.float32)) / np.asarray(STD, dtype=np.float32)
    return torch.from_numpy(values.transpose(2, 0, 1).copy())

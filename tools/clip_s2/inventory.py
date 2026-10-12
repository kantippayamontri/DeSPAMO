"""Admit completed absolute source row maps without rewriting inherited helpers."""

import json
from pathlib import Path

import numpy as np


def frame_paths(row: dict, frames_root: Path, split: str, clip: str) -> list[Path]:
    paths = [Path(p) if Path(p).is_absolute() else frames_root / p for p in row["paths"]]
    expected = frames_root / split / clip
    if (
        row["source_indices"] != list(range(row["frame_count"]))
        or len(paths) != row["frame_count"]
        or paths != sorted(set(paths))
        or any(p.parent != expected or p.is_symlink() for p in paths)
        or paths != sorted(expected.glob("*.png"))
    ):
        raise ValueError("source frame directory/order/count mismatch")
    return paths


def inventory(annotation_root: Path, row_map: Path, frames_root: Path) -> dict:
    rows = json.loads(row_map.read_text())["clips"]
    result, seen = {}, set()
    for split, count in (("train", 7096), ("dev", 519), ("test", 642)):
        raw = np.load(annotation_root / f"{split}_info_ml.npy", allow_pickle=True).item()
        records = [raw[k] for k in sorted(k for k in raw if type(k) is int)]
        ids = [r["fileid"] for r in records]
        if len(ids) != count or len(set(ids)) != count or seen.intersection(ids):
            raise ValueError("official split IDs/count mismatch")
        if {r["signer"] for r in records} != {f"Signer{i:02}" for i in range(1, 10)}:
            raise ValueError("official signer coverage mismatch")
        result[split] = {}
        for record in records:
            clip = record["fileid"]
            if clip in ("", ".", "..") or "/" in clip or "\\" in clip:
                raise ValueError("invalid clip ID")
            paths = frame_paths(rows[clip], frames_root, split, clip)
            if len(paths) != record["num_frames"]:
                raise ValueError("annotation/source frame count mismatch")
            result[split][clip] = dict(
                paths=[str(p) for p in paths],
                frame_count=len(paths),
                signer=record["signer"],
                expected_source_hash=rows[clip]["source_hash"],
            )
        seen.update(ids)
    if (
        set(rows) != seen
        or sum(r["frame_count"] for c in result.values() for r in c.values()) != 947756
    ):
        raise ValueError("official full source row-map coverage mismatch")
    return result

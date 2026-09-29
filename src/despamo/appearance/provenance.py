"""Version immutable five-frame sources without storing host-specific roots."""

import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from despamo.appearance.schema import POSITIONS


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def file_hash(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, mode="w", encoding="utf-8", prefix=".partial-", delete=False
        ) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sample_indices(count: int) -> list[int]:
    if type(count) is not int or count < 5:
        raise ValueError("five distinct source frames required")
    indices = [int((count - 1) * position + 0.5) for position in POSITIONS]
    if len(set(indices)) != 5:
        raise ValueError("five distinct source frames required")
    return indices


def build_sources(
    annotation: Path, frame_root: Path, split: str, frame_rows: Path | None = None
) -> dict:
    if split not in {"train", "dev", "test"}:
        raise ValueError("invalid annotation split")
    raw = np.load(annotation, allow_pickle=True).item()  # trusted local annotations
    if not isinstance(raw, dict):
        raise ValueError("invalid annotation mapping")
    root = frame_root.resolve()
    row_map = read_json(frame_rows) if frame_rows is not None else None
    if row_map is not None and not isinstance(row_map.get("clips"), dict):
        raise ValueError("invalid DINO frame-row map")
    clips = []
    for key in sorted(item for item in raw if type(item) is int):
        item = raw[key]
        clip_id = item["fileid"]
        if (
            not isinstance(clip_id, str)
            or clip_id in {"", ".", ".."}
            or "/" in clip_id
            or "\\" in clip_id
            or item["folder"] != f"{split}/{clip_id}/*.png"
        ):
            raise ValueError("annotation clip path is not the expected split/ID")
        paths = sorted(root.glob(item["folder"]))
        entry = dict(clip_id=clip_id, signer=item["signer"], frame_count=len(paths),
                     frames=[], error=None)
        try:
            if "num_frames" in item and item["num_frames"] != len(paths):
                raise ValueError("annotation PNG count mismatch")
            indices = sample_indices(len(paths))
            if row_map is not None:
                mapping = row_map["clips"].get(clip_id)
                ordered = [path.relative_to(root).as_posix() for path in paths]
                if (
                    not isinstance(mapping, dict)
                    or mapping.get("frame_count") != len(paths)
                    or mapping.get("source_indices") != list(range(len(paths)))
                    or mapping.get("paths") != ordered
                ):
                    raise ValueError(f"{split}/{clip_id}: DINO frame mapping mismatch")
            for position, index in zip(POSITIONS, indices, strict=True):
                path = paths[index].resolve()
                source_path = path.relative_to(root).as_posix()
                checksum = file_hash(path)
                if row_map is not None and mapping["sampled_images"].get(str(index)) != checksum:
                    raise ValueError(f"{split}/{clip_id}: sampled image differs from DINO map")
                entry["frames"].append(dict(position=position, index=index,
                                            path=source_path, sha256=checksum))
        except (OSError, ValueError) as error:
            if "DINO" in str(error):
                raise
            entry["error"] = f"{type(error).__name__}: incomplete source frames"
            entry["frames"] = []
        clips.append(entry)
    if not clips or len({clip["clip_id"] for clip in clips}) != len(clips):
        raise ValueError("empty or duplicate clip manifest")
    result = dict(split=split, clips=sorted(clips, key=lambda clip: clip["clip_id"]))
    if frame_rows is not None:
        result["frame_rows_hash"] = file_hash(frame_rows)
    return result


def create_dataset(root: Path, sources: dict, version: dict) -> Path:
    version = {**version, "source_split": sources["split"], "source_hash": digest(sources)}
    path = root / "datasets" / digest(version)
    if path.exists():
        if load_dataset(path) != (sources, version):
            raise ValueError("dataset collision or incomplete manifest; inspect before recovery")
    else:
        atomic_json(path / "source.json", sources)
        atomic_json(path / "version.json", version)
    return path


def load_dataset(path: Path) -> tuple[dict, dict]:
    sources = read_json(path / "source.json")
    version = read_json(path / "version.json")
    if digest(version) != path.name or digest(sources) != version["source_hash"]:
        raise ValueError("dataset provenance mismatch")
    return sources, version

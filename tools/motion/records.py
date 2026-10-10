"""Content identity and byte-bound records for the upcoming motion generator."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from despamo.appearance.provenance import digest, file_hash

MODEL = {
    "name": "MCG-NJU/videomae-large",
    "revision": "12da269a02d3e1fbbb7011e610e591fea8061dca",
    "dtype": "float32", "inference": True, "eval": True,
    "tokens": "hidden_states[-1][:,0]", "width": 1024,
}
PROCESSOR = {
    "color": "RGB", "resize": "bilinear_shortest_edge", "shortest_edge": 224,
    "centre_crop": 224, "rescale": 1 / 255, "mean": [0.5] * 3, "std": [0.5] * 3,
    "window_size": 16, "stride": 8, "tail_policy": "full_windows_only",
    "short_clip_policy": "repeat_last_to_16", "frame_order": "lexicographic",
    "motion_row_order": "ascending_window_start",
}


def regular_path(path: Path) -> Path:
    """Reject aliases, symlinks (including ancestors), and nonregular bytes."""
    if not path.is_absolute() or path != path.resolve():
        raise ValueError(f"absolute canonical path required: {path}")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"symlink forbidden: {path}")
    if not path.is_file() or path.stat().st_nlink != 1:
        raise ValueError(f"unique regular file required: {path}")
    return path


def window_rows(frame_count: int) -> tuple[dict, ...]:
    """Record encoder input separately from real support, including short padding."""
    if type(frame_count) is not int or frame_count <= 0:
        raise ValueError("positive integer frame count required")
    starts = [0] if frame_count < 16 else range(0, frame_count - 15, 8)
    return tuple({"row": row,
                  "input_source_indices": [min(start + i, frame_count - 1) for i in range(16)],
                  "real_start": start, "real_end": min(start + 15, frame_count - 1),
                  "centre": start + 7.5} for row, start in enumerate(starts))


def source_identity(paths: tuple[Path, ...], frame_sha256: tuple[str, ...]) -> str:
    """Match spatial digest of ordered (recorded POSIX path, raw SHA256) pairs."""
    if not paths or len(paths) != len(frame_sha256):
        raise ValueError("source path/hash count mismatch")
    if any(re.fullmatch(r"[0-9a-f]{64}", item) is None for item in frame_sha256):
        raise ValueError("invalid source SHA256")
    return digest([(path.as_posix(), sha) for path, sha in zip(paths, frame_sha256, strict=True)])


def make_motion_identity(inputs: dict, model: dict, processor: dict,
                         environment: dict, code: dict, hardware: dict) -> dict:
    """Return independent JSON content identity; runtime fields are rejected."""
    identity = dict(schema_version=1, origin="new_extraction_v1", inputs=inputs,
                    model=model, processor=processor, environment=environment,
                    code=code, hardware=hardware)

    def check(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key.lower() in {
                    "runtime", "runtime_policy", "runtime_status", "runtime_clock",
                    "budget", "budget_seconds", "budget_policy", "remaining_budget_seconds",
                    "elapsed", "elapsed_seconds", "extraction_elapsed_seconds",
                    "training_elapsed_seconds", "cap", "cap_seconds", "cap_transfer",
                    "extraction_cap_seconds", "training_cap_seconds",
                    "output", "output_directory", "output_dir", "output_root", "output_path",
                    "auth", "authorization", "authorization_receipt", "run_authorization",
                }:
                    raise ValueError("runtime policy must remain outside content identity")
                check(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                check(item)

    check(identity)
    if model != MODEL or processor != PROCESSOR:
        raise ValueError("frozen motion recipe mismatch")
    if set(inputs) != {"frame_root", "frame_rows", "annotation"}:
        raise ValueError("exact source inputs required")
    if not {"extractor", "helpers"} <= set(code) or "lock" not in environment:
        raise ValueError("extractor/helper/lock evidence required")
    digest(identity)  # Reject non-JSON values and nonfinite metadata.
    # Traverse the same representation that publication persists and admission reads.
    return json.loads(json.dumps(identity, allow_nan=False))


def make_clip_receipt(clip_id: str, spatial_row: dict, source_rows: list[dict],
                      windows: tuple[dict, ...], array_path: Path, identity: dict) -> dict:
    """Call immediately after generating array from recorded consumed source bytes."""
    if not isinstance(clip_id, str) or clip_id in {"", ".", ".."} or any(
        char in clip_id for char in "/\\"
    ):
        raise ValueError("invalid clip ID")
    count = spatial_row["frame_count"]
    expected = list(range(count))
    if (digest(spatial_row["source_indices"]) != digest(expected) or len(source_rows) != count
            or digest([r["index"] for r in source_rows]) != digest(expected)
            or [r["path"] for r in source_rows] != spatial_row["paths"]
            or digest(windows) != digest(window_rows(count))):
        raise ValueError("actual source/window order mismatch")
    paths = tuple(Path(r["path"]) for r in source_rows)
    source_root = Path(identity["inputs"]["frame_root"])
    if not source_root.is_absolute() or source_root != source_root.resolve():
        raise ValueError("canonical source root required")
    for path, row in zip(paths, source_rows, strict=True):
        if (".." in path.parts or path.as_posix() != row["path"]
                or set(row) != {"index", "path", "sha256"}):
            raise ValueError("source paths must preserve recorded spelling")
        actual_path = source_root / path
        if source_root not in actual_path.parents:
            raise ValueError("foreign source root")
        if file_hash(regular_path(actual_path)) != row["sha256"]:
            raise ValueError("consumed source bytes changed")
    source_hash = source_identity(paths, tuple(r["sha256"] for r in source_rows))
    if source_hash != spatial_row["source_hash"]:
        raise ValueError("source bytes differ from spatial receipt")
    regular_path(array_path)
    if array_path.parts[-2:] != ("train", f"{clip_id}.npy"):
        raise ValueError("array path does not match physical train clip")
    array = np.load(array_path, allow_pickle=False, mmap_mode="r")
    if (array.shape != (len(windows), 1024) or array.dtype != np.float32
            or not np.isfinite(array).all()):
        raise ValueError("invalid motion shape/dtype/finiteness")
    return dict(schema_version=1, origin="new_extraction_v1", clip_id=clip_id,
                featurekey=digest(identity), source_hash=source_hash, frame_count=count,
                source_indices=expected, source_rows=copy.deepcopy(source_rows),
                windows=copy.deepcopy(list(windows)),
                array={"path": f"train/{clip_id}.npy", "shape": list(array.shape),
                       "dtype": "float32", "sha256": file_hash(array_path)})

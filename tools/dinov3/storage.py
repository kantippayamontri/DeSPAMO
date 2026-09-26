import errno
import fcntl
import json
import os
import re
import stat
import subprocess
import tempfile
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

import numpy as np

from tools.dinov3.identity import sha256_file

NATIVE_FILESYSTEMS = {"ext4", "xfs", "btrfs", "overlay", "tmpfs", "zfs", "f2fs"}
RECEIPT_FIELDS = {
    "encoder_key",
    "source_hash",
    "paths",
    "source_indices",
    "sampled_images",
    "resolutions",
    "feature_hash",
}


def verify_location(path: Path, checkout: Path) -> None:
    resolved, project = path.resolve(strict=False), checkout.resolve(strict=False)
    if resolved == project or project in resolved.parents:
        raise ValueError("model cache/output inside Git checkout")
    ancestor = resolved
    while not ancestor.exists():
        ancestor = ancestor.parent
    filesystem = subprocess.check_output(
        ["findmnt", "-no", "FSTYPE", "-T", str(ancestor)],
        text=True,
        env={key: value for key, value in os.environ.items() if key != "HF_TOKEN"},
    ).strip()
    if filesystem not in NATIVE_FILESYSTEMS:
        raise ValueError("model cache/output must use Linux-native filesystem")


def _fsync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _mkdir_durable(path: Path) -> None:
    missing = []
    cursor = path
    while not cursor.exists():
        missing.append(cursor)
        cursor = cursor.parent
    for directory in reversed(missing):
        directory.mkdir(exist_ok=True)
        _fsync_directory(directory.parent)


@contextmanager
def writer(root: Path) -> Iterator[None]:
    verify_location(root, Path(__file__).resolve().parents[2])
    lock = root / ".writer.lock"
    if (root / "complete").is_dir() and not lock.is_file():
        raise ValueError("completed version missing writer lock")
    _mkdir_durable(root)
    if lock.is_symlink():
        raise ValueError("unsafe writer lock symlink")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o666)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError("unsafe writer lock symlink") from None
        raise
    with os.fdopen(fd, "a+b") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_dev != root.stat().st_dev:
            raise ValueError("unsafe writer lock type/filesystem")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another writer owns encoder key") from None
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic(path: Path, write: Callable[[BinaryIO], object]) -> None:
    _mkdir_durable(path.parent)
    name = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".partial-", delete=False
        ) as handle:
            name = Path(handle.name)
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
        _fsync_directory(path.parent)
    finally:
        if name is not None:
            name.unlink(missing_ok=True)


def atomic_json(path: Path, payload: object) -> None:
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    if json.loads(raw) != payload:
        raise ValueError(f"JSON round-trip mismatch: {path}")
    _atomic(path, lambda handle: handle.write(raw))
    if json.loads(path.read_text()) != payload:
        raise ValueError(f"JSON read-back mismatch: {path}")


def atomic_array(path: Path, array: np.ndarray) -> None:
    if (
        array.ndim != 2
        or array.shape[0] < 5
        or array.shape[1] != 2048
        or array.dtype != np.float32
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"invalid feature array: {path}")
    _atomic(path, lambda handle: np.save(handle, array, allow_pickle=False))
    saved = np.load(path, allow_pickle=False)
    if saved.shape != array.shape or saved.dtype != np.float32 or not np.array_equal(saved, array):
        raise ValueError(f"feature read-back mismatch: {path}")


def _clip_paths(root: Path, split: str, clip_id: str) -> tuple[Path, Path]:
    if split not in {"train", "dev", "test"} or (
        not isinstance(clip_id, str)
        or clip_id in {"", ".", ".."}
        or "/" in clip_id
        or "\\" in clip_id
    ):
        raise ValueError("invalid clip split/ID")
    base = root.resolve(strict=False)
    checkout = Path(__file__).resolve().parents[2]
    if base == checkout or checkout in base.parents:
        raise ValueError("unsafe output path inside Git checkout")
    ancestor = base
    while not ancestor.exists():
        ancestor = ancestor.parent
    device = ancestor.stat().st_dev

    def checked(*parts: str) -> Path:
        path = base
        for part in parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("unsafe output path symlink")
            if path.exists() and path.stat().st_dev != device:
                raise ValueError("unsafe output filesystem")
            resolved = path.resolve(strict=False)
            if base not in resolved.parents:
                raise ValueError("unsafe output path outside root")
        return path

    return checked(split, f"{clip_id}.npy"), checked("receipts", split, f"{clip_id}.json")


def _valid_samples(images: object, paths: object, indices: object) -> bool:
    if (
        not isinstance(paths, list)
        or not isinstance(indices, list)
        or not isinstance(images, dict)
        or len(paths) < 5
        or len(indices) != len(paths)
        or len(images) != 5
    ):
        return False
    ordinals = {index for index in indices if type(index) is int and 0 <= index < len(paths)}
    keys = {str(index) for index in ordinals}
    return (
        len(ordinals) == len(paths)
        and all(isinstance(path, str) for path in paths)
        and all(isinstance(value, str) for value in images.values())
        and all(
            isinstance(key, str)
            and key in keys
            and re.fullmatch(r"[0-9a-f]{64}", value) is not None
            for key, value in images.items()
        )
    )


def _valid_receipt(record: object) -> bool:
    if not isinstance(record, dict) or set(record) != RECEIPT_FIELDS:
        return False
    paths = record["paths"]
    indices = record["source_indices"]
    images = record["sampled_images"]
    resolutions = record["resolutions"]
    return (
        isinstance(record["encoder_key"], str)
        and isinstance(record["source_hash"], str)
        and isinstance(record["feature_hash"], str)
        and re.fullmatch(r"[0-9a-f]{64}", record["feature_hash"]) is not None
        and isinstance(paths, list)
        and len(paths) >= 5
        and all(isinstance(path, str) for path in paths)
        and isinstance(indices, list)
        and len(indices) == len(paths)
        and all(type(index) is int for index in indices)
        and _valid_samples(images, paths, indices)
        and isinstance(resolutions, list)
        and len(resolutions) == len(paths)
        and all(
            isinstance(size, list)
            and len(size) == 2
            and all(type(length) is int and length > 0 for length in size)
            for size in resolutions
        )
    )


def check_receipt(root: Path, split: str, clip_id: str, source: dict, key: str) -> dict | None:
    feature, path = _clip_paths(root, split, clip_id)
    if not path.exists() and not path.is_symlink():
        return None
    try:
        record = json.loads(path.read_text())
        if not _valid_receipt(record):
            raise ValueError("invalid receipt schema")
    except (ValueError, UnicodeError, OSError, TypeError):
        raise ValueError(f"{split}/{clip_id}: malformed receipt") from None
    if (
        record["encoder_key"] != key
        or record["source_hash"] != source["source_hash"]
        or record["paths"] != source["paths"]
        or record["source_indices"] != source["source_indices"]
        or record["sampled_images"] != source["sampled_images"]
        or (source["resolutions"] and record["resolutions"] != source["resolutions"])
    ):
        raise ValueError(f"{split}/{clip_id}: incompatible receipt (version/source changed)")
    try:
        if not feature.is_file() or sha256_file(feature) != record["feature_hash"]:
            raise ValueError("hash mismatch")
    except (ValueError, OSError):
        raise ValueError(f"{split}/{clip_id}: feature hash mismatch") from None
    try:
        array = np.load(feature, allow_pickle=False, mmap_mode="r")
        if (
            array.shape != (len(source["paths"]), 2048)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError("invalid shape/dtype/finiteness")
    except (ValueError, OSError, EOFError):
        raise ValueError(f"{split}/{clip_id}: feature shape/dtype/finiteness mismatch") from None
    return record


def publish_clip(
    root: Path, split: str, clip_id: str, source: dict, key: str, array: np.ndarray
) -> dict:
    feature, receipt_path = _clip_paths(root, split, clip_id)
    if not _valid_samples(
        source.get("sampled_images"), source.get("paths"), source.get("source_indices")
    ):
        raise ValueError(f"{split}/{clip_id}: invalid sampled_images")
    if check_receipt(root, split, clip_id, source, key) is not None:
        raise ValueError(f"{split}/{clip_id}: validated version already exists")
    if array.shape[0] != len(source["paths"]) or len(source["resolutions"]) != len(array):
        raise ValueError(f"{split}/{clip_id}: source/feature row mismatch")
    record = dict(
        encoder_key=key,
        source_hash=source["source_hash"],
        paths=source["paths"],
        source_indices=source["source_indices"],
        sampled_images=source["sampled_images"],
        resolutions=source["resolutions"],
        feature_hash="0" * 64,
    )
    if not _valid_receipt(record):
        raise ValueError(f"{split}/{clip_id}: invalid receipt metadata")
    atomic_array(feature, array)
    record["feature_hash"] = sha256_file(feature)
    _clip_paths(root, split, clip_id)
    atomic_json(receipt_path, record)
    check_receipt(root, split, clip_id, source, key)
    return record

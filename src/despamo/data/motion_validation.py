"""Consumer admission with identical byte checks and one identity digest per scan.

Generator files stay untouched: completed packages bind those exact source bytes.
Consumer checks the same receipt protocol, every source and finite array. Shared
ancestor checks and the 5MB identity digest are reused within one scan, while
metadata signatures are checked again before returning the admitted descriptor.
"""

import hashlib
import io
import os
import stat
from dataclasses import asdict
from pathlib import Path

import numpy as np

from despamo.appearance.provenance import digest, file_hash
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.motion_timing import build_motion_timing, validate_motion_timing
from tools.motion import package as producer
from tools.motion.records import source_identity, window_rows


class _Files:
    """One scan, with canonical ancestor checks reused and rechecked at the end."""

    def __init__(self):
        self.parents: dict[Path, tuple] = {}
        self.leaves: dict[Path, tuple] = {}

    @staticmethod
    def signature(s):
        return s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns

    def regular(self, path: Path):
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("absolute canonical file path required")
        for parent in path.parents:
            if parent not in self.parents:
                s = parent.lstat()
                if not stat.S_ISDIR(s.st_mode):
                    raise ValueError("symlink/non-directory ancestor")
                self.parents[parent] = (s.st_dev, s.st_ino, s.st_mode)
        s = path.lstat()
        if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
            raise ValueError("unique regular file required")
        self.leaves[path] = self.signature(s)
        return s

    def read(self, path: Path) -> bytes:
        before = self.regular(path)
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(fd, "rb") as handle:
            if self.signature(os.fstat(handle.fileno())) != self.signature(before):
                raise ValueError("file changed during admission")
            data = handle.read()
            if self.signature(os.fstat(handle.fileno())) != self.signature(before):
                raise ValueError("file changed during read")
        return data

    def finish(self):
        for path, sig in self.parents.items():
            s = path.lstat()
            if (s.st_dev, s.st_ino, s.st_mode) != sig:
                raise ValueError("ancestor changed during admission")
        for path, sig in self.leaves.items():
            if self.signature(path.lstat()) != sig:
                raise ValueError("file changed during admission")


class _Admission(dict):
    def __init__(self, descriptor: dict, files: _Files, spatial: dict):
        super().__init__(descriptor)
        self._files = files
        self._descriptor_hash = digest(descriptor)
        self._spatial_hash = digest(spatial)


def reuse_motion_admission(package: dict, root: Path, spatial: dict) -> dict:
    """Only an in-process full scan has reusable proof; plain JSON always scans."""
    if not isinstance(package, _Admission):
        return validate_motion_package(root, spatial, verify_features=True)
    if (package["root"] != str(root) or digest(package) != package._descriptor_hash
            or digest(spatial) != package._spatial_hash):
        raise ValueError("admitted package/source mapping changed")
    package._files.finish()
    return package


def _clip(clip: str, spatial: dict, receipt: dict, root: Path, featurekey: str,
          source_root: Path, files: _Files, bindings: dict) -> None:
    count = spatial["frame_count"]
    indices = list(range(count))
    rows, windows = receipt["source_rows"], receipt["windows"]
    if (not isinstance(clip, str) or clip in {"", ".", ".."} or "/" in clip or "\\" in clip
            or digest(spatial["source_indices"]) != digest(indices)
            or len(rows) != count or digest([r["index"] for r in rows]) != digest(indices)
            or [r["path"] for r in rows] != spatial["paths"]
            or digest(windows) != digest(window_rows(count))):
        raise ValueError("actual source/window order mismatch")
    for row in rows:
        path = Path(row["path"])
        actual = source_root / path
        if (set(row) != {"index", "path", "sha256"} or ".." in path.parts
                or path.as_posix() != row["path"] or source_root not in actual.parents):
            raise ValueError("foreign or aliased source path")
        sha = hashlib.sha256(files.read(actual)).hexdigest()
        if sha != row["sha256"]:
            raise ValueError("consumed source bytes changed")
        bindings[str(actual)] = sha
    source_hash = source_identity(tuple(Path(r["path"]) for r in rows),
                                  tuple(r["sha256"] for r in rows))
    if source_hash != spatial["source_hash"]:
        raise ValueError("source bytes differ from spatial receipt")
    array_path = root / "train" / f"{clip}.npy"
    raw = files.read(array_path)
    array = np.load(io.BytesIO(raw), allow_pickle=False)
    if (array.shape != (len(windows), 1024) or array.dtype != np.float32
            or not np.isfinite(array).all()):
        raise ValueError("invalid motion shape/dtype/finiteness")
    sha = hashlib.sha256(raw).hexdigest()
    expected = dict(schema_version=1, origin="new_extraction_v1", clip_id=clip,
                    featurekey=featurekey, source_hash=source_hash, frame_count=count,
                    source_indices=indices, source_rows=rows, windows=windows,
                    array={"path": f"train/{clip}.npy", "shape": list(array.shape),
                           "dtype": "float32", "sha256": sha})
    if digest(expected) != digest(receipt):
        raise ValueError("generated clip receipt mismatch")
    bindings[str(array_path)] = sha


def validate_motion_package(root: Path, expected_spatial: dict, *,
                            verify_features: bool = True) -> dict:
    """Reconstruct the full graph; never skip hashes, even with the optional flag."""
    try:
        identity = producer._read(root / "complete/identity.json")
        marker = producer._read(root / "complete/completion.json")
        producer._root(root, identity)
        featurekey = digest(identity)
        bindings = producer._evidence(identity)
        if producer._read(Path(identity["inputs"]["frame_rows"]["path"])) != expected_spatial:
            raise ValueError("spatial frame mapping differs from pinned source")
        clips = expected_spatial["clips"]
        if not clips:
            raise ValueError("empty package")
        run_path = root / "generating-run.json"
        run = producer._read(run_path)
        expected_run = {"schema_version": 1, "origin": "new_extraction_v1",
                        "featurekey": featurekey,
                        "authorization": {"extraction_cap_seconds": 18000,
                                          "training_cap_seconds": 64800, "cap_transfer": False},
                        "code": identity["code"], "clips": {}}
        files = _Files()
        source_root = Path(identity["inputs"]["frame_root"])
        if not source_root.is_absolute() or source_root != source_root.resolve():
            raise ValueError("canonical source root required")
        receipts, records = {}, []
        for clip in sorted(clips):
            receipt_path = root / "receipts/train" / f"{clip}.json"
            receipt = producer._read(receipt_path)
            _clip(clip, clips[clip], receipt, root, featurekey, source_root, files, bindings)
            receipts[clip] = receipt
            records.append(FeatureRecord(clip, "train", receipt["array"]["path"],
                                          len(receipt["windows"]), 1024, "float32"))
            receipt_hash = file_hash(receipt_path)
            expected_run["clips"][clip] = receipt_hash
            bindings[str(receipt_path)] = receipt_hash
        if digest(run) != digest(expected_run):
            raise ValueError("actual generating-run ledger/authorization mismatch")
        for directory, suffix in ((root / "train", ".npy"), (root / "receipts/train", ".json")):
            if {p.name for p in directory.iterdir()} != {f"{clip}{suffix}" for clip in clips}:
                raise ValueError("foreign clip artifacts")
        bindings[str(run_path)] = file_hash(run_path)
        manifest = FeatureManifest(1, "MCG-NJU/videomae-large", 1024, tuple(records))
        payload = {"schema_version": 1, "encoder": manifest.encoder, "expected_dim": 1024,
                   "records": [asdict(r) for r in manifest.records]}
        ledger = {"schema_version": 1, "origin": "new_extraction_v1",
                  "featurekey": featurekey, "clips": receipts}
        provenance = producer._provenance(root, identity, expected_spatial, payload)
        timing = build_motion_timing(expected_spatial, payload, provenance)
        for name, value in zip(producer.ARTIFACTS,
                               (identity, payload, ledger, provenance, timing), strict=True):
            path = root / "complete" / name
            if digest(producer._read(path)) != digest(value):
                raise ValueError(f"complete artifact content mismatch: {name}")
            bindings[str(path)] = file_hash(path)
        validate_motion_timing(timing, expected_spatial, payload)
        bindings = dict(sorted(bindings.items()))
        expected = {"schema_version": 1, "package_policy": producer.POLICY, "root": str(root),
                    "identity_hash": featurekey, "clip_ids": sorted(clips), "bindings": bindings}
        if digest(marker) != digest(expected):
            raise ValueError("completion requires full independently verified byte inventory")
        if {p.name for p in (root / "complete").iterdir()} != {*producer.ARTIFACTS,
                                                             "completion.json"}:
            raise ValueError("foreign complete artifacts")
        descriptor = producer._descriptor(root, identity, sorted(clips), bindings)
        descriptor = producer._bind_completion(descriptor)
        for name in descriptor["bindings"]:
            path = Path(name)
            if path not in files.leaves:
                files.regular(path)
        files.finish()
        return _Admission(descriptor, files, expected_spatial)
    except (OSError, KeyError, TypeError, EOFError) as exc:
        raise ValueError(f"invalid motion package: {exc}") from exc

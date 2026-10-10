"""Fail-closed complete-package publisher and independent CPU byte validator."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from despamo.appearance.provenance import atomic_json, digest, file_hash
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.motion_timing import CONVENTION, build_motion_timing, validate_motion_timing
from tools.motion.records import make_clip_receipt, make_motion_identity, regular_path

POLICY = "despamo-motion-features-v1"
ARTIFACTS = ("identity.json", "manifest.json", "source-window-records.json",
             "extraction-provenance.json", "motion-timing.json")


def _read(path: Path) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    value = json.loads(regular_path(path).read_text(), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("JSON object required")
    return value


def _root(root: Path, identity: dict) -> None:
    checkout = Path(__file__).resolve().parents[2]
    if (not root.is_absolute() or root != root.resolve()
            or any(p.is_symlink() for p in (root, *root.parents))
            or root == checkout or checkout in root.parents
            or root.name != f"motion-features-{digest(identity)}"):
        raise ValueError("foreign/aliased motion package root")
    rebuilt = make_motion_identity(*(identity[name] for name in
                                    ("inputs", "model", "processor", "environment",
                                     "code", "hardware")))
    if digest(rebuilt) != digest(identity):
        raise ValueError("identity schema mismatch")


def _evidence(identity: dict) -> dict[str, str]:
    bindings = {}
    entries = [identity["inputs"]["frame_rows"], identity["inputs"]["annotation"],
               *identity["code"].values(), identity["environment"]["lock"]]
    def collect(value):
        if isinstance(value, dict):
            if "path" in value or "sha256" in value:
                entries.append(value)
            else:
                for item in value.values():
                    collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    # Required entries above are checked separately; collect additional model/config bytes.
    extras = {k: v for k, v in identity["environment"].items() if k != "lock"}
    collect(extras)
    collect(identity["hardware"])
    for item in entries:
        if not isinstance(item, dict) or set(item) != {"path", "sha256"}:
            raise ValueError("explicit evidence path/hash required")
        path = regular_path(Path(item["path"]))
        if file_hash(path) != item["sha256"] or str(path) in bindings:
            raise ValueError("evidence hash mismatch/alias")
        bindings[str(path)] = item["sha256"]
    return bindings


def _generation(root: Path, identity: dict, frames: dict) -> tuple[dict, dict, dict]:
    """Rebuild all ledger facts from pinned inputs and generated clip receipts."""
    _root(root, identity)
    bindings = _evidence(identity)
    if _read(Path(identity["inputs"]["frame_rows"]["path"])) != frames:
        raise ValueError("spatial frame mapping differs from pinned source")
    clips = frames["clips"]
    if not clips:
        raise ValueError("empty package")
    run_path = root / "generating-run.json"
    run = _read(run_path)
    expected_run = {
        "schema_version": 1, "origin": "new_extraction_v1", "featurekey": digest(identity),
        "authorization": {"extraction_cap_seconds": 18000, "training_cap_seconds": 64800,
                          "cap_transfer": False},
        "code": identity["code"], "clips": {},
    }
    receipts, records = {}, []
    for clip in sorted(clips):
        receipt_path = root / "receipts/train" / f"{clip}.json"
        receipt = _read(receipt_path)
        array_path = root / "train" / f"{clip}.npy"
        expected = make_clip_receipt(clip, clips[clip], receipt["source_rows"],
                                     tuple(receipt["windows"]), array_path, identity)
        if digest(expected) != digest(receipt):
            raise ValueError("generated clip receipt mismatch")
        receipts[clip] = receipt
        records.append(FeatureRecord(clip, "train", receipt["array"]["path"],
                                     len(receipt["windows"]), 1024, "float32"))
        expected_run["clips"][clip] = file_hash(receipt_path)
        for path in (array_path, receipt_path):
            bindings[str(regular_path(path))] = file_hash(path)
        for row in receipt["source_rows"]:
            path = Path(identity["inputs"]["frame_root"]) / row["path"]
            bindings[str(regular_path(path))] = file_hash(path)
    if digest(run) != digest(expected_run):
        raise ValueError("actual generating-run ledger/authorization mismatch")
    for directory, suffix in ((root / "train", ".npy"), (root / "receipts/train", ".json")):
        if {p.name for p in directory.iterdir()} != {f"{clip}{suffix}" for clip in clips}:
            raise ValueError("foreign clip artifacts")
    bindings[str(run_path)] = file_hash(run_path)
    manifest = FeatureManifest(1, "MCG-NJU/videomae-large", 1024, tuple(records))
    manifest_payload = {"schema_version": 1, "encoder": manifest.encoder,
                        "expected_dim": 1024, "records": [asdict(r) for r in manifest.records]}
    ledger = {"schema_version": 1, "origin": "new_extraction_v1",
              "featurekey": digest(identity), "clips": receipts}
    return manifest_payload, ledger, bindings


def _provenance(root: Path, identity: dict, frames: dict, manifest: dict, *,
                manifest_hash: str | None = None) -> dict:
    complete = root / "complete"
    run_path = root / "generating-run.json"
    return {
        "schema_version": 1, "status": "original_receipt", "origin": "new_extraction_v1",
        **CONVENTION, "physical_split": "train", "spatial_crop_mode": "full",
        "input_sha256": {"frame_rows": digest(frames), "motion_manifest": digest(manifest)},
        "file_sha256": {"frame_rows": identity["inputs"]["frame_rows"]["sha256"],
                        "annotation": identity["inputs"]["annotation"]["sha256"],
                        "motion_manifest": (file_hash(complete / "manifest.json")
                                            if manifest_hash is None else manifest_hash),
                        "extractor": identity["code"]["extractor"]["sha256"],
                        "helpers": identity["code"]["helpers"]["sha256"]},
        "evidence": [{"path": str(run_path), "sha256": file_hash(run_path),
                      "claims": ["historical_window_parameters", "motion_row_order",
                                 "source_frame_order"]}],
        "clips": {r["clip_id"]: {"frame_count": frames["clips"][r["clip_id"]]["frame_count"],
                                 "source_hash": frames["clips"][r["clip_id"]]["source_hash"],
                                 "motion_record_sha256": digest(r)} for r in manifest["records"]},
    }


def _descriptor(root: Path, identity: dict, clips: list[str], bindings: dict) -> dict:
    result: dict = dict(package_policy=POLICY, root=str(root), identity=identity,
                  identity_hash=digest(identity), clip_ids=clips, bindings=bindings)
    for name, path in (
        ("manifest", root / "complete/manifest.json"),
        ("timing", root / "complete/motion-timing.json"),
        ("annotation", Path(identity["inputs"]["annotation"]["path"])),
        ("spatial_frame_rows", Path(identity["inputs"]["frame_rows"]["path"])),
    ):
        result[f"{name}_path"] = str(path)
        result[f"{name}_hash"] = bindings[str(path)]
    result["completion"] = str(root / "complete/completion.json")
    result["completion_hash"] = bindings.get(result["completion"])
    return result


def publish_motion_package(root: Path, identity: dict, frame_rows: dict,
                           annotation: Path, *,
                           before_commit: Callable[[], None] | None = None) -> dict:
    """Verify generator inputs first; atomically publish completion strictly last."""
    try:
        if str(annotation) != identity["inputs"]["annotation"]["path"]:
            raise ValueError("annotation identity mismatch")
        manifest, ledger, _ = _generation(root, identity, frame_rows)
        complete = root / "complete"
        if complete.exists() or complete.is_symlink():
            raise ValueError("complete directory already exists; inspect before recovery")
        atomic_json(complete / "identity.json", identity)
        atomic_json(complete / "manifest.json", manifest)
        atomic_json(complete / "source-window-records.json", ledger)
        provenance = _provenance(root, identity, frame_rows, manifest)
        timing = build_motion_timing(frame_rows, manifest, provenance)
        validate_motion_timing(timing, frame_rows, manifest)
        atomic_json(complete / "extraction-provenance.json", provenance)
        atomic_json(complete / "motion-timing.json", timing)
        return _commit_completion(root, identity, frame_rows, before_commit)
    except TimeoutError:
        raise
    except (OSError, KeyError, TypeError, EOFError) as exc:
        raise ValueError(f"invalid motion package: {exc}") from exc


def _check_artifacts(root: Path, identity: dict, frames: dict) -> dict:
    manifest, ledger, bindings = _generation(root, identity, frames)
    complete = root / "complete"
    provenance = _provenance(root, identity, frames, manifest)
    timing = build_motion_timing(frames, manifest, provenance)
    expected = (identity, manifest, ledger, provenance, timing)
    for name, value in zip(ARTIFACTS, expected, strict=True):
        path = complete / name
        if digest(_read(path)) != digest(value):
            raise ValueError(f"complete artifact content mismatch: {name}")
        bindings[str(path)] = file_hash(path)
    validate_motion_timing(timing, frames, manifest)
    return dict(sorted(bindings.items()))


def validate_motion_package(root: Path, expected_spatial: dict, *,
                            verify_features: bool = True) -> dict:
    """Independently rebuild inventory; even fast admission verifies feature bytes.

    Currently both modes verify finite arrays too. The flag never bypasses hashes.
    """
    try:
        marker_path = root / "complete/completion.json"
        marker = _read(marker_path)
        identity = _read(root / "complete/identity.json")
        _, descriptor = _completion_candidate(root, identity, expected_spatial, marker)
        return _bind_completion(descriptor)
    except (OSError, KeyError, TypeError, EOFError) as exc:
        raise ValueError(f"invalid motion package: {exc}") from exc


def _completion_candidate(root: Path, identity: dict, frames: dict,
                          marker: dict | None = None) -> tuple[dict, dict]:
    """Shared full admission of immutable bytes, directory inventory and marker facts."""
    bindings = _check_artifacts(root, identity, frames)
    expected = dict(schema_version=1, package_policy=POLICY, root=str(root),
                    identity_hash=digest(identity), clip_ids=sorted(frames["clips"]),
                    bindings=bindings)
    if marker is not None and digest(marker) != digest(expected):
        raise ValueError("completion requires full independently verified byte inventory")
    names = {*ARTIFACTS, "completion.json"} if marker is not None else set(ARTIFACTS)
    if {p.name for p in (root / "complete").iterdir()} != names:
        raise ValueError("foreign complete artifacts")
    # No completion hash exists before commit; never invent one for pending descriptors.
    return expected, _descriptor(root, identity, sorted(frames["clips"]), bindings)


def _bind_completion(descriptor: dict) -> dict:
    """Bind actual committed marker bytes; no second full admission pass."""
    checksum = file_hash(regular_path(Path(descriptor["completion"])))
    descriptor["bindings"][descriptor["completion"]] = checksum
    descriptor["completion_hash"] = checksum
    return descriptor


def _commit_completion(root: Path, identity: dict, frames: dict,
                       before_commit: Callable[[], None] | None) -> dict:
    marker, descriptor = _completion_candidate(root, identity, frames)
    if before_commit is not None:
        before_commit()
    atomic_json(root / "complete/completion.json", marker)
    return _bind_completion(descriptor)


def _json_bytes(value: dict) -> bytes:
    """Exact bytes emitted by neutral atomic_json, without publishing a file."""
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def resume_motion_package_publication(root: Path, identity: dict, frame_rows: dict,
                                     annotation: Path, *,
                                     before_commit: Callable[[], None] | None = None) -> dict:
    """Fill verified missing metadata only; caller must exclusively own/lock root.

    Complete packages are independently admitted without rewriting. Partial files
    must match expected neutral JSON bytes; conflicting bytes are never replaced.
    """
    try:
        if str(annotation) != identity["inputs"]["annotation"]["path"]:
            raise ValueError("annotation identity mismatch")
        manifest, ledger, _ = _generation(root, identity, frame_rows)
        provenance = _provenance(
            root, identity, frame_rows, manifest,
            manifest_hash=hashlib.sha256(_json_bytes(manifest)).hexdigest(),
        )
        timing = build_motion_timing(frame_rows, manifest, provenance)
        validate_motion_timing(timing, frame_rows, manifest)
        expected: dict[str, dict] = dict(zip(
            ARTIFACTS, (identity, manifest, ledger, provenance, timing), strict=True
        ))
        complete = root / "complete"
        if complete.is_symlink() or (complete.exists() and not complete.is_dir()):
            raise ValueError("unsafe complete directory")
        existing = {p.name for p in complete.iterdir()} if complete.exists() else set()
        if existing - {*ARTIFACTS, "completion.json"}:
            raise ValueError("foreign partial complete artifacts")
        if "completion.json" in existing:
            return validate_motion_package(root, frame_rows)
        # Inspect every present artifact before the first write, including later metadata.
        for name in existing:
            path = complete / name
            if (digest(_read(path)) != digest(expected[name])
                    or path.read_bytes() != _json_bytes(expected[name])):
                raise ValueError(f"partial complete artifact conflict: {name}")
        for name, value in expected.items():
            if name not in existing:
                path = complete / name
                if path.exists() or path.is_symlink():
                    raise ValueError("partial publication changed under exclusive owner")
                atomic_json(path, value)
        return _commit_completion(root, identity, frame_rows, before_commit)
    except TimeoutError:
        raise
    except (OSError, KeyError, TypeError, EOFError) as exc:
        raise ValueError(f"invalid motion package recovery: {exc}") from exc

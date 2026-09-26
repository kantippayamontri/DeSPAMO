import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path

from tools.dinov3.encoder import FATAL_REASONS, extract
from tools.dinov3.frames import annotations, sample_indices, scan
from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json, check_receipt, publish_clip, writer

EXPECTED = {"train": 7096, "dev": 519, "test": 642}
EXPECTED_PNG = {"train": 827354, "dev": 55775, "test": 64627}


def _row(receipt: dict, count: int) -> dict:
    if receipt["source_indices"] != list(range(count)) or set(receipt["sampled_images"]) != {
        str(index) for index in sample_indices(count)
    }:
        raise ValueError("invalid source-to-row mapping")
    return dict(
        feature_hash=receipt["feature_hash"],
        source_indices=receipt["source_indices"],
        sampled_images=receipt["sampled_images"],
        frame_count=count,
        paths=receipt["paths"],
        source_hash=receipt["source_hash"],
        resolutions=receipt["resolutions"],
    )


def _record(split: str, clip_id: str, count: int) -> dict:
    return dict(
        clip_id=clip_id,
        split=split,
        path=f"{split}/{clip_id}.npy",
        length=count,
        width=2048,
        dtype="float32",
    )


def _totals(records: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = Counter()
    for record in records:
        totals[record["split"]] += record["length"]
    return dict(totals)


def validate_complete(
    frame_root: Path, output: Path, published: Path, clips: dict, key: str
) -> dict:
    failure_path = output / "failures.json"
    if not failure_path.is_file() or json.loads(failure_path.read_text()) != {}:
        raise ValueError("complete version requires zero recorded failures")
    manifest_path = published / "manifest.json"
    row_path = published / "frame_rows.json"
    manifest = json.loads(manifest_path.read_text())
    row_map = json.loads(row_path.read_text())
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema_version", "encoder", "expected_dim", "records"}
        or manifest["schema_version"] != 1
        or manifest["expected_dim"] != 2048
        or manifest["encoder"] != f"dinov3:{key}"
        or not isinstance(manifest["records"], list)
        or not isinstance(row_map, dict)
        or set(row_map) != {"spatial_manifest_hash", "encoder_key", "clips"}
        or row_map["encoder_key"] != key
        or row_map["spatial_manifest_hash"] != sha256_file(manifest_path)
        or not isinstance(row_map["clips"], dict)
    ):
        raise ValueError("complete manifest identity/hash mismatch")
    records = manifest["records"]
    try:
        counts = Counter(record["split"] for record in records)
        indexed = {(record["split"], record["clip_id"]): record for record in records}
        totals = _totals(records)
    except (KeyError, TypeError):
        raise ValueError("complete manifest records malformed") from None
    expected_ids = {(split, clip_id) for split, entries in clips.items() for clip_id in entries}
    if (
        counts != EXPECTED
        or len(records) != sum(EXPECTED.values())
        or len(indexed) != len(records)
        or set(indexed) != expected_ids
        or set(row_map["clips"]) != {clip_id for _, clip_id in expected_ids}
        or totals != EXPECTED_PNG
    ):
        raise ValueError("complete split/row-map/PNG count mismatch")
    for split, entries in clips.items():
        for clip_id, expected in entries.items():
            try:
                source = scan(frame_root, split, clip_id, expected)
                receipt = check_receipt(output, split, clip_id, source, key)
                if receipt is None:
                    raise ValueError("missing validated receipt")
                if (
                    scan(frame_root, split, clip_id, expected)["source_hash"]
                    != source["source_hash"]
                ):
                    raise ValueError("source changed during final validation")
                if row_map["clips"][clip_id] != _row(receipt, expected[0]) or indexed[
                    split, clip_id
                ] != _record(split, clip_id, expected[0]):
                    raise ValueError("frame-row/feature record mismatch")
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise ValueError(f"{split}/{clip_id}: {error}") from None
    return dict(completed=len(records), complete=True, counts=dict(counts))


def run(
    frame_root: Path,
    annotation_root: Path,
    clip_manifest: Path,
    output: Path,
    key: str,
    metadata: dict,
    model: object,
    device: str,
    *,
    only: tuple[str, str] | None = None,
) -> dict:
    clips = annotations(annotation_root, clip_manifest)
    if only is None:
        for split, entries in clips.items():
            directory = frame_root / split
            actual = (
                {path.name for path in directory.iterdir() if path.is_dir()}
                if directory.is_dir()
                else set()
            )
            if actual - set(entries):
                raise ValueError(f"{split}: unexpected source clip directories")
    with writer(output):
        version = output / "version.json"
        if version.exists() and json.loads(version.read_text()) != metadata:
            raise ValueError("encoder key/version metadata mismatch")
        complete = output / "complete"
        if complete.exists():
            if not complete.is_dir() or not version.is_file():
                raise ValueError("malformed complete version")
            return validate_complete(frame_root, output, complete, clips, key)
        if not version.exists():
            atomic_json(version, metadata)
        failure_path = output / "failures.json"
        failures = json.loads(failure_path.read_text()) if failure_path.exists() else {}
        if not isinstance(failures, dict):
            raise ValueError("malformed failures journal")
        rows = {}
        records = []
        completed = 0
        for split, entries in clips.items():
            for clip_id, expected in sorted(entries.items()):
                if only is not None and (split, clip_id) != only:
                    continue
                stage = "source"
                try:
                    source = scan(frame_root, split, clip_id, expected)
                    stage = "resume"
                    receipt = check_receipt(output, split, clip_id, source, key)
                    if receipt is None:
                        stage = "encode"
                        batch_kwargs = (
                            {"batch_size": metadata["inference_batch_size"]}
                            if "inference_batch_size" in metadata
                            else {}
                        )
                        array, resolutions = extract(
                            [frame_root / path for path in source["paths"]],
                            model,
                            device,
                            **batch_kwargs,
                        )
                        if (
                            scan(frame_root, split, clip_id, expected)["source_hash"]
                            != source["source_hash"]
                        ):
                            raise ValueError("source PNG changed during extraction")
                        source["resolutions"] = resolutions
                        stage = "publish"
                        receipt = publish_clip(output, split, clip_id, source, key, array)
                    if (
                        scan(frame_root, split, clip_id, expected)["source_hash"]
                        != source["source_hash"]
                    ):
                        raise ValueError("source PNG changed before row validation")
                    rows[clip_id] = _row(receipt, expected[0])
                    records.append(_record(split, clip_id, expected[0]))
                    failures.pop(f"{split}/{clip_id}", None)
                    completed += 1
                except (ValueError, OSError, RuntimeError, MemoryError) as error:
                    fatal = isinstance(error, (RuntimeError, MemoryError))
                    reason = str(error)
                    if fatal and reason not in FATAL_REASONS:
                        reason = "DINOv3 extraction runtime failure"
                    failures[f"{split}/{clip_id}"] = dict(stage=stage, reason=reason)
                    atomic_json(failure_path, failures)
                    if fatal:
                        raise RuntimeError(
                            f"{split}/{clip_id}: {reason}; extraction stopped"
                        ) from None
        atomic_json(failure_path, failures)
        if only is not None:
            if completed != 1 or failures.get(f"{only[0]}/{only[1]}"):
                raise ValueError("bounded extraction incomplete; see failures.json")
            return dict(completed=completed, complete=False)
        counts = Counter(record["split"] for record in records)
        if failures or counts != EXPECTED or len(rows) != sum(EXPECTED.values()):
            raise ValueError(
                f"extraction incomplete; completed={dict(counts)}, failures={len(failures)}"
            )
        if _totals(records) != EXPECTED_PNG:
            raise ValueError("extraction incomplete; PNG totals differ from approved baseline")
        manifest = dict(
            schema_version=1,
            encoder=f"dinov3:{key}",
            expected_dim=2048,
            records=sorted(records, key=lambda record: (record["split"], record["clip_id"])),
        )
        staging = Path(tempfile.mkdtemp(prefix=".complete-staging-", dir=output))
        try:
            atomic_json(staging / "manifest.json", manifest)
            row_map = dict(
                spatial_manifest_hash=sha256_file(staging / "manifest.json"),
                encoder_key=key,
                clips=rows,
            )
            atomic_json(staging / "frame_rows.json", row_map)
            try:
                result = validate_complete(frame_root, output, staging, clips, key)
            except ValueError as error:
                clip_id = str(error).split(":", 1)[0]
                if clip_id in {
                    f"{split}/{clip}" for split, entries in clips.items() for clip in entries
                }:
                    failures[clip_id] = dict(stage="final_validation", reason=str(error))
                    atomic_json(failure_path, failures)
                raise ValueError("extraction incomplete; final validation failed") from error
            if complete.exists():
                raise ValueError("complete version appeared during staging")
            os.replace(staging, complete)
            dirfd = os.open(output, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(dirfd)
            finally:
                os.close(dirfd)
            return result
        finally:
            if staging.exists():
                shutil.rmtree(staging)

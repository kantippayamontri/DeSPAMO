"""Explicit-root 1024-wide plain CLIP admission and synchronous batches."""

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from despamo.data.batch import PhoenixBatch, collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from tools.clip_plain import policy
from tools.clip_plain.features import feature_identity, package_root
from tools.dinov3.identity import digest, sha256_file


def admit_data(config: dict, *, verify_arrays: bool = True) -> dict:
    spatial = package_root(config)
    expected = feature_identity(config)
    if json.loads((spatial / "complete/identity.json").read_text()) != expected:
        raise ValueError("Plain feature identity mismatch")
    pointer = json.loads((Path(config["feature_base"]) / "package-result.json").read_text())
    if pointer["root"] != str(spatial) or pointer["manifest_sha256"] != sha256_file(
        spatial / "complete/manifest.json"
    ):
        raise ValueError("Complete package pointer mismatch")
    roots = dict(spatial=spatial, motion=Path(config["motion_root"]))
    manifests, hashes, sources, bindings = {}, {}, {}, {}
    for kind, root in roots.items():
        path = root / "complete/manifest.json"
        if root.is_symlink() or path.is_symlink():
            raise ValueError("Native independent feature package required")
        if kind == "motion" and sha256_file(path) != config["motion_manifest_sha256"]:
            raise ValueError("Unchanged motion manifest required")
        manifest = FeatureManifest.load(path)
        identity = json.loads((root / "complete/identity.json").read_text())
        if manifest.expected_dim != 1024 or manifest.encoder != digest(identity):
            raise ValueError("1024 feature dimension/encoder mismatch")
        manifests[kind], hashes[kind], sources[kind] = manifest, {}, {}
        for record in manifest.records:
            array_path = root / record.path
            row = json.loads(
                (root / "receipts" / record.split / f"{record.clip_id}.json").read_text()
            )
            if array_path.is_symlink() or row["identity_hash"] != digest(identity):
                raise ValueError("Feature receipt mismatch")
            pair = record.split, record.clip_id
            hashes[kind][pair] = row["feature_hash"]
            sources[kind][pair] = row["source_hash"], row["source_indices"], row["frame_count"]
            if verify_arrays:
                a = np.load(array_path, allow_pickle=False, mmap_mode="r")
                if (
                    sha256_file(array_path) != row["feature_hash"]
                    or a.shape != (record.length, 1024)
                    or a.dtype != np.float32
                    or not np.isfinite(a).all()
                ):
                    raise ValueError("Feature array contract changed")
        bindings[str(path)] = sha256_file(path)
        bindings[str(root / "complete/identity.json")] = sha256_file(
            root / "complete/identity.json"
        )
    if sources["spatial"] != sources["motion"]:
        raise ValueError("Spatial/motion frame provenance differs")
    datasets, ids, lookup, seen = {}, {}, {}, set()
    for split, count in policy.COUNTS.items():
        annotation = Path(config["annotations"]) / f"{split}_info_ml.npy"
        ds = Phoenix14T(
            annotation,
            split,
            roots["spatial"],
            roots["motion"],
            manifests["spatial"],
            manifests["motion"],
            spatial_crop_mode="full",
        )
        clips = tuple(r["fileid"] for r in ds.records)
        if len(clips) != count or len(set(clips)) != count or seen.intersection(clips):
            raise ValueError("Official split coverage/order mismatch")
        if count == {"train": 7096, "dev": 519, "test": 642}[split] and {
            r["signer"] for r in ds.records
        } != {f"Signer{i:02}" for i in range(1, 10)}:
            raise ValueError("Official signer coverage mismatch")
        seen.update(clips)
        for clip in clips:
            s = manifests["spatial"].require(split, clip)
            m = manifests["motion"].require(split, clip)
            if m.length != max(1, (s.length - 16) // 8 + 1):
                raise ValueError("Full-source timing mismatch")
        datasets[split], ids[split] = ds, clips
        lookup[split] = {clip: i for i, clip in enumerate(clips)}
        bindings[str(annotation)] = sha256_file(annotation)
    ordered = [(s, i) for s in policy.COUNTS for i in ids[s]]
    for manifest in manifests.values():
        if [(r.split, r.clip_id) for r in manifest.records] != ordered:
            raise ValueError("Extra/missing/reordered feature records")
    identity = dict(
        bindings=bindings,
        ordered_ids={s: digest(v) for s, v in ids.items()},
        feature_hashes_digest=digest(
            {k: {f"{s}/{i}": h for (s, i), h in rows.items()} for k, rows in hashes.items()}
        ),
    )
    return dict(
        roots=roots,
        datasets=datasets,
        ids=ids,
        lookup=lookup,
        feature_hashes=hashes,
        bindings=bindings,
        identity=identity,
    )


def batch_for(admission: dict, split: str, clip_ids: tuple[str, ...]) -> PhoenixBatch:
    if (
        not clip_ids
        or len(set(clip_ids)) != len(clip_ids)
        or not set(clip_ids).issubset(admission["ids"][split])
    ):
        raise ValueError("Batch IDs outside admitted split")
    for clip in clip_ids:
        for kind, root in admission["roots"].items():
            path = root / split / f"{clip}.npy"
            if sha256_file(path) != admission["feature_hashes"][kind][split, clip]:
                raise ValueError("Feature changed during batch loading")
    return collate_phoenix(
        [admission["datasets"][split][admission["lookup"][split][c]] for c in clip_ids]
    )


def eval_batches(admission: dict, *, max_batches: int | None = None) -> Iterator[PhoenixBatch]:
    if max_batches is not None and (type(max_batches) is not int or max_batches < 1):
        raise ValueError("Positive bounded batch count required")
    for clip in admission["ids"]["dev"][:max_batches]:
        yield batch_for(admission, "dev", (clip,))

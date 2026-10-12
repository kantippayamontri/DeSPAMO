"""Explicit CLIP spatial / unchanged motion roots; profile packages cannot train."""

import json
from pathlib import Path

import numpy as np

from despamo.data.batch import collate_phoenix
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from tools.benchmark.translation_data import ANNOTATIONS, MOTION
from tools.dinov3.identity import digest, sha256_file

MOTION_SHA = "023a054fe0d3b9e04b5c0472ef77ebb8d031130e604841d67acfe7d81f66fd03"


def admit_data(
    spatial_root: Path,
    spatial_manifest_sha256: str,
    verify_arrays: bool = True,
    *,
    profile_only: bool = False,
) -> dict:
    if spatial_root.is_symlink():
        raise ValueError("native spatial package required")
    path = spatial_root / "complete/manifest.json"
    if sha256_file(path) != spatial_manifest_sha256:
        raise ValueError("CLIP manifest hash changed")
    identity = json.loads((spatial_root / "complete/identity.json").read_text())
    if identity["policy"] != "official-frozen-clip-s2-spatial-v1":
        raise ValueError("frozen CLIP/S2 identity required")
    if profile_only != (identity.get("purpose") == "bounded_profile_smoke"):
        raise ValueError("profile package cannot be admitted as full training corpus")
    roots = dict(spatial=spatial_root, motion=MOTION)
    manifests, hashes, sources, bindings = {}, {}, {}, {}
    spatial_manifest = FeatureManifest.load(path)
    admitted = {(r.split, r.clip_id) for r in spatial_manifest.records}
    for kind, root, width, expected_sha in (
        ("spatial", spatial_root, 2048, spatial_manifest_sha256),
        ("motion", MOTION, 1024, MOTION_SHA),
    ):
        p = root / "complete/manifest.json"
        if sha256_file(p) != expected_sha:
            raise ValueError("feature metadata changed")
        manifest = FeatureManifest.load(p)
        encoder = json.loads((root / "complete/identity.json").read_text())
        if manifest.expected_dim != width or manifest.encoder != digest(encoder):
            raise ValueError("feature encoder/dimension mismatch")
        manifests[kind] = manifest
        bindings[str(p)] = expected_sha
        bindings[str(root / "complete/identity.json")] = sha256_file(
            root / "complete/identity.json"
        )
        hashes[kind], sources[kind] = {}, {}
        for record in manifest.records:
            pair = record.split, record.clip_id
            if pair not in admitted:
                continue
            receipt = json.loads(
                (root / "receipts" / record.split / f"{record.clip_id}.json").read_text()
            )
            feature = root / record.path
            if receipt["identity_hash"] != digest(encoder) or feature.is_symlink():
                raise ValueError("feature receipt identity mismatch")
            if verify_arrays:
                array = np.load(feature, allow_pickle=False, mmap_mode="r")
                if (
                    sha256_file(feature) != receipt["feature_hash"]
                    or array.shape != (record.length, width)
                    or array.dtype != np.float32
                    or not np.isfinite(array).all()
                ):
                    raise ValueError("feature array shape/dtype/hash mismatch")
            hashes[kind][pair] = receipt["feature_hash"]
            sources[kind][pair] = (
                receipt["source_hash"],
                receipt["frame_count"],
                receipt["source_indices"],
            )
    if sources["spatial"] != sources["motion"]:
        raise ValueError("spatial/motion source rows changed or mismatched")
    datasets, ids, lookup, seen = {}, {}, {}, set()
    allowed = set()
    for split, count in (("train", 7096), ("dev", 519), ("test", 642)):
        annotation = ANNOTATIONS / f"{split}_info_ml.npy"
        ds = Phoenix14T(
            annotation,
            split,
            spatial_root,
            MOTION,
            manifests["spatial"],
            manifests["motion"],
            spatial_crop_mode="full",
        )
        all_ids = tuple(r["fileid"] for r in ds.records)
        if len(all_ids) != count or len(set(all_ids)) != count or seen.intersection(all_ids):
            raise ValueError("official split ID/count mismatch")
        if {r["signer"] for r in ds.records} != {f"Signer{i:02}" for i in range(1, 10)}:
            raise ValueError("official signer coverage mismatch")
        current = tuple(i for i in all_ids if (split, i) in admitted)
        if not profile_only and current != all_ids:
            raise ValueError("complete official spatial coverage required")
        for clip in current:
            s = manifests["spatial"].require(split, clip)
            m = manifests["motion"].require(split, clip)
            if m.length != max(1, (s.length - 16) // 8 + 1):
                raise ValueError("feature timing mismatch")
        datasets[split], ids[split] = ds, current
        allowed.update((split, i) for i in all_ids)
        lookup[split] = {i: n for n, i in enumerate(all_ids)}
        seen.update(all_ids)
        bindings[str(annotation)] = sha256_file(annotation)
    if not admitted.issubset(allowed):
        raise ValueError("extra spatial IDs outside official corpus")
    if not profile_only and [(r.split, r.clip_id) for r in spatial_manifest.records] != [
        (s, i) for s in ("train", "dev", "test") for i in ids[s]
    ]:
        raise ValueError("spatial manifest order differs from official corpus")
    return dict(
        roots=roots,
        datasets=datasets,
        ids=ids,
        lookup=lookup,
        bindings=bindings,
        feature_hashes=hashes,
        profile_only=profile_only,
        identity=dict(
            bindings=bindings,
            ordered_ids={s: digest(v) for s, v in ids.items()},
            feature_hashes_digest=digest(
                {k: {f"{s}/{i}": h for (s, i), h in v.items()} for k, v in hashes.items()}
            ),
        ),
    )


def batch_for(admission: dict, split: str, clip_ids: tuple[str, ...]):
    if len(set(clip_ids)) != len(clip_ids) or not set(clip_ids).issubset(admission["ids"][split]):
        raise ValueError("batch IDs outside admitted split")
    samples = []
    for clip in clip_ids:
        for kind, root in admission["roots"].items():
            p = root / split / f"{clip}.npy"
            if sha256_file(p) != admission["feature_hashes"][kind][split, clip]:
                raise ValueError("feature changed during batch loading")
        samples.append(admission["datasets"][split][admission["lookup"][split][clip]])
    return collate_phoenix(samples)


def eval_batches(admission: dict, split: str, batch_size: int = 4):
    ids = admission["ids"][split]
    for offset in range(0, len(ids), batch_size):
        yield batch_for(admission, split, ids[offset : offset + batch_size])

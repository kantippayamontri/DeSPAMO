from pathlib import Path

import numpy as np

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T
from tools.dinov3.identity import digest, sha256_file


def annotation_ids(path: Path, count: int) -> tuple[tuple[str, ...], str]:
    before = sha256_file(path)
    raw = np.load(path, allow_pickle=True).item()
    if not isinstance(raw, dict):
        raise ValueError("Original complete split IDs required")
    keys = {key for key in raw if type(key) is int}
    metadata_keys = {key for key in raw if type(key) is not int}
    if (
        keys != set(range(count))
        or metadata_keys - {"prefix"}
        or ("prefix" in raw and not isinstance(raw["prefix"], str))
    ):
        raise ValueError("Original complete split IDs required")
    names: list[str] = []
    for key in range(count):
        name = raw[key].get("fileid") if isinstance(raw[key], dict) else None
        if (
            not isinstance(name, str)
            or not name.strip()
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("Original complete split IDs required")
        names.append(name)
    if len(set(names)) != count:
        raise ValueError("Original complete split IDs required")
    if before != sha256_file(path):
        raise ValueError("Annotation changed during split IDs admission")
    return tuple(names), before


def require_disjoint(ids: dict[str, tuple[str, ...]]) -> None:
    seen: set[str] = set()
    for names in ids.values():
        if seen.intersection(names):
            raise ValueError("Dataset split overlap")
        seen.update(names)


def feature_hash(root: Path, record: FeatureRecord, width: int) -> str:
    path = root / record.path
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError(f"Invalid original feature path: {path}")
    before = sha256_file(path)
    array = np.load(path, allow_pickle=False, mmap_mode="r")
    if (
        record.width != width
        or array.shape != (record.length, width)
        or record.length > 512
        or str(array.dtype) != record.dtype
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"Invalid original feature: {path}")
    if before != sha256_file(path):
        raise ValueError(f"Original feature changed during admission: {path}")
    return before


def admit_data(config: dict) -> dict:
    data = config["data"]
    roots = {kind: Path(data[f"{kind}_root"]) for kind in ["spatial", "motion"]}
    manifests = {kind: FeatureManifest.load(Path(data[f"{kind}_manifest"])) for kind in roots}
    datasets, ids, lookup, hashes = {}, {}, {}, {kind: {} for kind in roots}
    bindings = {
        data[f"{kind}_manifest"]: sha256_file(Path(data[f"{kind}_manifest"])) for kind in roots
    }
    for split, count in [("train", 7096), ("dev", 519), ("test", 642)]:
        annotation = Path(data["annotation_root"]) / f"{split}_info_ml.npy"
        names, bindings[str(annotation)] = annotation_ids(annotation, count)
        ids[split] = names
        if split == "test":
            continue
        ds = Phoenix14T(
            annotation,
            split,
            roots["spatial"],
            roots["motion"],
            manifests["spatial"],
            manifests["motion"],
            max_frame_len=512,
            spatial_crop_mode="full",
        )
        if tuple(r["fileid"] for r in ds.records) != names or bindings[
            str(annotation)
        ] != sha256_file(annotation):
            raise ValueError("Original complete split IDs required")
        datasets[split], ids[split] = ds, names
        lookup[split] = {clip: index for index, clip in enumerate(names)}
        for kind, width in [("spatial", 2048), ("motion", 1024)]:
            hashes[kind][split] = {}
            for clip in names:
                record = manifests[kind].require(split, clip)
                hashes[kind][split][clip] = feature_hash(roots[kind], record, width)
    require_disjoint(ids)
    if any(sha256_file(Path(path)) != expected for path, expected in bindings.items()):
        raise ValueError("Data bindings changed during admission")
    return {
        "datasets": datasets,
        "ids": ids,
        "lookup": lookup,
        "roots": roots,
        "feature_hashes": hashes,
        "bindings": bindings,
        "ordered_ids": {split: digest(names) for split, names in ids.items()},
    }

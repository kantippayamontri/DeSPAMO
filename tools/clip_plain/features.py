"""CPU-only, parent-bound global224 cache derivation; no encoder execution."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from tools.benchmark.feature_package import finish_package, verify_clip, verify_package, write_clip
from tools.clip_plain.policy import FEATURE_POLICY, load_config, source_hashes
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.official_runtime import require_official_storage
from tools.dinov3.storage import atomic_json, writer


def global_columns(values: np.ndarray) -> np.ndarray:
    if (
        values.ndim != 2
        or not len(values)
        or values.shape[1] != 2048
        or values.dtype != np.float32
        or not np.isfinite(values).all()
    ):
        raise ValueError("Finite FP32 parent [frames,2048] required")
    return np.array(values[:, :1024], dtype=np.float32, order="C", copy=True)


def parent_metadata(config: dict) -> tuple[dict, dict]:
    root = Path(config["parent_root"])
    for name, expected in [
        ("manifest.json", config["parent_manifest_sha256"]),
        ("identity.json", config["parent_identity_sha256"]),
    ]:
        if sha256_file(root / "complete" / name) != expected:
            raise ValueError("Pinned parent metadata changed")
    identity = json.loads((root / "complete/identity.json").read_text())
    manifest = json.loads((root / "complete/manifest.json").read_text())
    encoder = identity["encoder"]
    if (
        identity["policy"] != "official-frozen-clip-s2-spatial-v1"
        or identity["kind"] != "spatial"
        or identity["width"] != 2048
        or manifest["expected_dim"] != 2048
        or manifest["encoder"] != digest(identity)
        or encoder["scales"] != [224, 448]
        or encoder["tokens"] != config["tokens"]
        or encoder["revision"] != config["revision"]
        or encoder["model"] != config["model"]
        or encoder["dtype"] != "float32"
        or encoder["tile_order"] != "row_major_tile_then_frame"
    ):
        raise ValueError("Parent global-first layout/model/identity mismatch")
    for p, sha in identity["inputs"]["source"].items():
        if sha256_file(Path(p)) != sha:
            raise ValueError("Parent producer source changed")
    return identity, manifest


def parent_inventory(config: dict) -> dict:
    _, manifest = parent_metadata(config)
    root = Path(config["parent_root"])
    result: dict = {}
    for row in manifest["records"]:
        split, clip = row["split"], row["clip_id"]
        receipts = result.setdefault(split, {})
        if clip in receipts or row["path"] != f"{split}/{clip}.npy" or row["width"] != 2048:
            raise ValueError("Invalid parent inventory")
        receipts[clip] = json.loads((root / "receipts" / split / f"{clip}.json").read_text())
    return result


def feature_identity(config: dict) -> dict:
    identity, _ = parent_metadata(config)
    return dict(
        policy=FEATURE_POLICY,
        kind="spatial",
        width=1024,
        dtype="float32",
        encoder={**identity["encoder"], "scales": [224]},
        derivation=dict(
            columns=[0, 1024],
            parent_root=config["parent_root"],
            parent_identity_sha256=config["parent_identity_sha256"],
            parent_manifest_sha256=config["parent_manifest_sha256"],
        ),
        producer_source=source_hashes(),
    )


def package_root(config: dict) -> Path:
    return Path(config["feature_base"]) / f"spatial-{digest(feature_identity(config))}"


def derive_clip(config: dict, split: str, clip_id: str) -> dict:
    parent = Path(config["parent_root"])
    identity, _ = parent_metadata(config)
    receipt_path = parent / "receipts" / split / f"{clip_id}.json"
    meta = json.loads(receipt_path.read_text())
    paths = [Path(p) for p in meta["paths"]]
    verify_clip(parent, split, clip_id, paths, identity)
    receipt_sha = sha256_file(receipt_path)
    path = parent / split / f"{clip_id}.npy"
    source_sha = sha256_file(path)
    values = global_columns(np.load(path, allow_pickle=False))
    if sha256_file(path) != source_sha or sha256_file(receipt_path) != receipt_sha:
        raise ValueError("Parent changed during conversion")
    child_root = package_root(config)
    child_identity = feature_identity(config)
    row = write_clip(
        child_root,
        split,
        clip_id,
        paths,
        values,
        child_identity,
        dict(
            parent_feature_sha256=source_sha,
            parent_receipt_sha256=receipt_sha,
            parent_path=str(path),
            columns=[0, 1024],
        ),
    )
    if (
        row["extra"].get("parent_feature_sha256") != source_sha
        or row["extra"].get("parent_receipt_sha256") != receipt_sha
    ):
        raise ValueError("Child parent binding changed")
    np.testing.assert_array_equal(
        np.load(child_root / split / f"{clip_id}.npy", allow_pickle=False), values
    )
    return row


def verify_derived_package(config: dict) -> dict:
    identity = feature_identity(config)
    root = package_root(config)
    inventory = parent_inventory(config)
    parent = Path(config["parent_root"])
    parent_identity, _ = parent_metadata(config)
    for split, clips in inventory.items():
        for clip, meta in clips.items():
            paths = [Path(p) for p in meta["paths"]]
            verify_clip(parent, split, clip, paths, parent_identity)
            child = verify_clip(root, split, clip, paths, identity)
            expected = dict(
                parent_feature_sha256=sha256_file(parent / split / f"{clip}.npy"),
                parent_receipt_sha256=sha256_file(parent / "receipts" / split / f"{clip}.json"),
                parent_path=str(parent / split / f"{clip}.npy"),
                columns=[0, 1024],
            )
            if child["extra"] != expected:
                raise ValueError("Derived parent receipt drift")
            np.testing.assert_array_equal(
                np.load(root / split / f"{clip}.npy", allow_pickle=False),
                global_columns(np.load(parent / split / f"{clip}.npy", allow_pickle=False)),
            )
    result = verify_package(root, identity, inventory, 1024)
    return dict(
        root=str(root), identity_sha256=sha256_file(root / "complete/identity.json"), **result
    )


def derive_package(config: dict, *, check_only: bool = False) -> dict:
    inventory = parent_inventory(config)
    frames = sum(r["frame_count"] for clips in inventory.values() for r in clips.values())
    overhead = sum(
        len(json.dumps(r).encode()) + 4096 for clips in inventory.values() for r in clips.values()
    )
    projected = frames * 1024 * 4 + overhead * 3 + 64 * 1024**2
    storage = require_official_storage(Path(config["feature_base"]), projected)
    if check_only:
        return dict(
            frames=frames,
            projected_bytes=projected,
            storage=storage,
            root=str(package_root(config)),
        )
    root = package_root(config)
    with writer(Path(config["feature_base"]) / "preparation"):
        budget_path = Path(config["feature_base"]) / "preparation-budget.json"
        prior = (
            json.loads(budget_path.read_text())
            if budget_path.exists()
            else dict(charged_seconds=0.0)
        )
        if prior.get("status") == "running":
            raise RuntimeError("Abandoned conversion accounting requires manual review")
        started = time.monotonic()
        status = "failed"
        error = None
        atomic_json(
            budget_path,
            dict(
                status="running", charged_seconds=prior["charged_seconds"], started_wall=time.time()
            ),
        )
        try:
            with writer(root):
                for split, clips in inventory.items():
                    for clip in clips:
                        derive_clip(config, split, clip)
                parent_metadata(config)
                finish_package(root, feature_identity(config), inventory, 1024)
                result = verify_derived_package(config)
                atomic_json(Path(config["feature_base"]) / "package-result.json", result)
            status = "complete"
            return result
        except BaseException as primary:
            error = primary
            raise
        finally:
            try:
                atomic_json(
                    budget_path,
                    dict(
                        status=status,
                        charged_seconds=prior["charged_seconds"] + time.monotonic() - started,
                    ),
                )
            except Exception as observer:
                if error is None:
                    raise
                error.add_note(f"Conversion accounting failed: {observer}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["check", "derive", "verify"], required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--authorize-feature-preparation", action="store_true")
    args = parser.parse_args()
    if args.mode == "derive" and not args.authorize_feature_preparation:
        raise RuntimeError("Explicit CPU feature-preparation approval required")
    config = load_config(args.config)
    result = (
        verify_derived_package(config)
        if args.mode == "verify"
        else derive_package(config, check_only=args.mode == "check")
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()

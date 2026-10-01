"""Versioned full physical-train spatial extraction from final E3 DINO LoRA."""

import argparse
import json
import os
import shutil
import signal
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from tools.dinov3.e3_model import attach_late_lora
from tools.dinov3.e3_train import BASE_SHA, DATASET_KEY, POLICY, PROTOCOL_HASH, _load_lora_state
from tools.dinov3.encoder import extract
from tools.dinov3.identity import MODEL, digest, sha256_file
from tools.dinov3.storage import atomic_array, atomic_json, verify_location, writer


def adapted_key(identity: dict) -> str:
    """Content identity of adapted weights + frozen source/preprocessing contract."""
    if not identity.get("checkpoint_hash") or not identity.get("source_hash"):
        raise ValueError("E3 adapted feature identity incomplete")
    return digest(identity)


def adapted_output_path(output_base: Path, identity: dict) -> Path:
    return output_base / f"e3-features-{adapted_key(identity)}"


def _clip_row(root: Path, clip_id: str, paths: list[Path], identity: dict) -> dict:
    feature = root / "train" / f"{clip_id}.npy"
    receipt_path = root / "receipts/train" / f"{clip_id}.json"
    if not feature.is_file() or not receipt_path.is_file():
        raise ValueError(f"E3 feature/receipt missing: {clip_id}")
    receipt = json.loads(receipt_path.read_text())
    hashes = [sha256_file(path) for path in paths]
    expected_source = digest([(path.as_posix(), value)
                              for path, value in zip(paths, hashes, strict=True)])
    if (receipt.get("clip_id") != clip_id
            or receipt.get("checkpoint_hash") != identity["checkpoint_hash"]
            or receipt.get("source_hash") != expected_source
            or receipt.get("feature_hash") != sha256_file(feature)
            or receipt.get("paths") != [path.as_posix() for path in paths]):
        raise ValueError(f"E3 feature receipt/hash drift: {clip_id}")
    values = np.load(feature, allow_pickle=False)
    if (values.shape != (len(paths), 2048) or values.dtype != np.float32
            or not np.isfinite(values).all()):
        raise ValueError(f"E3 adapted feature shape/values invalid: {clip_id}")
    return {key: receipt[key] for key in
            ("feature_hash", "source_hash", "paths", "source_indices", "frame_count")}


def write_adapted_clip(
    root: Path, clip_id: str, paths: list[Path], model, identity: dict, *,
    extract_fn=extract, batch_size: int = 8,
) -> dict:
    if (not clip_id or "/" in clip_id or "\\" in clip_id or not paths
            or any(not path.is_file() for path in paths)):
        raise ValueError("E3 clip paths invalid")
    feature = root / "train" / f"{clip_id}.npy"
    receipt_path = root / "receipts/train" / f"{clip_id}.json"
    if receipt_path.exists():
        return _clip_row(root, clip_id, paths, identity)
    hashes = [sha256_file(path) for path in paths]
    source_hash = digest([(path.as_posix(), value)
                          for path, value in zip(paths, hashes, strict=True)])
    array, resolutions = extract_fn(paths, model, "cuda", batch_size=batch_size)
    if (array.shape != (len(paths), 2048) or array.dtype != np.float32
            or not np.isfinite(array).all() or len(resolutions) != len(paths)):
        raise ValueError(f"E3 adapted CLS shape/values invalid: {clip_id}")
    atomic_array(feature, array)
    receipt = {"clip_id": clip_id, "checkpoint_hash": identity["checkpoint_hash"],
               "source_hash": source_hash, "feature_hash": sha256_file(feature),
               "paths": [path.as_posix() for path in paths],
               "source_indices": list(range(len(paths))), "frame_count": len(paths),
               "resolutions": resolutions}
    atomic_json(receipt_path, receipt)
    return {key: receipt[key] for key in
            ("feature_hash", "source_hash", "paths", "source_indices", "frame_count")}


def verify_adapted_version(root: Path, identity: dict, expected_ids: tuple[str, ...],
                           *, complete: Path | None = None) -> dict:
    complete = root / "complete" if complete is None else complete
    if not complete.is_dir() or json.loads((complete / "identity.json").read_text()) != identity:
        raise ValueError("E3 adapted version identity/checkpoint mismatch")
    manifest_path = complete / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    rows = json.loads((complete / "frame_rows.json").read_text())
    key = adapted_key(identity)
    records = manifest.get("records", ())
    record_ids = [record["clip_id"] for record in records]
    if (manifest.get("schema_version") != 1 or manifest.get("encoder") != key
            or manifest.get("expected_dim") != 2048 or record_ids != list(expected_ids)
            or rows.get("encoder_key") != key or set(rows.get("clips", {})) != set(expected_ids)
            or len(set(record_ids)) != len(record_ids)
            or set(path.stem for path in (root / "train").glob("*.npy")) != set(expected_ids)):
        raise ValueError("E3 adapted feature inventory mismatch")
    for record in records:
        clip_id = record["clip_id"]
        row = rows["clips"][clip_id]
        feature = root / "train" / f"{clip_id}.npy"
        receipt = json.loads((root / "receipts/train" / f"{clip_id}.json").read_text())
        if (record != {"clip_id": clip_id, "split": "train", "path": f"train/{clip_id}.npy",
                       "length": row["frame_count"], "width": 2048, "dtype": "float32"}
                or row["feature_hash"] != sha256_file(feature)
                or receipt.get("feature_hash") != row["feature_hash"]
                or receipt.get("checkpoint_hash") != identity["checkpoint_hash"]
                or receipt.get("source_hash") != row["source_hash"]):
            raise ValueError(f"E3 adapted feature/receipt integrity mismatch: {clip_id}")
        array = np.load(feature, mmap_mode="r", allow_pickle=False)
        if array.shape != (record["length"], 2048) or array.dtype != np.float32:
            raise ValueError(f"E3 adapted array shape mismatch: {clip_id}")
    return {"encoder_key": key, "manifest_hash": sha256_file(manifest_path),
            "frame_rows_hash": sha256_file(complete / "frame_rows.json"),
            "checkpoint_hash": identity["checkpoint_hash"], "clips": len(expected_ids)}


def publish_adapted_version(root: Path, identity: dict, ids: tuple[str, ...],
                            records: list[dict], rows: dict) -> dict:
    stage = root / f".complete-{os.getpid()}"
    if stage.exists() or (root / "complete").exists():
        raise ValueError("E3 complete staging already exists")
    stage.mkdir()
    atomic_json(stage / "manifest.json", {"schema_version": 1,
                                          "encoder": adapted_key(identity),
                                          "expected_dim": 2048, "records": records})
    atomic_json(stage / "frame_rows.json", {"encoder_key": adapted_key(identity),
                                             "clips": rows})
    atomic_json(stage / "identity.json", identity)
    result = verify_adapted_version(root, identity, ids, complete=stage)
    os.replace(stage, root / "complete")
    return result


def _run(args) -> dict:
    protocol = json.loads(args.protocol.read_text())
    if args.protocol.parent.name != PROTOCOL_HASH or protocol.get("protocol_hash") != PROTOCOL_HASH:
        raise ValueError("E3 extraction protocol mismatch")
    if args.dataset.name != DATASET_KEY:
        raise ValueError("E3 extraction dataset mismatch")
    source = json.loads((args.dataset / "source.json").read_text())
    if digest(source) != protocol["source_hash"] or len(source["clips"]) != 7096:
        raise ValueError("E3 extraction physical train source mismatch")
    selected = json.loads((args.adapt_run / "selected-checkpoint.json").read_text())
    adaptation = selected.get("identity", {})
    checkpoint = args.adapt_run / selected.get("checkpoint", "")
    if (selected.get("status") != "complete" or adaptation.get("protocol_hash") != PROTOCOL_HASH
            or adaptation.get("base_revision") != BASE_SHA
            or adaptation.get("supervision_policy") != POLICY
            or adaptation.get("human_review_status") != "not_assessed"
            or selected.get("step") != adaptation.get("adapt_steps")
            or not checkpoint.is_file()
            or sha256_file(checkpoint) != selected.get("checkpoint_hash")):
        raise ValueError("E3 final adaptation checkpoint identity mismatch")
    original_rows = json.loads((args.original_dino_root / "complete/frame_rows.json").read_text())
    if (sha256_file(args.original_dino_root / "complete/frame_rows.json")
            != protocol["frame_rows_hash"]):
        raise ValueError("E3 original frame map changed")
    root = Path(__file__).resolve().parent
    identity = {"protocol_hash": PROTOCOL_HASH, "source_hash": protocol["source_hash"],
                "checkpoint_hash": selected["checkpoint_hash"],
                "adaptation_run_key": adaptation["run_key"], "base_revision": BASE_SHA,
                "frame_rows_hash": protocol["frame_rows_hash"], "scales": [224, 448],
                "encoder_code_hashes": {
                    name: sha256_file(root / name) for name in
                    ("e3_extract.py", "e3_model.py", "encoder.py", "frames.py")
                },
                "lock_hash": sha256_file(root / "uv.lock"),
                "supervision_policy": POLICY, "human_review_status": "not_assessed"}
    key = adapted_key(identity)
    args.output = adapted_output_path(args.output_base, identity)
    verify_location(args.output, root.parents[1])
    ids = tuple(sorted(clip["clip_id"] for clip in source["clips"]))
    if len(set(ids)) != 7096 or not set(ids) <= original_rows["clips"].keys():
        raise ValueError("E3 full original row inventory mismatch")
    if args.mode == "preflight":
        return {"status": "ready", "feature_key": key, "clips": len(ids),
                "checkpoint_hash": selected["checkpoint_hash"]}
    if not args.authorize_e3_extraction or args.gpu_cap_seconds <= 0:
        raise ValueError("E3 extraction requires explicit authorization and GPU cap")
    if args.output.is_dir() and (args.output / "complete").exists():
        raise ValueError("E3 feature version already complete")
    reserve = sum(original_rows["clips"][clip_id]["frame_count"] for clip_id in ids) * 2048 * 8
    if shutil.disk_usage(args.output.parent).free < reserve:
        raise ValueError("E3 adapted feature storage reserve insufficient")
    if not torch.cuda.is_available() or torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        raise RuntimeError("E3 extraction requires 12 GiB free CUDA memory")
    with writer(args.output):
        spent = 0.0
        budget_path = args.output / "budget.json"
        if budget_path.exists():
            budget = json.loads(budget_path.read_text())
            if budget.get("identity") != identity or budget.get("status") != "running":
                raise ValueError("E3 extraction resume identity/status mismatch")
            spent = budget["gpu_seconds"]
        else:
            atomic_json(budget_path, {"identity": identity, "status": "running",
                                      "gpu_seconds": 0.0})
        started = time.monotonic()
        model = AutoModel.from_pretrained(MODEL, revision=BASE_SHA, cache_dir=args.cache,
                                          local_files_only=True, use_safetensors=True,
                                          trust_remote_code=False).eval().to("cuda")
        attach_late_lora(model)
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if state.get("identity") != adaptation or state.get("step") != selected["step"]:
            raise ValueError("E3 extraction checkpoint snapshot identity mismatch")
        _load_lora_state(model, state["model"])
        model.eval()
        rows, records = {}, []
        try:
            for clip_id in ids:
                if spent + time.monotonic() - started >= args.gpu_cap_seconds - 30:
                    raise TimeoutError("E3 extraction cap near; preserve partial version")
                original = original_rows["clips"][clip_id]
                paths = [(args.frames / path).resolve() for path in original["paths"]]
                if (len(paths) != original["frame_count"] or
                        any(args.frames.resolve() not in path.parents for path in paths)):
                    raise ValueError(f"E3 source frame path/inventory mismatch: {clip_id}")
                row = write_adapted_clip(args.output, clip_id, paths, model, identity,
                                         batch_size=args.batch_size)
                rows[clip_id] = row
                records.append({"clip_id": clip_id, "split": "train",
                                "path": f"train/{clip_id}.npy", "length": row["frame_count"],
                                "width": 2048, "dtype": "float32"})
            result = publish_adapted_version(args.output, identity, ids, records, rows)
            atomic_json(budget_path, {"identity": identity, "status": "closed",
                                      "gpu_seconds": spent + time.monotonic() - started})
            return {"status": "complete", **result}
        except BaseException:
            atomic_json(budget_path, {"identity": identity, "status": "running",
                                      "gpu_seconds": spent + time.monotonic() - started})
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "run"))
    for name in ("protocol", "dataset", "adapt-run", "original-dino-root",
                 "frames", "cache", "output-base"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--gpu-cap-seconds", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--authorize-e3-extraction", action="store_true")
    args = parser.parse_args()

    def timeout(signum, frame):
        raise TimeoutError("E3 extraction cap")

    if args.mode == "run":
        signal.signal(signal.SIGALRM, timeout)
        signal.alarm(args.gpu_cap_seconds)
    try:
        print(json.dumps(_run(args), sort_keys=True))
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()

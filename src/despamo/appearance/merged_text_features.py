"""Offline CPU encoding with sealed parents and manifest-last publication."""

from importlib.metadata import version
from pathlib import Path
from typing import cast

import numpy as np
import torch

from despamo.appearance.merged_targets import REVISION, merged_target_key
from despamo.appearance.provenance import digest, file_hash, read_json
from despamo.appearance.text_features import MODEL, TextEncoder
from tools.dinov3.storage import _atomic, atomic_json, writer

ENCODING: dict = dict(model=MODEL, revision=REVISION, tokenizer=MODEL, tokenizer_revision=REVISION,
                width=768, dtype="float32", device="cpu", batch_size=64,
                stable_vector_policy="copy-verbatim", canonical_version="merged-targets-v1")
_SOURCE_ROOT = Path(__file__).resolve().parents[3]
PRODUCERS = [Path(__file__).resolve().with_name(name) for name in
             ("merged_targets.py", "merged_text_features.py", "text_features.py",
              "provenance.py", "schema.py")]
PRODUCERS += [_SOURCE_ROOT / p for p in ("tools/dinov3/storage.py", "tools/dinov3/identity.py",
                                       "scripts/prepare_official_factor_targets.py")]
ENCODING["producer_sha256"] = {p.name: file_hash(p) for p in PRODUCERS}
ENCODING["runtime_versions"] = {name: version(name) for name in
                                ("torch", "transformers", "numpy", "huggingface-hub")}


def load_pinned_encoder(cache_root: Path) -> TextEncoder:
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL, revision=REVISION, cache_dir=str(cache_root), local_files_only=True)
    model = cast(CLIPTextModelWithProjection, CLIPTextModelWithProjection.from_pretrained(
        MODEL, revision=REVISION, cache_dir=str(cache_root), local_files_only=True))
    model.cpu()
    model.float()
    encoder = TextEncoder(tokenizer, model, REVISION)
    if encoder.width != 768:
        raise ValueError("pinned CLIP projection width mismatch")
    return encoder


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
            or root.resolve() not in path.resolve().parents):
        raise ValueError("cache/input path escapes owned root")
    return path


def _key(document: dict, encoding: dict | None = None) -> str:
    return digest(dict(document_key=merged_target_key(document),
                       encoding=ENCODING if encoding is None else encoding))


def _arrays(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        vectors, valid = saved["vectors"].copy(), saved["valid"].copy()
    if (vectors.shape != (19, 768) or vectors.dtype != np.float32
            or valid.shape != (19,) or valid.dtype != np.bool_
            or not np.isfinite(vectors).all() or np.any(vectors[~valid])):
        raise ValueError("invalid text vector shape/dtype/mask/values")
    return vectors, valid


def read_stable_vectors(document: dict, old_text_root: Path) -> dict:
    result = {}
    for row in document["records"]:
        old = row["old_vector"]
        path = _inside(old_text_root, old["path"])
        if file_hash(path) != old["file_hash"]:
            raise ValueError("previous text NPZ hash mismatch")
        vectors, valid = _arrays(path)
        expected = np.array([t["text"] is not None for t in row["targets"][:4]])
        if not np.array_equal(valid[:4], expected) or file_hash(path) != old["file_hash"]:
            raise ValueError("previous stable vector mask/hash changed")
        result[row["clip_id"]] = vectors[:4].copy(), valid[:4].copy()
    return result


def verify_merged_manifest(path: Path) -> dict:
    root = path.parent
    complete = read_json(root / "complete.json")
    if complete["manifest_sha256"] != file_hash(path):
        raise ValueError("merged manifest hash changed")
    document_path = root / "merged_targets.json"
    if complete["document_sha256"] != file_hash(document_path):
        raise ValueError("merged target document hash changed")
    document, manifest = read_json(document_path), read_json(path)
    encoding = manifest["metadata"]
    contract = {k: v for k, v in ENCODING.items()
                if k not in {"producer_sha256", "runtime_versions"}}
    if (root.name != _key(document, encoding) or complete["key"] != root.name
            or any(encoding.get(k) != v for k, v in contract.items())
            or set(encoding["producer_sha256"]) != {p.name for p in PRODUCERS}
            or manifest["dataset_key"] != root.name
            or manifest["merged_document_key"] != merged_target_key(document)
            or manifest["supervision_policy"] != document["supervision_policy"]
            or manifest["human_review_status"] != "not_assessed"
            or document["human_review_status"] != "not_assessed"):
        raise ValueError("merged cache identity mismatch")
    for name, parent in document["parents"].items():
        if name not in {"old_manifest", "crop_export", "source"}:
            raise ValueError("unknown source snapshot")
        if file_hash(_inside(root, f"inputs/{name}.json")) != parent["sha256"]:
            raise ValueError("sealed input snapshot changed")
    for name, sha in encoding["producer_sha256"].items():
        if file_hash(_inside(root, f"producer/{name}")) != sha:
            raise ValueError("sealed producer source changed")
    if [r["clip_id"] for r in manifest["records"]] != [
            r["clip_id"] for r in document["records"]]:
        raise ValueError("merged record inventory changed")
    for row, original in zip(manifest["records"], document["records"], strict=True):
        array_path = _inside(root, row["path"])
        if row["targets"] != original["targets"] or file_hash(array_path) != row["file_hash"]:
            raise ValueError("merged vector/target hash mismatch")
        _, valid = _arrays(array_path)
        if not np.array_equal(valid, [t["text"] is not None for t in row["targets"]]):
            raise ValueError("merged target mask mismatch")
    return manifest


def encode_merged_targets(document: dict, old_text_root: Path, encoder: TextEncoder,
                          output_base: Path) -> Path:
    if encoder.revision != REVISION or encoder.width != 768:
        raise ValueError("text encoder revision/width mismatch")
    if hasattr(encoder, "model") and any(
            p.device.type != "cpu" or p.dtype != torch.float32 or p.requires_grad
            for p in encoder.model.parameters()):
        raise ValueError("encoding requires frozen CPU/FP32 model")
    root = output_base / _key(document)
    manifest_path = root / "manifest.json"
    with writer(root):
        if (root / "complete.json").exists():
            verify_merged_manifest(manifest_path)
            return manifest_path
        for source in PRODUCERS:
            dest = root / "producer" / source.name
            sha = ENCODING["producer_sha256"][source.name]
            if file_hash(source) != sha:
                raise ValueError("producer changed during encoding")
            if not dest.exists():
                _atomic(dest, lambda handle, p=source: handle.write(p.read_bytes()))
            if file_hash(dest) != sha or file_hash(source) != sha:
                raise ValueError("producer source snapshot hash mismatch")
        for name, parent in document["parents"].items():
            if name not in {"old_manifest", "crop_export", "source"}:
                raise ValueError("unknown source snapshot")
            dest = root / "inputs" / f"{name}.json"
            if not dest.exists():
                source = Path(parent["path"])
                if file_hash(source) != parent["sha256"]:
                    raise ValueError("parent snapshot source hash mismatch")
                _atomic(dest, lambda handle, p=source: handle.write(p.read_bytes()))
                if file_hash(source) != parent["sha256"]:
                    raise ValueError("parent changed during snapshot copy")
            if file_hash(dest) != parent["sha256"]:
                raise ValueError("parent snapshot hash mismatch")
        doc_path = root / "merged_targets.json"
        if doc_path.exists() and read_json(doc_path) != document:
            raise ValueError("existing merged document differs")
        if not doc_path.exists():
            atomic_json(doc_path, document)
        stable = read_stable_vectors(document, old_text_root)
        texts = sorted({t["text"] for row in document["records"]
                        for t in row["targets"][4:] if t["text"] is not None})
        encoded = {}
        for start in range(0, len(texts), 64):
            batch = texts[start:start+64]
            values = encoder.encode(batch)
            if (values.shape != (len(batch), 768) or values.dtype != np.float32
                    or not np.isfinite(values).all()):
                raise ValueError("invalid new text embeddings")
            encoded.update(zip(batch, values, strict=True))
        records = []
        for row in document["records"]:
            vectors = np.zeros((19, 768), np.float32)
            valid = np.array([t["text"] is not None for t in row["targets"]], dtype=np.bool_)
            vectors[:4] = stable[row["clip_id"]][0]
            for i, target in enumerate(row["targets"][4:], 4):
                if target["text"] is not None:
                    vectors[i] = encoded[target["text"]]
            relative = f"{digest(row['clip_id'])}.npz"
            dest = root / relative
            if dest.exists():
                prior, prior_valid = _arrays(dest)
                if not np.array_equal(prior, vectors) or not np.array_equal(prior_valid, valid):
                    raise ValueError("existing vector cache differs")
            else:
                _atomic(dest, lambda handle, v=vectors, m=valid:
                        np.savez_compressed(handle, vectors=v, valid=m))
            records.append(dict(clip_id=row["clip_id"], path=relative,
                                file_hash=file_hash(dest), targets=row["targets"],
                                record_hash=digest(row)))
        manifest = dict(dataset_key=root.name, metadata=ENCODING,
                        merged_document_key=merged_target_key(document),
                        supervision_policy=document["supervision_policy"],
                        human_review_status="not_assessed", records=records)
        if manifest_path.exists() and read_json(manifest_path) != manifest:
            raise ValueError("existing manifest differs")
        atomic_json(manifest_path, manifest)
        complete = dict(key=root.name, manifest_sha256=file_hash(manifest_path),
                        document_sha256=file_hash(doc_path), clip_count=len(records))
        atomic_json(root / "complete.json", complete)
        verify_merged_manifest(manifest_path)
    return manifest_path

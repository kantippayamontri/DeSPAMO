"""Official train-only DINO admission without main-runtime/Pydantic imports."""

import json
from pathlib import Path

import numpy as np
import torch

from tools.dinov3.frames import preprocess, sample_indices
from tools.dinov3.identity import digest, sha256_file

POLICY = "qwen-stable-v1-crop-articulators-v2-unreviewed-v1"
REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"
STABLE = ("biometric", "clothing", "hair", "background")
ARTICULATORS = ("left_hand", "right_hand", "mouth")
SIGNERS = {f"Signer{i:02}" for i in range(1, 10)}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _inside(root: Path, relative: str) -> Path:
    path = root / relative
    if (Path(relative).is_absolute() or ".." in Path(relative).parts or path.is_symlink()
            or root.resolve() not in path.resolve().parents):
        raise ValueError("input path escapes admitted root")
    return path


def _official_inventory(annotations: dict) -> dict:
    result, seen = {}, set()
    for split, count in (("train", 7096), ("dev", 519), ("test", 642)):
        raw = annotations[split]
        records = [raw[k] for k in sorted(k for k in raw if type(k) is int)]
        ids = [r["fileid"] for r in records]
        if (len(records) != count or len(set(ids)) != count
                or any(not isinstance(i, str) or not i or "/" in i or "\\" in i
                       or i in {".", ".."} for i in ids)
                or {r["signer"] for r in records} != SIGNERS or seen.intersection(ids)):
            raise ValueError(f"official {split} count/ID/signer/overlap mismatch")
        for r in records:
            if "folder" in r and r["folder"] != f"{split}/{r['fileid']}/*.png":
                raise ValueError("annotation physical split/path mismatch")
        seen.update(ids)
        result[split] = dict(zip(ids, records, strict=True))
    return result


def _index(records: list[dict], train: dict, label: str) -> dict:
    indexed = {r["clip_id"]: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(train):
        raise ValueError(f"{label} source inventory mismatch")
    return indexed


def _join_records(source: dict, document: dict, teacher: dict, manifest: dict,
                  train: dict, teacher_root: Path, merged_root: Path) -> dict:
    sources = _index(source["clips"], train, "frames")
    docs = _index(document["records"], train, "merged document")
    records = _index(manifest["records"], train, "merged vectors")
    result = {}
    for clip_id, annotation in train.items():
        clip, doc, record = sources[clip_id], docs[clip_id], records[clip_id]
        row = teacher["clips"].get(clip_id)
        count = clip["frame_count"]
        indices = sample_indices(count)
        frames = clip["frames"]
        paths, hashes = [f["path"] for f in frames], [f["sha256"] for f in frames]
        if (row is None or clip["signer"] != annotation["signer"]
                or doc["signer"] != annotation["signer"] or doc["frame_count"] != count
                or doc["frame_indices"] != indices or doc["frame_paths"] != paths
                or doc["source_frame_hashes"] != hashes
                or [f["index"] for f in frames] != indices
                or row.get("frame_count") != count
                or row.get("source_indices") != list(range(count))
                or len(row.get("paths", [])) != count
                or [row["paths"][i] for i in indices] != paths
                or any(row.get("sampled_images", {}).get(str(i)) != sha
                       for i, sha in zip(indices, hashes, strict=True))
                or any(not p.startswith(f"train/{clip_id}/") or ".." in Path(p).parts
                       for p in paths)):
            raise ValueError(f"official source/teacher/target alignment mismatch: {clip_id}")
        targets = record["targets"]
        slots = [(f, -1) for f in STABLE] + [(f, i) for i in indices for f in ARTICULATORS]
        if (targets != doc["targets"]
                or [(t.get("factor"), t.get("frame_index")) for t in targets] != slots
                or any(t["text"] is not None and (
                    not isinstance(t["text"], str) or not t["text"].strip()) for t in targets)):
            raise ValueError("merged factor/frame/text contract mismatch")
        result[clip_id] = dict(signer=annotation["signer"], frame_count=count,
                              frame_indices=list(indices), frame_paths=list(paths),
                              frame_hashes=list(hashes), targets=targets,
                              teacher_path=_inside(teacher_root, f"train/{clip_id}.npy"),
                              teacher_sha256=row["feature_hash"],
                              target_path=_inside(merged_root, record["path"]),
                              target_sha256=record["file_hash"])
    return result


def _signature(path: Path) -> tuple:
    st = path.stat()
    return st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns


def _descriptor_hash(row: dict) -> str:
    return digest({k: str(v) if isinstance(v, Path) else v for k, v in row.items()})


def _seal_admission(clips: dict, frames_root: Path, bindings: dict) -> dict:
    sealed = {}
    for path, sha in bindings.items():
        before = _signature(path)
        if sha256_file(path) != sha or _signature(path) != before:
            raise ValueError("metadata changed during admission")
        sealed[str(path.resolve())] = dict(sha256=sha, signature=before)
    return dict(clips=clips, frames_root=frames_root.resolve(),
                descriptor_hashes={i: _descriptor_hash(r) for i, r in clips.items()},
                bindings=sealed)


def _checked_arrays(row: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature, target = row["teacher_path"], row["target_path"]
    if (sha256_file(feature) != row["teacher_sha256"]
            or sha256_file(target) != row["target_sha256"]):
        raise ValueError("teacher/target byte hash changed")
    original = np.load(feature, allow_pickle=False)
    with np.load(target, allow_pickle=False) as saved:
        vectors, masks = saved["vectors"].copy(), saved["valid"].copy()
    expected = np.array([t["text"] is not None for t in row["targets"]], dtype=np.bool_)
    if (original.shape != (row["frame_count"], 2048) or original.dtype != np.float32
            or vectors.shape != (19, 768) or vectors.dtype != np.float32
            or masks.shape != (19,) or masks.dtype != np.bool_
            or not np.array_equal(masks, expected) or np.any(vectors[~masks])
            or not np.isfinite(original).all() or not np.isfinite(vectors).all()):
        raise ValueError("teacher/target shape/dtype/mask/values mismatch")
    if (sha256_file(feature) != row["teacher_sha256"]
            or sha256_file(target) != row["target_sha256"]):
        raise ValueError("teacher/target changed while reading")
    return original, vectors, masks


def _checked_frame_paths(row: dict, root: Path) -> list[Path]:
    paths = [_inside(root, p) for p in row["frame_paths"]]
    if any(sha256_file(p) != sha for p, sha in zip(paths, row["frame_hashes"], strict=True)):
        raise ValueError("source PNG hash changed")
    return paths


def load_official_batch(ids: tuple[str, ...], admission: dict) -> dict:
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= admission["clips"].keys():
        raise ValueError("batch must contain unique admitted official train clips")
    for path, binding in admission["bindings"].items():
        if _signature(Path(path)) != binding["signature"]:
            raise ValueError("admitted metadata signature changed")
    images224, images448, teachers, vectors, masks, labels = [], [], [], [], [], []
    for clip_id in ids:
        row = admission["clips"][clip_id]
        if _descriptor_hash(row) != admission["descriptor_hashes"][clip_id]:
            raise ValueError("admitted clip descriptor changed")
        original, target, mask = _checked_arrays(row)
        paths = _checked_frame_paths(row, admission["frames_root"])
        images224.append(torch.stack([preprocess(p, 224) for p in paths]))
        images448.append(torch.stack([preprocess(p, 448) for p in paths]))
        _checked_frame_paths(row, admission["frames_root"])
        teachers.append(torch.from_numpy(original[row["frame_indices"]].copy()))
        vectors.append(torch.from_numpy(target))
        masks.append(torch.from_numpy(mask))
        labels.append(tuple(t["text"] or "" for t in row["targets"]))
    return dict(ids=ids, images224=torch.stack(images224), images448=torch.stack(images448),
                teacher=torch.stack(teachers), vectors=torch.stack(vectors),
                masks=torch.stack(masks), labels=tuple(labels))


def _read_merged(manifest_path: Path, bindings: dict) -> tuple[dict, dict]:
    root = manifest_path.parent
    for path in (manifest_path, root / "merged_targets.json", root / "complete.json"):
        bindings[path] = sha256_file(path)
    manifest, document, complete = (_read(manifest_path), _read(root / "merged_targets.json"),
                                    _read(root / "complete.json"))
    encoding = manifest["metadata"]
    key = digest(dict(document_key=digest(document), encoding=encoding))
    if (complete["manifest_sha256"] != bindings[manifest_path]
            or complete["document_sha256"] != bindings[root / "merged_targets.json"]
            or root.name != key or complete["key"] != key or manifest["dataset_key"] != key
            or manifest["merged_document_key"] != digest(document)
            or manifest["supervision_policy"] != POLICY
            or document["supervision_policy"] != POLICY
            or manifest["human_review_status"] != "not_assessed"
            or document["human_review_status"] != "not_assessed"
            or encoding.get("model") != "openai/clip-vit-large-patch14"
            or encoding.get("revision") != REVISION
            or encoding.get("tokenizer_revision") != REVISION
            or encoding.get("width") != 768 or encoding.get("dtype") != "float32"
            or encoding.get("device") != "cpu"):
        raise ValueError("completed merged-cache contract/hash mismatch")
    for name, parent in document["parents"].items():
        path = _inside(root, f"inputs/{name}.json")
        if sha256_file(path) != parent["sha256"]:
            raise ValueError("sealed merged parent changed")
        bindings[path] = parent["sha256"]
    for name, sha in encoding["producer_sha256"].items():
        path = _inside(root, f"producer/{name}")
        if sha256_file(path) != sha:
            raise ValueError("sealed merged producer changed")
        bindings[path] = sha
    return document, manifest


def admit_official_data(train_annotation: Path, dev_annotation: Path, test_annotation: Path,
                       source_path: Path, merged_manifest: Path, teacher_root: Path,
                       frames_root: Path) -> dict:
    paths = dict(train=train_annotation, dev=dev_annotation, test=test_annotation)
    bindings = {p: sha256_file(p) for p in paths.values()}
    annotations = {name: np.load(p, allow_pickle=True).item() for name, p in paths.items()}
    split = _official_inventory(annotations)
    rows_path = teacher_root / "complete/frame_rows.json"
    for path in (source_path, rows_path):
        bindings[path] = sha256_file(path)
    source, rows = _read(source_path), _read(rows_path)
    document, manifest = _read_merged(merged_manifest, bindings)
    if (source.get("split") != "train"
            or source.get("frame_rows_hash") != bindings[rows_path]
            or document["parents"]["source"]["sha256"] != bindings[source_path]
            or document.get("target_slots") != 19 or len(manifest["records"]) != 7096):
        raise ValueError("official source/merged/teacher lineage mismatch")
    clips = _join_records(source, document, rows, manifest, split["train"],
                          teacher_root, merged_manifest.parent)
    for row in clips.values():
        _checked_arrays(row)
        _checked_frame_paths(row, frames_root)
    admission = _seal_admission(clips, frames_root, bindings)
    admission.update(train_ids=tuple(split["train"]),
                     split_ids={name: tuple(records) for name, records in split.items()},
                     merged_manifest=str(merged_manifest.resolve()), policy=POLICY,
                     human_review_status="not_assessed")
    bound_hashes = {p: v["sha256"] for p, v in admission["bindings"].items()}
    admission["identity"] = dict(policy="official-phoenix14t-adaptation-data-v1",
                                 train_ids_hash=digest(admission["train_ids"]),
                                 manifest_sha256=bindings[merged_manifest],
                                 binding_hashes=bound_hashes,
                                 descriptor_hash=digest(admission["descriptor_hashes"]),
                                 frames_root=str(frames_root.resolve()))
    return admission

"""Frozen train-only image/teacher/CLIP joins for E3 adaptation."""

from pathlib import Path

import numpy as np
import torch

from tools.dinov3.frames import preprocess, sample_indices
from tools.dinov3.identity import digest, sha256_file


def validate_e3_sources(protocol: dict, source: dict, text: dict) -> tuple[str, ...]:
    if (protocol.get("protocol_hash") != digest({key: value for key, value in protocol.items()
                                                if key != "protocol_hash"})
            or protocol.get("source_hash") != digest(source)):
        raise ValueError("E3 protocol/source identity mismatch")
    groups = protocol["split"]["groups"]
    train, dev, test = (tuple(groups[name]["clip_ids"]) for name in ("train", "dev", "test"))
    if len(set(train + dev + test)) != len(train + dev + test):
        raise ValueError("E3 logical train/dev/test overlap")
    clips = {item["clip_id"]: item for item in source["clips"]}
    records = {item["clip_id"]: item for item in text["records"]}
    if (len(clips) != len(source["clips"]) or len(records) != len(text["records"])
            or not set(train) <= clips.keys() or not set(train) <= records.keys()):
        raise ValueError("E3 missing/duplicate train source or target")
    valid = failed = 0
    for clip_id in train:
        clip, entry = clips[clip_id], records[clip_id]
        if (clip.get("signer") in {"Signer03", "Signer07"} or len(clip["frames"]) != 5
                or len(entry["targets"]) != 19):
            raise ValueError(f"E3 train signer/frame/target mismatch: {clip_id}")
        values = [target.get("text") for target in entry["targets"]]
        if any(value is not None and (not isinstance(value, str) or not value) for value in values):
            raise ValueError(f"E3 invalid target text: {clip_id}")
        if any(value is not None for value in values):
            valid += 1
        else:
            failed += 1
    if valid != groups["train"]["valid"] or failed != groups["train"]["failed"]:
        raise ValueError("E3 valid/failed train count mismatch")
    return train


def load_e3_batch(
    ids: tuple[str, ...], source: dict, text: dict, rows: dict, *,
    frames: Path, dino_root: Path, text_manifest: Path,
) -> dict:
    clips = {item["clip_id"]: item for item in source["clips"]}
    records = {item["clip_id"]: item for item in text["records"]}
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("E3 train batch clip inventory invalid")
    images224, images448, teachers, targets, masks, labels = [], [], [], [], [], []
    for clip_id in ids:
        clip = clips.get(clip_id)
        if (clip is None or clip.get("signer") in {"Signer03", "Signer07"}
                or clip_id not in records or clip_id not in rows):
            raise ValueError(f"E3 train-only clip required: {clip_id}")
        entry, row = records[clip_id], rows[clip_id]
        count = clip["frame_count"]
        selected = clip["frames"]
        if (len(selected) != 5 or len(entry["targets"]) != 19
                or [frame["index"] for frame in selected] != sample_indices(count)
                or row["source_indices"] != list(range(count))):
            raise ValueError(f"E3 source frame alignment failed: {clip_id}")
        feature = dino_root / "train" / f"{clip_id}.npy"
        if sha256_file(feature) != row["feature_hash"]:
            raise ValueError(f"E3 original DINO feature hash mismatch: {clip_id}")
        original = np.load(feature, allow_pickle=False)
        if original.shape != (count, 2048) or not np.isfinite(original).all():
            raise ValueError(f"E3 original DINO feature shape/values invalid: {clip_id}")
        teachers.append(torch.from_numpy(original[[frame["index"] for frame in selected]].copy()))
        paths = []
        for frame in selected:
            path = (frames / frame["path"]).resolve()
            if frames.resolve() not in path.parents or sha256_file(path) != frame["sha256"]:
                raise ValueError(f"E3 source PNG hash mismatch: {clip_id}")
            paths.append(path)
        images224.append(torch.stack([preprocess(path, 224) for path in paths]))
        images448.append(torch.stack([preprocess(path, 448) for path in paths]))
        target_path = (text_manifest.parent / entry["path"]).resolve()
        if (target_path.parent != text_manifest.parent.resolve()
                or sha256_file(target_path) != entry["file_hash"]):
            raise ValueError(f"E3 factor target hash mismatch: {clip_id}")
        with np.load(target_path, allow_pickle=False) as saved:
            values, active = saved["vectors"].copy(), saved["valid"].copy()
        expected = np.array([item.get("text") is not None for item in entry["targets"]])
        if (values.shape != (19, 768) or active.shape != (19,) or active.dtype != np.bool_
                or not np.array_equal(active, expected) or np.any(values[~active])
                or not np.isfinite(values).all()):
            raise ValueError(f"E3 factor mask/target alignment invalid: {clip_id}")
        targets.append(torch.from_numpy(values))
        masks.append(torch.from_numpy(active))
        labels.append(tuple(item["text"] or "" for item in entry["targets"]))
    return {"ids": ids, "images224": torch.stack(images224),
            "images448": torch.stack(images448), "teacher": torch.stack(teachers).float(),
            "vectors": torch.stack(targets).float(), "masks": torch.stack(masks),
            "labels": tuple(labels)}

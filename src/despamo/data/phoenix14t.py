import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from despamo.data.batch import PhoenixSample
from despamo.data.manifest import FeatureManifest, FeatureRecord


def normalize_text(text: str) -> str:
    stripped = text.strip().lower()
    return stripped if stripped.endswith(".") else f"{stripped}."


def _load_feature(root: Path, record: FeatureRecord, modality: str) -> torch.Tensor:
    path = root / record.path
    try:
        array = np.load(path, allow_pickle=False)
    except (OSError, ValueError, EOFError) as exc:
        raise ValueError(f"invalid {modality} feature for {record.clip_id}: {path}") from exc
    expected_shape = (record.length, record.width)
    if array.shape != expected_shape:
        raise ValueError(
            f"{modality} shape mismatch for {record.clip_id}: "
            f"expected {expected_shape}, got {array.shape}"
        )
    if str(array.dtype) != record.dtype:
        raise ValueError(
            f"{modality} dtype mismatch for {record.clip_id}: "
            f"expected {record.dtype}, got {array.dtype}"
        )
    return torch.from_numpy(array.astype(np.float32, copy=False))


class Phoenix14T(Dataset[PhoenixSample]):
    def __init__(
        self,
        annotation_path: Path,
        split: str,
        spatial_root: Path,
        motion_root: Path,
        spatial_manifest: FeatureManifest,
        motion_manifest: FeatureManifest,
        max_frame_len: int | None = None,
        spatial_crop_mode: str = "full",
    ) -> None:
        if spatial_crop_mode not in {"upstream_random", "center", "full"}:
            raise ValueError(f"unsupported spatial_crop_mode: {spatial_crop_mode}")
        if (spatial_crop_mode != "full" and max_frame_len is None) or (
            max_frame_len is not None and (type(max_frame_len) is not int or max_frame_len <= 0)
        ):
            raise ValueError("max_frame_len must be a positive integer")
        try:
            raw = np.load(annotation_path, allow_pickle=True).item()
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"invalid annotation: {annotation_path}") from exc
        if not isinstance(raw, dict):
            raise ValueError("invalid annotation: expected dictionary")
        keys = sorted(key for key in raw if type(key) is int)
        if not keys:
            raise ValueError("invalid annotation: no integer records")
        records: list[dict[str, str]] = []
        seen: set[str] = set()
        for key in keys:
            item = raw[key]
            if not isinstance(item, dict):
                raise ValueError(f"invalid annotation record: {key}")
            for field in ("fileid", "signer", "gloss", "text", "en_text", "es_text", "fr_text"):
                value = item.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"invalid annotation {field}: {key}")
            clip_id = item["fileid"]
            if "/" in clip_id or "\\" in clip_id:
                raise ValueError(f"invalid annotation fileid: {clip_id}")
            if clip_id in seen:
                raise ValueError(f"duplicate annotation clip_id: {clip_id}")
            seen.add(clip_id)
            records.append(item)
        self.records = records
        self.split = split
        self.spatial_root = spatial_root
        self.motion_root = motion_root
        self.spatial_manifest = spatial_manifest
        self.motion_manifest = motion_manifest
        self.max_frame_len = max_frame_len
        self.spatial_crop_mode = spatial_crop_mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> PhoenixSample:
        item = self.records[index]
        clip_id = item["fileid"]
        spatial_record = self.spatial_manifest.require(self.split, clip_id)
        motion_record = self.motion_manifest.require(self.split, clip_id)
        spatial = _load_feature(self.spatial_root, spatial_record, "spatial")
        if (
            self.spatial_crop_mode != "full"
            and self.max_frame_len is not None
            and len(spatial) > self.max_frame_len
        ):
            excess = len(spatial) - self.max_frame_len
            start = excess // 2 if self.spatial_crop_mode == "center" else random.randint(0, excess)
            spatial = spatial[start : start + self.max_frame_len]
        return PhoenixSample(
            clip_id=clip_id,
            signer=item["signer"],
            text=normalize_text(item["text"]),
            gloss=item["gloss"],
            en_text=item["en_text"],
            es_text=item["es_text"],
            fr_text=item["fr_text"],
            spatial=spatial,
            motion=_load_feature(self.motion_root, motion_record, "motion"),
        )

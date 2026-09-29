"""Strict clip/frame joins for unreviewed Qwen appearance targets."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from despamo.appearance.audit_policy import schema_eligibility
from despamo.appearance.canonical import targets
from despamo.appearance.generation import load_records
from despamo.appearance.provenance import digest, file_hash, load_dataset, read_json
from despamo.appearance.schema import ARTICULATORS, FACTORS, STABLE, Record
from despamo.appearance.text_features import MODEL
from despamo.data.batch import PhoenixBatch, PhoenixSample, collate_phoenix
from despamo.data.manifest import FeatureManifest

UNREVIEWED_POLICY = "qwen-schema98-unreviewed-v1"
CLIP_REVISION = "32bd64288804d66eefd0ccbe215aa642df71cc41"


@dataclass(frozen=True)
class FactorBatch:
    base: PhoenixBatch
    rows: torch.Tensor  # [B, 5] indices into uncropped DINO spatial rows
    vectors: dict[str, torch.Tensor]  # stable [B, D], articulator [B, 5, D]
    masks: dict[str, torch.Tensor]  # stable [B], articulator [B, 5]
    labels: dict[str, tuple[str, ...]]


def map_rows(requested: list[int], source: list[int], length: int) -> list[int]:
    if (
        type(length) is not int
        or length <= 0
        or len(source) != length
        or len(set(source)) != length
        or not requested
        or len(set(requested)) != len(requested)
    ):
        raise ValueError("invalid source-to-feature-row mapping")
    lookup = {index: row for row, index in enumerate(source)}
    if any(index not in lookup for index in requested):
        raise ValueError("sampled source ordinal has no feature row")
    return [lookup[index] for index in requested]


def masked_targets(clip: dict) -> list[dict]:
    return [{"factor": f, "frame_index": -1, "text": None} for f in STABLE] + [
        {"factor": factor, "frame_index": frame["index"], "text": None}
        for frame in clip["frames"]
        for factor in ARTICULATORS
    ]


def load_factor_targets(
    clip: dict, journal: dict, entry: dict, text_root: Path, width: int
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    if journal["clip_id"] != clip["clip_id"] or entry["clip_id"] != clip["clip_id"]:
        raise ValueError("factor clip identity mismatch")
    if entry["record_hash"] != digest(journal):
        raise ValueError("factor source-record hash mismatch")
    expected = (
        targets(Record.model_validate(journal["record"]))
        if journal["status"] == "valid"
        else masked_targets(clip)
    )
    if journal["status"] not in {"valid", "failed"} or len(expected) != 19:
        raise ValueError("factor target status or count mismatch")
    if entry["targets"] != expected:
        raise ValueError("canonical targets differ from saved Qwen record")
    path = (text_root / entry["path"]).resolve()
    if path.parent != text_root.resolve() or file_hash(path) != entry["file_hash"]:
        raise ValueError("text feature file hash/path mismatch")
    with np.load(path, allow_pickle=False) as saved:
        values, valid = saved["vectors"].copy(), saved["valid"].copy()
    if (
        values.shape != (19, width)
        or values.dtype != np.float32
        or not np.isfinite(values).all()
        or valid.shape != (19,)
        or valid.dtype != np.bool_
    ):
        raise ValueError("invalid factor feature shape/dtype/values")
    mask = np.array([target["text"] is not None for target in expected], dtype=np.bool_)
    if not np.array_equal(valid, mask) or np.any(values[~valid]):
        raise ValueError("factor mask and zero-valued targets disagree")
    return values, valid, tuple(target["text"] or "" for target in expected)


def collate_factors(
    samples: list[tuple[PhoenixSample, torch.Tensor, np.ndarray, np.ndarray, tuple[str, ...]]],
) -> FactorBatch:
    if not samples:
        raise ValueError("empty factor batch")
    base = collate_phoenix([sample[0] for sample in samples])
    vectors, masks, labels = {}, {}, {}
    per_factor: dict[str, list[tuple[torch.Tensor, torch.Tensor, str]]] = {
        factor: [] for factor in FACTORS
    }
    frame_slots = [(-1, factor) for factor in STABLE] + [
        (position, factor) for position in range(5) for factor in ARTICULATORS
    ]
    for _sample, mapped, values, valid, strings in samples:
        if (
            mapped.shape != (5,)
            or values.shape[0] != 19
            or valid.shape != (19,)
            or len(strings) != 19
        ):
            raise ValueError("invalid factor batch shape")
        for i, (_, factor) in enumerate(frame_slots):
            per_factor[factor].append(
                (torch.from_numpy(values[i].copy()), torch.tensor(bool(valid[i])), strings[i])
            )
    for factor, entries in per_factor.items():
        data = torch.stack([item[0] for item in entries])
        active = torch.stack([item[1] for item in entries])
        shape = (len(samples),) if factor in STABLE else (len(samples), 5)
        vectors[factor] = data.reshape(*shape, -1)
        masks[factor] = active.reshape(*shape)
        labels[factor] = tuple(item[2] for item in entries)
    return FactorBatch(base, torch.stack([sample[1] for sample in samples]), vectors, masks, labels)


class FactorDataset(Dataset):
    """Join all translation clips to exact source-indexed factor targets."""

    def __init__(
        self,
        base,
        dataset: Path,
        text_manifest: Path,
        frame_rows: Path,
        spatial_manifest_path: Path,
        supervision_policy: str,
        *,
        logical_clip_ids: tuple[str, ...] | None = None,
        expected_parent_records_hash: str | None = None,
    ) -> None:
        if supervision_policy != UNREVIEWED_POLICY:
            raise ValueError("unsupported factor supervision policy")
        self.base = base
        self.source, version = load_dataset(dataset)
        if self.source["split"] != "train" or base.split != "train":
            raise ValueError("factor training requires complete train split")
        all_clips = {clip["clip_id"]: clip for clip in self.source["clips"]}
        selected_ids = (
            tuple(clip["clip_id"] for clip in self.source["clips"])
            if logical_clip_ids is None
            else logical_clip_ids
        )
        self.clips = {
            clip_id: all_clips[clip_id] for clip_id in selected_ids if clip_id in all_clips
        }
        if (
            len(all_clips) != len(self.source["clips"])
            or not selected_ids
            or len(self.clips) != len(selected_ids)
            or set(self.clips) != {record["fileid"] for record in base.records}
            or len(base.records) != len(self.clips)
        ):
            raise ValueError("logical caption/translation clip ID mismatch")
        if (
            logical_clip_ids is not None
            and tuple(record["fileid"] for record in base.records) != selected_ids
        ):
            raise ValueError("logical split order differs from train view")
        parent_journals = load_records(dataset)
        parent_hash = digest(parent_journals)
        if expected_parent_records_hash is not None and parent_hash != expected_parent_records_hash:
            raise ValueError("parent records hash differs from frozen pilot protocol")
        if len(parent_journals) != len(all_clips):
            raise ValueError("incomplete parent journal inventory")
        self.journals = {
            item["clip_id"]: item for item in parent_journals if item["clip_id"] in self.clips
        }
        if set(self.journals) != set(self.clips):
            raise ValueError("incomplete/duplicate caption journals")
        journals = [self.journals[clip_id] for clip_id in selected_ids]
        eligibility = schema_eligibility(
            sum(row["status"] == "valid" for row in journals), len(journals)
        )
        if not eligibility["schema_pass"]:
            raise ValueError("98% schema validity required for unreviewed factor policy")
        self.manifest = read_json(text_manifest)
        meta = self.manifest["metadata"]
        if (
            self.manifest["dataset_key"] != dataset.name
            or self.manifest["encoder_key"] != digest(meta)
            or meta["schema_version"] != 2
            or meta["model"] != MODEL
            or meta["revision"] != meta["tokenizer_revision"]
            or meta["revision"] != CLIP_REVISION
            or meta["dtype"] != "float32"
            or meta["width"] <= 0
        ):
            raise ValueError("text manifest model/revision/dataset identity mismatch")
        self.width = meta["width"]
        self.text_root = text_manifest.parent
        all_entries = {item["clip_id"]: item for item in self.manifest["records"]}
        if len(all_entries) != len(self.manifest["records"]) or set(all_entries) != set(all_clips):
            raise ValueError("missing/duplicate/extra text clips")
        self.entries = {clip_id: all_entries[clip_id] for clip_id in selected_ids}
        for clip_id, item in self.entries.items():
            journal = self.journals[clip_id]
            if item["record_hash"] != digest(journal):
                raise ValueError("text source-record hash mismatch")
            expected = (
                targets(Record.model_validate(journal["record"]))
                if journal["status"] == "valid"
                else masked_targets(self.clips[clip_id])
            )
            if item["targets"] != expected or len(expected) != 19:
                raise ValueError("text targets differ from validated source record")
            path = (self.text_root / item["path"]).resolve()
            if path.parent != self.text_root.resolve() or file_hash(path) != item["file_hash"]:
                raise ValueError(f"text feature file hash/path mismatch: {clip_id}")
        if self.source.get("frame_rows_hash", file_hash(frame_rows)) != file_hash(frame_rows):
            raise ValueError("caption/DINO frame-row map mismatch")
        self.mapping = read_json(frame_rows)["clips"]
        row_data = read_json(frame_rows)
        spatial = FeatureManifest.load(spatial_manifest_path)
        if (
            row_data["spatial_manifest_hash"] != file_hash(spatial_manifest_path)
            or spatial.encoder != "dinov3:" + row_data["encoder_key"]
            or set(self.clips) - set(self.mapping)
        ):
            raise ValueError("DINO source map/manifest identity mismatch")
        for clip_id, clip in self.clips.items():
            mapping = self.mapping[clip_id]
            feature = spatial.require("train", clip_id)
            if (
                mapping["feature_hash"] != file_hash(base.spatial_root / feature.path)
                or mapping["source_indices"] != list(range(clip["frame_count"]))
                or feature.length != len(mapping["source_indices"])
            ):
                raise ValueError(f"DINO feature/source index mismatch: {clip_id}")
            for frame in clip["frames"]:
                if mapping["sampled_images"].get(str(frame["index"])) != frame["sha256"]:
                    raise ValueError(f"caption/DINO source image mismatch: {clip_id}")
        schema_fields = {
            key: eligibility[key]
            for key in (
                "valid_count",
                "failed_count",
                "total_count",
                "schema_rate",
                "minimum_schema_rate",
                "minimum_valid_count",
                "maximum_failed_count",
                "schema_pass",
                "original_minimum_schema_rate",
                "original_schema_pass",
            )
        }
        self.provenance = {
            **schema_fields,
            "schema_eligibility": schema_fields,
            "policy_version": supervision_policy,
            "supervision_policy": supervision_policy,
            "human_review_status": "not_assessed",
            "dataset_key": dataset.name,
            "source_hash": digest(self.source),
            "version_hash": digest(version),
            "records_hash": parent_hash,
            "logical_train_ids_hash": digest(selected_ids),
            "training_records_hash": digest(journals),
            "text_manifest_hash": file_hash(text_manifest),
            "frame_rows_hash": file_hash(frame_rows),
            "spatial_manifest_hash": file_hash(spatial_manifest_path),
        }

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        sample = self.base[index]
        clip = self.clips[sample.clip_id]
        mapping = self.mapping[sample.clip_id]
        requested = [frame["index"] for frame in clip["frames"]]
        mapped = map_rows(requested, mapping["source_indices"], sample.spatial.shape[0])
        values, mask, labels = load_factor_targets(
            clip,
            self.journals[sample.clip_id],
            self.entries[sample.clip_id],
            self.text_root,
            self.width,
        )
        return sample, torch.tensor(mapped, dtype=torch.long), values, mask, labels

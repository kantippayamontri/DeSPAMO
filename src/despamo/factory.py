from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from despamo.data.batch import collate_phoenix
from despamo.data.datamodule import PhoenixDataModule
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.losses.vt_align import VTAlignLoss
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.models.visual_adapter import SpaMoVisualAdapter
from despamo.training.baseline_module import SpaMoBaselineModule


def _preflight_features(
    dataset: Phoenix14T,
    modality: str,
    root: Path,
    manifest: FeatureManifest,
    expected_width: int,
) -> None:
    annotated = {item["fileid"] for item in dataset.records}
    for item in dataset.records:
        clip_id = item["fileid"]
        clip = f"{dataset.split}/{clip_id}"
        try:
            record = manifest.require(dataset.split, clip_id)
        except KeyError as exc:
            raise ValueError(f"missing {modality} feature for {clip} in manifest") from exc
        if record.width != expected_width:
            raise ValueError(
                f"{modality} width mismatch for {clip}: "
                f"expected {expected_width}, got {record.width} in manifest"
            )

        path = root / record.path
        try:
            feature = np.lib.format.open_memmap(path, mode="r")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"missing {modality} feature file for {clip}: {path}") from exc
        except (OSError, ValueError, EOFError) as exc:
            raise ValueError(f"invalid {modality} feature for {clip}: {path}") from exc
        if feature.ndim != 2 or feature.shape[0] == 0:
            raise ValueError(f"invalid {modality} feature shape for {clip}: {feature.shape}")
        if feature.shape[1] != expected_width:
            raise ValueError(
                f"{modality} file width mismatch for {clip}: "
                f"expected {expected_width}, got {feature.shape[1]}"
            )
        if feature.shape[0] != record.length or str(feature.dtype) != record.dtype:
            raise ValueError(
                f"{modality} feature metadata mismatch for {clip}: expected "
                f"({record.length}, {record.width}) {record.dtype}, "
                f"got {feature.shape} {feature.dtype}"
            )
        del feature

    for record in manifest.records:
        if record.split == dataset.split and record.clip_id not in annotated:
            raise ValueError(f"unexpected {modality} feature for {dataset.split}/{record.clip_id}")


def build_data(config: DictConfig) -> PhoenixDataModule:
    spatial_manifest = FeatureManifest.load(Path(config.data.spatial_manifest))
    motion_manifest = FeatureManifest.load(Path(config.data.motion_manifest))
    datasets: dict[str, Phoenix14T] = {}
    for name, annotation_name in (
        ("train", "train_info_ml.npy"),
        ("validation", "dev_info_ml.npy"),
        ("test", "test_info_ml.npy"),
    ):
        split = "dev" if name == "validation" else name
        datasets[name] = Phoenix14T(
            Path(config.data.annotation_root) / annotation_name,
            split,
            Path(config.data.spatial_root),
            Path(config.data.motion_root),
            spatial_manifest,
            motion_manifest,
            max_frame_len=config.model.get("max_frame_len"),
            spatial_crop_mode=config.model.get("spatial_crop_mode", "full"),
        )
        for modality, manifest in (("spatial", spatial_manifest), ("motion", motion_manifest)):
            _preflight_features(
                datasets[name],
                modality,
                Path(config.data[f"{modality}_root"]),
                manifest,
                config.model[f"{modality}_dim"],
            )
    return PhoenixDataModule(
        datasets["train"],
        datasets["validation"],
        datasets["test"],
        config.data.batch_size,
        config.data.num_workers,
        collate_phoenix,
    )


def build_model(config: DictConfig) -> SpaMoBaselineModule:
    language_model = FlanT5Backbone.from_pretrained(
        config.model.name,
        config.model.cache_dir,
        config.model.max_text_length,
        config.model.get("tuning_type", "lora"),
        config.model.lora_rank,
        config.model.lora_alpha,
        config.model.lora_dropout,
    )
    return SpaMoBaselineModule(
        visual_adapter=SpaMoVisualAdapter(
            config.model.spatial_dim,
            config.model.motion_dim,
            config.model.adapter_dim,
            config.model.language_dim,
        ),
        language_model=language_model,
        vt_align=VTAlignLoss(),
        prompt_template=config.model.prompt,
        use_in_context=config.model.use_in_context,
        num_in_context=config.model.num_in_context,
        vt_pooling=config.model.vt_pooling,
        vt_weight=config.model.vt_weight,
        warm_up_steps=config.model.warm_up_steps,
        learning_rate=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
        seed=config.seed,
    )

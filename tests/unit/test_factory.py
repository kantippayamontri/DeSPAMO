from pathlib import Path

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf
from torch import nn

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.factory import build_data, build_model
from despamo.models.flan_t5 import FlanT5Backbone


class DummyLanguageModel(nn.Module):
    pass


def test_build_model_uses_configured_visual_dimensions(monkeypatch: pytest.MonkeyPatch) -> None:
    loader_calls = []

    def fake_loader(*args, **kwargs):
        loader_calls.append((args, kwargs))
        return DummyLanguageModel()

    monkeypatch.setattr(FlanT5Backbone, "from_pretrained", fake_loader)
    config = OmegaConf.create(
        {
            "seed": 0,
            "model": {
                "name": "google/flan-t5-xl",
                "cache_dir": "/tmp/cache",
                "spatial_dim": 2048,
                "motion_dim": 1024,
                "adapter_dim": 768,
                "language_dim": 2048,
                "max_text_length": 64,
                "lora_rank": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "prompt": "Translate the given sentence into {}.",
                "use_in_context": False,
                "num_in_context": 0,
                "vt_pooling": "legacy_mean",
                "vt_weight": 1.0,
                "warm_up_steps": None,
            },
            "optimizer": {"learning_rate": 6.0e-4, "weight_decay": 0.01},
        }
    )

    model = build_model(config)

    assert model.visual_adapter.spatial_projector.in_features == 2048
    assert model.visual_adapter.motion_projector.in_features == 1024
    assert model.visual_adapter.multimodal_projector[-1].out_features == 2048
    assert loader_calls == [(("google/flan-t5-xl", "/tmp/cache", 64, "lora", 16, 32, 0.1), {})]

    config.model.tuning_type = "freeze"
    build_model(config)
    assert loader_calls[-1] == (("google/flan-t5-xl", "/tmp/cache", 64, "freeze", 16, 32, 0.1), {})


def _synthetic_data_config(tmp_path: Path) -> DictConfig:
    annotation_root = tmp_path / "annotations"
    annotation_root.mkdir()
    records = []
    for split, filename in (
        ("train", "train_info_ml.npy"),
        ("dev", "dev_info_ml.npy"),
        ("test", "test_info_ml.npy"),
    ):
        clip_id = f"{split}-clip"
        np.save(
            annotation_root / filename,
            {
                0: {
                    "fileid": clip_id,
                    "signer": "Signer01",
                    "gloss": "WIND",
                    "text": "wind",
                    "en_text": "wind",
                    "es_text": "viento",
                    "fr_text": "vent",
                }
            },
        )
        records.append((split, clip_id))

    manifests = {}
    for modality, width in (("spatial", 2048), ("motion", 1024)):
        root = tmp_path / modality
        feature_records = []
        for split, clip_id in records:
            (root / split).mkdir(parents=True)
            np.save(root / split / f"{clip_id}.npy", np.ones((2, width), dtype=np.float32))
            feature_records.append(
                FeatureRecord(clip_id, split, f"{split}/{clip_id}.npy", 2, width, "float32")
            )
        path = tmp_path / f"{modality}.json"
        FeatureManifest(1, modality, width, tuple(feature_records)).save(path)
        manifests[modality] = (root, path)

    return OmegaConf.create(
        {
            "data": {
                "annotation_root": str(annotation_root),
                "spatial_root": str(manifests["spatial"][0]),
                "motion_root": str(manifests["motion"][0]),
                "spatial_manifest": str(manifests["spatial"][1]),
                "motion_manifest": str(manifests["motion"][1]),
                "batch_size": 2,
                "num_workers": 0,
            },
            "model": {"spatial_dim": 2048, "motion_dim": 1024},
        }
    )


def test_build_data_uses_split_annotations_and_collates_features(tmp_path: Path) -> None:
    config = _synthetic_data_config(tmp_path)
    module = build_data(config)

    assert [
        module.train_dataset.split,
        module.validation_dataset.split,
        module.test_dataset.split,
    ] == ["train", "dev", "test"]
    assert [
        next(iter(loader)).clip_ids
        for loader in (module.train_dataloader(), module.val_dataloader(), module.test_dataloader())
    ] == [("train-clip",), ("dev-clip",), ("test-clip",)]
    assert next(iter(module.test_dataloader())).spatial.shape == (1, 2, 2048)
    assert next(iter(module.test_dataloader())).motion.shape == (1, 2, 1024)


def test_build_data_passes_spatial_crop_policy_to_all_splits(tmp_path: Path) -> None:
    config = _synthetic_data_config(tmp_path)
    config.model.max_frame_len = 512
    config.model.spatial_crop_mode = "center"

    module = build_data(config)

    for dataset in (module.train_dataset, module.validation_dataset, module.test_dataset):
        assert dataset.max_frame_len == 512
        assert dataset.spatial_crop_mode == "center"


@pytest.mark.parametrize("split", ["train", "dev", "test"])
@pytest.mark.parametrize("modality,width", [("spatial", 2048), ("motion", 1024)])
def test_build_data_rejects_stale_file_width_for_any_split(
    tmp_path: Path, split: str, modality: str, width: int
) -> None:
    config = _synthetic_data_config(tmp_path)
    np.save(
        tmp_path / modality / split / f"{split}-clip.npy",
        np.ones((2, width - 1), dtype=np.float32),
    )

    with pytest.raises(ValueError, match=rf"{modality}.*width.*{split}/{split}-clip") as error:
        build_data(config)
    assert f"expected {width}, got {width - 1}" in str(error.value)


@pytest.mark.parametrize("modality,expected", [("spatial", 1024), ("motion", 2048)])
def test_build_data_rejects_manifest_width_against_model(
    tmp_path: Path, modality: str, expected: int
) -> None:
    config = _synthetic_data_config(tmp_path)
    config.model[f"{modality}_dim"] = expected

    with pytest.raises(ValueError, match=rf"{modality}.*width.*train/train-clip") as error:
        build_data(config)
    assert f"expected {expected}" in str(error.value)


def test_build_data_rejects_unannotated_feature_for_split(tmp_path: Path) -> None:
    config = _synthetic_data_config(tmp_path)
    path = Path(config.data.motion_manifest)
    manifest = FeatureManifest.load(path)
    FeatureManifest(
        1,
        manifest.encoder,
        manifest.expected_dim,
        (*manifest.records, FeatureRecord("extra", "dev", "dev/extra.npy", 2, 1024, "float32")),
    ).save(path)

    with pytest.raises(ValueError, match=r"unexpected motion feature for dev/extra"):
        build_data(config)

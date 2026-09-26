import hashlib
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from despamo.comparison import load_pair, preflight_run, preflight_sources, validate_pair_configs
from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.factory import build_model
from despamo.models.flan_t5 import FlanT5Backbone
from despamo.utils.hashing import sha256_file

FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def fake_version():
    # JSON equivalent of tools/dinov3/identity.py output; baseline never imports extractor.
    return {
        "schema_version": 1,
        "feature_format": "npy-float32-Tx2048",
        "model": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "model_sha": "b" * 40,
        "lock_sha256": "d" * 64,
        "source_resolution_policy": "record-per-frame",
        "scales": [224, 448],
        "color": "RGB",
        "resize": "Pillow.BICUBIC square",
        "range": "float32/255",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "tokens": "last_hidden_state[:,0,:]",
        "augmentation": "none",
        "libraries": {
            "torch": "2.5.1+cu121",
            "transformers": "4.56.2",
            "pillow": "11.3.0",
            "numpy": "1.26.4",
        },
    }


def make_pair(tmp_path, monkeypatch, *, version=None):
    import despamo.comparison as comparison

    monkeypatch.setattr(comparison, "EXPECTED", {"train": 1, "dev": 1, "test": 1})
    ann = tmp_path / "annotations"
    ann.mkdir()
    if version is None:
        version = fake_version()
    key = hashlib.sha256(
        json.dumps(version, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()
    clip_root, dino_root = tmp_path / "clip", tmp_path / "dino" / key
    clip_records, dino_records, rows = [], [], {}
    for split in ("train", "dev", "test"):
        clip_id = f"{split}-id"
        np.save(
            ann / f"{split}_info_ml.npy",
            {0: {"fileid": clip_id, "num_frames": 5, "text": "original text"}},
        )
        for root, records in ((clip_root, clip_records), (dino_root, dino_records)):
            (root / split).mkdir(parents=True)
            np.save(root / split / f"{clip_id}.npy", np.ones((5, 2048), np.float32))
            records.append(
                FeatureRecord(clip_id, split, f"{split}/{clip_id}.npy", 5, 2048, "float32")
            )
        rows[clip_id] = {
            "feature_hash": sha256_file(dino_root / split / f"{clip_id}.npy"),
            "frame_count": 5,
            "source_indices": list(range(5)),
            "sampled_images": {str(i): "a" * 64 for i in range(5)},
            "source_hash": "c" * 64,
            "paths": [f"{split}/{clip_id}/{i:06}.png" for i in range(5)],
            "resolutions": [[210, 260] for _ in range(5)],
        }
    clip_manifest = tmp_path / "clip.json"
    FeatureManifest(1, "clip", 2048, tuple(clip_records)).save(clip_manifest)
    (dino_root / "complete").mkdir()
    dino_manifest = dino_root / "complete/manifest.json"
    FeatureManifest(1, f"dinov3:{dino_root.name}", 2048, tuple(dino_records)).save(dino_manifest)
    (dino_root / "complete/frame_rows.json").write_text(
        json.dumps(
            {
                "encoder_key": dino_root.name,
                "spatial_manifest_hash": sha256_file(dino_manifest),
                "clips": rows,
            }
        )
    )
    (dino_root / "version.json").write_text(json.dumps(version))
    (dino_root / "failures.json").write_text("{}")
    common = {
        "seed": 0,
        "comparison": {
            "enabled": True,
            "smoke": False,
            "clip_root": str(clip_root),
            "clip_manifest": str(clip_manifest),
            "dino_root": str(dino_root),
            "dino_manifest": str(dino_manifest),
        },
        "data": {"annotation_root": str(ann), "batch_size": 2},
        "model": {
            "spatial_dim": 2048,
            "motion_dim": 1024,
            "vt_pooling": "masked_mean",
            "spatial_crop_mode": "full",
            "name": "google/flan-t5-xl",
            "lora_rank": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.1,
            "revision": FLAN_SHA,
            "prompt": "Translate the given sentence into {}.",
            "vt_weight": 1.0,
            "use_in_context": True,
            "num_in_context": 3,
            "warm_up_steps": 0,
        },
        "trainer": {
            "max_steps": 1000,
            "max_epochs": -1,
            "accumulate_grad_batches": 2,
            "precision": "bf16",
            "default_root_dir": str(tmp_path / "run"),
        },
        "optimizer": {"learning_rate": 6e-4, "weight_decay": 0.01},
        "evaluation": {"generation": "deterministic", "beam_size": 5, "expected_test_items": 642},
    }
    clip = OmegaConf.merge(
        common, {"data": {"spatial_root": str(clip_root), "spatial_manifest": str(clip_manifest)}}
    )
    dino = OmegaConf.merge(
        common, {"data": {"spatial_root": str(dino_root), "spatial_manifest": str(dino_manifest)}}
    )
    return clip, dino, ann, dino_root


def test_dino_content_hash_fails_on_same_shape_feature_change_and_rowmap_drift(
    tmp_path, monkeypatch
):
    from despamo.comparison import validate_dino_content

    clip, _, _, root = make_pair(tmp_path, monkeypatch)
    mapping = root / "complete/frame_rows.json"
    original = validate_dino_content(clip)
    assert original == sha256_file(mapping)
    feature = root / "test/test-id.npy"
    np.save(feature, np.zeros((5, 2048), dtype=np.float32))
    with pytest.raises(ValueError, match="test/test-id.*feature hash"):
        validate_dino_content(clip)
    rows = json.loads(mapping.read_text())
    rows["clips"]["test-id"]["feature_hash"] = sha256_file(feature)
    mapping.write_text(json.dumps(rows))
    assert validate_dino_content(clip) != original


def test_two_step_comparison_smoke_reaches_post_warmup_step(tmp_path, monkeypatch):
    from despamo.comparison import validate_comparison_protocol
    from despamo.training.baseline_module import combine_losses

    clip, dino, _, _ = make_pair(tmp_path, monkeypatch)
    for config in (clip, dino):
        config.comparison.smoke = True
        config.trainer.max_steps = 2
        validate_comparison_protocol(config)
    translation, vt = torch.tensor(3.0), torch.tensor(2.0)
    assert combine_losses(translation, vt, 0, 0, 1.0) == vt
    assert combine_losses(translation, vt, 1, 0, 1.0) == translation + vt


def test_real_overlays_differ_only_by_spatial_source(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", str(tmp_path))
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", str(tmp_path))
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path))
    monkeypatch.setenv("DINO_ROOT", str(tmp_path / "dino"))
    monkeypatch.setenv("COMPARISON_RUN_DIR", str(tmp_path / "run"))
    configs = Path(__file__).resolve().parents[2] / "configs"
    base = [
        configs / "data/phoenix14t.yaml",
        configs / "model/spamo_flan_t5_xl.yaml",
        configs / "experiment/phoenix14t_baseline.yaml",
        configs / "experiment/phoenix14t_encoder_comparison.yaml",
    ]
    clip, dino = load_pair(base)
    validate_pair_configs(clip, dino)
    assert clip.model.vt_pooling == dino.model.vt_pooling == "masked_mean"
    assert clip.model.spatial_crop_mode == dino.model.spatial_crop_mode == "full"
    assert clip.trainer.max_steps == dino.trainer.max_steps == 1000
    assert clip.model.lora_rank == dino.model.lora_rank == 16
    assert clip.model.lora_alpha == dino.model.lora_alpha == 32
    assert clip.model.lora_dropout == dino.model.lora_dropout == 0.1
    assert clip.model.name == dino.model.name == "google/flan-t5-xl"
    assert clip.model.revision == dino.model.revision == FLAN_SHA
    assert clip.model.prompt == dino.model.prompt == "Translate the given sentence into {}."
    assert clip.model.vt_weight == dino.model.vt_weight == 1.0
    assert clip.data.batch_size == dino.data.batch_size == 2
    assert clip.trainer.accumulate_grad_batches == dino.trainer.accumulate_grad_batches == 2
    assert clip.optimizer.learning_rate == dino.optimizer.learning_rate == 6e-4
    assert clip.evaluation.generation == dino.evaluation.generation == "deterministic"
    assert clip.data.motion_root == dino.data.motion_root

    class FakeLoRA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lora_A = torch.nn.Linear(1, 4, bias=False)
            self.lora_B = torch.nn.Linear(4, 1, bias=False)

    monkeypatch.setattr(FlanT5Backbone, "from_pretrained", lambda *args, **kwargs: FakeLoRA())
    torch.manual_seed(0)
    clip_model = build_model(clip)
    torch.manual_seed(0)
    dino_model = build_model(dino)
    assert clip_model is not dino_model
    assert "language_model.lora_A.weight" in clip_model.state_dict()
    for key, value in clip_model.state_dict().items():
        torch.testing.assert_close(value, dino_model.state_dict()[key])
    seeded_clip, seeded_dino = load_pair(base, ["seed=2"])
    assert seeded_clip.seed == seeded_dino.seed == 2
    validate_pair_configs(seeded_clip, seeded_dino)


def test_all_splits_parity_and_corruption_before_training(tmp_path, monkeypatch):
    clip, dino, ann, root = make_pair(tmp_path, monkeypatch)
    validate_pair_configs(clip, dino)
    counts = {"clip": {"train": 1, "dev": 1, "test": 1}, "dino": {"train": 1, "dev": 1, "test": 1}}
    assert preflight_sources(clip) == preflight_sources(dino) == counts
    raw = np.load(ann / "dev_info_ml.npy", allow_pickle=True).item()
    raw[0]["num_frames"] = 6
    np.save(ann / "dev_info_ml.npy", raw)
    with pytest.raises(ValueError, match="dev/dev-id.*row"):
        preflight_sources(clip)
    raw[0]["num_frames"] = 5
    np.save(ann / "dev_info_ml.npy", raw)
    path = root / "test/test-id.npy"
    np.save(path, np.ones((4, 2048), np.float32))
    with pytest.raises(ValueError, match="test/test-id.*row"):
        preflight_sources(dino)


def test_incomplete_or_unmatched_configuration_fails_closed(tmp_path, monkeypatch):
    clip, dino, _, root = make_pair(tmp_path, monkeypatch)
    dino.optimizer = {"learning_rate": 0.0001}
    with pytest.raises(ValueError, match="only spatial"):
        validate_pair_configs(clip, dino)
    dino.optimizer = {"learning_rate": 6e-4, "weight_decay": 0.01}
    dino.model.prompt = "different prompt"
    with pytest.raises(ValueError, match="only spatial"):
        validate_pair_configs(clip, dino)
    (root / "version.json").write_text('{"model_sha":"main"}')
    with pytest.raises(ValueError, match="version key"):
        preflight_sources(clip)
    (root / "version.json").write_text(json.dumps(fake_version()))
    (root / "complete/frame_rows.json").unlink()
    with pytest.raises((ValueError, FileNotFoundError), match="frame_rows"):
        preflight_sources(clip)


def test_fake_full_length_batches_preserve_both_source_masks(tmp_path, monkeypatch):
    from despamo.factory import build_data

    clip, dino, ann, root = make_pair(tmp_path, monkeypatch)
    clip.model.max_frame_len = dino.model.max_frame_len = 512
    for split in ("train", "dev", "test"):
        path = ann / f"{split}_info_ml.npy"
        raw = np.load(path, allow_pickle=True).item()
        raw[0].update(
            signer="Signer01",
            gloss="WIND",
            text="wind",
            en_text="wind",
            es_text="viento",
            fr_text="vent",
        )
        if split == "train":
            raw[0]["num_frames"] = 520
        np.save(path, raw)
    train_id = "train-id"
    for config in (clip, dino):
        source = Path(config.data.spatial_root)
        np.save(source / "train" / f"{train_id}.npy", np.ones((520, 2048), np.float32))
        path = Path(config.data.spatial_manifest)
        manifest = FeatureManifest.load(path)
        records = tuple(
            FeatureRecord(
                r.clip_id, r.split, r.path, 520 if r.split == "train" else r.length, 2048, "float32"
            )
            for r in manifest.records
        )
        FeatureManifest(1, manifest.encoder, 2048, records).save(path)
    rows_path = root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    train_row = rows["clips"][train_id]
    train_row["frame_count"] = 520
    train_row["source_indices"] = list(range(520))
    train_row["paths"] = [f"train/{train_id}/{i:06}.png" for i in range(520)]
    train_row["resolutions"] = [[210, 260] for _ in range(520)]
    train_row["sampled_images"] = {
        str(int(519 * fraction + 0.5)): "a" * 64 for fraction in (0.1, 0.3, 0.5, 0.7, 0.9)
    }
    train_row["feature_hash"] = sha256_file(root / "train/train-id.npy")
    rows["spatial_manifest_hash"] = sha256_file(root / "complete/manifest.json")
    rows_path.write_text(json.dumps(rows))
    motion_root = tmp_path / "motion"
    motion_records = []
    for split in ("train", "dev", "test"):
        (motion_root / split).mkdir(parents=True)
        np.save(motion_root / split / f"{split}-id.npy", np.ones((5, 1024), np.float32))
        motion_records.append(
            FeatureRecord(f"{split}-id", split, f"{split}/{split}-id.npy", 5, 1024, "float32")
        )
    motion_manifest = tmp_path / "motion.json"
    FeatureManifest(1, "videomae", 1024, tuple(motion_records)).save(motion_manifest)
    for config in (clip, dino):
        config.data.motion_root = str(motion_root)
        config.data.motion_manifest = str(motion_manifest)
        config.data.num_workers = 0
    validate_pair_configs(clip, dino)
    assert preflight_sources(clip) == preflight_sources(dino)
    clip_batch = next(iter(build_data(clip).train_dataloader()))
    dino_batch = next(iter(build_data(dino).train_dataloader()))
    assert clip_batch.spatial.shape == dino_batch.spatial.shape == (1, 520, 2048)
    assert clip_batch.spatial_mask.equal(dino_batch.spatial_mask)
    assert clip_batch.spatial_mask.all()
    assert clip_batch.motion_mask.equal(dino_batch.motion_mask)


def test_run_preflight_checks_pair_selected_config_and_both_sources(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    clip, dino, _, root = make_pair(tmp_path, monkeypatch)
    monkeypatch.setattr(comparison, "load_pair", lambda base, overrides=(): (clip, dino))
    paths = [tmp_path / "base.yaml", comparison.CONFIGS / "phoenix14t_clip_control.yaml"]
    counts = {"clip": {"train": 1, "dev": 1, "test": 1}, "dino": {"train": 1, "dev": 1, "test": 1}}
    assert preflight_run(clip, paths, ["seed=0"]) == counts
    dino.model.vt_weight = 0.5
    with pytest.raises(ValueError, match="only spatial"):
        preflight_run(clip, paths, ["seed=0"])
    dino.model.vt_weight = 1.0
    changed = OmegaConf.merge(clip, {"model": {"prompt": "wrong prompt"}})
    with pytest.raises(ValueError, match="selected config"):
        preflight_run(changed, paths, ["seed=0"])
    (root / "complete/frame_rows.json").unlink()
    with pytest.raises(FileNotFoundError, match="frame_rows"):
        preflight_run(clip, paths, ["seed=0"])


def test_dotlist_cannot_bind_dino_overlay_to_clip_features(tmp_path, monkeypatch):
    import despamo.comparison as comparison

    clip, dino, _, _ = make_pair(tmp_path, monkeypatch)
    overrides = [
        f"data.spatial_root={clip.data.spatial_root}",
        f"data.spatial_manifest={clip.data.spatial_manifest}",
    ]
    swapped_clip = OmegaConf.merge(clip, OmegaConf.from_dotlist(overrides))
    swapped_dino = OmegaConf.merge(dino, OmegaConf.from_dotlist(overrides))
    with pytest.raises(ValueError, match="dino spatial source binding"):
        validate_pair_configs(swapped_clip, swapped_dino)
    monkeypatch.setattr(
        comparison, "load_pair", lambda base, overrides=(): (swapped_clip, swapped_dino)
    )
    monkeypatch.setattr(
        comparison,
        "preflight_sources",
        lambda config: pytest.fail("feature I/O started before source binding"),
    )
    paths = [tmp_path / "base.yaml", comparison.CONFIGS / "phoenix14t_dinov3.yaml"]
    with pytest.raises(ValueError, match="dino spatial source binding"):
        preflight_run(swapped_dino, paths, overrides)


def test_pair_rejects_both_names_pointing_to_same_spatial_source(tmp_path, monkeypatch):
    clip, dino, _, _ = make_pair(tmp_path, monkeypatch)
    clip.comparison.clip_root = dino.comparison.dino_root
    clip.comparison.clip_manifest = dino.comparison.dino_manifest
    dino.comparison.clip_root = dino.comparison.dino_root
    dino.comparison.clip_manifest = dino.comparison.dino_manifest
    clip.data.spatial_root = dino.data.spatial_root
    clip.data.spatial_manifest = dino.data.spatial_manifest
    with pytest.raises(ValueError, match="distinct spatial"):
        validate_pair_configs(clip, dino)
    with pytest.raises(ValueError, match="distinct spatial"):
        preflight_sources(clip)


def test_rejects_invalid_sampled_image_sha_and_feature_bytes(tmp_path, monkeypatch):
    clip, _, _, root = make_pair(tmp_path, monkeypatch)
    rows_path = root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    rows["clips"]["dev-id"]["sampled_images"]["2"] = "not-a-sha256"
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="dev/dev-id.*frame_rows"):
        preflight_sources(clip)
    rows["clips"]["dev-id"]["sampled_images"]["2"] = "a" * 64
    rows_path.write_text(json.dumps(rows))
    np.save(root / "dev/dev-id.npy", np.zeros((5, 2048), np.float32))
    with pytest.raises(ValueError, match="dev/dev-id.*frame_rows"):
        preflight_sources(clip)


def test_rejects_missing_source_provenance_and_duplicate_source_paths(tmp_path, monkeypatch):
    clip, _, _, root = make_pair(tmp_path, monkeypatch)
    rows_path = root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    row = rows["clips"]["train-id"]
    row["source_hash"] = "mutable-source"
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="train/train-id.*frame_rows"):
        preflight_sources(clip)
    row["source_hash"] = "c" * 64
    row["paths"][1] = row["paths"][0]
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="train/train-id.*frame_rows"):
        preflight_sources(clip)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scales", [448, 224]),
        ("augmentation", "horizontal-flip"),
        ("schema_version", 2),
        ("feature_format", "npy-float16-Tx2048"),
        ("source_resolution_policy", "fixed"),
        ("color", "BGR"),
        ("resize", "Pillow.BILINEAR square"),
        ("range", "float32/256"),
        ("mean", [0.5, 0.456, 0.406]),
        ("std", [0.229, 0.224, 0.23]),
        ("tokens", "last_hidden_state[:,1,:]"),
        ("lock_sha256", "not-a-sha"),
        (
            "libraries",
            {"torch": "", "transformers": "4.56.2", "pillow": "11.3.0", "numpy": "1.26.4"},
        ),
        ("libraries", {"torch": "2.5.1"}),
    ],
)
def test_rejects_self_hashed_dino_version_with_different_s2_recipe(
    tmp_path, monkeypatch, field, value
):
    version = fake_version()
    version[field] = value
    clip, _, _, root = make_pair(tmp_path, monkeypatch, version=version)
    assert (
        root.name
        == hashlib.sha256(
            json.dumps(version, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
        ).hexdigest()
    )
    with pytest.raises(ValueError, match="version.*recipe"):
        preflight_sources(clip)


def test_rejects_source_paths_out_of_lexicographic_order(tmp_path, monkeypatch):
    clip, _, _, root = make_pair(tmp_path, monkeypatch)
    rows_path = root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    paths = rows["clips"]["dev-id"]["paths"]
    paths[0], paths[1] = paths[1], paths[0]
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="dev/dev-id.*frame_rows"):
        preflight_sources(clip)


@pytest.mark.parametrize("source", ["clip", "dino"])
def test_npz_disguised_as_npy_fails_with_clip_context_and_closes(tmp_path, monkeypatch, source):
    import despamo.comparison as comparison

    clip, dino, _, _ = make_pair(tmp_path, monkeypatch)
    path = Path(clip.comparison[f"{source}_root"]) / "train/train-id.npy"
    with path.open("wb") as handle:
        np.savez(handle, data=np.ones((5, 2048), np.float32))
    original_load = comparison.np.load
    archives = []

    def tracked_load(file, *args, **kwargs):
        value = original_load(file, *args, **kwargs)
        if Path(file) == path:
            archives.append(value)
        return value

    monkeypatch.setattr(comparison.np, "load", tracked_load)
    with pytest.raises(ValueError, match=rf"{source} train/train-id: invalid feature"):
        preflight_sources({"clip": clip, "dino": dino}[source])
    assert len(archives) == 1
    assert archives[0].zip is None


def test_comparison_checkpoint_requires_measured_language_revision(tmp_path, monkeypatch):
    from despamo.comparison import (
        annotation_hashes,
        comparison_code_sha256,
        validate_comparison_checkpoint,
    )

    clip, _, ann, dino_root = make_pair(tmp_path, monkeypatch)
    motion = tmp_path / "motion.json"
    motion.write_text("motion")
    clip.data.motion_manifest = str(motion)
    metadata = {
        "seed": 0,
        "config": OmegaConf.to_container(clip, resolve=True),
        "spatial_manifest_sha256": sha256_file(Path(clip.data.spatial_manifest)),
        "motion_manifest_sha256": sha256_file(motion),
        "annotation_sha256": annotation_hashes(clip),
        "comparison_code_sha256": comparison_code_sha256(Path(__file__).resolve().parents[2]),
        "dino_frame_rows_sha256": sha256_file(dino_root / "complete/frame_rows.json"),
        "resume_reproducibility": "fresh",
        "model_source": {
            "identifier": "google/flan-t5-xl",
            "tuning_type": "lora",
            "revision_status": "resolved",
            "revision": FLAN_SHA,
        },
    }
    checkpoint = _full_checkpoint(metadata)
    validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    checkpoint["global_step"] = 999
    with pytest.raises(ValueError, match="step"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    checkpoint["global_step"] = 1000
    motion.write_text("changed motion feature manifest")
    with pytest.raises(ValueError, match="manifest provenance mismatch"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    motion.write_text("motion")
    checkpoint["run_metadata"]["model_source"]["revision"] = "c" * 40
    with pytest.raises(ValueError, match="revision"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")
    checkpoint["run_metadata"]["model_source"]["revision"] = FLAN_SHA
    original = np.load(ann / "train_info_ml.npy", allow_pickle=True).item()
    original[0]["text"] = "different text"
    np.save(ann / "train_info_ml.npy", original)
    with pytest.raises(ValueError, match="annotation SHA256 mismatch"):
        validate_comparison_checkpoint(clip, checkpoint, "deterministic")


def _full_checkpoint(metadata):
    return {
        "pytorch-lightning_version": "1.9.5",
        "global_step": 1000,
        "state_dict": {"weight": torch.ones(1)},
        "optimizer_states": [
            {
                "state": {
                    0: {
                        "step": torch.tensor(1000.0),
                        "exp_avg": torch.ones(1),
                        "exp_avg_sq": torch.ones(1),
                    }
                },
                "param_groups": [{"params": [0]}],
            }
        ],
        "lr_schedulers": [{"last_epoch": 1000}],
        "run_metadata": metadata,
    }


@pytest.mark.parametrize("missing", ["exp_avg", "exp_avg_sq"])
def test_full_checkpoint_rejects_missing_adamw_moments(tmp_path, monkeypatch, missing):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    checkpoint["optimizer_states"][0]["state"][0].pop(missing, None)
    with pytest.raises(ValueError, match="optimizer.*moment"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")


def test_full_checkpoint_rejects_stale_scheduler(tmp_path, monkeypatch):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    checkpoint["lr_schedulers"][0]["last_epoch"] = 0
    with pytest.raises(ValueError, match="scheduler.*step"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")


def _comparison_checkpoint_fixture(tmp_path, monkeypatch):
    from despamo.comparison import annotation_hashes, comparison_code_sha256

    clip, _, _, dino_root = make_pair(tmp_path, monkeypatch)
    motion = tmp_path / "motion.json"
    motion.write_text("motion")
    clip.data.motion_manifest = str(motion)
    metadata = {
        "seed": clip.seed,
        "config": OmegaConf.to_container(clip, resolve=True),
        "spatial_manifest_sha256": sha256_file(Path(clip.data.spatial_manifest)),
        "motion_manifest_sha256": sha256_file(motion),
        "annotation_sha256": annotation_hashes(clip),
        "comparison_code_sha256": comparison_code_sha256(Path(__file__).resolve().parents[2]),
        "dino_frame_rows_sha256": sha256_file(dino_root / "complete/frame_rows.json"),
        "resume_reproducibility": "fresh",
        "model_source": {
            "identifier": "google/flan-t5-xl",
            "tuning_type": "lora",
            "revision_status": "resolved",
            "revision": FLAN_SHA,
        },
    }
    return clip, _full_checkpoint(metadata)


def test_checkpoint_rejects_rehashed_dino_features_from_different_run(tmp_path, monkeypatch):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    dino_root = Path(config.comparison.dino_root)
    feature = dino_root / "test/test-id.npy"
    np.save(feature, np.zeros((5, 2048), dtype=np.float32))
    rows_path = dino_root / "complete/frame_rows.json"
    rows = json.loads(rows_path.read_text())
    rows["clips"]["test-id"]["feature_hash"] = sha256_file(feature)
    rows_path.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="DINO content SHA256 mismatch"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")


def test_comparison_checkpoint_rejects_code_drift_before_model(tmp_path, monkeypatch):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    checkpoint["run_metadata"]["comparison_code_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="comparison code SHA256 mismatch"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")


@pytest.mark.parametrize(
    "case",
    [
        "missing-version",
        "invalid-version",
        "missing-state",
        "non-tensor-state",
        "missing-optimizers",
        "empty-optimizer-state",
        "missing-param-groups",
        "missing-optimizer-step",
        "wrong-optimizer-step",
        "orphan-optimizer-step",
        "missing-schedulers",
    ],
)
def test_comparison_rejects_non_lightning_or_untrained_checkpoint(tmp_path, monkeypatch, case):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    if case == "missing-version":
        del checkpoint["pytorch-lightning_version"]
    elif case == "invalid-version":
        checkpoint["pytorch-lightning_version"] = ""
    elif case == "missing-state":
        checkpoint["state_dict"] = {}
    elif case == "non-tensor-state":
        checkpoint["state_dict"] = {"weight": "not a tensor"}
    elif case == "missing-optimizers":
        checkpoint["optimizer_states"] = []
    elif case == "empty-optimizer-state":
        checkpoint["optimizer_states"][0]["state"] = {}
    elif case == "missing-param-groups":
        checkpoint["optimizer_states"][0]["param_groups"] = []
    elif case == "missing-optimizer-step":
        del checkpoint["optimizer_states"][0]["state"][0]["step"]
    elif case == "wrong-optimizer-step":
        checkpoint["optimizer_states"][0]["state"][0]["step"] = torch.tensor(1.0)
    elif case == "orphan-optimizer-step":
        checkpoint["optimizer_states"][0]["param_groups"][0]["params"] = [1]
    else:
        checkpoint["lr_schedulers"] = []
    with pytest.raises(ValueError, match="comparison.*(Lightning|state_dict|optimizer|scheduler)"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("model.vt_pooling", "legacy_mean"),
        ("model.spatial_crop_mode", "center"),
        ("model.lora_rank", 8),
        ("model.lora_dropout", 0.2),
        ("model.tuning_type", "freeze"),
        ("optimizer.learning_rate", 1e-4),
        ("trainer.precision", "32"),
        ("trainer.max_steps", 999),
        ("data.spatial_root", "other-source"),
    ],
)
def test_comparison_rejects_mutated_self_consistent_protocol(tmp_path, monkeypatch, path, value):
    from despamo.comparison import validate_comparison_checkpoint

    config, checkpoint = _comparison_checkpoint_fixture(tmp_path, monkeypatch)
    OmegaConf.update(config, path, value)
    checkpoint["run_metadata"]["config"] = deepcopy(OmegaConf.to_container(config, resolve=True))
    with pytest.raises(ValueError, match="comparison.*(protocol|source binding)"):
        validate_comparison_checkpoint(config, checkpoint, "deterministic")

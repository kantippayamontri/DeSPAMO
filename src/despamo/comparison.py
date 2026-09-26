"""CPU-only, fail-closed source parity gate for matched encoder comparisons."""

import hashlib
import json
import math
import re
import statistics
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from despamo.config import load_config
from despamo.data.manifest import FeatureManifest
from despamo.utils.hashing import sha256_file

EXPECTED = {"train": 7096, "dev": 519, "test": 642}
EXPECTED_ROWS = {"train": 827354, "dev": 55775, "test": 64627}
FLAN_SHA = "7d6315df2c2fb742f0f5b556879d730926ca9001"
CONFIGS = Path(__file__).resolve().parents[2] / "configs/experiment"


def comparison_code_sha256(root: Path) -> str:
    """Hash current comparison code/config bytes, including uncommitted changes."""
    paths = [root / "pyproject.toml", root / "uv.lock"]
    for pattern in ("src/despamo/**/*.py", "scripts/*.py", "configs/**/*.yaml"):
        paths.extend(path for path in root.glob(pattern) if path.is_file() or path.is_symlink())
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix()):
        if path.is_symlink():
            raise ValueError(f"comparison code input must not be a symlink: {path}")
        contents = path.read_bytes()
        name = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(len(contents).to_bytes(8, "big"))
        digest.update(contents)
    return digest.hexdigest()


def load_pair(base: list[Path], overrides: Sequence[str] = ()) -> tuple[DictConfig, DictConfig]:
    return (
        load_config([*base, CONFIGS / "phoenix14t_clip_control.yaml"], overrides),
        load_config([*base, CONFIGS / "phoenix14t_dinov3.yaml"], overrides),
    )


def _require_source_binding(config: DictConfig, source: str) -> None:
    actual = (config.data.spatial_root, config.data.spatial_manifest)
    expected = (config.comparison[f"{source}_root"], config.comparison[f"{source}_manifest"])
    if actual != expected:
        raise ValueError(f"{source} spatial source binding mismatch: root/manifest")


def _require_distinct_sources(config: DictConfig) -> None:
    if any(
        Path(config.comparison[f"clip_{field}"]).resolve()
        == Path(config.comparison[f"dino_{field}"]).resolve()
        for field in ("root", "manifest")
    ):
        raise ValueError("comparison requires distinct spatial roots and manifests")


def validate_pair_configs(clip: DictConfig, dino: DictConfig) -> None:
    _require_source_binding(clip, "clip")
    _require_source_binding(dino, "dino")
    _require_distinct_sources(clip)
    left, right = (OmegaConf.to_container(config, resolve=True) for config in (clip, dino))
    for config in (left, right):
        for name in ("spatial_root", "spatial_manifest"):
            del config["data"][name]
    if left != right:
        raise ValueError("only spatial root/manifest may differ between CLIP and DINO")
    if (
        not clip.comparison.enabled
        or clip.seed != dino.seed
        or clip.comparison.smoke != dino.comparison.smoke
    ):
        raise ValueError("paired comparison requires same seed and protocol")


def preflight_run(
    config: DictConfig, paths: list[Path], overrides: Sequence[str]
) -> dict[str, dict[str, int]]:
    if len(paths) < 2:
        raise ValueError("comparison needs common config followed by source overlay")
    overlay = paths[-1].resolve()
    overlays = {
        (CONFIGS / "phoenix14t_clip_control.yaml").resolve(): "clip",
        (CONFIGS / "phoenix14t_dinov3.yaml").resolve(): "dino",
    }
    if overlay not in overlays:
        raise ValueError("comparison source overlay must be last --config layer")
    clip, dino = load_pair(paths[:-1], overrides)
    validate_pair_configs(clip, dino)
    source = overlays[overlay]
    expected = {"clip": clip, "dino": dino}[source]
    _require_source_binding(expected, source)
    _require_source_binding(config, source)
    if OmegaConf.to_container(config, resolve=True) != OmegaConf.to_container(
        expected, resolve=True
    ):
        raise ValueError("selected config differs from paired source overlay")
    return preflight_sources(expected)


def _annotations(root: Path) -> dict[str, dict[str, int]]:
    result: dict[str, dict[str, int]] = {}
    all_ids: set[str] = set()
    for split in EXPECTED:
        raw = np.load(root / f"{split}_info_ml.npy", allow_pickle=True).item()
        if not isinstance(raw, dict):
            raise ValueError(f"{split}: annotation is not a mapping")
        entries: dict[str, int] = {}
        for index in sorted(key for key in raw if type(key) is int):
            item = raw[index]
            clip_id, count = item["fileid"], item["num_frames"]
            if (
                not isinstance(clip_id, str)
                or not clip_id
                or "/" in clip_id
                or "\\" in clip_id
                or type(count) is not int
                or count < 5
                or clip_id in all_ids
            ):
                raise ValueError(f"{split}/{clip_id}: invalid/duplicate annotation")
            entries[clip_id] = count
            all_ids.add(clip_id)
        if len(entries) != EXPECTED[split]:
            raise ValueError(f"{split}: expected {EXPECTED[split]} annotation IDs")
        if EXPECTED == {"train": 7096, "dev": 519, "test": 642} and (
            sum(entries.values()) != EXPECTED_ROWS[split]
        ):
            raise ValueError(f"{split}: approved annotation row total mismatch")
        result[split] = entries
    return result


def validate_comparison_protocol(config: DictConfig) -> None:
    """Check controlled comparison settings and source binding without feature I/O."""
    if not config.comparison.enabled:
        raise ValueError("comparison preflight requires comparison.enabled")
    _require_distinct_sources(config)
    if (
        config.model.vt_pooling != "masked_mean"
        or config.model.spatial_crop_mode != "full"
        or config.evaluation.generation != "deterministic"
        or config.evaluation.beam_size != 5
        or config.model.name != "google/flan-t5-xl"
        or config.model.revision != FLAN_SHA
        or config.model.prompt != "Translate the given sentence into {}."
        or config.model.vt_weight != 1.0
        or config.model.use_in_context is not True
        or config.model.num_in_context != 3
        or config.model.warm_up_steps != 0
        or config.model.lora_rank != 16
        or config.model.lora_alpha != 32
        or config.model.lora_dropout != 0.1
        or config.model.get("tuning_type", "lora") != "lora"
        or config.model.spatial_dim != 2048
        or config.model.motion_dim != 1024
        or config.trainer.max_epochs != -1
        or (
            config.trainer.max_steps not in (1, 2)
            if config.comparison.smoke
            else config.trainer.max_steps != 1000
        )
    ):
        raise ValueError("comparison protocol mismatch")
    if (
        config.data.batch_size != 2
        or config.trainer.accumulate_grad_batches != 2
        or config.trainer.precision != "bf16"
        or config.optimizer.learning_rate != 6e-4
        or config.optimizer.weight_decay != 0.01
    ):
        raise ValueError("comparison optimizer/batch/precision protocol mismatch")
    sources = {
        "clip": (Path(config.comparison.clip_root), Path(config.comparison.clip_manifest)),
        "dino": (Path(config.comparison.dino_root), Path(config.comparison.dino_manifest)),
    }
    if (Path(config.data.spatial_root), Path(config.data.spatial_manifest)) not in sources.values():
        raise ValueError("comparison selected spatial source binding mismatch")
    dino_root, dino_manifest = sources["dino"]
    if dino_manifest != dino_root / "complete/manifest.json":
        raise ValueError("DINO manifest must be complete/manifest.json at encoder-key root")


def preflight_sources(config: DictConfig) -> dict[str, dict[str, int]]:
    validate_comparison_protocol(config)
    sources = {
        "clip": (Path(config.comparison.clip_root), Path(config.comparison.clip_manifest)),
        "dino": (Path(config.comparison.dino_root), Path(config.comparison.dino_manifest)),
    }
    dino_root, dino_manifest = sources["dino"]
    for file in (
        dino_root / "version.json",
        dino_root / "complete/frame_rows.json",
        dino_root / "failures.json",
    ):
        if not file.is_file():
            raise FileNotFoundError(f"DINO complete file missing: {file}")
    version = json.loads((dino_root / "version.json").read_text())
    if not isinstance(version, dict):
        raise ValueError("DINO version key/model revision mismatch")
    canonical = json.dumps(version, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    if (
        version.get("model") != "facebook/dinov3-vitl16-pretrain-lvd1689m"
        or not re.fullmatch(r"[0-9a-f]{40}", str(version.get("model_sha")))
        or hashlib.sha256(canonical.encode()).hexdigest() != dino_root.name
    ):
        raise ValueError("DINO version key/model revision mismatch")
    recipe = {
        "feature_format": "npy-float32-Tx2048",
        "source_resolution_policy": "record-per-frame",
        "scales": [224, 448],
        "color": "RGB",
        "resize": "Pillow.BICUBIC square",
        "range": "float32/255",
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "tokens": "last_hidden_state[:,0,:]",
        "augmentation": "none",
    }
    libraries = version.get("libraries")
    if (
        type(version.get("schema_version")) is not int
        or version["schema_version"] != 1
        or any(version.get(name) != value for name, value in recipe.items())
        or not isinstance(version.get("lock_sha256"), str)
        or not re.fullmatch(r"[0-9a-f]{64}", version["lock_sha256"])
        or not isinstance(libraries, dict)
        or set(libraries) != {"torch", "transformers", "pillow", "numpy"}
        or any(not isinstance(value, str) or not value.strip() for value in libraries.values())
    ):
        raise ValueError("DINO version recipe mismatch")
    if json.loads((dino_root / "failures.json").read_text()) != {}:
        raise ValueError("DINO extraction failures.json must be empty")
    rows = json.loads((dino_root / "complete/frame_rows.json").read_text())
    if (
        not isinstance(rows, dict)
        or rows.get("encoder_key") != dino_root.name
        or rows.get("spatial_manifest_hash") != sha256_file(dino_manifest)
    ):
        raise ValueError("DINO frame_rows manifest hash/encoder_key mismatch")
    if not isinstance(rows.get("clips"), dict):
        raise ValueError("DINO frame_rows clips mapping missing")
    annotations = _annotations(Path(config.data.annotation_root))
    manifests = {name: FeatureManifest.load(path) for name, (_, path) in sources.items()}
    if manifests["dino"].encoder != f"dinov3:{dino_root.name}":
        raise ValueError("DINO manifest encoder key mismatch")
    seen: dict[str, dict[tuple[str, str], int]] = {name: {} for name in sources}
    for name, manifest in manifests.items():
        if manifest.expected_dim != 2048:
            raise ValueError(f"{name}: expected spatial width 2048")
        counts = Counter(record.split for record in manifest.records)
        if counts != EXPECTED:
            raise ValueError(f"{name}: complete split counts mismatch: {counts}")
        for record in manifest.records:
            split, clip_id = record.split, record.clip_id
            if split not in EXPECTED or clip_id not in annotations[split]:
                raise ValueError(f"{name} {split}/{clip_id}: extra feature ID")
            expected = annotations[split][clip_id]
            if record.length != expected:
                raise ValueError(f"{name} {split}/{clip_id}: annotation/feature row mismatch")
            feature = sources[name][0] / record.path
            try:
                array = np.load(feature, allow_pickle=False, mmap_mode="r")
            except (OSError, ValueError, EOFError) as exc:
                raise ValueError(f"{name} {split}/{clip_id}: invalid feature") from exc
            if not isinstance(array, np.ndarray):
                close = getattr(array, "close", None)
                if callable(close):
                    close()
                raise ValueError(f"{name} {split}/{clip_id}: invalid feature")
            if array.shape != (expected, 2048) or str(array.dtype) != record.dtype:
                raise ValueError(f"{name} {split}/{clip_id}: file/manifest row or dtype mismatch")
            if name == "dino" and (str(array.dtype) != "float32" or not np.isfinite(array).all()):
                raise ValueError(f"dino {split}/{clip_id}: feature dtype/finiteness mismatch")
            del array
            seen[name][split, clip_id] = expected
            if name == "dino":
                row = rows["clips"].get(clip_id)
                sample_indices = {
                    str(int((expected - 1) * position + 0.5))
                    for position in (0.1, 0.3, 0.5, 0.7, 0.9)
                }
                if (
                    not isinstance(row, dict)
                    or row.get("frame_count") != expected
                    or row.get("source_indices") != list(range(expected))
                    or not isinstance(row.get("sampled_images"), dict)
                    or set(row["sampled_images"]) != sample_indices
                    or any(
                        not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest)
                        for digest in row["sampled_images"].values()
                    )
                    or not isinstance(row.get("source_hash"), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", row["source_hash"])
                    or not isinstance(row.get("paths"), list)
                    or len(row["paths"]) != expected
                    or any(
                        not isinstance(path, str)
                        or not path.startswith(f"{split}/{clip_id}/")
                        or not path.endswith(".png")
                        for path in row["paths"]
                    )
                    or len(set(row["paths"])) != expected
                    or row["paths"] != sorted(row["paths"])
                    or not isinstance(row.get("resolutions"), list)
                    or len(row["resolutions"]) != expected
                    or any(
                        not isinstance(size, list)
                        or len(size) != 2
                        or any(type(side) is not int or side <= 0 for side in size)
                        for size in row["resolutions"]
                    )
                    or row.get("feature_hash") != sha256_file(feature)
                ):
                    raise ValueError(f"dino {split}/{clip_id}: frame_rows/feature hash mismatch")
    reference = {
        (split, clip_id): count
        for split, items in annotations.items()
        for clip_id, count in items.items()
    }
    if seen["clip"] != reference or seen["dino"] != reference:
        raise ValueError("CLIP/DINO full split ID/row parity mismatch")
    if set(rows["clips"]) != {clip for _, clip in reference}:
        raise ValueError("DINO frame_rows extra/missing IDs")
    return {
        name: {split: sum(key[0] == split for key in records) for split in EXPECTED}
        for name, records in seen.items()
    }


def annotation_hashes(config: DictConfig) -> dict[str, str]:
    root = Path(config.data.annotation_root)
    return {split: sha256_file(root / f"{split}_info_ml.npy") for split in ("train", "dev", "test")}


def dino_frame_rows_hash(config: DictConfig) -> str:
    return sha256_file(Path(config.comparison.dino_root) / "complete/frame_rows.json")


def validate_dino_content(config: DictConfig) -> str:
    """Bind all DINO feature bytes to the complete frame-row map."""
    root = Path(config.comparison.dino_root)
    manifest_path = Path(config.comparison.dino_manifest)
    row_path = root / "complete/frame_rows.json"
    rows = json.loads(row_path.read_text())
    manifest = FeatureManifest.load(manifest_path)
    if (
        not isinstance(rows, dict)
        or rows.get("encoder_key") != root.name
        or rows.get("spatial_manifest_hash") != sha256_file(manifest_path)
        or not isinstance(rows.get("clips"), dict)
        or manifest.encoder != f"dinov3:{root.name}"
        or set(rows["clips"]) != {record.clip_id for record in manifest.records}
    ):
        raise ValueError("DINO frame rows/manifest identity mismatch")
    for record in manifest.records:
        row = rows["clips"][record.clip_id]
        if not isinstance(row, dict) or row.get("feature_hash") != sha256_file(root / record.path):
            raise ValueError(f"dino {record.split}/{record.clip_id}: feature hash mismatch")
    return dino_frame_rows_hash(config)


def _require_full_lightning_checkpoint(checkpoint: dict) -> None:
    version = checkpoint.get("pytorch-lightning_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("comparison requires full Lightning checkpoint version")
    state = checkpoint.get("state_dict")
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        )
    ):
        raise ValueError("comparison checkpoint state_dict must contain tensors")
    optimizers = checkpoint.get("optimizer_states")
    if not isinstance(optimizers, list) or not optimizers:
        raise ValueError("comparison checkpoint requires optimizer_states")
    for optimizer in optimizers:
        if not isinstance(optimizer, Mapping):
            raise ValueError("comparison checkpoint optimizer state invalid")
        groups = optimizer.get("param_groups")
        state = optimizer.get("state")
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, Mapping)
                or not isinstance(group.get("params"), list)
                or not group["params"]
                or any(type(param) is not int for param in group["params"])
                for group in groups
            )
            or not isinstance(state, Mapping)
            or not state
        ):
            raise ValueError("comparison checkpoint optimizer state/param_groups missing")
        if not any(
            isinstance(moment, Mapping)
            and (
                (
                    isinstance(moment.get("step"), torch.Tensor)
                    and moment["step"].numel() == 1
                    and bool(moment["step"] == 1000)
                )
                or (type(moment.get("step")) in (int, float) and moment["step"] == 1000)
            )
            for group in groups
            for param in group["params"]
            for moment in (state.get(param),)
        ):
            raise ValueError("comparison checkpoint optimizer step must equal 1000")
        for moment in state.values():
            if (
                not isinstance(moment, Mapping)
                or not isinstance(moment.get("exp_avg"), torch.Tensor)
                or not isinstance(moment.get("exp_avg_sq"), torch.Tensor)
                or moment["exp_avg"].shape != moment["exp_avg_sq"].shape
                or not torch.isfinite(moment["exp_avg"]).all()
                or not torch.isfinite(moment["exp_avg_sq"]).all()
            ):
                raise ValueError("comparison checkpoint optimizer moments invalid")
    schedulers = checkpoint.get("lr_schedulers")
    if (
        not isinstance(schedulers, list)
        or not schedulers
        or not all(
            isinstance(scheduler, Mapping)
            and type(scheduler.get("last_epoch")) is int
            and scheduler["last_epoch"] == 1000
            for scheduler in schedulers
        )
    ):
        raise ValueError("comparison checkpoint scheduler step must equal 1000")


def validate_comparison_checkpoint(config: DictConfig, checkpoint: dict, generation: str) -> None:
    if generation != "deterministic" or config.evaluation.generation != "deterministic":
        raise ValueError("comparison requires deterministic generation")
    if config.comparison.smoke:
        raise ValueError("comparison test evaluation requires full 1000-step checkpoint")
    if type(checkpoint.get("global_step")) is not int or checkpoint["global_step"] != 1000:
        raise ValueError("comparison checkpoint step must equal 1000")
    _require_full_lightning_checkpoint(checkpoint)
    validate_comparison_protocol(config)
    saved = checkpoint.get("run_metadata")
    if not isinstance(saved, dict) or saved.get("resume_reproducibility") != "fresh":
        raise ValueError("comparison requires fresh training run_metadata")
    source = saved.get("model_source")
    if (
        not isinstance(source, dict)
        or source.get("identifier") != "google/flan-t5-xl"
        or source.get("tuning_type") != "lora"
        or source.get("revision_status") != "resolved"
        or source.get("revision") != FLAN_SHA
        or config.model.revision != FLAN_SHA
    ):
        raise ValueError("comparison Flan revision mismatch in checkpoint provenance")
    if (
        saved.get("seed") != config.seed
        or saved.get("config") != OmegaConf.to_container(config, resolve=True)
        or saved.get("spatial_manifest_sha256") != sha256_file(Path(config.data.spatial_manifest))
        or saved.get("motion_manifest_sha256") != sha256_file(Path(config.data.motion_manifest))
    ):
        raise ValueError("comparison checkpoint/config/manifest provenance mismatch")
    if saved.get("annotation_sha256") != annotation_hashes(config):
        raise ValueError("comparison annotation SHA256 mismatch")
    if saved.get("comparison_code_sha256") != comparison_code_sha256(
        Path(__file__).resolve().parents[2]
    ):
        raise ValueError("comparison code SHA256 mismatch")
    if saved.get("dino_frame_rows_sha256") != validate_dino_content(config):
        raise ValueError("comparison DINO content SHA256 mismatch")


def summarize(root: Path) -> dict:
    """Validate six offline comparison artifacts and summarize paired seed scores."""
    expected_paths = {
        root / source / f"seed-{seed}" / "test.json"
        for seed in (0, 1, 2)
        for source in ("clip", "dino")
    }
    extras = set(root.glob("*/seed-*/test.json")) - expected_paths
    if extras:
        raise ValueError(
            f"comparison requires exactly six run artifacts; unexpected: {sorted(extras)}"
        )

    runs: list[dict] = []
    reference_items = None
    reference_config = None
    spatial_hashes: dict[str, str] = {}
    motion_hash = None
    lock_hash = None
    annotation_reference = None
    code_hash = None
    git_revision = None
    git_dirty = None
    dino_content_hash = None
    for seed in (0, 1, 2):
        for source in ("clip", "dino"):
            path = root / source / f"seed-{seed}" / "test.json"
            payload = json.loads(path.read_text())
            meta, metrics, items = payload["metadata"], payload["metrics"], payload["items"]
            identities = [(item["clip_id"], item["reference"]) for item in items]
            if len(identities) != EXPECTED["test"] or len({row[0] for row in identities}) != len(
                identities
            ):
                raise ValueError(f"{path}: test must contain unique full split IDs")
            if reference_items is None:
                reference_items = identities
            elif identities != reference_items:
                raise ValueError(f"{path}: paired test IDs/references differ")
            if (
                type(meta.get("seed")) is not int
                or meta["seed"] != seed
                or meta.get("comparison_source") != source
                or meta.get("generation") != "deterministic"
                or meta.get("baseline_accepted") is not False
                or meta.get("spatial_crop_mode") != "full"
                or type(meta.get("checkpoint_global_step")) is not int
                or meta["checkpoint_global_step"] != 1000
            ):
                raise ValueError(f"{path}: comparison requires deterministic 1000 steps")
            config = meta["config"]
            try:
                validate_comparison_protocol(OmegaConf.create(config))
            except (AttributeError, KeyError, ValueError) as exc:
                raise ValueError(f"{path}: comparison protocol mismatch: {exc}") from exc
            if (
                type(config.get("seed")) is not int
                or config["seed"] != seed
                or type(config["trainer"].get("max_steps")) is not int
                or config["trainer"]["max_steps"] != 1000
                or config["evaluation"].get("generation") != "deterministic"
                or config["evaluation"].get("beam_size") != 5
                or config["model"].get("vt_pooling") != "masked_mean"
                or config["model"].get("spatial_crop_mode") != "full"
                or config["model"].get("revision") != FLAN_SHA
                or config["comparison"].get("enabled") is not True
                or config["comparison"].get("smoke") is not False
            ):
                raise ValueError(f"{path}: protocol mismatch")
            if (
                config["data"].get("spatial_root") != config["comparison"][f"{source}_root"]
                or config["data"].get("spatial_manifest")
                != config["comparison"][f"{source}_manifest"]
            ):
                raise ValueError(f"{path}: spatial source/config mismatch")
            if any(
                Path(config["comparison"][f"clip_{field}"]).resolve()
                == Path(config["comparison"][f"dino_{field}"]).resolve()
                for field in ("root", "manifest")
            ):
                raise ValueError(f"{path}: comparison requires distinct spatial roots/manifests")
            if (
                Path(config["trainer"]["default_root_dir"]).resolve()
                != (root / source / f"seed-{seed}").resolve()
            ):
                raise ValueError(f"{path}: run root differs from seed/source output layout")
            saved = meta["checkpoint_metadata"]
            if (
                saved.get("seed") != seed
                or saved.get("config") != config
                or saved.get("resume_reproducibility") != "fresh"
                or saved.get("spatial_manifest_sha256") != meta.get("spatial_manifest_sha256")
                or saved.get("motion_manifest_sha256") != meta.get("motion_manifest_sha256")
                or saved.get("package_lock_sha256") != meta.get("package_lock_sha256")
            ):
                raise ValueError(f"{path}: checkpoint provenance mismatch")
            current_code = meta.get("comparison_code_sha256")
            if (
                not isinstance(current_code, str)
                or not re.fullmatch(r"[0-9a-f]{64}", current_code)
                or saved.get("comparison_code_sha256") != current_code
            ):
                raise ValueError(f"{path}: checkpoint code SHA256 mismatch")
            if code_hash is not None and current_code != code_hash:
                raise ValueError(f"{path}: comparison_code_sha256 drift across runs")
            code_hash = current_code
            current_dino_hash = meta.get("dino_frame_rows_sha256")
            if (
                not isinstance(current_dino_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", current_dino_hash)
                or saved.get("dino_frame_rows_sha256") != current_dino_hash
            ):
                raise ValueError(f"{path}: checkpoint DINO content SHA256 mismatch")
            if dino_content_hash is not None and current_dino_hash != dino_content_hash:
                raise ValueError(f"{path}: DINO content SHA256 drift across runs")
            dino_content_hash = current_dino_hash
            current_revision = meta.get("git_revision")
            if not isinstance(current_revision, str) or not current_revision:
                raise ValueError(f"{path}: invalid git_revision")
            current_dirty = meta.get("git_dirty")
            if type(current_dirty) is not bool:
                raise ValueError(f"{path}: invalid git_dirty")
            if (
                saved.get("git_revision") != current_revision
                or type(saved.get("git_dirty")) is not bool
                or saved["git_dirty"] != current_dirty
            ):
                raise ValueError(f"{path}: checkpoint git provenance mismatch")
            if git_revision is not None and current_revision != git_revision:
                raise ValueError(f"{path}: git_revision drift across runs")
            git_revision = current_revision
            if git_dirty is not None and current_dirty != git_dirty:
                raise ValueError(f"{path}: git_dirty drift across runs")
            git_dirty = current_dirty
            current_annotations = meta.get("annotation_sha256")
            if (
                not isinstance(current_annotations, dict)
                or set(current_annotations) != {"train", "dev", "test"}
                or any(
                    not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)
                    for value in current_annotations.values()
                )
                or saved.get("annotation_sha256") != current_annotations
            ):
                raise ValueError(f"{path}: checkpoint annotation SHA256 mismatch")
            if annotation_reference is None:
                annotation_reference = current_annotations.copy()
            else:
                for split in ("train", "dev", "test"):
                    if current_annotations[split] != annotation_reference[split]:
                        raise ValueError(f"{path}: {split} annotation SHA256 drift across runs")
            model_source = saved["model_source"]
            if (
                model_source.get("identifier") != "google/flan-t5-xl"
                or model_source.get("tuning_type") != "lora"
                or model_source.get("revision_status") != "resolved"
                or model_source.get("revision") != FLAN_SHA
            ):
                raise ValueError(f"{path}: Flan revision mismatch")
            for field in (
                "checkpoint_sha256",
                "spatial_manifest_sha256",
                "motion_manifest_sha256",
                "package_lock_sha256",
            ):
                digest = meta.get(field)
                if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                    raise ValueError(f"{path}: invalid {field}")
            if (
                source in spatial_hashes
                and spatial_hashes[source] != meta["spatial_manifest_sha256"]
            ):
                raise ValueError(f"{path}: spatial manifest drift across seeds")
            spatial_hashes[source] = meta["spatial_manifest_sha256"]
            if len(spatial_hashes) == 2 and spatial_hashes["clip"] == spatial_hashes["dino"]:
                raise ValueError(f"{path}: comparison requires distinct spatial manifest SHA256")
            if motion_hash is not None and motion_hash != meta["motion_manifest_sha256"]:
                raise ValueError(f"{path}: motion manifest drift across runs")
            motion_hash = meta["motion_manifest_sha256"]
            if lock_hash is not None and lock_hash != meta["package_lock_sha256"]:
                raise ValueError(f"{path}: baseline lock drift across runs")
            lock_hash = meta["package_lock_sha256"]
            matched = json.loads(json.dumps(config))
            for key in ("spatial_root", "spatial_manifest"):
                del matched["data"][key]
            del matched["trainer"]["default_root_dir"]
            del matched["seed"]
            if reference_config is None:
                reference_config = matched
            elif matched != reference_config:
                raise ValueError(f"{path}: training configs differ beyond source/seed/output")
            for metric, maximum in (("bleu4", 100), ("rougeL_f1", 1)):
                value = metrics.get(metric)
                if (
                    type(value) not in (float, int)
                    or not math.isfinite(value)
                    or not 0 <= value <= maximum
                ):
                    raise ValueError(f"{path}: invalid {metric}")
            runs.append(
                {
                    "source": source,
                    "seed": seed,
                    "budget_optimizer_steps": 1000,
                    "generation": "deterministic",
                    "beam_size": 5,
                    "test_items": len(items),
                    "bleu4": metrics["bleu4"],
                    "rougeL_f1": metrics["rougeL_f1"],
                    "checkpoint_sha256": meta["checkpoint_sha256"],
                    "spatial_manifest_sha256": meta["spatial_manifest_sha256"],
                    "motion_manifest_sha256": meta["motion_manifest_sha256"],
                    "package_lock_sha256": meta["package_lock_sha256"],
                    "comparison_code_sha256": current_code,
                    "dino_frame_rows_sha256": current_dino_hash,
                    "git_revision": current_revision,
                    "git_dirty": current_dirty,
                    "flan_revision": model_source["revision"],
                    "annotation_sha256": current_annotations.copy(),
                    "result_path": str(path),
                }
            )
    sources = {}
    for source in ("clip", "dino"):
        sources[source] = {}
        for metric in ("bleu4", "rougeL_f1"):
            values = [run[metric] for run in runs if run["source"] == source]
            score = {
                "mean": statistics.mean(values),
                "std": statistics.stdev(values),
            }
            if not all(math.isfinite(value) for value in score.values()):
                raise ValueError(f"invalid {source} {metric} summary")
            sources[source][metric] = score
    paired_delta = {}
    for metric in ("bleu4", "rougeL_f1"):
        values = [
            next(run[metric] for run in runs if run["source"] == "dino" and run["seed"] == seed)
            - next(run[metric] for run in runs if run["source"] == "clip" and run["seed"] == seed)
            for seed in (0, 1, 2)
        ]
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"invalid paired {metric} delta")
        score = {"mean": statistics.mean(values), "std": statistics.stdev(values)}
        if not all(math.isfinite(value) for value in score.values()):
            raise ValueError(f"invalid paired {metric} summary")
        paired_delta[metric] = score
    return {
        "runs": runs,
        "annotation_sha256": annotation_reference,
        "comparison_code_sha256": code_hash,
        "dino_frame_rows_sha256": dino_content_hash,
        "git_revision": git_revision,
        "git_dirty": git_dirty,
        "sources": sources,
        "paired_delta": paired_delta,
    }

"""Fresh pilot configuration and immutable logical-split admission."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from despamo.appearance.generation import load_records
from despamo.appearance.provenance import digest, file_hash, load_dataset, read_json
from despamo.data.manifest import FeatureManifest
from despamo.data.phoenix14t import Phoenix14T
from despamo.data.signer_split import PhoenixTrainView, verify_split
from despamo.pilot import bind_protocol, validate_live_counts

ROOT = Path(__file__).resolve().parents[3]
FLAN_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def require_protocol(protocol: dict) -> None:
    payload = {key: value for key, value in protocol.items() if key != "protocol_hash"}
    if (
        protocol.get("version") != "signer-pilot-seed0-v1"
        or protocol.get("seed") != 0
        or protocol.get("gpu_hour_ceiling") != 24
        or protocol.get("supervision_policy") != "qwen-schema98-unreviewed-v1"
        or protocol.get("human_review_status") != "not_assessed"
        or protocol.get("decoder")
        != {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False}
        or protocol.get("protocol_hash") != digest(payload)
        or not protocol.get("split_hash")
        or protocol.get("split_hash") != protocol.get("split", {}).get("split_hash")
    ):
        raise ValueError("pilot protocol policy/content hash mismatch")
    try:
        validate_live_counts(protocol["split"])
        groups = protocol["split"]["groups"]
        ids = [clip_id for name in ("train", "dev", "test") for clip_id in groups[name]["clip_ids"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("pilot protocol split/count mismatch") from exc
    if len(ids) != 7096 or len(set(ids)) != len(ids):
        raise ValueError("pilot protocol clip inventory overlaps")


def load_pilot_views(
    protocol_path: Path,
    dataset: Path,
    text_manifest: Path,
    dino_root: Path,
    motion_root: Path,
    motion_manifest: Path,
    annotation: Path,
) -> tuple[dict, dict[str, PhoenixTrainView]]:
    """Rebind every frozen input before constructing physical-train views."""
    if annotation.name != "train_info_ml.npy" or not all(
        path.is_file()
        for path in (
            protocol_path,
            text_manifest,
            motion_manifest,
            annotation,
            dino_root / "complete/frame_rows.json",
            dino_root / "complete/manifest.json",
        )
    ):
        raise ValueError("pilot preflight requires frozen physical train inputs")
    protocol = read_json(protocol_path)
    require_protocol(protocol)
    if protocol_path.parent.name != protocol["protocol_hash"] or dataset.name != protocol.get(
        "dataset_key"
    ):
        raise ValueError("pilot protocol/dataset identity mismatch")
    source, version = load_dataset(dataset)
    journals = load_records(dataset)
    verify_split(protocol["split"], source, journals)
    text = read_json(text_manifest)
    spatial_path = dino_root / "complete/manifest.json"
    rows_path = dino_root / "complete/frame_rows.json"
    rebuilt = bind_protocol(
        protocol["split"],
        dataset.name,
        digest(source),
        digest(version),
        digest(journals),
        file_hash(text_manifest),
        text["metadata"],
        file_hash(rows_path),
        file_hash(spatial_path),
        file_hash(motion_manifest),
        file_hash(annotation),
    )
    if rebuilt != protocol or text["dataset_key"] != dataset.name:
        raise ValueError("pilot frozen Qwen/CLIP/DINO/motion/annotation identity changed")
    spatial = FeatureManifest.load(spatial_path)
    motion = FeatureManifest.load(motion_manifest)
    parent = Phoenix14T(
        annotation,
        "train",
        dino_root,
        motion_root,
        spatial,
        motion,
        spatial_crop_mode="full",
    )
    source_by_id = {clip["clip_id"]: clip["signer"] for clip in source["clips"]}
    record_signers = {record["fileid"]: record["signer"] for record in parent.records}
    if source_by_id != record_signers or len(parent) != 7096:
        raise ValueError("pilot Qwen/physical-train annotation signer inventory changed")
    for clip_id in source_by_id:
        if (
            spatial.require("train", clip_id).width != 2048
            or motion.require("train", clip_id).width != 1024
        ):
            raise ValueError(f"pilot train feature width mismatch: {clip_id}")
    views = {
        role: PhoenixTrainView(parent, tuple(protocol["split"]["groups"][role]["clip_ids"]))
        for role in ("train", "dev", "test")
    }
    return protocol, views


def pilot_config(
    model_cache: Path,
    spatial_root: Path,
    motion_root: Path,
    annotation_root: Path,
    spatial_manifest: Path,
    motion_manifest: Path,
    steps: int,
    physical_batch: int,
    output: Path,
) -> DictConfig:
    if (
        steps not in (4000, 6000, 8000)
        or physical_batch not in (2, 4)
        or spatial_manifest != spatial_root / "complete/manifest.json"
        or not output.is_absolute()
    ):
        raise ValueError("invalid pilot tier/batch/source root")
    config = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/model/spamo_flan_t5_xl.yaml"),
        OmegaConf.load(ROOT / "configs/experiment/phoenix14t_baseline.yaml"),
    )
    config.seed = 0
    config.data = {
        "annotation_root": str(annotation_root),
        "spatial_root": str(spatial_root),
        "motion_root": str(motion_root),
        "spatial_manifest": str(spatial_manifest),
        "motion_manifest": str(motion_manifest),
        "batch_size": physical_batch,
        "num_workers": 0,
    }
    config.model.cache_dir = str(model_cache)
    config.model.revision = FLAN_REVISION
    config.model.use_in_context = False
    config.model.num_in_context = 0
    config.model.spatial_crop_mode = "full"
    config.model.vt_pooling = "masked_mean"
    config.model.warm_up_steps = {4000: 1000, 6000: 2000, 8000: 2000}[steps]
    config.trainer.max_steps = steps
    config.trainer.max_epochs = -1
    config.trainer.accumulate_grad_batches = 1 if physical_batch == 4 else 2
    config.trainer.default_root_dir = str(output)
    config.trainer.enable_checkpointing = False
    config.trainer.enable_progress_bar = False
    config.trainer.logger = False
    config.trainer.num_sanity_val_steps = 0
    config.comparison = {"enabled": True}
    config.evaluation.generation = "deterministic"
    return config

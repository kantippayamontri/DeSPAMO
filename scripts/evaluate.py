import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

from despamo.config import load_config, validate_baseline_config
from despamo.evaluation.acceptance import validate_baseline_result
from despamo.evaluation.artifact import collect_runtime_metadata, write_result_artifact
from despamo.evaluation.metrics import evaluate_translations
from despamo.factory import build_data, build_model
from despamo.models.prompts import build_prompts
from despamo.utils.checkpoint_snapshot import checkpoint_snapshot
from despamo.utils.config_policy import reject_secret_keys
from despamo.utils.hashing import sha256_file
from despamo.utils.output_policy import require_ignored_git_output as _require_ignored_git_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _reject_checkpoint_output_alias(checkpoint_path: Path, output: Path) -> None:
    if checkpoint_path.resolve() == output.resolve() or (
        checkpoint_path.exists() and output.exists() and checkpoint_path.samefile(output)
    ):
        raise ValueError("checkpoint and output must be different files")


def evaluate(
    config: DictConfig,
    checkpoint_path: Path,
    generation: str,
    output: Path,
    *,
    device: str = "cuda",
    accept_baseline: bool = False,
) -> dict[str, float]:
    _reject_checkpoint_output_alias(checkpoint_path, output)
    _require_ignored_git_output(output, PROJECT_ROOT)
    if generation not in {"upstream", "deterministic"}:
        raise ValueError(f"unsupported generation mode: {generation}")
    if accept_baseline and generation != "upstream":
        raise ValueError("--accept-baseline requires --generation upstream")
    if config.evaluation.expected_test_items != 642:
        raise ValueError("expected 642 test items in evaluation config")

    pl.seed_everything(config.seed, workers=True)
    data_config = OmegaConf.merge(config)
    if config.model.get("spatial_crop_mode") == "upstream_random" and generation == "deterministic":
        data_config.model.spatial_crop_mode = "center"
    resolved_config = OmegaConf.to_container(data_config, resolve=True)
    reject_secret_keys(resolved_config)
    data = build_data(data_config)
    expected_ids = [record["fileid"] for record in data.test_dataset.records]
    if len(expected_ids) != 642:
        raise ValueError(f"expected 642 test items, got {len(expected_ids)}")
    if len(set(expected_ids)) != len(expected_ids):
        raise ValueError("duplicate clip_id in test split")

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint must be an existing file: {checkpoint_path}")
    with checkpoint_snapshot(checkpoint_path, output.parent) as snapshot:
        checkpoint = torch.load(snapshot, map_location="cpu", weights_only=True)
        checkpoint_sha256 = sha256_file(snapshot)
        checkpoint_metadata = (
            checkpoint["metadata"] if "metadata" in checkpoint else checkpoint["run_metadata"]
        )
        reject_secret_keys(checkpoint_metadata, "checkpoint_metadata")
        model = build_model(config)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.to(device).eval()

        clip_ids: list[str] = []
        predictions: list[str] = []
        references: list[str] = []
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if device == "cuda"
            else nullcontext()
        )
        with torch.inference_mode(), autocast:
            for batch in data.test_dataloader():
                visual, visual_mask = model.visual_adapter(
                    batch.spatial.to(device),
                    batch.spatial_mask.to(device),
                    batch.motion.to(device),
                    batch.motion_mask.to(device),
                )
                prompts = build_prompts(
                    batch,
                    model.prompt_template,
                    model.use_in_context,
                    model.num_in_context,
                    model.rng,
                )
                generated = model.language_model.generate_text(
                    visual,
                    visual_mask,
                    prompts,
                    generation,
                    config.evaluation.beam_size,
                    config.model.max_text_length,
                )
                if len(generated) != len(batch.clip_ids):
                    raise ValueError("predictions must match each batch of test clips")
                clip_ids.extend(batch.clip_ids)
                predictions.extend(generated)
                references.extend(text.lower() for text in batch.texts)

        if clip_ids != expected_ids:
            raise ValueError("test split clips must be evaluated once in annotation order")
        metrics = evaluate_translations(predictions, references)
        if accept_baseline:
            validate_baseline_result(len(clip_ids), metrics)

        metadata = {
            **collect_runtime_metadata(
                config.seed,
                PROJECT_ROOT / "uv.lock",
                Path(config.data.spatial_manifest),
                Path(config.data.motion_manifest),
            ),
            "generation": generation,
            "baseline_accepted": accept_baseline,
            "spatial_crop_mode": data.test_dataset.spatial_crop_mode,
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_metadata": checkpoint_metadata,
            "config": resolved_config,
        }
        write_result_artifact(output, clip_ids, predictions, references, metrics, metadata)
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate SpaMo baseline with ordered config layers"
    )
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--generation", choices=("upstream", "deterministic"), required=True)
    parser.add_argument("--accept-baseline", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.accept_baseline and args.generation != "upstream":
        parser.error("--accept-baseline requires --generation upstream")
    config = load_config(args.config)
    validate_baseline_config(config)
    reject_secret_keys(OmegaConf.to_container(config, resolve=True))
    _reject_checkpoint_output_alias(args.checkpoint, args.output)
    _require_ignored_git_output(args.output, PROJECT_ROOT)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required for checkpoint evaluation; no evaluation was run")
    metrics = evaluate(
        config, args.checkpoint, args.generation, args.output, accept_baseline=args.accept_baseline
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

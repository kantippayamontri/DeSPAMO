import argparse
import json
import os
import re
import stat
import tempfile
from collections.abc import Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf

from despamo.comparison import (
    annotation_hashes,
    comparison_code_sha256,
    dino_frame_rows_hash,
    preflight_run,
)
from despamo.config import load_config, validate_baseline_config
from despamo.evaluation.artifact import collect_runtime_metadata
from despamo.factory import build_data, build_model
from despamo.training.baseline_module import validate_run_metadata
from despamo.utils.checkpoint_snapshot import checkpoint_snapshot
from despamo.utils.config_policy import reject_secret_keys
from despamo.utils.hashing import sha256_file
from despamo.utils.output_policy import require_ignored_git_output

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _model_revision(model: Any) -> tuple[str | None, str]:
    pretrained = getattr(getattr(model, "language_model", None), "model", None)
    if callable(getattr(pretrained, "get_base_model", None)):
        pretrained = pretrained.get_base_model()
    commit = getattr(getattr(pretrained, "config", None), "_commit_hash", None)
    if isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit):
        return commit, "resolved"
    return None, "unresolved"


def _write_run_metadata(root: Path, metadata: dict) -> tuple[int, int]:
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=root, prefix=".run_metadata-", suffix=".tmp", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            json.dump(metadata, temporary, indent=2, sort_keys=True, allow_nan=False)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
            file_stat = os.fstat(temporary.fileno())
            identity = file_stat.st_dev, file_stat.st_ino
        except BaseException:
            temporary_path.unlink()
            raise
    try:
        os.replace(temporary_path, root / "run_metadata.json")
    finally:
        temporary_path.unlink(missing_ok=True)
    return identity


def _remove_owned_provisional_manifest(
    manifest: Path, identity: tuple[int, int], contents: bytes
) -> None:
    try:
        file_stat = manifest.lstat()
        if (file_stat.st_dev, file_stat.st_ino) != identity or not stat.S_ISREG(file_stat.st_mode):
            return
        if manifest.read_bytes() == contents:
            # Recheck after reading so observed replacements are left intact.
            file_stat = manifest.lstat()
            if (file_stat.st_dev, file_stat.st_ino) == identity:
                manifest.unlink()
    except FileNotFoundError:
        pass


def _validate_resume_checkpoint(checkpoint: Any, metadata: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        raise ValueError("resume checkpoint must be a dict")
    state = checkpoint.get("state_dict")
    if (
        not isinstance(state, Mapping)
        or not state
        or not all(
            isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
        )
    ):
        raise ValueError("resume checkpoint state_dict must contain tensor values")
    optimizers = checkpoint.get("optimizer_states")
    if not isinstance(optimizers, list) or not optimizers:
        raise ValueError("resume checkpoint optimizer_states must be a nonempty list")
    for optimizer in optimizers:
        if not isinstance(optimizer, Mapping) or not isinstance(optimizer.get("state"), Mapping):
            raise ValueError("resume checkpoint optimizer_states missing optimizer state")
        groups = optimizer.get("param_groups")
        if (
            not isinstance(groups, list)
            or not groups
            or any(
                not isinstance(group, Mapping)
                or not isinstance(group.get("params"), list)
                or not group["params"]
                for group in groups
            )
        ):
            raise ValueError("resume checkpoint optimizer_states missing param_groups")
    schedulers = checkpoint.get("lr_schedulers")
    if (
        not isinstance(schedulers, list)
        or not schedulers
        or not all(isinstance(scheduler, Mapping) and scheduler for scheduler in schedulers)
    ):
        raise ValueError("resume checkpoint lr_schedulers must be a nonempty list")
    version = checkpoint.get("pytorch-lightning_version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("resume checkpoint pytorch-lightning_version must be a nonempty string")
    step = checkpoint.get("global_step")
    if type(step) is not int or step < 0:
        raise ValueError("resume checkpoint global_step must be a nonnegative integer")
    saved_metadata = checkpoint.get("run_metadata")
    validate_run_metadata(metadata, saved_metadata, check_revision=False)
    return saved_metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SpaMo baseline with ordered config layers")
    parser.add_argument(
        "--config",
        type=Path,
        action="append",
        required=True,
        help="config layer; repeat in merge order",
    )
    parser.add_argument("override", nargs="*", help="OmegaConf key=value overrides")
    parser.add_argument("--resume", type=Path, help="explicit Lightning checkpoint to resume")
    parser.add_argument(
        "--trusted-resume", action="store_true", help="trust resume checkpoint pickle"
    )
    args = parser.parse_args()
    if args.resume is not None and not args.trusted_resume:
        raise ValueError("--trusted-resume is required with --resume before reading the checkpoint")
    if args.trusted_resume and args.resume is None:
        raise ValueError("--trusted-resume requires --resume")
    config = load_config(args.config, args.override)
    validate_baseline_config(config)
    comparison = bool(config.get("comparison", {}).get("enabled", False))
    if comparison and args.resume is not None:
        raise ValueError("comparison runs must start fresh; --resume forbidden")
    if comparison:
        preflight_run(config, args.config, args.override)
    if args.resume is not None and not args.resume.is_file():
        raise FileNotFoundError(f"resume checkpoint must be an existing file: {args.resume}")
    pl.seed_everything(config.seed, workers=True)
    trainer = pl.Trainer(**dict(config.trainer))
    data = build_data(config)
    root = Path(config.trainer.default_root_dir)
    require_ignored_git_output(root, PROJECT_ROOT)
    if comparison and (root.resolve() == PROJECT_ROOT or PROJECT_ROOT in root.resolve().parents):
        raise ValueError("comparison artifacts must live outside Git checkout")
    if comparison and (root / "checkpoints/final.ckpt").exists():
        raise ValueError("comparison checkpoint already exists; use a new run directory")
    resolved_config = OmegaConf.to_container(config, resolve=True)
    reject_secret_keys(resolved_config)
    metadata = {
        **collect_runtime_metadata(
            config.seed,
            PROJECT_ROOT / "uv.lock",
            Path(config.data.spatial_manifest),
            Path(config.data.motion_manifest),
        ),
        "config": resolved_config,
        "resume_reproducibility": "optimizer_state_only" if args.resume is not None else "fresh",
        "model_source": {
            "identifier": config.model.name,
            "tuning_type": config.model.get("tuning_type", "lora"),
            "revision": None,
            "revision_status": "unresolved",
        },
    }
    if comparison:
        metadata["annotation_sha256"] = annotation_hashes(config)
        metadata["comparison_code_sha256"] = comparison_code_sha256(PROJECT_ROOT)
        metadata["dino_frame_rows_sha256"] = dino_frame_rows_hash(config)
    snapshot_context = checkpoint_snapshot(args.resume, root) if args.resume else nullcontext(None)
    with snapshot_context as snapshot:
        saved_metadata = None
        if snapshot is not None:
            metadata["resume_checkpoint_sha256"] = sha256_file(snapshot)
            try:
                checkpoint = torch.load(snapshot, map_location="cpu", weights_only=True)
            except Exception as exc:
                raise ValueError(
                    f"safe resume preflight failed (weights_only=True): {exc}"
                ) from exc
            saved_metadata = _validate_resume_checkpoint(checkpoint, metadata)
            del checkpoint

        manifest = root / "run_metadata.json"
        require_ignored_git_output(manifest, PROJECT_ROOT)
        existing_metadata = None
        provisional_identity = None
        if manifest.exists():
            if args.resume is None:
                raise ValueError(f"run_metadata.json already exists; use --resume: {manifest}")
            existing_metadata = json.loads(manifest.read_text())
            validate_run_metadata(metadata, existing_metadata, check_revision=False)
        else:
            provisional_identity = _write_run_metadata(root, metadata)
            provisional_contents = manifest.read_bytes()

        try:
            model = build_model(config)
            revision, status = _model_revision(model)
            if comparison and (status != "resolved" or revision != config.model.revision):
                raise ValueError("comparison Flan revision mismatch")
            metadata["model_source"].update(revision=revision, revision_status=status)
            if saved_metadata is not None:
                validate_run_metadata(metadata, saved_metadata)
            if existing_metadata is not None:
                validate_run_metadata(metadata, existing_metadata)
            final_identity = _write_run_metadata(root, metadata)
            if provisional_identity is not None:
                provisional_identity = final_identity
                provisional_contents = manifest.read_bytes()
            model.run_metadata = metadata
        except BaseException:
            if provisional_identity is not None:
                _remove_owned_provisional_manifest(
                    manifest, provisional_identity, provisional_contents
                )
            raise
        if snapshot is None:
            trainer.fit(model, datamodule=data)
        else:
            trainer.fit(model, datamodule=data, ckpt_path=str(snapshot))
        if comparison:
            if trainer.global_step != config.trainer.max_steps:
                raise ValueError("comparison stopped before exact optimizer-step budget")
            target = root / "checkpoints/final.ckpt"
            require_ignored_git_output(target, PROJECT_ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            trainer.save_checkpoint(target)
            if not target.is_file():
                raise ValueError("comparison trainer did not write final checkpoint")


if __name__ == "__main__":
    main()

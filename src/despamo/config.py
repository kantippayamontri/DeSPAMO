from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


def load_config(paths: Sequence[Path], overrides: Sequence[str] = ()) -> DictConfig:
    configs = [OmegaConf.load(path) for path in paths]
    if overrides:
        configs.append(OmegaConf.from_dotlist(list(overrides)))
    merged = OmegaConf.merge(*configs) if configs else OmegaConf.create()
    OmegaConf.resolve(merged)
    return merged


def validate_baseline_config(config: DictConfig) -> None:
    spatial_dim = OmegaConf.select(config, "model.spatial_dim")
    motion_dim = OmegaConf.select(config, "model.motion_dim")
    if spatial_dim != 2048:
        raise ValueError(f"model.spatial_dim must be 2048, got {spatial_dim}")
    if motion_dim != 1024:
        raise ValueError(f"model.motion_dim must be 1024, got {motion_dim}")
    mode = OmegaConf.select(config, "model.vt_pooling", default="legacy_mean")
    if mode not in {"legacy_mean", "masked_mean"}:
        raise ValueError(f"unsupported model.vt_pooling: {mode}")
    generation = OmegaConf.select(config, "evaluation.generation", default="upstream")
    if generation not in {"upstream", "deterministic"}:
        raise ValueError(f"unsupported evaluation.generation: {generation}")

from pathlib import Path

import pytest
from omegaconf.errors import InterpolationResolutionError

from despamo.config import load_config, validate_baseline_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_load_config_merges_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("model:\n  spatial_dim: 2048\ntrainer:\n  max_epochs: 40\n")
    override.write_text("trainer:\n  max_epochs: 1\n")

    config = load_config([base, override], ["trainer.max_epochs=2"])

    assert config.model.spatial_dim == 2048
    assert config.trainer.max_epochs == 2


def test_load_config_resolves_layered_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", "/example/annotations")
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", "/example/features")
    monkeypatch.setenv("DESPAMO_HF_CACHE", "/example/cache")
    monkeypatch.setenv("DESPAMO_CHECKPOINT", "/example/checkpoint.pt")

    config = load_config(
        [
            CONFIGS / "data/phoenix14t.yaml",
            CONFIGS / "model/spamo_flan_t5_xl.yaml",
            CONFIGS / "experiment/phoenix14t_baseline.yaml",
            CONFIGS / "experiment/phoenix14t_smoke.yaml",
            CONFIGS / "local.example.yaml",
        ],
        ["model.vt_pooling=masked_mean", "evaluation.generation=deterministic"],
    )

    assert config.data.annotation_root == "/example/annotations"
    assert config.data.spatial_root == "/example/features/vit_feat_Phoenix14T"
    assert config.data.motion_root == "/example/features/mae_feat_Phoenix14T"
    assert config.data.spatial_manifest == "/example/features/manifests/phoenix14t_spatial.json"
    assert config.data.motion_manifest == "/example/features/manifests/phoenix14t_motion.json"
    assert config.model.cache_dir == "/example/cache"
    assert config.model.spatial_dim == 2048
    assert config.model.motion_dim == 1024
    assert config.model.tuning_type == "freeze"
    assert config.model.warm_up_steps is None
    assert config.trainer.max_epochs == 1
    assert config.trainer.accumulate_grad_batches == 1
    assert config.optimizer.learning_rate == 6.0e-4
    assert config.evaluation.expected_test_items == 642
    assert config.evaluation.checkpoint == "/example/checkpoint.pt"
    assert config.model.vt_pooling == "masked_mean"
    assert config.evaluation.generation == "deterministic"
    validate_baseline_config(config)


def test_load_config_requires_referenced_environment_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DESPAMO_FEATURE_ROOT", raising=False)
    config_path = tmp_path / "env.yaml"
    config_path.write_text("data:\n  spatial_root: ${oc.env:DESPAMO_FEATURE_ROOT}\n")

    with pytest.raises(InterpolationResolutionError, match="DESPAMO_FEATURE_ROOT"):
        load_config([config_path])


@pytest.mark.parametrize(
    ("spatial_dim", "motion_dim", "error"),
    [
        (1024, 1024, "model.spatial_dim must be 2048"),
        (2048, 512, "model.motion_dim must be 1024"),
    ],
)
def test_validate_rejects_wrong_baseline_width(
    spatial_dim: int, motion_dim: int, error: str
) -> None:
    config = load_config([], [f"model.spatial_dim={spatial_dim}", f"model.motion_dim={motion_dim}"])

    with pytest.raises(ValueError, match=error):
        validate_baseline_config(config)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ("model.vt_pooling=maximum", "unsupported model.vt_pooling: maximum"),
        ("evaluation.generation=sampling", "unsupported evaluation.generation: sampling"),
    ],
)
def test_validate_rejects_unsupported_modes(override: str, error: str) -> None:
    config = load_config([], ["model.spatial_dim=2048", "model.motion_dim=1024", override])

    with pytest.raises(ValueError, match=error):
        validate_baseline_config(config)


def test_validate_accepts_default_modes() -> None:
    config = load_config([], ["model.spatial_dim=2048", "model.motion_dim=1024"])

    validate_baseline_config(config)

from pathlib import Path

import pytest

from despamo.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_smoke_config_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PHOENIX14T_ANNOTATION_ROOT", "/example/annotations")
    monkeypatch.setenv("DESPAMO_FEATURE_ROOT", "/example/features")
    monkeypatch.setenv("DESPAMO_HF_CACHE", "/example/cache")

    config = load_config(
        [
            CONFIGS / "data/phoenix14t.yaml",
            CONFIGS / "model/spamo_flan_t5_xl.yaml",
            CONFIGS / "experiment/phoenix14t_baseline.yaml",
            CONFIGS / "experiment/phoenix14t_smoke.yaml",
        ]
    )

    assert config.model.tuning_type == "freeze"
    assert config.trainer.max_epochs == 1
    assert config.trainer.limit_train_batches == 0.01
    assert config.trainer.limit_val_batches == 1
    assert config.trainer.num_sanity_val_steps == 0
    assert config.trainer.check_val_every_n_epoch == 1
    assert config.trainer.log_every_n_steps == 1
    assert config.trainer.default_root_dir == "artifacts/phoenix14t_smoke"

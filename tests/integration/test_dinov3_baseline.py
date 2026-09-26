import os
from collections import Counter
from pathlib import Path

import pytest

from despamo.config import load_config
from despamo.data.manifest import FeatureManifest
from despamo.factory import build_data


@pytest.mark.integration
def test_external_dino_manifest_loads_without_new_transformers(tmp_path, monkeypatch):
    if os.environ.get("DESPAMO_RUN_DINO_BASELINE") != "1":
        pytest.skip("opt in only after full approved extraction")
    root = Path(os.environ["DESPAMO_DINO_ROOT"])
    monkeypatch.setenv("DESPAMO_HF_CACHE", str(tmp_path / "synthetic-hf-cache"))
    manifest = FeatureManifest.load(root / "complete/manifest.json")
    assert Counter(record.split for record in manifest.records) == {
        "train": 7096,
        "dev": 519,
        "test": 642,
    }
    configs = Path(__file__).resolve().parents[2] / "configs"
    config = load_config(
        [
            configs / "data/phoenix14t.yaml",
            configs / "model/spamo_flan_t5_xl.yaml",
            configs / "experiment/phoenix14t_baseline.yaml",
        ],
        [
            f"data.spatial_root={root}",
            f"data.spatial_manifest={root / 'complete/manifest.json'}",
            "model.spatial_crop_mode=full",
        ],
    )
    assert config.model.cache_dir == str(tmp_path / "synthetic-hf-cache")
    batch = next(iter(build_data(config).test_dataloader()))
    assert batch.spatial.shape[-1] == 2048 and batch.motion.shape[-1] == 1024

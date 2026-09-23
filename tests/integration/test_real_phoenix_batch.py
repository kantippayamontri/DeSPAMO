import os
from collections import Counter
from pathlib import Path

import pytest

from despamo.config import load_config
from despamo.data.manifest import FeatureManifest
from despamo.factory import build_data


@pytest.mark.integration
def test_real_test_batch_has_expected_widths() -> None:
    if os.environ.get("DESPAMO_RUN_REAL_BATCH") != "1":
        pytest.skip("set DESPAMO_RUN_REAL_BATCH=1 after full external manifest indexing")
    required = ("PHOENIX14T_ANNOTATION_ROOT", "DESPAMO_FEATURE_ROOT", "DESPAMO_HF_CACHE")
    if any(not os.environ.get(name) for name in required):
        pytest.skip("local PHOENIX14T paths are not configured")

    configs = Path(__file__).resolve().parents[2] / "configs"
    config = load_config(
        [
            configs / "data/phoenix14t.yaml",
            configs / "model/spamo_flan_t5_xl.yaml",
            configs / "experiment/phoenix14t_baseline.yaml",
        ]
    )
    expected_counts = {"train": 7096, "dev": 519, "test": 642}
    for manifest_path in (config.data.spatial_manifest, config.data.motion_manifest):
        path = Path(manifest_path)
        if not path.is_file():
            pytest.skip(f"full feature manifest not indexed: {path}")
        manifest = FeatureManifest.load(path)
        if Counter(record.split for record in manifest.records) != expected_counts:
            pytest.skip(f"full feature manifest not indexed: {path}")

    batch = next(iter(build_data(config).test_dataloader()))
    assert batch.spatial.shape[0] == 2
    assert batch.spatial.shape[2] == 2048
    assert batch.motion.shape[2] == 1024

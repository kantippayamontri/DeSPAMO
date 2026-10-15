import json
import math
from pathlib import Path

import pytest

from tools.spamo_repro.policy import OWN_CLIP_POLICY, recipe_for, validate_config
from tools.spamo_repro.recipe import joint_for, lr_for

CONFIG = Path(__file__).parents[1] / "config_own_clip200.json"


def own():
    return json.loads(CONFIG.read_text())


def test_own_clip_config_admitted_with_200_epoch_horizon():
    config = own()
    assert config["policy"] == OWN_CLIP_POLICY
    assert validate_config(config) == 88800
    recipe = recipe_for(config)
    assert recipe["max_epochs"] == 200 and recipe["lr_warmup_updates"] == 22200
    assert recipe["patience_checks"] == 50 and recipe["dev_every"] == 2


def test_own_clip_uses_own_spatial_and_motion_features():
    data = own()["data"]
    assert "official-phoenix14t-clip-s2-features-v1/spatial-f0ea5e0b" in data["spatial_root"]
    assert "official-phoenix14t-features-v1/motion-1123ff44" in data["motion_root"]
    assert data["spatial_manifest"] == data["spatial_root"] + "/complete/manifest.json"
    assert data["motion_manifest"] == data["motion_root"] + "/complete/manifest.json"


def test_own_clip_lr_warmup_22200_then_cosine_to_zero_at_88800():
    recipe = recipe_for(own())
    assert lr_for(1, recipe) == 0.0
    assert lr_for(11101, recipe) == pytest.approx(3e-4)
    assert lr_for(22201, recipe) == pytest.approx(6e-4)
    mid = 22200 + (88800 - 22200) // 2
    assert lr_for(mid + 1, recipe) == pytest.approx(3e-4)
    assert lr_for(88800, recipe) == pytest.approx(
        6e-4 * 0.5 * (1 + math.cos(math.pi * 66599 / 66600)), rel=1e-9
    )


def test_own_clip_joint_loss_from_second_update():
    recipe = recipe_for(own())
    assert joint_for(1, recipe) is False and joint_for(2, recipe) is True


def test_upstream_recipe_unchanged_by_explicit_warmup():
    upstream = json.loads((CONFIG.parent / "config_upstream.json").read_text())
    recipe = recipe_for(upstream)
    assert recipe["lr_warmup_updates"] == 22200
    assert lr_for(42625, recipe) == pytest.approx(0.0005846628034933489, rel=1e-12)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("recipe", "max_epochs", 500),
        ("recipe", "lr_warmup_updates", 22000),
        ("recipe", "learning_rate", 1e-4),
        ("data", "spatial_root", "/home/kan/datasets/spamo/features/vit_feat_Phoenix14T"),
    ],
)
def test_own_clip_drift_rejected(section, key, value):
    config = own()
    config[section][key] = value
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


def test_training_icl_mode_follows_recipe_not_policy_name():
    from tools.spamo_repro.worker import training_icl

    assert training_icl(own()) == "derangement"
    assert training_icl(json.loads((CONFIG.parent / "config_upstream.json").read_text())) == (
        "derangement"
    )
    assert training_icl(json.loads((CONFIG.parent / "config.json").read_text())) == "pool"

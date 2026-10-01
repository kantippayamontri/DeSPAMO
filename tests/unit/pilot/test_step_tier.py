from copy import deepcopy

import pytest
from omegaconf import OmegaConf

from despamo.training.pilot_runtime import pilot_config
from scripts.train_e3_pilot import (
    E1_SHARED_HASH,
    require_matched_e3_config,
    select_e3_checkpoint,
)
from scripts.train_signer_pilot import checkpoint_steps

TIERS = {4000: (1000, (1750, 2800, 4000)), 8000: (2000, (3500, 5600, 8000))}


def _config(steps: int, warmup: int, spatial: str) -> dict:
    return {
        "data": {"spatial_root": spatial, "spatial_manifest": f"{spatial}/complete/manifest.json",
                 "motion_root": "/motion", "batch_size": 4, "num_workers": 0},
        "model": {"use_in_context": False, "warm_up_steps": warmup},
        "trainer": {"max_steps": steps, "accumulate_grad_batches": 1,
                    "default_root_dir": "/external"},
        "seed": 0,
    }


@pytest.mark.parametrize("steps", sorted(TIERS))
def test_supported_tiers_keep_their_own_dev_schedule(steps):
    warmup, expected = TIERS[steps]
    assert checkpoint_steps(steps, warmup) == expected


@pytest.mark.parametrize("steps", (5000, 15000))
def test_unsupported_step_tier_is_rejected(steps):
    with pytest.raises(ValueError, match="tier|schedule"):
        checkpoint_steps(steps, 2000)
    with pytest.raises(ValueError, match="tier|batch|source"):
        pilot_config(**_pilot_kwargs(steps))


def _pilot_kwargs(steps: int) -> dict:
    from pathlib import Path
    return {"model_cache": Path("/cache"), "spatial_root": Path("/adapted"),
            "motion_root": Path("/motion"), "annotation_root": Path("/annotations"),
            "spatial_manifest": Path("/adapted/complete/manifest.json"),
            "motion_manifest": Path("/motion-manifest.json"), "steps": steps,
            "physical_batch": 4, "output": Path("/external")}


def test_pilot_config_emits_requested_tier_warmup():
    for steps, (warmup, _) in TIERS.items():
        config = pilot_config(**_pilot_kwargs(steps))
        assert config.trainer.max_steps == steps
        assert config.model.warm_up_steps == warmup


def test_e3_config_match_requires_equal_step_tier():
    e1_8000 = {"config": _config(8000, 2000, "/frozen"), "initial_shared_hash": E1_SHARED_HASH}
    matched = _config(8000, 2000, "/adapted")
    require_matched_e3_config(e1_8000, matched, "/adapted", steps=8000)

    mismatched = deepcopy(matched)
    mismatched["trainer"]["max_steps"] = 4000
    mismatched["model"]["warm_up_steps"] = 1000
    with pytest.raises(ValueError, match="matched|config|steps"):
        require_matched_e3_config(e1_8000, mismatched, "/adapted", steps=8000)


def test_e3_config_match_still_accepts_original_4000_tier():
    e1 = {"config": _config(4000, 1000, "/frozen"), "initial_shared_hash": E1_SHARED_HASH}
    require_matched_e3_config(e1, _config(4000, 1000, "/adapted"), "/adapted", steps=4000)


@pytest.mark.parametrize("steps", sorted(TIERS))
def test_dev_selection_validates_the_tier_schedule(steps):
    ids = ("a", "b")
    reports = [{"split": "dev", "split_hash": "s", "items": [{"clip_id": i} for i in ids],
                "metrics": {"bleu4": 1.0 + n}, "checkpoint_step": step,
                "checkpoint_hash": str(step)}
               for n, step in enumerate(TIERS[steps][1])]
    assert select_e3_checkpoint(reports, ids, steps)["checkpoint_step"] == TIERS[steps][1][-1]
    wrong = deepcopy(reports)
    wrong[0]["checkpoint_step"] = 999
    with pytest.raises(ValueError, match="schedule"):
        select_e3_checkpoint(wrong, ids, steps)


def test_omegaconf_tier_roundtrip_matches_plain_dict():
    config = pilot_config(**_pilot_kwargs(8000))
    assert OmegaConf.to_container(config, resolve=True)["trainer"]["max_steps"] == 8000

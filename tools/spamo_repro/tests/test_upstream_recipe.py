import json
import math
import random
from pathlib import Path

import pytest

from tools.followup.recipe import PromptPool
from tools.spamo_repro.policy import recipe_for, validate_config
from tools.spamo_repro.recipe import joint_for, lr_for
from tools.spamo_repro.study import should_stop

CONFIG = Path(__file__).parents[1] / "config_upstream.json"


def upstream():
    return json.loads(CONFIG.read_text())


def test_upstream_config_admitted_with_released_checkpoint_horizon():
    assert validate_config(upstream()) == 222000
    recipe = recipe_for(upstream())
    assert recipe["updates_per_epoch"] == 444 and recipe["max_epochs"] == 500
    assert recipe["dev_every"] == 2 and recipe["patience_checks"] == 50


def test_paper_config_still_admitted_unchanged():
    paper = json.loads((CONFIG.parent / "config.json").read_text())
    assert validate_config(paper) == 17760
    assert recipe_for(paper)["max_epochs"] == 40


def test_upstream_lr_matches_released_lightning_scheduler_state():
    recipe = recipe_for(upstream())
    assert lr_for(1, recipe) == 0.0
    assert lr_for(22201, recipe) == pytest.approx(6e-4)
    # Released checkpoint: last_epoch 42624, _last_lr for next update.
    assert lr_for(42625, recipe) == pytest.approx(0.0005846628034933489, rel=1e-12)
    assert lr_for(222000, recipe) == pytest.approx(
        6e-4 * 0.5 * (1 + math.cos(math.pi * 199799 / 199800)), rel=1e-9
    )


def test_upstream_joint_loss_from_second_update():
    recipe = recipe_for(upstream())
    assert joint_for(1, recipe) is False
    assert joint_for(2, recipe) is True


def test_paper_schedule_behaviour_preserved():
    paper = recipe_for(json.loads((CONFIG.parent / "config.json").read_text()))
    assert lr_for(1, paper) == pytest.approx(1e-8)
    assert lr_for(17760, paper) == pytest.approx(5e-5)
    assert joint_for(4000, paper) is False and joint_for(4001, paper) is True


@pytest.mark.parametrize("key,value", [("learning_rate", 1e-4), ("max_epochs", 200)])
def test_upstream_drift_rejected(key, value):
    config = upstream()
    config["recipe"][key] = value
    with pytest.raises(ValueError, match="approved"):
        validate_config(config)


def test_early_stop_after_fifty_checks_without_strict_improvement():
    history = [10.0] + [10.0] * 49
    assert should_stop(history, 50) is False
    assert should_stop(history + [9.0], 50) is True
    assert should_stop(history + [10.5], 50) is False
    assert should_stop([1.0, 2.0], 50) is False


def test_upstream_in_batch_derangement_never_uses_own_reference():
    from tools.spamo_repro.model import derangement_prompts

    rows = [
        dict(fileid=f"c{i}", text=f"satz {i}", en_text=f"e{i}", fr_text=f"f{i}", es_text=f"s{i}")
        for i in range(8)
    ]
    pool = PromptPool(rows, 0)
    prompts = derangement_prompts(tuple(r["fileid"] for r in rows), pool, random.Random(0))
    assert len(prompts) == 8
    for row, prompt in zip(rows, prompts, strict=True):
        assert prompt.startswith("Translate the given sentence into German. ")
        assert f"={row['text']}" not in prompt
    assert sorted(prompts) == sorted(pool.render(r) for r in rows)

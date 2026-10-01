from copy import deepcopy

import pytest

from despamo.appearance.provenance import digest
from scripts.train_e3_pilot import (
    E1_SHARED_HASH,
    build_e3_identity,
    require_matched_e3_config,
    select_e3_checkpoint,
)


def _configs():
    e1 = {"data": {"spatial_root": "/frozen", "spatial_manifest": "/frozen/manifest.json",
                   "motion_root": "/motion", "batch_size": 4, "num_workers": 0},
          "model": {"use_in_context": False, "warm_up_steps": 1000},
          "trainer": {"max_steps": 4000, "accumulate_grad_batches": 1,
                      "default_root_dir": "/external"}, "seed": 0}
    e3 = deepcopy(e1)
    e3["data"].update(spatial_root="/adapted",
                      spatial_manifest="/adapted/complete/manifest.json")
    return e1, e3


def test_e3_config_only_changes_spatial_source_from_matched_e1():
    e1, e3 = _configs()
    require_matched_e3_config({"config": e1, "initial_shared_hash": E1_SHARED_HASH}, e3,
                              "/adapted", steps=4000)
    e3["model"]["warm_up_steps"] = 1001
    with pytest.raises(ValueError, match="matched|config"):
        require_matched_e3_config({"config": e1, "initial_shared_hash": E1_SHARED_HASH}, e3,
                                  "/adapted", steps=4000)


def test_e3_run_identity_binds_checkpoint_and_shared_initialization():
    e1, e3 = _configs()
    protocol = {"protocol_hash": "protocol", "split_hash": "split",
                "supervision_policy": "qwen-schema98-unreviewed-v1"}
    summary = {"checkpoint_hash": "a" * 64, "manifest_hash": "b" * 64,
               "frame_rows_hash": "c" * 64, "encoder_key": "d" * 64, "clips": 7096}
    result = build_e3_identity(protocol, summary, e3, 3600)
    assert result["initial_shared_hash"] == E1_SHARED_HASH
    assert result["adaptation_checkpoint_hash"] == "a" * 64
    assert result["variant"] == "E3_dino_lora"
    assert result["supervision_policy"] == "qwen-schema98-unreviewed-v1"
    assert result["run_key"].startswith("e3-only-")
    assert result["run_key"] == "e3-only-" + digest({k: v for k, v in result.items()
                                                      if k != "run_key"})


def test_e3_dev_selection_uses_full_dev_and_earliest_tie_only():
    ids = ("a", "b")
    reports = [{"split": "dev", "split_hash": "split", "items": [{"clip_id": clip}
                for clip in ids], "metrics": {"bleu4": 2.0},
                "checkpoint_step": step, "checkpoint_hash": str(step)}
               for step in (1750, 2800, 4000)]
    assert select_e3_checkpoint(reports, ids, 4000)["checkpoint_step"] == 1750
    reports[2]["metrics"]["bleu4"] = 3.0
    assert select_e3_checkpoint(reports, ids, 4000)["checkpoint_step"] == 4000
    reports[2]["items"].append({"clip_id": "Signer07"})
    with pytest.raises(ValueError, match="dev|clip"):
        select_e3_checkpoint(reports, ids, 4000)

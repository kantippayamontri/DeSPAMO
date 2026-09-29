from copy import deepcopy
from pathlib import Path

import pytest

from despamo.appearance.provenance import digest
from despamo.training.pilot_runtime import load_pilot_views, pilot_config, require_protocol

DATASET = Path(
    "/home/kan/datasets/despamo/appearance/datasets/"
    "b4d52678db150326ce22d1b73811883a99ec2b8100f258e3690b4d90a004297a"
)
ORIGINAL = Path(
    "/home/kan/datasets/despamo/features/"
    "1e84e43d142abc6242cce75c3e65abe9e1e03a6c61e3770181d7be381f3bc31d"
)
PROTOCOL = Path(
    "/home/kan/datasets/despamo/signer-pilot/protocol/"
    "0d7df319799e2c98883fdc6a494970be3edb4b1e62b09034c2c7742d1b2352c7/protocol.json"
)


def protocol_fixture():
    groups = {
        "train": {"clip_ids": [f"train-{i}" for i in range(5746)], "valid": 5641, "failed": 105},
        "dev": {"clip_ids": [f"dev-{i}" for i in range(582)], "valid": 571, "failed": 11},
        "test": {"clip_ids": [f"test-{i}" for i in range(768)], "valid": 765, "failed": 3},
    }
    payload = {
        "version": "signer-pilot-seed0-v1",
        "seed": 0,
        "supervision_policy": "qwen-schema98-unreviewed-v1",
        "human_review_status": "not_assessed",
        "gpu_hour_ceiling": 24,
        "split_hash": "split",
        "split": {"groups": groups, "split_hash": "split"},
        "decoder": {"mode": "deterministic", "beam_size": 5, "max_length": 64, "in_context": False},
    }
    return {**payload, "protocol_hash": digest(payload)}


def test_protocol_requires_exact_frozen_counts_content_hash_and_no_icl():
    protocol = protocol_fixture()
    require_protocol(protocol)
    for mutation in (
        lambda x: x["split"]["groups"]["dev"]["clip_ids"].pop(),
        lambda x: x["decoder"].update(in_context=True),
        lambda x: x.update(human_review_status="reviewed"),
        lambda x: x["split"]["groups"]["train"]["clip_ids"].append("test-0"),
        lambda x: x["split"].pop("split_hash"),
    ):
        tampered = deepcopy(protocol)
        mutation(tampered)
        tampered["protocol_hash"] = digest(
            {key: value for key, value in tampered.items() if key != "protocol_hash"}
        )
        with pytest.raises(ValueError, match="pilot|protocol|count"):
            require_protocol(tampered)


def test_config_fixes_flan_revision_reference_free_prompt_and_physical_batch():
    config = pilot_config(
        model_cache=Path("/cache"),
        spatial_root=Path("/dino"),
        motion_root=Path("/motion"),
        annotation_root=Path("/annotations"),
        spatial_manifest=Path("/dino/complete/manifest.json"),
        motion_manifest=Path("/motion-manifest.json"),
        steps=4000,
        physical_batch=2,
        output=Path("/external"),
    )
    assert config.model.name == "google/flan-t5-xl"
    assert config.model.revision == "7d6315df2c2fb742f0f5b556879d730926ca9001"
    assert config.model.warm_up_steps == 1000
    assert config.model.use_in_context is False
    assert config.model.num_in_context == 0
    assert config.model.spatial_crop_mode == "full"
    assert config.model.vt_pooling == "masked_mean"
    assert config.trainer.accumulate_grad_batches == 2
    assert config.data.batch_size == 2 and config.data.num_workers == 0
    assert config.trainer.max_steps == 4000
    with pytest.raises(ValueError, match="tier|batch"):
        pilot_config(
            Path("/cache"), Path("/dino"), Path("/motion"), Path("/annotations"),
            Path("/dino/complete/manifest.json"), Path("/motion-manifest.json"),
            4001, 3, Path("/external"),
        )


@pytest.mark.skipif(not PROTOCOL.is_file(), reason="frozen local signer-pilot corpus unavailable")
def test_live_pilot_views_and_join_remain_on_physical_train_and_exclude_holdouts():
    protocol, views = load_pilot_views(
        PROTOCOL,
        DATASET,
        DATASET
        / "text/24ad915ab5bde47335bfc8839f654b0a54ad66ac1253fc4218e1c415951fa571/manifest.json",
        ORIGINAL,
        Path("/home/kan/datasets/spamo/features/mae_feat_Phoenix14T"),
        Path("/home/kan/datasets/spamo/features/manifests/phoenix14t_motion.json"),
        Path("/home/kan/Research/SpaMo/preprocess/Phoenix14T/train_info_ml.npy"),
    )
    assert [len(views[name]) for name in ("train", "dev", "test")] == [5746, 582, 768]
    assert all(view.split == "train" for view in views.values())
    assert {row["signer"] for row in views["dev"].records} == {"Signer03"}
    assert {row["signer"] for row in views["test"].records} == {"Signer07"}
    assert protocol["protocol_hash"] == PROTOCOL.parent.name

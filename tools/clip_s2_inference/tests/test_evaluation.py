import numpy as np
import pytest
from torch import nn

from despamo.data.manifest import FeatureManifest, FeatureRecord
from despamo.data.phoenix14t import Phoenix14T
from tools.clip_s2_inference.evaluation import score_dev_cached
from tools.clip_s2_inference.tests.helpers import language
from tools.dinov3.identity import sha256_file
from tools.followup.recipe import PromptPool


class Visual(nn.Module):
    def forward(self, spatial, mask, motion, motion_mask):
        return spatial[:, :, :32], mask


def fixture(tmp_path):
    roots = {k: tmp_path / k for k in ("spatial", "motion")}
    rows = {
        n: dict(
            fileid=f"dev{n}",
            signer="Signer01",
            gloss="G",
            text=f"held out {n}",
            en_text="en",
            fr_text="fr",
            es_text="es",
        )
        for n in range(3)
    }
    annotation = tmp_path / "dev.npy"
    np.save(annotation, rows)
    hashes, manifests = {}, {}
    for kind, width, length in (("spatial", 2048, 5), ("motion", 1024, 1)):
        folder = roots[kind] / "dev"
        folder.mkdir(parents=True)
        records, hashes[kind] = [], {}
        for n in range(3):
            clip = f"dev{n}"
            p = folder / f"{clip}.npy"
            np.save(p, np.ones((length, width), dtype=np.float32) * n)
            hashes[kind]["dev", clip] = sha256_file(p)
            records.append(FeatureRecord(clip, "dev", f"dev/{clip}.npy", length, width, "float32"))
        manifests[kind] = FeatureManifest(1, "fixture", width, tuple(records))
    ds = Phoenix14T(
        annotation,
        "dev",
        roots["spatial"],
        roots["motion"],
        manifests["spatial"],
        manifests["motion"],
    )
    admission = dict(
        roots=roots,
        datasets={"dev": ds},
        ids={"dev": ("dev0", "dev1", "dev2")},
        lookup={"dev": {f"dev{n}": n for n in range(3)}},
        feature_hashes=hashes,
    )
    pool = PromptPool(
        [
            dict(fileid=f"train{n}", text=f"german {n}", en_text="en", fr_text="fr", es_text="es")
            for n in range(8)
        ],
        0,
    )
    model = nn.Module()
    model.visual_adapter = Visual()
    model.language_model = language()
    return model, admission, pool


def test_full_dev_uses_one_clip_per_batch_and_preserves_order_modes_and_config(tmp_path):
    model, admission, pool = fixture(tmp_path)
    model.train()
    modes = [m.training for m in model.modules()]
    events = []
    report = score_dev_cached(model, admission, pool, "split", on_event=events.append)
    assert report["complete"] and not report["diagnostic_only"]
    assert tuple(i["clip_id"] for i in report["items"]) == admission["ids"]["dev"]
    assert report["decoding"]["batch_size"] == 1 and report["decoding"]["use_cache"]
    assert report["decoding"]["beam_size"] == 5 and report["decoding"]["max_length"] == 64
    starts = [e for e in events if e["phase"] == "generation_start"]
    assert len(starts) == 3 and all(e["effective_beam_sequences"] == 5 for e in starts)
    assert all(e["joint_shape"][0] == 1 for e in starts)
    assert all(e["memory"]["allocated_bytes"] is None for e in events)
    assert [m.training for m in model.modules()] == modes
    assert model.language_model.model.config.use_cache is False


def test_bounded_probe_is_not_reported_as_full_corpus_or_scored(tmp_path):
    model, admission, pool = fixture(tmp_path)
    report = score_dev_cached(model, admission, pool, "split", max_batches=1)
    assert report["diagnostic_only"] and not report["complete"]
    assert len(report["items"]) == 1 and report["metrics"] is None
    assert report["processed_clips"] == 1 and report["expected_clips"] == 3


def test_reference_overlap_rejected_before_generation(tmp_path):
    model, admission, pool = fixture(tmp_path)
    pool.by_id[pool.eval_id]["text"] = admission["datasets"]["dev"].records[0]["text"]
    events = []
    with pytest.raises(ValueError, match="overlap"):
        score_dev_cached(model, admission, pool, "split", on_event=events.append)
    assert not events


def test_failure_records_error_and_restores_modes_without_retry(tmp_path, monkeypatch):
    import tools.clip_s2_inference.evaluation as e

    model, admission, pool = fixture(tmp_path)
    flags = [m.training for m in model.modules()]
    calls = []

    def fail(*args, **kwargs):
        calls.append(1)
        raise RuntimeError("simulated device failure")

    monkeypatch.setattr(e, "generate_cached_text", fail)
    events = []
    with pytest.raises(RuntimeError, match="simulated device failure"):
        score_dev_cached(model, admission, pool, "split", on_event=events.append)
    assert len(calls) == 1 and events[-1]["phase"] == "batch_error"
    assert [m.training for m in model.modules()] == flags


def test_error_observer_cannot_mask_original_device_fault(tmp_path, monkeypatch):
    import tools.clip_s2_inference.evaluation as e

    model, admission, pool = fixture(tmp_path)

    def fail(*args, **kwargs):
        raise RuntimeError("original device fault")

    def observer(event):
        if event["phase"] == "batch_error":
            raise OSError("logging failed")

    monkeypatch.setattr(e, "generate_cached_text", fail)
    with pytest.raises(RuntimeError, match="original device fault") as caught:
        score_dev_cached(model, admission, pool, "split", on_event=observer)
    assert any("logging failed" in note for note in caught.value.__notes__)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_probe_limits_rejected_before_loading_batches(tmp_path, limit):
    model, admission, pool = fixture(tmp_path)
    with pytest.raises(ValueError, match="positive"):
        score_dev_cached(model, admission, pool, "split", max_batches=limit)


def test_unexpected_short_loader_cannot_be_reported_as_full_or_bounded_success(
    tmp_path, monkeypatch
):
    import itertools

    import tools.clip_s2_inference.evaluation as e

    model, admission, pool = fixture(tmp_path)
    original = e.eval_batches
    monkeypatch.setattr(
        e, "eval_batches", lambda *args, **kwargs: itertools.islice(original(*args, **kwargs), 1)
    )
    with pytest.raises(ValueError, match="incomplete"):
        score_dev_cached(model, admission, pool, "split")
    with pytest.raises(ValueError, match="incomplete"):
        score_dev_cached(model, admission, pool, "split", max_batches=2)

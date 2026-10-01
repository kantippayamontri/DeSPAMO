import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_e3_features as cli

TRAIN_SIGNERS = (1, 2, 4, 5, 6, 8, 9)


def _setup(tmp_path, monkeypatch):
    train_ids = [f"s{s:02d}-{i}" for s in TRAIN_SIGNERS for i in range(12)]
    protocol = {
        "protocol_hash": cli.PROTOCOL_HASH,
        "split_hash": "split-h",
        "source_hash": "src-h",
        "split": {"groups": {"train": {"clip_ids": train_ids}}},
    }
    source = {"clips": [
        {"clip_id": c, "signer": f"Signer{int(c[1:3]):02d}",
         "frames": [{"index": i, "path": f"p{i}"} for i in range(5)]}
        for c in train_ids
    ]}
    text = {"records": [
        {"clip_id": c, "targets": [
            {"text": (f"hand: finger configuration shape_"
                      f"{TRAIN_SIGNERS.index(int(c[1:3]))}; palm orient")
                     if idx in (10, 11) else f"val-{idx}"}
            for idx in range(19)
        ]}
        for c in train_ids
    ]}
    monkeypatch.setattr(cli, "_load_sources", lambda *a: (protocol, source, text))
    adapt = tmp_path / "adapt"
    adapt.mkdir()
    identity = {
        "run_policy": "signer-pilot-e3-dino-lora-v1",
        "protocol_hash": cli.PROTOCOL_HASH,
        "source_hash": "src-h",
        "base_revision": "rev",
        "run_key": "adapt",
    }
    (adapt / "selected-checkpoint.json").write_text(json.dumps({
        "status": "complete", "step": 100, "checkpoint": "resume-100.pt",
        "checkpoint_hash": "ckpt-" + "a" * 60, "identity": identity,
    }))
    args = SimpleNamespace(
        mode="preflight", protocol=tmp_path / "p", dataset=tmp_path / "d",
        text_manifest=tmp_path / "t", original_dino_root=tmp_path / "orig",
        frames=tmp_path / "f", cache=tmp_path / "c", adapt_run=adapt,
        output=tmp_path / "report.json", clips_per_signer=8,
    )
    return args, train_ids


def test_gate_preflight_checks_identity_and_writes_nothing(tmp_path, monkeypatch):
    args, _ = _setup(tmp_path, monkeypatch)
    res = cli.run_gate(args)
    assert res["status"] == "ready"
    assert res["clips"] == 56
    assert not args.output.exists()


def test_gate_rejects_mismatched_adaptation_protocol(tmp_path, monkeypatch):
    args, _ = _setup(tmp_path, monkeypatch)
    sel = args.adapt_run / "selected-checkpoint.json"
    data = json.loads(sel.read_text())
    data["identity"]["protocol_hash"] = "wrong"
    sel.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="protocol"):
        cli.run_gate(args)


def test_gate_refuses_to_overwrite_existing_output(tmp_path, monkeypatch):
    args, _ = _setup(tmp_path, monkeypatch)
    args.output.write_text("exists")
    with pytest.raises(ValueError, match="output"):
        cli.run_gate(args)


def test_gate_handles_full_length_original_and_five_frame_adapted(tmp_path, monkeypatch):
    """Original features hold every frame; adapted features hold only the 5 sampled ones."""
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    rng = np.random.default_rng(1)
    anchor = rng.normal(size=(1, 2048)).astype(np.float32)
    anchor = 2.0 * anchor / np.linalg.norm(anchor)
    # 40 absolute frames, so frames[2]["index"] == 2 is valid but row 2 of a 5-row
    # array is a different frame; the gate must not index the 5-row array by index 26.
    orig_feats = {c: rng.normal(size=(40, 2048)).astype(np.float32) for c in ids}
    adap_feats = {}
    for c in ids:
        sampled = orig_feats[c][[0, 10, 20, 30, 39]]
        u = sampled / np.linalg.norm(sampled, axis=-1, keepdims=True)
        adap_feats[c] = (anchor + u).astype(np.float32)
    for rec in source_frames_with_spread(args):
        rec["frames"] = [{"index": i, "path": f"p{i}"} for i in (0, 10, 20, 30, 39)]
    monkeypatch.setattr(cli, "_load_subset_features",
                        lambda clips, a, model: (orig_feats, adap_feats))
    monkeypatch.setattr(cli, "_load_adaptation_model", lambda run, cache: object())
    res = cli.run_gate(args)
    assert res["status"] == "complete"
    assert set(res["handshape_probe"]) == {"left", "right"}


def source_frames_with_spread(args):
    return cli._load_sources(args)[1]["clips"]


def test_gate_computes_std_ratio_and_accuracy_without_clip_predictions(tmp_path, monkeypatch):
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    # Model cone collapse: add anchor of norm 2.0 to unit-normalised frames
    rng = np.random.default_rng(0)
    anchor = rng.normal(size=(1, 2048)).astype(np.float32)
    anchor = 2.0 * anchor / np.linalg.norm(anchor)
    orig_feats = {c: rng.normal(size=(5, 2048)).astype(np.float32) for c in ids}
    # Per-frame unit normalisation then add anchor
    adap_feats = {}
    for c in ids:
        u = orig_feats[c] / np.linalg.norm(orig_feats[c], axis=-1, keepdims=True)
        adap_feats[c] = (anchor + u).astype(np.float32)
    monkeypatch.setattr(cli, "_load_subset_features",
                        lambda clips, args, model: (orig_feats, adap_feats))
    monkeypatch.setattr(cli, "_load_adaptation_model", lambda run, cache: object())
    res = cli.run_gate(args)
    assert res["status"] == "complete"
    assert "predictions" not in args.output.read_text()
    assert 0.0 < res["geometry"]["std_ratio"] < 0.35, "std_ratio must capture cone collapse"
    assert set(res["signer_probe"]) == {"original", "adapted", "delta_top1"}
    assert set(res["handshape_probe"]) == {"left", "right"}

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import probe_e3_features as cli
from tools.dinov3.identity import digest, sha256_file

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
    assert res["gate_pass"] is False
    assert res["gate_status"] == "FAIL"


def _write_feature_inputs(args, ids):
    source = cli._load_sources(args)[1]
    indices = [0, 10, 20, 30, 39]
    rng = np.random.default_rng(17)
    arrays, rows = {}, {}
    (args.original_dino_root / "train").mkdir(parents=True)
    for clip in source["clips"]:
        cid = clip["clip_id"]
        clip["frame_count"] = 40
        clip["frames"] = [{"index": i, "path": f"{cid}/p{i}"} for i in indices]
        values = rng.normal(size=(40, 2048)).astype(np.float32) + 10
        values[indices] = rng.normal(size=(5, 2048)).astype(np.float32)
        values[:, 0] = np.arange(40)
        arrays[cid] = values
        path = args.original_dino_root / "train" / f"{cid}.npy"
        np.save(path, values)
        rows[cid] = {
            "frame_count": 40, "source_indices": list(range(40)),
            "feature_hash": sha256_file(path),
        }
    args.dataset.mkdir()
    (args.dataset / "source.json").write_text(json.dumps(source))
    complete = args.original_dino_root / "complete"
    complete.mkdir()
    (complete / "frame_rows.json").write_text(json.dumps({"clips": rows}))
    return arrays


def _fake_extract(monkeypatch, arrays):
    def extract(paths, model, device, *, batch_size):
        cid = paths[0].parent.name
        indices = [int(path.name[1:]) for path in paths]
        return arrays[cid][indices].copy(), [[210, 260] for _ in paths]

    monkeypatch.setitem(sys.modules, "tools.dinov3.encoder", SimpleNamespace(extract=extract))


def test_subset_loader_selects_same_five_original_and_adapted_frames(tmp_path, monkeypatch):
    args, ids = _setup(tmp_path, monkeypatch)
    arrays = _write_feature_inputs(args, ids)
    _fake_extract(monkeypatch, arrays)
    orig, adapted = cli._load_subset_features((ids[0],), args, object())
    assert orig[ids[0]].shape == (5, 2048)
    np.testing.assert_array_equal(orig[ids[0]][:, 0], [0, 10, 20, 30, 39])
    np.testing.assert_array_equal(orig[ids[0]], adapted[ids[0]])


def test_identical_encoder_has_unit_spread_and_zero_signer_delta(tmp_path, monkeypatch):
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    arrays = _write_feature_inputs(args, ids)
    _fake_extract(monkeypatch, arrays)
    monkeypatch.setattr(cli, "_load_adaptation_model", lambda *a: object())
    report = cli.run_gate(args)
    assert report["geometry"]["std_ratio"] == pytest.approx(1.0)
    assert report["signer_probe"]["delta_top1"] == 0.0
    assert report["gate_pass"] is True
    assert report["gate_status"] == "PASS"


@pytest.mark.parametrize("missing", [("left",), ("right",), ("left", "right")])
def test_unavailable_hand_probe_is_incomplete_not_passed(tmp_path, monkeypatch, missing):
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    text = cli._load_sources(args)[2]
    for rec in text["records"]:
        for side in missing:
            rec["targets"][10 if side == "left" else 11]["text"] = None
    rng = np.random.default_rng(0)
    features = {cid: rng.normal(size=(5, 2048)).astype(np.float32) for cid in ids}
    monkeypatch.setattr(cli, "_load_subset_features", lambda *a: (features, features))
    monkeypatch.setattr(cli, "_load_adaptation_model", lambda *a: object())
    report = cli.run_gate(args)
    assert report["gate_pass"] is None
    assert report["gate_status"] == "INCOMPLETE"
    for side in missing:
        assert report["handshape_probe"][side]["status"] == "unavailable"
        assert report["handshape_probe"][side]["reason"]


def test_hand_probe_with_under_supported_class_is_unavailable(tmp_path, monkeypatch):
    args, ids = _setup(tmp_path, monkeypatch)
    source, text = cli._load_sources(args)[1:]
    selected = tuple(cid for cid in ids if not cid.startswith("s09-") or cid == "s09-0")
    rng = np.random.default_rng(0)
    features = {cid: rng.normal(size=(5, 2048)).astype(np.float32) for cid in selected}
    assert cli._probe_handshape(features, source, text, selected, "left") is None


def _write_cached_features(args, arrays):
    source = cli._load_sources(args)[1]
    selected = json.loads((args.adapt_run / "selected-checkpoint.json").read_text())
    args.adapted_root = args.output.parent / "cached"
    for folder in ("train", "complete", "receipts/train"):
        (args.adapted_root / folder).mkdir(parents=True)
    identity = {
        "checkpoint_hash": selected["checkpoint_hash"],
        "adaptation_run_key": args.adapt_run.name,
        "protocol_hash": cli.PROTOCOL_HASH,
        "source_hash": "src-h",
        "frame_rows_hash": sha256_file(args.original_dino_root / "complete/frame_rows.json"),
    }
    records, rows = [], {}
    for clip in source["clips"]:
        cid = clip["clip_id"]
        path = args.adapted_root / "train" / f"{cid}.npy"
        np.save(path, arrays[cid])
        row = {
            "frame_count": 40, "source_indices": list(range(40)),
            "paths": [str((args.frames / f"{cid}/p{i}").resolve()) for i in range(40)],
            "feature_hash": sha256_file(path), "source_hash": f"source-{cid}",
        }
        rows[cid] = row
        records.append({"clip_id": cid, "split": "train", "path": f"train/{cid}.npy",
                        "length": 40, "width": 2048, "dtype": "float32"})
        receipt = {**row, "checkpoint_hash": identity["checkpoint_hash"], "clip_id": cid}
        (args.adapted_root / "receipts/train" / f"{cid}.json").write_text(json.dumps(receipt))
    complete = args.adapted_root / "complete"
    (complete / "identity.json").write_text(json.dumps(identity))
    (complete / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "encoder": digest(identity), "expected_dim": 2048,
        "records": records,
    }))
    (complete / "frame_rows.json").write_text(json.dumps({
        "encoder_key": digest(identity), "clips": rows,
    }))
    return identity


def test_cached_gate_runs_same_frame_control_without_loading_cuda_model(tmp_path, monkeypatch):
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    arrays = _write_feature_inputs(args, ids)
    identity = _write_cached_features(args, arrays)

    def reject_cuda(*a):
        raise AssertionError("Cached probes must not load a CUDA model")

    monkeypatch.setattr(cli, "_load_adaptation_model", reject_cuda)
    report = cli.run_gate(args)
    assert report["geometry"]["std_ratio"] == pytest.approx(1.0)
    assert report["signer_probe"]["delta_top1"] == 0.0
    assert report["gate_status"] == "PASS"
    assert report["adapted_encoder_key"] == digest(identity)
    assert report["schema_version"] == 2
    assert report["sampling"]["frames_per_clip"] == 5
    assert report["code_hashes"]["scripts/probe_e3_features.py"]


@pytest.mark.parametrize("drift", ["checkpoint", "frame_order", "feature"])
def test_cached_gate_rejects_checkpoint_alignment_or_feature_drift(tmp_path, monkeypatch, drift):
    args, ids = _setup(tmp_path, monkeypatch)
    args.mode = "run"
    arrays = _write_feature_inputs(args, ids)
    _write_cached_features(args, arrays)
    monkeypatch.setattr(cli, "_load_adaptation_model",
                        lambda *a: pytest.fail("Cached probes must not load a CUDA model"))
    if drift == "checkpoint":
        path = args.adapted_root / "complete/identity.json"
        value = json.loads(path.read_text())
        value["checkpoint_hash"] = "wrong"
        path.write_text(json.dumps(value))
    elif drift == "frame_order":
        path = args.adapted_root / "complete/frame_rows.json"
        value = json.loads(path.read_text())
        for row in value["clips"].values():
            row["source_indices"].reverse()
        path.write_text(json.dumps(value))
    else:
        for cid in ids:
            np.save(args.adapted_root / "train" / f"{cid}.npy", arrays[cid] + 1)
    with pytest.raises(ValueError, match="cached"):
        cli.run_gate(args)
    assert not args.output.exists()

from types import SimpleNamespace

import pytest

from despamo.appearance.provenance import atomic_json
from scripts import evaluate_signer_pilot as cli


def _setup(monkeypatch, tmp_path):
    ids = tuple(f"clip-{i}" for i in range(768))
    protocol = {
        "protocol_hash": cli.PROTOCOL_HASH,
        "split_hash": "frozen-split",
        "decoder": {"mode": "deterministic", "beam_size": 5, "max_length": 64,
                    "in_context": False},
        "split": {"groups": {"test": {"clip_ids": list(ids), "signers": ["Signer07"]}}},
    }
    records = [{"fileid": clip_id, "signer": "Signer07"} for clip_id in ids]
    monkeypatch.setattr(cli, "load_pilot_views", lambda *args: (protocol, {"test":
                                                                SimpleNamespace(records=records)}))
    monkeypatch.setattr(cli, "_validate_run", lambda root, variant, proto: {
        "checkpoint": root / "step-4000.ckpt", "checkpoint_hash": variant + "-hash",
        "run_key": root.name,
    })
    args = SimpleNamespace(
        mode="preflight", protocol=tmp_path / "protocol", dataset=tmp_path / "dataset",
        text_manifest=tmp_path / "text", dino_root=tmp_path / "dino",
        motion_root=tmp_path / "motion", motion_manifest=tmp_path / "motion-manifest",
        annotation=tmp_path / "annotation", e1_run=tmp_path / "e1",
        e2_run=tmp_path / "e2", output=tmp_path / "new-report",
    )
    return args, ids


def test_preflight_requires_768_signer07_clips_and_writes_nothing(tmp_path, monkeypatch):
    args, ids = _setup(monkeypatch, tmp_path)
    result = cli.evaluate(args)
    assert result["test_clips"] == 768
    assert result["split_hash"] == "frozen-split"
    assert not args.output.exists()
    monkeypatch.setattr(cli, "load_pilot_views", lambda *unused: (
        {"protocol_hash": cli.PROTOCOL_HASH, "split_hash": "frozen-split",
         "decoder": {"mode": "deterministic", "beam_size": 5, "max_length": 64,
                     "in_context": False},
         "split": {"groups": {"test": {"clip_ids": list(ids[:-1]),
                                         "signers": ["Signer07"]}}}},
        {"test": SimpleNamespace(records=[{"fileid": clip_id, "signer": "Signer07"}
                                          for clip_id in ids[:-1]])},
    ))
    with pytest.raises(ValueError, match="768"):
        cli.evaluate(args)


def test_run_writes_ordered_paired_reports_once(tmp_path, monkeypatch):
    args, ids = _setup(monkeypatch, tmp_path)
    args.mode = "run"
    events = []

    def fake_score(variant, run, view, expected_ids, split_hash):
        events.append(variant)
        assert expected_ids == ids
        assert split_hash == "frozen-split"
        return {"split": "test", "split_hash": split_hash,
                "checkpoint_hash": run["checkpoint_hash"],
                "model_variant": "E1_frozen" if variant == "e1" else "E2_projector",
                "decoding": {"mode": "deterministic", "beam_size": 5,
                             "max_length": 64, "in_context": False},
                "items": [{"clip_id": clip_id, "reference": "same", "prediction": "hyp"}
                          for clip_id in ids],
                "metrics": {"bleu4": 2.0 if variant == "e1" else 1.75}}

    monkeypatch.setattr(cli, "_score", fake_score)
    result = cli.evaluate(args)
    assert events == ["e1", "e2"]
    assert result["e2_minus_e1_bleu4"] == -0.25
    assert len(result["clip_ids_hash"]) == 64
    assert sorted(path.name for path in args.output.iterdir()) == [
        "e1-test.json", "e2-test.json", "paired.json"
    ]
    with pytest.raises(ValueError, match="output"):
        cli.evaluate(args)


def test_pair_rejects_reference_drift_without_summary(tmp_path, monkeypatch):
    args, ids = _setup(monkeypatch, tmp_path)
    args.mode = "run"

    def fake_score(variant, run, view, expected_ids, split_hash):
        return {"split": "test", "split_hash": split_hash,
                "checkpoint_hash": run["checkpoint_hash"],
                "model_variant": "E1_frozen" if variant == "e1" else "E2_projector",
                "decoding": {"mode": "deterministic", "beam_size": 5,
                             "max_length": 64, "in_context": False},
                "items": [{"clip_id": clip_id, "reference": "drift" if variant == "e2" else "same"}
                          for clip_id in ids],
                "metrics": {"bleu4": 2.0}}

    monkeypatch.setattr(cli, "_score", fake_score)
    with pytest.raises(ValueError, match="references"):
        cli.evaluate(args)
    assert not (args.output / "paired.json").exists()


def test_factor_provenance_accepts_expanded_checkpoint_metadata():
    base = {"supervision_policy": "qwen-schema98-unreviewed-v1"}
    model = SimpleNamespace(factor_provenance={**base, "total_steps": 4000,
                                               "enabled": ["attire"]})
    checkpoint = {"factor_provenance": dict(model.factor_provenance)}
    cli.require_factor_provenance(checkpoint, model, {"factor_provenance": base})
    checkpoint["factor_provenance"]["total_steps"] = 6000
    with pytest.raises(ValueError, match="factor provenance"):
        cli.require_factor_provenance(checkpoint, model, {"factor_provenance": base})


def test_resume_keeps_verified_e1_score_and_only_scores_e2(tmp_path, monkeypatch):
    args, ids = _setup(monkeypatch, tmp_path)
    args.mode = "run"
    args.output.mkdir()
    e1_report = {
        "split": "test", "split_hash": "frozen-split", "checkpoint_hash": "e1-hash",
        "model_variant": "E1_frozen", "decoding": cli.DECODER,
        "items": [{"clip_id": clip_id, "reference": "same"} for clip_id in ids],
        "metrics": {"bleu4": 2.0},
    }
    atomic_json(args.output / "e1-test.json", e1_report)
    seen = []

    def fake_score(variant, run, view, expected_ids, split_hash):
        seen.append(variant)
        return {**e1_report, "checkpoint_hash": "e2-hash", "model_variant": "E2_projector",
                "metrics": {"bleu4": 1.5}}

    monkeypatch.setattr(cli, "_score", fake_score)
    result = cli.evaluate(args)
    assert seen == ["e2"]
    assert result["e2_minus_e1_bleu4"] == -0.5
    assert cli.read_json(args.output / "e1-test.json") == e1_report

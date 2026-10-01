import argparse
import json

import pytest
import torch

from despamo.appearance.provenance import file_hash
from scripts import probe_signer_pilot as cli


def _fixtures(tmp_path, monkeypatch):
    clips = [(f"s{s}-{i}", f"Signer{s:02d}") for s in range(7)
             for i in range(820 + (s < 6))]
    protocol = {"protocol_hash": cli.PROTOCOL_HASH,
                "split_hash": "split-123",
                "split": {"groups": {"train": {"clip_ids": [clip_id for clip_id, _ in clips],
                                                 "signers": [f"Signer{s:02d}" for s in range(7)]}}}}
    protocol.update({name: "fixture-hash" for name in (
        "source_hash", "version_hash", "records_hash", "annotation_hash", "frame_rows_hash",
        "spatial_manifest_hash", "motion_manifest_hash", "text_manifest_hash"
    )})
    class View:
        records = [{"fileid": clip_id, "signer": signer} for clip_id, signer in clips]
    monkeypatch.setattr(cli, "load_pilot_views", lambda *args: (protocol, {"train": View()}))
    runs = []
    hashes = {}
    for name, variant, policy in (("e1", "E1_frozen", "signer-pilot-e1-fresh-retry-v3"),
                                  ("e2", "E2_projector", "signer-pilot-e2-projector-bounded-v1")):
        root = tmp_path / name
        root.mkdir()
        torch.save({"state_dict": {"visual_adapter.spatial_projector.weight":
                                   torch.ones(2, 2048),
                                   "visual_adapter.spatial_projector.bias": torch.zeros(2)}},
                   root / "step-4000.ckpt")
        hashes[name] = file_hash(root / "step-4000.ckpt")
        (root / "run-status.json").write_text(
            json.dumps({"status": "complete", "global_step": 4000})
        )
        identity = {"run_policy": policy, "variant": variant, "protocol_hash": cli.PROTOCOL_HASH,
                    "initial_shared_hash": cli.SHARED_HASH}
        if name == "e2":
            identity["matched_e1_checkpoint_hash"] = hashes["e1"]
            identity["factor_provenance"] = {"supervision_policy": "qwen-schema98-unreviewed-v1",
                                            "human_review_status": "not_assessed"}
        (root / "budget.json").write_text(json.dumps({"status": "closed", "identity": identity}))
        (root / "selected-checkpoint.json").write_text(
            json.dumps({"step": 4000, "checkpoint_hash": hashes[name]})
        )
        runs.append(root)
    monkeypatch.setattr(cli, "CHECKPOINT_HASHES", {"e1": hashes["e1"], "e2": hashes["e2"]})
    args = argparse.Namespace(protocol=tmp_path / "protocol", dataset=tmp_path / "dataset",
                              text_manifest=tmp_path / "text", dino_root=tmp_path / "dino",
                              motion_root=tmp_path / "motion",
                              motion_manifest=tmp_path / "motion-json",
                              annotation=tmp_path / "annotation", e1_run=runs[0], e2_run=runs[1],
                              output=tmp_path / "report.json", mode="preflight")
    return args


def test_preflight_checks_both_checkpoints_without_writing_or_loading(tmp_path, monkeypatch):
    args = _fixtures(tmp_path, monkeypatch)
    monkeypatch.setattr(
        cli.torch, "load", lambda *a, **kw: pytest.fail("preflight loaded checkpoint")
    )
    result = cli.probe(args)
    assert result["split_hash"] == "split-123"
    assert not args.output.exists()


def test_preflight_rejects_changed_run_identity_or_checkpoint(tmp_path, monkeypatch):
    args = _fixtures(tmp_path, monkeypatch)
    budget = args.e2_run / "budget.json"
    contents = json.loads(budget.read_text())
    contents["identity"]["protocol_hash"] = "wrong"
    budget.write_text(json.dumps(contents))
    with pytest.raises(ValueError, match="identity"):
        cli.probe(args)
    contents["identity"]["protocol_hash"] = cli.PROTOCOL_HASH
    budget.write_text(json.dumps(contents))
    (args.e2_run / "step-4000.ckpt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checkpoint"):
        cli.probe(args)


def test_preflight_rejects_output_inside_source_run(tmp_path, monkeypatch):
    args = _fixtures(tmp_path, monkeypatch)
    args.output = args.e1_run / "report.json"
    with pytest.raises(ValueError, match="output"):
        cli.probe(args)


def test_run_writes_aggregate_paired_scores_without_clip_predictions(tmp_path, monkeypatch):
    args = _fixtures(tmp_path, monkeypatch)
    args.mode = "run"
    monkeypatch.setattr(cli, "pool_spatial", lambda view, ids: torch.ones(len(ids), 2048))
    monkeypatch.setattr(cli, "score_probe", lambda features, labels, split, ids: {
        "accuracy": float(features[0, 0] > 0), "balanced_accuracy": 0.5,
        "counts": {f"Signer{i:02d}": len(split["test"]) // 7 for i in range(7)},
        "majority_baseline": 1 / 7, "weight_decay": 0., "epoch": 1,
    })
    result = cli.probe(args)
    assert json.loads(args.output.read_text()) == result
    assert result["purpose"] == "diagnostic_only"
    assert result["partition_counts"]["fit"] > 0
    assert result["e2_minus_e1"]["accuracy"] == 0
    assert "predictions" not in args.output.read_text()
    with pytest.raises(ValueError, match="output"):
        cli.probe(args)

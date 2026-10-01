import json
from types import SimpleNamespace

import pytest
import torch

from despamo.appearance.provenance import digest
from despamo.evaluation.pilot_probe import split_probe_clips
from scripts import probe_e3_pilot as cli


def _fixture(tmp_path, monkeypatch):
    records = [{"fileid": f"{signer}-{i}", "signer": f"Signer{signer:02d}"}
               for signer in (1, 2, 4, 5, 6, 8, 9) for i in range(20)]
    train_ids = [record["fileid"] for record in records]
    protocol = {"protocol_hash": cli.PROTOCOL_HASH, "split_hash": "split",
                "split": {"groups": {"train": {"clip_ids": train_ids},
                                      "dev": {"clip_ids": ["dev"]}}}}
    monkeypatch.setattr(cli, "load_e3_views", lambda *args: (
        protocol, {"train": SimpleNamespace(records=records)}
    ))
    monkeypatch.setattr(cli, "read_json", lambda path: {"checkpoint_hash": "adapt-hash"}
                        if path.name == "identity.json" else json.loads(path.read_text()))
    monkeypatch.setattr(cli, "_validate_e3_selection", lambda *args: {
        "checkpoint": tmp_path / "e3/step-4000.ckpt", "checkpoint_hash": "e3-hash",
        "identity": {"adaptation_checkpoint_hash": "adapt-hash"}, "step": 4000,
    })
    split = split_probe_clips([(row["fileid"], row["signer"]) for row in records])
    manifest = tmp_path / "e1-e2-probe.json"
    manifest.write_text(json.dumps({"protocol_hash": cli.PROTOCOL_HASH,
                                    "split_hash": "split", "partition_hash": digest(split)}))
    args = SimpleNamespace(mode="preflight", adapted_root=tmp_path / "adapted",
                           e3_run=tmp_path / "e3", split_manifest=manifest,
                           output=tmp_path / "probe.json", protocol=tmp_path / "protocol",
                           dataset=tmp_path / "dataset", text_manifest=tmp_path / "text",
                           dino_root=tmp_path / "dino", motion_root=tmp_path / "motion",
                           motion_manifest=tmp_path / "motion-manifest",
                           annotation=tmp_path / "annotation")
    return args, split


def test_e3_probe_preflight_requires_same_train_partition(tmp_path, monkeypatch):
    args, split = _fixture(tmp_path, monkeypatch)
    result = cli.probe(args)
    assert result["partition_hash"] == digest(split)
    assert not args.output.exists()
    args.split_manifest.write_text(json.dumps({"protocol_hash": cli.PROTOCOL_HASH,
                                               "split_hash": "split",
                                               "partition_hash": "wrong"}))
    with pytest.raises(ValueError, match="partition"):
        cli.probe(args)


def test_e3_probe_reports_aggregate_only_and_refuses_repeat(tmp_path, monkeypatch):
    args, split = _fixture(tmp_path, monkeypatch)
    args.mode = "run"
    monkeypatch.setattr(cli, "pool_spatial", lambda view, ids: torch.ones(len(ids), 2048))
    monkeypatch.setattr(cli, "_load_projector", lambda path: (torch.ones(2, 2048), torch.zeros(2)))
    monkeypatch.setattr(cli, "score_probe", lambda x, labels, partition, ids: {
        "accuracy": 0.5, "balanced_accuracy": 0.5, "counts": {"Signer01": 3},
    })
    result = cli.probe(args)
    assert result["purpose"] == "diagnostic_only"
    assert result["partition_hash"] == digest(split)
    assert "predictions" not in args.output.read_text()
    with pytest.raises(ValueError, match="output"):
        cli.probe(args)

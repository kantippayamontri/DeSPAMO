import json
from types import SimpleNamespace

import pytest

from scripts import evaluate_e3_pilot as cli


def _args(tmp_path, monkeypatch):
    ids = tuple(f"test-{i}" for i in range(768))
    protocol = {"protocol_hash": cli.PROTOCOL_HASH, "split_hash": "split",
                "decoder": cli.DECODER,
                "split": {"groups": {"test": {"clip_ids": list(ids),
                                               "signers": ["Signer07"]},
                                      "dev": {"clip_ids": ["dev"]}}}}
    views = {"test": SimpleNamespace(records=[{"fileid": clip, "signer": "Signer07"}
                                               for clip in ids])}
    monkeypatch.setattr(cli, "load_e3_views", lambda original, root, identity: (protocol, views))
    monkeypatch.setattr(cli, "_validate_e3_selection", lambda run, protocol, ids: {
        "checkpoint": run / "step-4000.ckpt", "checkpoint_hash": "checkpoint-hash",
        "step": 4000, "identity": {"protocol_hash": cli.PROTOCOL_HASH,
                                   "adaptation_checkpoint_hash": "adapt-hash"},
    })
    args = SimpleNamespace(mode="preflight", protocol=tmp_path / "protocol",
                           dataset=tmp_path / "dataset", text_manifest=tmp_path / "text",
                           dino_root=tmp_path / "dino", motion_root=tmp_path / "motion",
                           motion_manifest=tmp_path / "motion-manifest",
                           annotation=tmp_path / "annotation", adapted_root=tmp_path / "adapted",
                           e3_run=tmp_path / "run", output=tmp_path / "report.json")
    monkeypatch.setattr(cli, "read_json", lambda path: {"checkpoint_hash": "adapt-hash"})
    return args, ids


def test_e3_preflight_requires_exact_signer07_inventory(tmp_path, monkeypatch):
    args, ids = _args(tmp_path, monkeypatch)
    result = cli.evaluate(args)
    assert result["test_clips"] == 768
    assert result["checkpoint_hash"] == "checkpoint-hash"
    assert not args.output.exists()
    wrong = ids[:-1]
    monkeypatch.setattr(cli, "load_e3_views", lambda *args: (
        {"protocol_hash": cli.PROTOCOL_HASH, "split_hash": "split", "decoder": cli.DECODER,
         "split": {"groups": {"test": {"clip_ids": list(wrong), "signers": ["Signer07"]}}}},
        {"test": SimpleNamespace(records=[{"fileid": item, "signer": "Signer07"}
                                          for item in wrong])},
    ))
    with pytest.raises(ValueError, match="768"):
        cli.evaluate(args)


def test_e3_heldout_run_writes_once_with_checkpoint_provenance(tmp_path, monkeypatch):
    args, ids = _args(tmp_path, monkeypatch)
    args.mode = "run"
    calls = []

    def score(run, view, expected_ids, split_hash):
        calls.append((expected_ids, split_hash))
        return {"split": "test", "split_hash": split_hash, "decoding": cli.DECODER,
                "items": [{"clip_id": clip, "prediction": "hyp", "reference": "ref"}
                          for clip in expected_ids], "metrics": {"bleu4": 1.5}}

    monkeypatch.setattr(cli, "_score_e3", score)
    result = cli.evaluate(args)
    assert calls == [(ids, "split")]
    assert json.loads(args.output.read_text()) == result
    assert result["adaptation_checkpoint_hash"] == "adapt-hash"
    with pytest.raises(ValueError, match="output"):
        cli.evaluate(args)

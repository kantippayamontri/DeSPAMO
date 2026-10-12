import importlib

import pytest
import torch

from tools.dinov3.identity import sha256_file
from tools.dinov3.storage import atomic_json


def module():
    return importlib.import_module("tools.benchmark.official_spatial")


def test_epoch40_compact_checkpoint_admission(tmp_path):
    run=tmp_path/"run"
    run.mkdir()
    identity=dict(config=dict(model=dict(revision="ea8dc2863c51be0a264bab82070e3e8836b02d51")))
    checkpoint=run/"model-epoch-40.pt"
    torch.save(dict(identity=identity,epoch=40,completed_updates=11800,
                    model={"x.up.weight":torch.ones(2)},heads={}),checkpoint)
    atomic_json(checkpoint.with_suffix(".json"),dict(sha256=sha256_file(checkpoint),
                                                   bytes=checkpoint.stat().st_size))
    atomic_json(run/"selected-checkpoint.json",dict(status="complete",epoch=40,
        checkpoint=checkpoint.name,checkpoint_hash=sha256_file(checkpoint),identity=identity))
    atomic_json(run/"completion.json",dict(status="complete",identity=identity))
    result=module().admit_epoch40(run)
    assert result["checkpoint_sha256"]==sha256_file(checkpoint)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError):
        module().admit_epoch40(run)


def test_nonterminal_selection_cannot_extract(tmp_path):
    atomic_json(tmp_path/"selected-checkpoint.json",dict(status="complete",epoch=30))
    with pytest.raises(ValueError):
        module().admit_epoch40(tmp_path)

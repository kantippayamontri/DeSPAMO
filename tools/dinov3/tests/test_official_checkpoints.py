import importlib
import random
from copy import deepcopy

import numpy as np
import pytest
import torch

from tools.dinov3.official_sampler import OfficialBatchSampler
from tools.dinov3.official_schedule import AdaptSchedule
from tools.dinov3.storage import atomic_json


def module():
    return importlib.import_module("tools.dinov3.official_checkpoints")


def state_for(run, epoch=1):
    schedule = AdaptSchedule(train_clips=8, batch_size=4, epochs=4, calibration_epochs=1,
                             lora_warmup_epochs=1, grl_ramp_epochs=1, checkpoint_epochs=1)
    step = epoch * 2
    sampler = OfficialBatchSampler(tuple(str(i) for i in range(8)), batch_size=4)
    for _ in range(step):
        sampler.next_ids()
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([parameter])
    parameter.sum().backward()
    optimizer.step()
    return dict(schema_version=1, identity=dict(run_key=run.name, run_root=str(run),
                policy="official-phoenix14t-merged-dino-adaptation-v1"), epoch=epoch,
                completed_updates=step, applied_update=step,
                model={"test.up.weight": torch.ones(2)},
                heads={"test.weight": torch.ones(2)}, optimizer=optimizer.state_dict(),
                schedule=schedule.state_dict(step), sampler=sampler.state_dict(),
                rng=dict(python=random.getstate(), numpy=np.random.get_state(),
                         torch=torch.get_rng_state(), cuda=[]), charged_seconds=10.0+epoch)


def test_milestone_publication_resume_and_model_retention(tmp_path):
    run = tmp_path / "official-dino-adapt-test"
    run.mkdir()
    for epoch in (1, 2, 3):
        state = state_for(run, epoch)
        receipt = module().publish_milestone(run, state, epoch)
        assert receipt["epoch"] == epoch
        path = run / f"resume-epoch-{epoch}.pt"
        restored = module().validate_official_resume(path, state["identity"],
            dict(charged_seconds=100., completed_updates=epoch*2))
        assert restored["sampler"] == state["sampler"]
        compact = torch.load(run/f"model-epoch-{epoch}.pt", weights_only=False)
        assert "optimizer" not in compact and compact["model"].keys() == state["model"].keys()
        module().prune_owned_resumes(run)
    assert sorted(p.name for p in run.glob("resume-*.pt")) == [
        "resume-epoch-2.pt", "resume-epoch-3.pt"]
    assert len(list(run.glob("model-*.pt"))) == 3


@pytest.mark.parametrize("bad", ["optimizer", "rng", "sampler", "nan", "step", "smoke", "time",
                                "cursor", "optimizer_state", "bool_schema"])
def test_invalid_snapshot_rejected_before_publication(tmp_path, bad):
    run = tmp_path / "official-dino-adapt-test"
    run.mkdir()
    state = state_for(run)
    if bad in ("optimizer", "rng"):
        state[bad] = {}
    elif bad == "sampler":
        state["sampler"]["batches_yielded"] = 1
    elif bad == "nan":
        state["model"]["test.up.weight"][:] = float("nan")
    elif bad == "step":
        state["applied_update"] += 1
    elif bad == "smoke":
        state["identity"]["policy"] = "official-adaptation-smoke-v1"
    elif bad == "cursor":
        state["sampler"]["cursor"] = 1
    elif bad == "optimizer_state":
        state["optimizer"]["state"] = {}
    elif bad == "bool_schema":
        state["schema_version"] = True
    else:
        state["charged_seconds"] = float("nan")
    with pytest.raises(ValueError):
        module().publish_milestone(run, state, 1)
    assert not list(run.glob("*.pt"))


def test_resume_rejects_stale_hash_and_accounting_rewind(tmp_path):
    run = tmp_path / "official-dino-adapt-test"
    run.mkdir()
    one, two = state_for(run), state_for(run, 2)
    module().publish_milestone(run, one, 1)
    module().publish_milestone(run, two, 2)
    ledger = dict(charged_seconds=100., completed_updates=4)
    with pytest.raises(ValueError):
        module().validate_official_resume(run/"resume-epoch-1.pt", one["identity"], ledger)
    with pytest.raises(ValueError):
        module().validate_official_resume(run/"resume-epoch-2.pt", two["identity"],
                                          dict(charged_seconds=1., completed_updates=4))
    (run/"resume-epoch-2.pt").write_bytes(b"bad")
    with pytest.raises(ValueError):
        module().validate_official_resume(run/"resume-epoch-2.pt", two["identity"], ledger)


def test_receipt_failure_cannot_prune_older_resume(tmp_path, monkeypatch):
    run = tmp_path / "official-dino-adapt-test"
    run.mkdir()
    for epoch in (1, 2):
        module().publish_milestone(run, state_for(run, epoch), epoch)
    def failure(path, value):
        raise OSError("receipt fsync failed")
    monkeypatch.setattr(module(), "atomic_json", failure)
    with pytest.raises(OSError):
        module().publish_milestone(run, state_for(run, 3), 3)
    with pytest.raises(ValueError):
        module().prune_owned_resumes(run)
    assert (run/"resume-epoch-1.pt").is_file()


def test_duplicate_publication_fails_without_overwrite(tmp_path):
    run = tmp_path / "official-dino-adapt-test"
    run.mkdir()
    state = state_for(run)
    module().publish_milestone(run, state, 1)
    with pytest.raises(ValueError):
        module().publish_milestone(run, deepcopy(state), 1)
    atomic_json(run/"resume-epoch-1.json", {"sha256": "wrong"})
    with pytest.raises(ValueError):
        module().validate_official_resume(run/"resume-epoch-1.pt", state["identity"],
                                          dict(charged_seconds=100., completed_updates=2))

"""An authorized time waiver keeps clocks monotonic while disabling duration stops."""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts import train_fusion_study as runner
from scripts.fusion_time_waiver import run_without_deadline, validate_waived_checkpoint
from scripts.run_projector_overnight import Study, command_signature


def test_waived_ledger_keeps_elapsed_work_above_original_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "time_waived", lambda identity=None: True, raising=False)
    clock = {"now": 0.0}
    ledger = runner.FusionLedger(tmp_path, {"gpu_cap_seconds": 10800}, 10790, 13000,
                                lambda: clock["now"], lambda: 1000.0)
    monkeypatch.setattr(ledger, "_write", lambda *a: None)
    clock["now"] = 12000.0
    assert ledger.require_remaining(300, step=13001, phase="train") == 22790.0
    assert ledger.step == 13001


def test_time_waiver_source_mapping_preserves_checkpoint_identity():
    from scripts.fusion_time_waiver import compatible_sources

    value = {"original_runner_sources": {"a.py": "old"},
             "current_sources": {"a.py": "new"}}
    assert compatible_sources({"a.py": "new"}, "runner", record=value) == {"a.py": "old"}


def test_owned_waived_stage_resumes_above_cap_without_refunding(tmp_path):
    previous = [sys.executable, "-c", "pass"]
    command = [*previous, "--resume-checkpoint", str(tmp_path / "resume-12500.ckpt")]
    record = {"status": "failed", "owned_arm": "reference", "elapsed_seconds": 50000.,
              "attempts": 1, "command": previous, "command_hash": command_signature(previous)}
    phase = SimpleNamespace(root=tmp_path, active_arm="reference", env=dict(os.environ),
                            boot_id=Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                            state={"spent_seconds": 50000., "controller_sources": {},
                                   "stages": {"reference-train": record}},
                            save=lambda: None, _sources=lambda: {})
    phase._charge = lambda *args: Study._charge(phase, *args)
    run_without_deadline(phase, "reference-train", command)
    assert record["status"] == "complete" and record["exit_code"] == 0
    assert phase.state["spent_seconds"] > 50000
    assert record["elapsed_seconds"] == phase.state["spent_seconds"]
    assert record["attempts"] == 2
    assert record["previous_attempt_commands"][0]["command"] == previous
    assert record["watchdog_command"][3] == "0"


def test_waived_checkpoint_above_24h_preserves_actual_time_and_replay_checks(tmp_path):
    identity = {"protocol_hash": "frozen"}
    checkpoint = {"pilot_identity": identity, "run_metadata": identity, "global_step": 12500,
                  "optimizer_states": [{"state": {}}], "lr_schedulers": [{}],
                  "pilot_rng": {"state": "rng"}, "pilot_sampler": {"batches_yielded": 12500},
                  "gpu_seconds_cumulative": 90000.}
    path = tmp_path / "resume-12500.ckpt"
    torch.save(checkpoint, path)
    assert validate_waived_checkpoint(path, tmp_path, identity, 91000) == checkpoint
    with pytest.raises(ValueError, match="accounting"):
        validate_waived_checkpoint(path, tmp_path, identity, 89000)
    checkpoint["pilot_sampler"]["batches_yielded"] = 12000
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="sampler"):
        validate_waived_checkpoint(path, tmp_path, identity, 91000)

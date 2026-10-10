"""Forecasts cannot veto a measured, validated extraction smoke."""

from pathlib import Path

import pytest

from tools.motion import extractor as e


def profile(identity):
    return {
        "schema_version": 3,
        "admission_policy": "measured-smoke-hard-cap-v1",
        "featurekey": e.digest(identity),
        "parity": [{"batch_size": n, "bit_equal": True} for n in (1, 8, 16)],
        "samples": [{"windows": 1, "seconds": 99.0}],
        "observation_sha256": {"clip": "a" * 64},
        "projected_remaining_seconds": 999999999,
    }


def test_valid_smoke_is_not_rejected_by_forecast():
    identity = {"test": "identity"}
    assert e.validate_measured_profile(profile(identity), identity) is None


@pytest.mark.parametrize("mutation", ["identity", "parity", "empty"])
def test_measured_admission_still_requires_actual_smoke(mutation):
    identity = {"test": "identity"}
    value = profile(identity)
    if mutation == "identity":
        value["featurekey"] = "b" * 64
    elif mutation == "parity":
        value["parity"][0]["bit_equal"] = False
    else:
        value["samples"] = []
    with pytest.raises(ValueError):
        e.validate_measured_profile(value, identity)


def test_inherited_work_cannot_be_refunded_on_restart(tmp_path: Path):
    clock = e.ExtractionClock(tmp_path / "budget.json", "key", resume=False,
                              inherited_seconds=216.61064642599376)
    clock.save()
    assert clock.spent == 216.61064642599376
    with pytest.raises(ValueError, match="inherited"):
        e.ExtractionClock(tmp_path / "budget.json", "key", resume=True,
                          inherited_seconds=0)


def test_preparation_child_preserves_recovery_binding(tmp_path, monkeypatch):
    recovery = str(tmp_path / "recovery.json")
    launch = {"paths": {}, "recovery": {"path": recovery}}
    request = {"launch": launch, "mode": "profile", "resume": False,
               "adopt_complete": False}
    path = tmp_path / "request.json"
    e.atomic_json(path, request)
    monkeypatch.setattr(e, "_deadline", lambda request: None)

    def bind(args):
        assert args.recovery_record == recovery
        return launch

    monkeypatch.setattr(e, "bootstrap_launch_binding", bind)
    monkeypatch.setattr(e, "preparation_inputs", lambda args: {})
    monkeypatch.setattr(e, "prepare", lambda args: ({}, {"clips": {}}, tmp_path / "cache"))
    e._prepare_entry(str(path))
    assert e.read_json(tmp_path / "prepared.json")["root"] == str(tmp_path / "cache")

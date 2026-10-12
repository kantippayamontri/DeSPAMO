"""A new code version inherits the stopped attempt's actual charged work."""

import json

import pytest

from despamo.appearance.provenance import file_hash


def test_recovery_inherits_maximum_charge_without_touching_old_state(tmp_path):
    from tools.motion.recovery import validate_recovery_record

    old = tmp_path / "motion-fusion-pipeline-v1"
    boot = tmp_path / "motion-extraction-bootstrap-v1"
    old.mkdir()
    boot.mkdir()
    state = {"status": "failed",
             "active": {"bucket": "extraction", "owner": {
                 "pid": 2147483647, "identity": {"start_ticks": "1"}}}, "stages": [],
             "clocks": {"extraction": {"outer_seconds": 216.61,
                                       "inner_bound_seconds": 127.51},
                        "training": {"outer_seconds": 0, "inner_bound_seconds": 0}},
             "identity": {"authorization": {"extraction_cap_seconds": 18000,
                            "training_cap_seconds": 64800, "cap_transfer": False}}}
    budget = {"status": "capped", "active": None, "child": None,
              "spent_seconds": 127.51, "authorization": state["identity"]["authorization"]}
    status = {"error": "TimeoutError: profile projects beyond remaining extraction cap"}
    files = {old / "state.json": state, boot / "budget.json": budget,
             boot / "status.json": status}
    for path, value in files.items():
        path.write_text(json.dumps(value))
    record = old / "measured-profile-recovery.json"
    record.write_text(json.dumps({"schema_version": 1,
        "policy": "measured-profile-recovery-v1",
        "authorization": budget["authorization"],
        "artifacts": {str(p): file_hash(p) for p in files},
        "credit": {"extraction": 216.61, "training": 0}}))
    before = {p: p.read_bytes() for p in files}
    value = validate_recovery_record(record, pilot=tmp_path)
    assert value["credit"] == {"extraction": 216.61, "training": 0}
    assert {p: p.read_bytes() for p in files} == before
    value = json.loads(record.read_text())
    value["credit"]["extraction"] = 127.51
    record.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="credit"):
        validate_recovery_record(record, pilot=tmp_path)


def test_new_pipeline_starts_at_inherited_charge_and_rejects_refund(tmp_path):
    from scripts.run_motion_fusion_pipeline import AUTHORIZATION, POLICY, MotionFusionPipeline

    root = tmp_path / "recovered-pipeline"
    identity = {"policy": POLICY, "authorization": AUTHORIZATION,
                "paths": {"pipeline_root": str(root)}, "sources": {},
                "recovery": {"credit": {"extraction": 216.61, "training": 0}}}
    with MotionFusionPipeline(root, identity) as run:
        assert run.charged("extraction") == 216.61
        assert run.remaining("extraction") == 18000 - 216.61
        run.state["clocks"]["extraction"]["outer_seconds"] = 0
        with pytest.raises(ValueError, match="cumulative"):
            run.validate_state()
        run.state["clocks"]["extraction"]["outer_seconds"] = 216.61

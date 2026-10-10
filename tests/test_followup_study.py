import pytest

from tools.followup.study import recovery_allowed, validate_selections


def test_six_distinct_frozen_selections_required():
    rows = [
        dict(seed=s, arm=a, selection_hash="hash")
        for s in (0, 1)
        for a in ("reference", "aligned_fixed", "aligned_learned")
    ]
    validate_selections(rows)
    with pytest.raises(ValueError):
        validate_selections(rows[:-1] + [rows[0]])


def test_manual_stop_and_live_workers_block_recovery():
    assert not recovery_allowed("failed", "signal15", manual_stop=True, any_live=False)
    assert not recovery_allowed("failed", "signal15", manual_stop=False, any_live=True)
    assert recovery_allowed(
        "failed", "KeyboardInterrupt: signal15", manual_stop=False, any_live=False
    )
    assert not recovery_allowed(
        "failed", "checkpoint hash mismatch", manual_stop=False, any_live=False
    )
    assert not recovery_allowed("complete", "signal15", manual_stop=False, any_live=False)


def test_old_transient_trace_does_not_authorize_new_unknown_failure(tmp_path):
    from tools.followup.study import failure_evidence

    (tmp_path / "active.log").write_text(
        "Traceback (most recent call last):\nKeyboardInterrupt: signal15\n"
        "Traceback (most recent call last):\nValueError: checkpoint hash mismatch\n"
    )
    state = {"error": "stage failed", "stages": {"active": {"status": "failed"}}}
    error = failure_evidence(state, tmp_path)
    assert not recovery_allowed("failed", error, manual_stop=False, any_live=False)

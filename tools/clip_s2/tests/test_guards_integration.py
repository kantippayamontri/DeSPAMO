import pytest

from tools.clip_s2.guards import RecoverableFault, check_dev_report
from tools.clip_s2.study import recovery_allowed


def report(predictions):
    return dict(items=[dict(prediction=p) for p in predictions])


def test_identical_dev_output_after_joint_training_is_recoverable_fault():
    recipe = dict(vt_warmup_steps=4000, steps_per_epoch=887)
    with pytest.raises(RecoverableFault, match="identical"):
        check_dev_report(report(["sss"] * 519), 6, recipe)


def test_identical_output_during_vt_only_warmup_is_allowed():
    recipe = dict(vt_warmup_steps=4000, steps_per_epoch=887)
    check_dev_report(report(["echo"] * 519), 4, recipe)


def test_varied_output_passes():
    recipe = dict(vt_warmup_steps=4000, steps_per_epoch=887)
    check_dev_report(report(["a", "b"] * 200), 6, recipe)


def test_recoverable_fault_permits_bounded_recovery():
    evidence = (
        "Traceback ...\ntools.clip_s2.guards.RecoverableFault: translation CE collapse near uniform"
    )
    assert recovery_allowed("failed", evidence, manual_stop=False, any_live=False)
    assert not recovery_allowed("failed", evidence, manual_stop=True, any_live=False)
    assert not recovery_allowed(
        "failed", "ValueError: checkpoint hash", manual_stop=False, any_live=False
    )

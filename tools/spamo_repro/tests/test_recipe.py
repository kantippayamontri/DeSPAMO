import pytest


def test_paper_lr_warmup_peak_and_nonzero_floor():
    from tools.spamo_repro.recipe import lr_at_update

    assert lr_at_update(0, 17760) == 0
    assert lr_at_update(5000, 17760) == 5e-5
    assert lr_at_update(10000, 17760) == 1e-4
    assert lr_at_update(17760, 17760) == 5e-5
    with pytest.raises(ValueError):
        lr_at_update(0, 9999)


def test_epoch_accumulation_tail_is_framework_flush_not_extra_negatives():
    from tools.spamo_repro.recipe import optimizer_updates_per_epoch

    assert optimizer_updates_per_epoch(7096, 8, 2) == 444
    assert optimizer_updates_per_epoch(16, 8, 2) == 1


def test_exact_VT4000_then_joint4001():
    from tools.spamo_repro.recipe import joint_update

    assert joint_update(4000) is False
    assert joint_update(4001) is True

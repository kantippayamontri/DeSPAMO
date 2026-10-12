import importlib
import json

import pytest


def module():
    return importlib.import_module("tools.dinov3.official_schedule")


def test_epoch_accounting_and_milestones():
    s = module().AdaptSchedule()
    assert s.batch_size == 24 and s.steps_per_epoch == 295
    assert s.total_steps == 11800 and s.calibration_steps == 1475
    assert s.lora_warmup_steps == 590 and s.grl_ramp_steps == 590
    assert s.milestone_steps == (2950, 5900, 8850, 11800)
    assert s.total_steps - s.calibration_steps == 10325


@pytest.mark.parametrize("k,head,lora,grl", [
    (1, 1/2215, 0, 0), (2215, 1, 0, 0),
    (2216, None, 1/886, 1/886), (3101, None, 1, 1), (17720, 0, 0, 1),
])
def test_applied_update_boundaries(k, head, lora, grl):
    m, s = module(), module().AdaptSchedule(batch_size=16)
    if head is not None:
        assert m.head_factor(k, s) == pytest.approx(head, abs=1e-12)
    else:
        assert 0 < m.head_factor(k, s) < 1
    assert m.lora_factor(k, s) == pytest.approx(lora, abs=1e-12)
    assert m.grl_factor(k, s) == pytest.approx(grl, abs=1e-12)


def test_independent_cosines_decrease_after_different_peaks():
    m, s = module(), module().AdaptSchedule(batch_size=16)
    heads = [m.head_factor(k, s) for k in (2215, 3101, 8000, 16000, 17720)]
    lora = [m.lora_factor(k, s) for k in (3101, 8000, 16000, 17720)]
    assert all(a > b for a, b in zip(heads, heads[1:], strict=False))
    assert all(a > b for a, b in zip(lora, lora[1:], strict=False))
    assert m.lora_factor(2215, s) == 0 and m.head_factor(2215, s) == 1


@pytest.mark.parametrize("k", [0, -1, 17721, 2.5, True])
def test_invalid_applied_update_rejected(k):
    m, s = module(), module().AdaptSchedule(batch_size=16)
    for function in (m.head_factor, m.lora_factor, m.grl_factor):
        with pytest.raises(ValueError):
            function(k, s)


@pytest.mark.parametrize("kwargs", [
    dict(train_clips=15), dict(batch_size=0), dict(epochs=5),
    dict(lora_warmup_epochs=35), dict(grl_ramp_epochs=36),
    dict(checkpoint_epochs=11), dict(calibration_epochs=False),
])
def test_invalid_schedule_rejected(kwargs):
    with pytest.raises(ValueError):
        module().AdaptSchedule(**kwargs)


def test_serialized_resume_uses_next_one_based_update():
    m, original = module(), module().AdaptSchedule(batch_size=16)
    for completed in (0, 2215, 3101, 4430, 17720):
        state = json.loads(json.dumps(original.state_dict(completed)))
        restored, next_k = m.AdaptSchedule.restore(state)
        assert restored == original
        assert next_k == (completed + 1 if completed < 17720 else None)
        if next_k is not None:
            assert m.lora_factor(next_k, restored) == m.lora_factor(next_k, original)
    state = original.state_dict(2215)
    state["applied_update"] = 2216
    with pytest.raises(ValueError):
        m.AdaptSchedule.restore(state)


def test_drop_last_then_resume_at_epoch_boundary():
    from tools.dinov3.e3_train import E3BatchSampler

    s = module().AdaptSchedule(batch_size=16)
    sampler = E3BatchSampler(tuple(str(i) for i in range(7096)), seed=0, batch_size=16)
    seen = {i for _ in range(s.steps_per_epoch) for i in sampler.next_ids()}
    assert len(seen) == 7088 and sampler.cursor == 7088
    assert set(sampler.order[7088:]).isdisjoint(seen)
    state = sampler.state_dict()
    expected = sampler.next_ids()
    restored = E3BatchSampler(tuple(str(i) for i in range(7096)), seed=0, batch_size=16)
    restored.load_state_dict(state)
    assert restored.next_ids() == expected
    assert len(expected) == 16 and sampler.cursor == 16


def test_small_cpu_schedule_supported_without_changing_default_recipe():
    m = module()
    small = m.AdaptSchedule(train_clips=32, batch_size=4, epochs=4,
                           calibration_epochs=1, lora_warmup_epochs=1,
                           grl_ramp_epochs=1, checkpoint_epochs=1)
    assert small.total_steps == 32 and small.milestone_steps == (8, 16, 24, 32)
    assert m.lora_factor(9, small) == pytest.approx(1/8)
    assert m.AdaptSchedule().total_steps == 11800


@pytest.mark.parametrize("field,value", [("applied_update", 2215.0), ("next_update", 2216.0)])
def test_restore_rejects_float_counters_even_if_numerically_equal(field, value):
    s = module().AdaptSchedule()
    state = s.state_dict(2215)
    state[field] = value
    with pytest.raises(ValueError):
        module().AdaptSchedule.restore(state)


def test_batch24_applied_clocks_and_replay():
    m, s = module(), module().AdaptSchedule()
    assert m.head_factor(1475, s) == 1 and m.lora_factor(1475, s) == 0
    assert m.lora_factor(1476, s) == pytest.approx(1/590)
    assert m.grl_factor(1476, s) == pytest.approx(1/590)
    assert m.lora_factor(2065, s) == m.grl_factor(2065, s) == 1
    assert m.head_factor(11800, s) == m.lora_factor(11800, s) == 0
    state = json.loads(json.dumps(s.state_dict(1475)))
    restored, next_k = m.AdaptSchedule.restore(state)
    assert restored == s and next_k == 1476


def test_batch24_sampler_drops16_and_replays_next_batch():
    sampler_module = importlib.import_module("tools.dinov3.official_sampler")
    ids = tuple(str(i) for i in range(7096))
    sampler = sampler_module.OfficialBatchSampler(ids, seed=0, batch_size=24)
    seen = {i for _ in range(295) for i in sampler.next_ids()}
    assert len(seen) == 7080 and sampler.cursor == 7080
    assert set(sampler.order[7080:]).isdisjoint(seen)
    state = sampler.state_dict()
    expected = sampler.next_ids()
    restored = sampler_module.OfficialBatchSampler(ids, seed=0, batch_size=24)
    restored.load_state_dict(state)
    assert restored.next_ids() == expected and len(expected) == 24
    incompatible = sampler_module.OfficialBatchSampler(ids, seed=0, batch_size=16)
    with pytest.raises(ValueError):
        incompatible.load_state_dict(state)


def test_production_config_uses_approved_batch24_and_epoch_clocks():
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    config = json.loads((root / "configs/adaptation/phoenix14t_dino_merged40.json").read_text())
    schedule = module().AdaptSchedule(**config["schedule"])
    assert schedule.batch_size == 24 and schedule.total_steps == 11800
    assert schedule.calibration_steps == 1475 and schedule.lora_warmup_steps == 590
    assert schedule.grl_ramp_steps == 590
    assert schedule.milestone_steps == (2950, 5900, 8850, 11800)
    assert config["optimizer"]["lora_lr"] == 1e-5
    assert config["losses"]["relational_weight"] == .1


@pytest.mark.parametrize("field,value", [("schema_version", True), ("cursor", 23),
                                       ("batches_yielded", True), ("rng", ())])
def test_batch24_sampler_rejects_corrupt_replay_state(field, value):
    m = importlib.import_module("tools.dinov3.official_sampler")
    sampler = m.OfficialBatchSampler(tuple(str(i) for i in range(48)), batch_size=24)
    sampler.next_ids()
    state = sampler.state_dict()
    state[field] = value
    with pytest.raises(ValueError):
        sampler.load_state_dict(state)

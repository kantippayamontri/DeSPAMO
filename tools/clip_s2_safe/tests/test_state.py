from pathlib import Path

import pytest
import torch

from tools.benchmark.compact_checkpoint import capture_state
from tools.clip_s2_safe.tests.helpers import original_fixture, tiny
from tools.dinov3.identity import digest


def binding_fixture(tmp_path):
    from tools.clip_s2_safe import state as s
    from tools.clip_s2_safe.policy import verify_original

    c, i, m, o, sampler = original_fixture(tmp_path)
    b = s.make_binding(c, verify_original(c))
    s.publish_migration(c, b)
    return s, c, b, m, o, sampler


def test_anchor_overlay_original_identity_and_complete_restore(tmp_path):
    s, c, b, m, o, sampler = binding_fixture(tmp_path)
    fresh, other, new_sampler = tiny()
    before = Path(c["anchor"]).read_bytes()
    loaded = s.load_weights(Path(c["anchor"]), fresh, b, base_hash="frozen")
    s.restore_training(loaded, fresh, other, new_sampler, device=torch.device("cpu"))
    assert (
        new_sampler.state_dict() == sampler.state_dict()
        and fresh.rng.getstate() == m.rng.getstate()
    )
    for k, v in m.state_dict().items():
        assert torch.equal(v, fresh.state_dict()[k])
    for k, v in o.state_dict()["state"].items():
        for name, value in v.items():
            assert torch.equal(value, other.state_dict()["state"][k][name])
    assert Path(c["anchor"]).read_bytes() == before


def test_new_snapshot_needs_both_sidecars_and_carries_old_identity(tmp_path):
    s, c, b, m, o, sampler = binding_fixture(tmp_path)
    sampler.next_ids()
    snap = capture_state(m, o, b["original_identity"], 3, sampler.state_dict(), base_hash="frozen")
    saved = s.publish_checkpoint(c, snap, dict(last_update=3, ce_values=[]), mode="production")
    p = Path(saved["checkpoint"])
    loaded = s.load_weights(p, m, b, base_hash="frozen")
    assert loaded["identity"] == b["original_identity"]
    p.with_suffix(".runtime.json").unlink()
    with pytest.raises(ValueError):
        s.load_weights(p, m, b, base_hash="frozen")
    assert p.exists()


@pytest.mark.parametrize("kind", ["checkpoint", "binding", "step", "history"])
def test_guard_state_rejects_drift(tmp_path, kind):
    s, c, b, *_ = binding_fixture(tmp_path)
    state = dict(
        schema_version=1,
        checkpoint_sha256="cp",
        binding_hash=digest(b),
        last_update=6,
        ce_values=[1.0, 2.0],
    )
    s.validate_guard_state(state, checkpoint_sha256="cp", binding=b, step=6)
    field, value = dict(
        checkpoint=("checkpoint_sha256", "bad"),
        binding=("binding_hash", "bad"),
        step=("last_update", 5),
        history=("ce_values", []),
    )[kind]
    state[field] = value
    with pytest.raises(ValueError):
        s.validate_guard_state(state, checkpoint_sha256="cp", binding=b, step=6)


def test_weights_only_never_restore_global_rng_or_optimizer(tmp_path, monkeypatch):
    s, c, b, m, *_ = binding_fixture(tmp_path)

    def forbidden(*a, **k):
        pytest.fail("restore on evaluator")

    monkeypatch.setattr(s, "restore_rng", forbidden)
    state = s.load_weights(Path(c["anchor"]), m, b, base_hash="frozen")
    assert "optimizer" in state  # read on CPU, not restored or copied to GPU


def test_migration_is_immutable_and_keeps_original_budget_copy(tmp_path):
    s, c, b, *_ = binding_fixture(tmp_path)
    original = (Path(c["run_root"]) / "budget.json").read_bytes()
    assert (Path(c["runtime_root"]) / "original-budget.json").read_bytes() == original
    assert s.publish_migration(c, b) == b
    changed = {**b, "evaluation_policy_hash": "bad"}
    with pytest.raises(ValueError):
        s.publish_migration(c, changed)


def test_anchor_copy_elsewhere_not_exempt(tmp_path):
    s, c, b, m, *_ = binding_fixture(tmp_path)
    p = Path(c["runtime_root"]) / "copy.pt"
    p.write_bytes(Path(c["anchor"]).read_bytes())
    p.with_suffix(".json").write_bytes(Path(c["anchor"]).with_suffix(".json").read_bytes())
    with pytest.raises(ValueError):
        s.load_weights(p, m, b, base_hash="frozen")


def test_native_cuda_rng_cannot_be_restored_on_cpu(tmp_path):
    s, c, b, m, o, sampler = binding_fixture(tmp_path)
    state = s.load_weights(Path(c["anchor"]), m, b, base_hash="frozen")
    state["rng"]["cuda"] = [torch.zeros(20, dtype=torch.uint8)]
    with pytest.raises(RuntimeError, match="CUDA RNG"):
        s.restore_training(state, m, o, sampler, device=torch.device("cpu"))

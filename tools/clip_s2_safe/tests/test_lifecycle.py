import json
import os
import sys
from pathlib import Path

import torch

from tools.clip_s2_safe.tests.test_train import setup as training_setup
from tools.dinov3.identity import digest
from tools.dinov3.storage import atomic_json


def test_resumed_cpu_updates_match_continuous_rng_optimizer_sampler_guard(tmp_path, monkeypatch):
    from tools.clip_s2_safe import state

    t, c, b, m, o, s, g, j = training_setup(tmp_path, monkeypatch)
    first = t.train_chunk(
        m, o, s, {}, None, b, g, j, device=torch.device("cpu"), on_progress=lambda e: None
    )
    saved = state.load_weights(Path(first["checkpoint"]), m, b, base_hash="frozen")
    j.update(start_step=4, target_step=6)
    # Continuous path; store reference final state independently before fresh construction.
    t.train_chunk(
        m,
        o,
        s,
        {},
        None,
        b,
        saved["safe_guard"],
        j,
        device=torch.device("cpu"),
        on_progress=lambda e: None,
    )
    expected = {k: v.clone() for k, v in m.state_dict().items()}
    expected_prompt = m.rng.getstate()
    expected_sampler = s.state_dict()
    from tools.clip_s2_safe.tests.helpers import tiny

    fresh, other, fs = tiny()
    fresh.load_state_dict(saved["weights"], strict=False)
    state.restore_training(saved, fresh, other, fs, device=torch.device("cpu"))
    # Capture publications separately without touching original identity or source files.
    records = []
    monkeypatch.setattr(
        t,
        "publish_checkpoint",
        lambda c, snapshot, guard, **kwargs: records.append((snapshot, guard)) or {},
    )
    t.train_chunk(
        fresh,
        other,
        fs,
        {},
        None,
        b,
        saved["safe_guard"],
        j,
        device=torch.device("cpu"),
        on_progress=lambda e: None,
    )
    for k, v in expected.items():
        assert torch.equal(v, fresh.state_dict()[k])
    assert fresh.rng.getstate() == expected_prompt and fs.state_dict() == expected_sampler
    for k, v in o.state_dict()["state"].items():
        for name, value in v.items():
            assert torch.equal(value, other.state_dict()["state"][k][name])
    assert records[-1][1]["ce_values"] == [0.1, 0.1]


def test_original_anchor_native_schema_audit_is_cpu_only(tmp_path):
    from tools.clip_s2_safe import state
    from tools.clip_s2_safe.tests.helpers import original_fixture

    c, *_ = original_fixture(tmp_path)
    result = state.audit_native_schema(c)
    assert result["step"] == 2 and result["sampler_batches_yielded"] == 2
    assert result["weights_count"] > 1 and result["optimizer_state_count"] > 0
    assert result["cuda_rng_restored"] is False


def test_real_file_backed_same_root_workers(tmp_path, monkeypatch):
    from tools.clip_s2_safe import accounting, policy, state, study
    from tools.clip_s2_safe.tests.cpu_child import lifecycle_fixture, load_inputs
    from tools.dinov3.official_runtime import process_owner

    c = lifecycle_fixture(tmp_path)
    original = policy.verify_original(c)
    b = state.make_binding(c, original)
    state.publish_migration(c, b)
    original_files = [
        Path(c["anchor"]),
        Path(c["anchor"]).with_suffix(".json"),
        Path(c["study_root"]) / "manual-stop.json",
    ]
    before = {p: p.read_bytes() for p in original_files}
    config_path = tmp_path / "cpu-config.json"
    atomic_json(config_path, c)
    scope = dict(
        approved=True,
        mode="production",
        token="real-cpu",
        binding_hash=digest(b),
        max_train_updates=8,
        max_dev_clips=9,
    )
    attempt = accounting.begin_attempt(c, b, scope, process_owner(os.getpid()))
    monkeypatch.setattr(study, "require_scope", lambda *a, **k: scope)
    monkeypatch.setattr(study, "admit_original", lambda *a: load_inputs(c))
    monkeypatch.setattr(
        study,
        "worker_command",
        lambda c, j, a: [
            sys.executable,
            "-B",
            "-m",
            "tools.clip_s2_safe.tests.cpu_child",
            j["stage"],
            str(config_path),
            j["ticket_path"],
        ],
    )
    a = load_inputs(c)
    steps = []
    for stage, target in [("dev", 4), ("train", 8), ("dev", 8), ("train", 12), ("dev", 12)]:
        pending = study.pending_stage(c, b, a)
        assert (pending["stage"], pending["target_step"]) == (stage, target)
        job = study._job(b, pending, "production")
        result = study.execute_job(c, b, attempt, job, tmp_path / "fake-approval")
        steps.append((stage, result["completed_updates"]))
    assert study.pending_stage(c, b, a)["stage"] == "complete"
    assert steps == [("dev", 4), ("train", 8), ("dev", 8), ("train", 12), ("dev", 12)]
    for p, blob in before.items():
        assert p.read_bytes() == blob
    budget = json.loads((Path(c["run_root"]) / "budget.json").read_text())
    assert budget["completed_updates"] == 12 and budget["identity"] == b["original_identity"]

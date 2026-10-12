import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tools.clip_s2_safe.tests.test_state import binding_fixture
from tools.dinov3.identity import digest, sha256_file
from tools.dinov3.storage import atomic_json


def test_failure_progress_keeps_newly_saved_durable_checkpoint(tmp_path, monkeypatch):
    from tools.clip_s2_safe import train as t
    from tools.clip_s2_safe.policy import boot_id

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    scope = dict(approved=True, boot_id=boot_id(), max_train_updates=4)
    monkeypatch.setattr(t, "require_scope", lambda *a, **k: scope)
    for name in [
        "require_exclusive_gpu",
        "verify_original",
        "require_storage",
        "require_prior_dev",
    ]:
        monkeypatch.setattr(t, name, lambda *a, **k: None)
    rows = [
        dict(fileid=f"train{i}", text=f"german{i}", en_text="en", fr_text="fr", es_text="es")
        for i in range(8)
    ]
    a = dict(
        ids={"train": tuple(r["fileid"] for r in rows)},
        datasets={
            "train": SimpleNamespace(records=rows),
            "dev": SimpleNamespace(records=[{"text": "dev"}]),
            "test": SimpleNamespace(records=[{"text": "test"}]),
        },
    )
    monkeypatch.setattr(t, "admit_original", lambda *args: a)
    monkeypatch.setattr(t, "build_model", lambda *args: (m, "frozen"))
    monkeypatch.setattr(t, "optimizer_for", lambda *args: o)
    monkeypatch.setattr(m, "to", lambda *a, **k: m)
    error = RuntimeError("primary")

    def chunk(*args, **kwargs):
        kwargs["on_progress"](dict(completed_updates=3, durable_updates=3))
        raise error

    monkeypatch.setattr(t, "train_chunk", chunk)
    job = dict(
        mode="production",
        start_step=2,
        target_step=4,
        checkpoint=c["anchor"],
        checkpoint_sha256=c["anchor_sha256"],
        progress_path=str(tmp_path / "progress.json"),
    )
    with pytest.raises(RuntimeError) as got:
        t.run_train_job(c, job, tmp_path / "approval")
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert got.value is error and progress["completed_updates"] == progress["durable_updates"] == 3


def test_runtime_mode_matches_directory_and_partial_adoption_required(tmp_path):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_s2_safe import state as s

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    sampler.next_ids()
    p = s.publish_checkpoint(
        c,
        capture_state(m, o, b["original_identity"], 3, sampler.state_dict(), base_hash="frozen"),
        dict(last_update=3, ce_values=[]),
        mode="handoff_probe",
    )
    path = Path(p["checkpoint"])
    side = path.with_suffix(".runtime.json")
    r = json.loads(side.read_text())
    r["mode"] = "production"
    atomic_json(side, r)
    with pytest.raises(ValueError):
        s.checkpoint_bundle(path, b)


def test_promotion_rejects_nonanchor_progress_before_any_copy(tmp_path, monkeypatch):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_s2_safe import accounting as a
    from tools.clip_s2_safe.state import publish_checkpoint

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    sampler.next_ids()
    p = publish_checkpoint(
        c,
        capture_state(m, o, b["original_identity"], 3, sampler.state_dict(), base_hash="frozen"),
        dict(last_update=3, ce_values=[]),
        mode="handoff_probe",
    )
    approval = tmp_path / "adopt.json"
    atomic_json(approval, dict(approved=True))
    monkeypatch.setattr(
        a,
        "require_scope",
        lambda *a, **k: dict(
            approve_adoption=True, probe_checkpoint_sha256=p["checkpoint_sha256"], token="adopt"
        ),
    )
    budget = Path(c["run_root"]) / "budget.json"
    r = json.loads(budget.read_text())
    r["completed_updates"] = 3
    atomic_json(budget, r)
    report = tmp_path / "report.json"
    atomic_json(report, {})
    atomic_json(
        Path(c["runtime_root"]) / "production-anchor-dev-gate.json",
        dict(binding_hash=digest(b), report=str(report), report_sha256=sha256_file(report)),
    )
    with pytest.raises((ValueError, RuntimeError)):
        a.adopt_probe(c, b, p, approval)
    assert not (Path(c["runtime_root"]) / "checkpoints/production").exists()


def test_probe_adoption_needs_completed_handoff_evidence(tmp_path, monkeypatch):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_s2_safe import accounting as a
    from tools.clip_s2_safe.state import publish_checkpoint

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    sampler.next_ids()
    p = publish_checkpoint(
        c,
        capture_state(m, o, b["original_identity"], 3, sampler.state_dict(), base_hash="frozen"),
        dict(last_update=3, ce_values=[]),
        mode="handoff_probe",
    )
    approval = tmp_path / "adopt.json"
    atomic_json(approval, dict(approved=True))
    monkeypatch.setattr(
        a,
        "require_scope",
        lambda *a, **k: dict(
            approve_adoption=True, probe_checkpoint_sha256=p["checkpoint_sha256"], token="adopt"
        ),
    )
    report = tmp_path / "report.json"
    atomic_json(report, {})
    atomic_json(
        Path(c["runtime_root"]) / "production-anchor-dev-gate.json",
        dict(binding_hash=digest(b), report=str(report), report_sha256=sha256_file(report)),
    )
    with pytest.raises(RuntimeError, match="handoff"):
        a.adopt_probe(c, b, p, approval)


def test_successful_explicit_adoption_charges_probe_once(tmp_path, monkeypatch):
    from tools.clip_s2_safe import accounting, evaluate, policy, state, study, train
    from tools.clip_s2_safe.tests.cpu_child import lifecycle_fixture, load_inputs, model
    from tools.dinov3.official_sampler import OfficialBatchSampler
    from tools.followup.recipe import PromptPool

    c = lifecycle_fixture(tmp_path)
    b = state.make_binding(c, policy.verify_original(c))
    state.publish_migration(c, b)
    inputs = load_inputs(c)
    m = model()
    cp = state.make_anchor_candidate(c, b)
    state.load_weights(cp, m, b, base_hash="frozen")
    pool = PromptPool(inputs["datasets"]["train"].records, 0)
    job = dict(
        stage="dev",
        seed=0,
        mode="production",
        epoch=2,
        start_step=4,
        target_step=4,
        checkpoint=str(cp),
        checkpoint_sha256=sha256_file(cp),
        binding_hash=digest(b),
        max_batches=None,
    )
    report = evaluate.score_dev(m, inputs, pool, job, b, on_event=lambda e: None)
    atomic_json(Path(c["runtime_root"]) / "dev/epoch-2.json", report)
    monkeypatch.setattr(study, "archive_candidate", lambda *a: {})
    study.select_checkpoint(c, b, inputs)
    # Real CPU training update and provisional ledger from the original anchor.
    m = model()
    loaded = state.load_weights(Path(c["anchor"]), m, b, base_hash="frozen")
    opt = torch.optim.AdamW(m.parameters(), lr=0.001)
    sampler = OfficialBatchSampler(inputs["ids"]["train"], seed=0, batch_size=8)
    state.restore_training(loaded, m, opt, sampler, device=torch.device("cpu"))
    now = [100.0]
    monkeypatch.setattr(accounting.time, "time", lambda: now[0])
    owner = dict(pid=99999999, boot_id="old", start_ticks="0", command=["dead"])
    scope = dict(
        approved=True,
        token="probe",
        mode="handoff_probe",
        binding_hash=digest(b),
        max_train_updates=1,
        max_dev_clips=1,
    )
    attempt = accounting.begin_attempt(c, b, scope, owner)
    tj = dict(
        mode="handoff_probe",
        stage="train",
        seed=0,
        stage_id="train4-5",
        start_step=4,
        target_step=5,
        binding_hash=digest(b),
    )
    accounting.begin_stage(c, attempt, tj)

    def update(model, opt, batch, k, r, pool):
        opt.zero_grad()
        loss = model.language_model.model.encoder.embed_tokens.weight.square().mean()
        loss.backward()
        opt.step()
        return dict(translation_loss=float(loss.detach()))

    monkeypatch.setattr(train, "update", update)
    monkeypatch.setattr(train, "check_guard", lambda *a: None)
    probe = train.train_chunk(
        m,
        opt,
        sampler,
        inputs,
        pool,
        b,
        loaded["safe_guard"],
        tj,
        device=torch.device("cpu"),
        on_progress=lambda e: None,
    )
    now[0] = 101.0
    accounting.finish_stage(c, attempt, tj, probe, status="complete")
    dj = {
        **job,
        "mode": "handoff_probe",
        "epoch": None,
        "stage_id": "dev5",
        "start_step": 5,
        "target_step": 5,
        "checkpoint": probe["candidate"],
        "checkpoint_sha256": probe["candidate_sha256"],
        "max_batches": 1,
    }
    accounting.begin_stage(c, attempt, dj)
    d = evaluate.score_dev(m, inputs, pool, dj, b, on_event=lambda e: None)
    path = Path(c["runtime_root"]) / "validation/probe/dev.json"
    atomic_json(path, d)
    now[0] = 102.0
    accounting.finish_stage(
        c,
        attempt,
        dj,
        dict(completed_updates=5, durable_updates=5, completed_clips=1),
        status="complete",
    )
    accounting.finish_attempt(c, attempt, status="complete")
    handoff = path.parent / "handoff.json"
    atomic_json(
        handoff,
        dict(
            status="completed",
            binding_hash=digest(b),
            checkpoint_sha256=probe["checkpoint_sha256"],
            end_step=5,
            token="probe",
            dev_report=str(path),
            dev_report_sha256=sha256_file(path),
        ),
    )
    probe.update(handoff=str(handoff), handoff_sha256=sha256_file(handoff))
    approval = tmp_path / "adopt.json"
    atomic_json(approval, dict(approved=True))
    monkeypatch.setattr(
        accounting,
        "require_scope",
        lambda *a, **k: dict(
            approve_adoption=True,
            probe_checkpoint_sha256=probe["checkpoint_sha256"],
            handoff_sha256=probe["handoff_sha256"],
            token="adopt",
        ),
    )
    monkeypatch.setattr(evaluate, "admit_original", lambda *a: inputs)
    before = json.loads((Path(c["run_root"]) / "budget.json").read_text())
    adopted = accounting.adopt_probe(c, b, probe, approval)
    after = json.loads((Path(c["run_root"]) / "budget.json").read_text())
    assert before["completed_updates"] == 4 and after["completed_updates"] == 5
    assert before["charged_seconds"] == after["charged_seconds"] == 499.5533489369991
    state.checkpoint_bundle(Path(adopted["checkpoint"]), b)


def test_bool_seed_scope_rejected(tmp_path, monkeypatch):
    from tools.clip_s2_safe.tests.test_policy import authorization

    p, c, r, path = authorization(tmp_path, monkeypatch)
    r["seed"] = False
    atomic_json(path, r)
    with pytest.raises(RuntimeError):
        p.require_scope(c, "train", path)


def test_empty_probe_result_cannot_pass_handoff_admission(tmp_path, monkeypatch):
    from tools.clip_s2_safe import study as s
    from tools.clip_s2_safe.evaluate import score_dev
    from tools.clip_s2_safe.tests.test_evaluate import setup

    c, b, m, a, p, job = setup(tmp_path, monkeypatch)
    job.update(mode="handoff_probe", max_batches=1)
    report = score_dev(m, a, p, job, b, on_event=lambda e: None)
    report["items"] = []
    path = Path(c["runtime_root"]) / "validation/dev.json"
    atomic_json(path, report)
    result = dict(
        status="complete",
        stage="dev",
        mode="handoff_probe",
        completed_updates=2,
        durable_updates=2,
        completed_clips=0,
        report=str(path),
        report_sha256=sha256_file(path),
    )
    monkeypatch.setattr(s, "admit_original", lambda *a0: a)
    with pytest.raises(ValueError):
        s._validate_result(c, b, job, result)


@pytest.mark.parametrize("bad", ["shape", "nan", "negative_variance"])
def test_bad_optimizer_state_rejected_before_restore(tmp_path, bad):
    from tools.clip_s2_safe import state as s

    _, c, b, m, o, sampler = binding_fixture(tmp_path)
    loaded = s.load_weights(Path(c["anchor"]), m, b, base_hash="frozen")
    first = next(iter(loaded["optimizer"]["state"].values()))
    if bad == "shape":
        first["exp_avg"] = torch.zeros(100)
    elif bad == "nan":
        first["exp_avg"].fill_(float("nan"))
    else:
        first["exp_avg_sq"].fill_(-1.0)
    before = m.rng.getstate()
    with pytest.raises(ValueError):
        s.restore_training(loaded, m, o, sampler, device=torch.device("cpu"))
    assert m.rng.getstate() == before

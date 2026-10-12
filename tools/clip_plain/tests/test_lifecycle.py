import json
from pathlib import Path

import torch

from tools.clip_plain.tests.test_evaluate import fixture
from tools.clip_plain.tests.test_train_chunk import setup_loop


def test_interrupted_training_rng_sampler_optimizer_equal(tmp_path, monkeypatch):
    from tools.clip_plain import state

    t, a, oa, sa, i = setup_loop(tmp_path / "continuous", monkeypatch)
    t.train_to_boundary(
        a,
        oa,
        sa,
        {},
        None,
        i,
        tmp_path / "continuous",
        2,
        "base",
        torch.device("cpu"),
        lambda e: None,
    )
    saved = state.load_weights(tmp_path / "continuous/resume-epoch-2.pt", a, i, "base")
    i["recipe"]["dev_epochs"] = [2, 4]
    # Identity remains the same; same model continues after isolated dev.
    t.train_to_boundary(
        a,
        oa,
        sa,
        {},
        None,
        i,
        tmp_path / "continuous",
        4,
        "base",
        torch.device("cpu"),
        lambda e: None,
    )
    t, b, ob, sb, _ = setup_loop(tmp_path / "resumed", monkeypatch)
    b.load_state_dict(saved["weights"], strict=False)
    state.restore_training(saved, b, ob, sb)
    t.train_to_boundary(
        b, ob, sb, {}, None, i, tmp_path / "resumed", 4, "base", torch.device("cpu"), lambda e: None
    )
    assert sa.batches_yielded == sb.batches_yielded == 4
    assert a.rng.getstate() == b.rng.getstate()
    for k, v in a.state_dict().items():
        assert torch.equal(v, b.state_dict()[k])
    for k, v in oa.state_dict()["state"].items():
        for name, value in v.items():
            assert torch.equal(value, ob.state_dict()["state"][k][name])


def test_recover_missing_handoff_from_verified_checkpoint(tmp_path, monkeypatch):
    from tools.clip_plain import study

    t, m, o, s, i = setup_loop(tmp_path, monkeypatch)
    t.train_to_boundary(
        m, o, s, {}, None, i, tmp_path, 2, "base", torch.device("cpu"), lambda e: None
    )
    (tmp_path / "candidate-epoch-2.pt").unlink()
    (tmp_path / "candidate-epoch-2.json").unlink()
    # Explicit reviewed CPU repair must use preserved resume, not train again.
    result = study.recover_candidate(tmp_path, i, 2, reviewed=True)
    assert result["step"] == 2 and Path(result["candidate"]).exists()
    assert study.pending_stage(tmp_path, i, {})["stage"] == "dev"


def test_verified_dev_report_reconciliation_does_not_generate_again(tmp_path, monkeypatch):
    from tools.benchmark.compact_checkpoint import capture_state
    from tools.clip_plain import study
    from tools.clip_plain.evaluate import score_dev
    from tools.clip_plain.policy import POLICY, evaluation_policy
    from tools.clip_plain.state import publish_epoch
    from tools.dinov3.identity import digest
    from tools.dinov3.storage import atomic_json

    m, a, p, j = fixture(tmp_path, monkeypatch)
    i = dict(
        policy=POLICY,
        seed=0,
        recipe=dict(
            steps_per_epoch=1,
            dev_epochs=[2, 4],
            vt_warmup_steps=4000,
            min_delta=0.1,
            min_epochs=40,
            patience=10,
            epochs=200,
        ),
        evaluation_policy_hash=digest(evaluation_policy()),
        data=a["identity"],
    )
    import random

    m.rng = random.Random(0)
    opt = torch.optim.AdamW(m.parameters())
    run = tmp_path / "run"
    pub = publish_epoch(run, capture_state(m, opt, i, 2, {}, base_hash="f"), 2, dev_candidate=True)
    j.update(
        checkpoint_sha256=pub["candidate_sha256"],
        run_identity_hash=digest(i),
        data_identity_hash=digest(a["identity"]),
    )
    r = score_dev(m, a, p, j, on_event=lambda e: None)
    r["checkpoint_path"] = pub["candidate"]
    atomic_json(run / "dev/dev-epoch-2.json", r)
    assert study.pending_stage(run, i, a)["stage"] == "train"


def test_first_worker_identity_binds_existing_charged_ledger(tmp_path, monkeypatch):
    from tools.clip_plain import state, study
    from tools.dinov3.storage import atomic_json

    run = tmp_path / "run"
    initial = dict(policy="uninitialized", seed=0)
    state.begin_stage(run, initial, dict(token="a", stage="train"))
    actual = dict(policy="official-clip-global224-safe-dev-spamo200-v1", seed=0)
    atomic_json(run / "identity.json", actual)
    before = json.loads((run / "budget.json").read_text())["charged_seconds"]
    study.bind_initialized_budget(run)
    b = json.loads((run / "budget.json").read_text())
    assert b["identity"] == actual and b["charged_seconds"] == before


def test_file_backed_train_dev_train_dev_in_fresh_processes(tmp_path, monkeypatch):
    import subprocess
    import sys

    from tools.clip_plain import study
    from tools.clip_plain.tests.test_data import admission_fixture
    from tools.dinov3.identity import digest
    from tools.dinov3.storage import atomic_json

    c, a = admission_fixture(tmp_path, monkeypatch)
    c["recipe"].update(
        steps_per_epoch=1, total_steps=4, epochs=4, dev_every=2, min_epochs=4, vt_warmup_steps=0
    )
    run = Path(c["study_root"]) / "runs/seed0"
    config_path = tmp_path / "cpu-config.json"
    atomic_json(config_path, c)
    init = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "tools.clip_plain.tests.cpu_child",
            "initialize",
            str(config_path),
            str(run),
        ],
        capture_output=True,
        text=True,
    )
    assert init.returncode == 0, init.stderr
    i = json.loads((run / "identity.json").read_text())
    monkeypatch.setattr(study, "require_gpu_scope", lambda *a, **k: {})
    monkeypatch.setattr(
        study,
        "worker_command",
        lambda config, job, approval, path: [
            sys.executable,
            "-B",
            "-m",
            "tools.clip_plain.tests.cpu_child",
            job["stage"],
            str(config_path),
            str(path),
        ],
    )
    trace = []
    for stage, epoch in [("train", 2), ("dev", 2), ("train", 4), ("dev", 4)]:
        pending = study.pending_stage(run, i, a)
        assert (pending["stage"], pending["epoch"]) == (stage, epoch)
        directory = run / "jobs" / f"{stage}{epoch}"
        job = dict(
            seed=0,
            run_root=str(run),
            attempt_token=f"{stage}{epoch}",
            result_path=str(directory / "result.json"),
            progress_path=str(directory / "progress.json"),
            **pending,
        )
        if stage == "dev":
            job.update(
                split="dev",
                run_identity_hash=digest(i),
                evaluation_policy_hash=i["evaluation_policy_hash"],
                data_identity_hash=digest(a["identity"]),
                max_batches=None,
            )
        r = study.execute_job(c, job, tmp_path / "approval")
        trace.append((stage, r["completed_updates"]))
        if stage == "dev":
            from tools.clip_plain.evaluate import validate_report

            validate_report(json.loads(Path(r["report"]).read_text()), job, a)
    assert trace == [("train", 2), ("dev", 2), ("train", 4), ("dev", 4)]
    assert study.pending_stage(run, i, a)["stage"] == "complete"

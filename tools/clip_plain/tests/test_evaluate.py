import pytest
from torch import nn

from tools.clip_plain.tests.test_data import admission_fixture
from tools.clip_s2_inference.tests.helpers import language
from tools.followup.recipe import PromptPool


def fixture(tmp_path, monkeypatch):
    from tools.clip_plain.policy import evaluation_policy
    from tools.dinov3.identity import digest

    c, a = admission_fixture(tmp_path, monkeypatch)

    class Visual(nn.Module):
        def forward(self, s, sm, m, mm):
            return s[:, :, :32], sm

    model = nn.Module()
    model.visual_adapter = Visual()
    model.language_model = language()
    pool = PromptPool(a["datasets"]["train"].records, 0)
    job = dict(
        stage="dev",
        split="dev",
        epoch=2,
        checkpoint_sha256="cp",
        run_identity_hash="run",
        evaluation_policy_hash=digest(evaluation_policy()),
        data_identity_hash=digest(a["identity"]),
        max_batches=None,
    )
    return model, a, pool, job


def test_full_dev_complete_order_and_modes(tmp_path, monkeypatch):
    from tools.clip_plain.evaluate import score_dev, validate_report

    m, a, p, j = fixture(tmp_path, monkeypatch)
    flags = [x.training for x in m.modules()]
    events = []
    r = score_dev(m, a, p, j, on_event=events.append)
    validate_report(r, j, a)
    assert r["complete"] and not r["diagnostic_only"] and r["metrics"] is not None
    assert [x["clip_id"] for x in r["items"]] == list(a["ids"]["dev"])
    assert [x.training for x in m.modules()] == flags
    assert sum(e["phase"] == "generation_start" for e in events) == 3


@pytest.mark.parametrize("limit", [1, 3, 8])
def test_bounded_even_full_corpus_not_scored(tmp_path, monkeypatch, limit):
    from tools.clip_plain.evaluate import score_dev, validate_report

    m, a, p, j = fixture(tmp_path, monkeypatch)
    j["max_batches"] = limit
    r = score_dev(m, a, p, j, on_event=lambda e: None)
    assert not r["complete"] and r["diagnostic_only"] and r["metrics"] is None
    with pytest.raises(ValueError):
        validate_report(r, j, a)


@pytest.mark.parametrize("wrong", ["order", "policy", "checkpoint", "split"])
def test_report_rejects_wrong_binding(tmp_path, monkeypatch, wrong):
    from tools.clip_plain.evaluate import score_dev, validate_report

    m, a, p, j = fixture(tmp_path, monkeypatch)
    r = score_dev(m, a, p, j, on_event=lambda e: None)
    if wrong == "order":
        r["items"].reverse()
    elif wrong == "policy":
        r["evaluation_policy_hash"] = "bad"
    elif wrong == "checkpoint":
        r["checkpoint_sha256"] = "bad"
    else:
        r["split"] = "test"
    with pytest.raises(ValueError):
        validate_report(r, j, a)


def test_reference_overlap_before_generation(tmp_path, monkeypatch):
    from tools.clip_plain.evaluate import score_dev

    m, a, p, j = fixture(tmp_path, monkeypatch)
    p.by_id[p.eval_id]["text"] = a["datasets"]["dev"].records[0]["text"]
    events = []
    with pytest.raises(ValueError):
        score_dev(m, a, p, j, on_event=events.append)
    assert not events


def test_disk_full_before_generation_and_primary_after_failure(tmp_path, monkeypatch):
    from tools.clip_plain import evaluate as e

    m, a, p, j = fixture(tmp_path, monkeypatch)
    calls = []

    def generate(*a, **k):
        calls.append(1)
        raise RuntimeError("primary")

    monkeypatch.setattr(e, "generate_cached_text", generate)

    def bad(event):
        raise OSError("disk full")

    with pytest.raises(OSError):
        e.score_dev(m, a, p, j, on_event=bad)
    assert not calls

    def error_logger(event):
        if event["phase"] == "batch_error":
            raise OSError("logger")

    with pytest.raises(RuntimeError, match="primary"):
        e.score_dev(m, a, p, j, on_event=error_logger)
    assert len(calls) == 1


def test_worker_rejects_gpu_before_optimizer_or_model(tmp_path, monkeypatch):
    from tools.clip_plain.evaluate import run_dev_job
    from tools.clip_plain.tests.helpers import config_at

    with pytest.raises(RuntimeError, match="approval"):
        run_dev_job(config_at(tmp_path), dict(stage="dev"), None)


def test_report_rejects_missing_token_evidence_and_wrong_exemplar(tmp_path, monkeypatch):
    from tools.clip_plain.evaluate import score_dev, validate_report

    m, a, p, j = fixture(tmp_path, monkeypatch)
    r = score_dev(m, a, p, j, on_event=lambda e: None)
    r["items"][0]["exemplar_ids"] = ["dev0"]
    with pytest.raises(ValueError):
        validate_report(r, j, a)


def test_report_metric_tampering_rejected(tmp_path, monkeypatch):
    from tools.clip_plain.evaluate import score_dev, validate_report

    m, a, p, j = fixture(tmp_path, monkeypatch)
    r = score_dev(m, a, p, j, on_event=lambda e: None)
    r["metrics"]["bleu4"] += 1
    with pytest.raises(ValueError):
        validate_report(r, j, a)


def test_post_warmup_identical_output_is_fail_stop():
    from tools.clip_s2.guards import check_dev_report
    from tools.followup.recipe import schedule

    report = dict(items=[dict(prediction="same")] * 519)
    check_dev_report(report, 2, schedule(8))
    with pytest.raises(RuntimeError):
        check_dev_report(report, 6, schedule(8))

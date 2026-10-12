import pytest

from tools.clip_s2_inference.tests.test_evaluation import fixture as input_fixture
from tools.clip_s2_safe.tests.test_state import binding_fixture
from tools.dinov3.identity import digest


def setup(tmp_path, monkeypatch):
    _, c, b, *_ = binding_fixture(tmp_path / "original")
    (tmp_path / "inputs").mkdir()
    model, a, pool = input_fixture(tmp_path / "inputs")
    a["identity"] = dict(bindings={}, ordered_ids=dict(dev=digest(a["ids"]["dev"])))
    b["original_identity"]["data"] = a["identity"]
    b["original_identity"]["decoder"]["exemplar_id"] = pool.eval_id
    b["original_identity_hash"] = digest(b["original_identity"])
    job = dict(
        mode="production",
        stage="dev",
        seed=0,
        epoch=2,
        start_step=2,
        target_step=2,
        checkpoint="candidate.pt",
        checkpoint_sha256="cp",
        binding_hash=digest(b),
        max_batches=None,
    )
    return c, b, model, a, pool, job


def test_full_report_explicit_s2_policy_and_original_identity(tmp_path, monkeypatch):
    from tools.clip_s2_safe.evaluate import score_dev, validate_report

    c, b, m, a, p, j = setup(tmp_path, monkeypatch)
    events = []
    r = score_dev(m, a, p, j, b, on_event=events.append)
    validate_report(r, j, b, a)
    assert r["complete"] and not r["diagnostic_only"] and r["metrics"] is not None
    assert r["evaluation_policy"]["id"] == "clip-s2-cached-b1-beam5-v1"
    assert r["original_identity_hash"] == b["original_identity_hash"]
    assert all(e["decoding"]["use_cache"] for e in events if e["phase"] == "generation_start")


@pytest.mark.parametrize("limit", [1, 3, 8])
def test_bounds_never_production_selection(tmp_path, monkeypatch, limit):
    from tools.clip_s2_safe.evaluate import score_dev, validate_report

    c, b, m, a, p, j = setup(tmp_path, monkeypatch)
    j["max_batches"] = limit
    r = score_dev(m, a, p, j, b, on_event=lambda e: None)
    assert r["diagnostic_only"] and not r["complete"] and r["metrics"] is None
    with pytest.raises(ValueError):
        validate_report(r, j, b, a)


def test_probe_mode_full_count_is_still_diagnostic(tmp_path, monkeypatch):
    from tools.clip_s2_safe.evaluate import score_dev, validate_report

    c, b, m, a, p, j = setup(tmp_path, monkeypatch)
    j["mode"] = "handoff_probe"
    r = score_dev(m, a, p, j, b, on_event=lambda e: None)
    assert r["metrics"] is None and r["diagnostic_only"]
    with pytest.raises(ValueError):
        validate_report(r, j, b, a)


@pytest.mark.parametrize(
    "field",
    ["binding_hash", "checkpoint_sha256", "original_identity_hash", "metrics", "order", "exemplar"],
)
def test_changed_report_binding_or_metrics_rejected(tmp_path, monkeypatch, field):
    from tools.clip_s2_safe.evaluate import score_dev, validate_report

    c, b, m, a, p, j = setup(tmp_path, monkeypatch)
    r = score_dev(m, a, p, j, b, on_event=lambda e: None)
    if field == "order":
        r["items"].reverse()
    elif field == "exemplar":
        r["items"][0]["exemplar_ids"] = ["dev0"]
    elif field == "metrics":
        r["metrics"]["bleu4"] += 1
    else:
        r[field] = "wrong"
    with pytest.raises(ValueError):
        validate_report(r, j, b, a)


def test_primary_survives_failing_error_observer(tmp_path, monkeypatch):
    from tools.clip_s2_safe import evaluate as e

    c, b, m, a, p, j = setup(tmp_path, monkeypatch)
    error = RuntimeError("primary")
    calls = []

    def fail(*a, **k):
        calls.append(1)
        raise error

    def observer(event):
        if event["phase"] == "batch_error":
            raise OSError("log failed")

    monkeypatch.setattr(e, "generate_cached_text", fail)
    with pytest.raises(RuntimeError) as got:
        e.score_dev(m, a, p, j, b, on_event=observer)
    assert got.value is error and len(calls) == 1 and error.__notes__


def test_worker_scope_blocks_model_and_optimizer(tmp_path):
    from tools.clip_s2_safe.evaluate import run_dev_job
    from tools.clip_s2_safe.tests.helpers import original_fixture

    c, *_ = original_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="scope"):
        run_dev_job(c, {}, None)

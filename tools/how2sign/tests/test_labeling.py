import json

import how2sign_probe.labeling as labeling
import numpy as np
import pytest
from how2sign_probe.labeling import make_job, run_job

CROP = np.zeros((224, 224, 3), dtype=np.uint8)


@pytest.fixture
def env(tmp_path):
    root = tmp_path / "journal"
    crops = tmp_path / "crops"
    crops.mkdir()
    import cv2

    job = make_job("hand", "LEFT", "clipA", 10, json.dumps([0, 0, 100, 100]))
    cv2.imwrite(str(crops / f"{job.key}.png"), CROP)
    return root, crops, job


VALID_HAND = json.dumps(
    {
        "handshape_class": "fist",
        "finger_spread": "together",
        "palm_direction": "up",
        "thumb_position": "extended",
        "visibility": "clear",
    }
)


def test_run_job_journals_done_and_resumes(env, monkeypatch):
    root, crops, job = env
    calls = []

    def fake(prompt_text, image_b64, model, port, kind, timeout=900):
        calls.append(model)
        return VALID_HAND, {"eval_count": 991}

    monkeypatch.setattr(labeling, "request_label", fake)
    state = run_job(root, job, crops, "qwen3-vl:8b", 11434, sleeper=lambda _s: None)
    assert state["status"] == "done"
    assert state["label"]["handshape_class"] == "fist"
    assert state["attempts"][0]["eval_count"] == 991
    assert calls == ["qwen3-vl:8b"]
    calls.clear()
    again = run_job(root, job, crops, "qwen3-vl:8b", 11434, sleeper=lambda _s: None)
    assert again["status"] == "done"
    assert calls == []


def test_run_job_reserves_attempt_before_request(env, monkeypatch):
    root, crops, job = env
    seen = {}

    def fake(prompt_text, image_b64, model, port, kind, timeout=900):
        seen.setdefault("reserved", labeling.journal_path(root, job).read_text())
        raise RuntimeError("boom")

    monkeypatch.setattr(labeling, "request_label", fake)
    state = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    reserved = json.loads(seen["reserved"])
    assert reserved["attempts"][0]["error"] == "interrupted"
    assert reserved["status"] == "pending"
    assert state["status"] == "failed"
    assert len(state["attempts"]) == 3
    assert all("request: boom" in attempt["error"] for attempt in state["attempts"])


def test_run_job_retries_invalid_labels(env, monkeypatch):
    root, crops, job = env
    responses = iter(["not json", VALID_HAND])
    monkeypatch.setattr(
        labeling,
        "request_label",
        lambda *a, **k: (next(responses), {"eval_count": 100}),
    )
    state = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    assert state["status"] == "done"
    assert len(state["attempts"]) == 2
    assert "parse:" in state["attempts"][0]["error"]


def test_exhausted_attempts_fails(env, monkeypatch):
    root, crops, job = env
    monkeypatch.setattr(labeling, "request_label", lambda *a, **k: ("junk", {}))
    state = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    assert state["status"] == "failed"


def test_failed_job_is_retried_on_next_pass(env, monkeypatch):
    root, crops, job = env
    monkeypatch.setattr(
        labeling, "request_label", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("500"))
    )
    state = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    assert state["status"] == "failed"
    monkeypatch.setattr(labeling, "request_label", lambda *a, **k: (VALID_HAND, {"eval_count": 1}))
    retried = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    assert retried["status"] == "done"
    assert len(retried["attempts"]) == 1


def test_request_payload_skips_thinking(env, monkeypatch):
    root, crops, job = env
    seen = {}

    class FakeStream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __iter__(self):
            yield (
                json.dumps(
                    {"message": {"content": VALID_HAND}, "done": True, "eval_count": 34}
                ).encode()
                + b"\n"
            )

    def fake_urlopen(url, data, timeout=None):
        seen["url"] = url
        seen["payload"] = json.loads(data)
        return FakeStream()

    monkeypatch.setattr(labeling.urllib.request, "urlopen", fake_urlopen)
    state = run_job(root, job, crops, "qwen3-vl:8b", 11434, sleeper=lambda _s: None)
    payload = seen["payload"]
    assert state["status"] == "done"
    assert seen["url"] == "http://127.0.0.1:11434/api/chat"
    assert payload["raw"] is True
    assert payload["messages"][1] == {"role": "assistant", "content": "\n\n"}
    assert payload["messages"][0]["images"]
    assert "format" not in payload
    assert "think" not in payload
    assert payload["options"]["num_gpu"] == 99


def test_missing_crop_journals_failed_instead_of_crashing(tmp_path, monkeypatch):
    root = tmp_path / "journal"
    crops = tmp_path / "crops"
    crops.mkdir()
    job = make_job("hand", "LEFT", "clipX", 3, json.dumps([0, 0, 50, 50]))
    called = []
    monkeypatch.setattr(labeling, "request_label", lambda *a, **k: called.append(1))
    state = run_job(root, job, crops, "model", 11434, sleeper=lambda _s: None)
    assert state["status"] == "failed"
    assert "crop read failed" in state["attempts"][0]["error"]
    assert called == []

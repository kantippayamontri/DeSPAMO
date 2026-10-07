import copy
import json

import cv2
import numpy as np
import pytest
from how2sign_probe import stable
from how2sign_probe.labeling import atomic_json

LABELS = {
    "biometric": dict(visibility="partial", apparent_age_band="uncertain",
                      gender_presentation="uncertain", apparent_height="uncertain",
                      apparent_build="average"),
    "clothing": dict(visibility="clear", upper_garment="shirt", upper_color="orange",
                     lower_garment=None, lower_color=None, pattern="plain", accessories=[]),
    "hair": dict(visibility="clear", color="brown", length="short", style="straight"),
    "background": dict(visibility="clear", scene="studio", dominant_colors=["green"],
                       static_objects=[], lighting="even"),
}


def response(clip="A", indices=None):
    return json.dumps(dict(clip_id=clip, frame_indices=indices or [2, 6, 10, 14, 18],
                           stable=LABELS))


def test_stable_response_uses_original_schema_and_canonical_strings():
    record = stable.validate_response(response(), "A", [2, 6, 10, 14, 18])
    targets = stable.target_list(record)
    assert [t["factor"] for t in targets] == ["biometric", "clothing", "hair", "background"]
    assert all(t["frame_index"] == -1 for t in targets)
    assert targets[0]["text"] == "biometric: apparent build average"
    assert targets[1]["text"] == (
        "clothing: upper garment shirt; upper color orange; pattern plain"
    )


@pytest.mark.parametrize("clip,indices", [("wrong", [2, 6, 10, 14, 18]), ("A", [1, 2, 3, 4, 5])])
def test_response_rejects_wrong_clip_or_frame_alignment(clip, indices):
    with pytest.raises(ValueError):
        stable.validate_response(response(clip, indices), "A", [2, 6, 10, 14, 18])


def test_privacy_guard_and_visibility_contract():
    labels = copy.deepcopy(LABELS)
    labels["clothing"]["upper_garment"] = "skin tone"
    with pytest.raises(ValueError, match="forbidden"):
        stable.validate_response(json.dumps(dict(clip_id="A", frame_indices=[2, 6, 10, 14, 18],
                                                 stable=labels)), "A", [2, 6, 10, 14, 18])
    labels["clothing"]["upper_garment"] = "shirt"
    labels["clothing"]["visibility"] = "not_visible"
    with pytest.raises(ValueError, match="not_visible"):
        stable.validate_response(json.dumps(dict(clip_id="A", frame_indices=[2, 6, 10, 14, 18],
                                                 stable=labels)), "A", [2, 6, 10, 14, 18])


def test_request_boundary_sends_five_whole_frames_without_thinking(monkeypatch):
    seen = {}

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __iter__(self):
            yield json.dumps(dict(message=dict(content=response()), done=True,
                                  eval_count=180)).encode()

    def transport(url, data, timeout):
        seen.update(json.loads(data))
        return Stream()

    monkeypatch.setattr(stable.urllib.request, "urlopen", transport)
    version = dict(model="qwen3-vl:8b", options=stable.OPTIONS)
    raw, _ = stable.request(stable.build_prompt("A", [2, 6, 10, 14, 18]),
                            ["image"] * 5, version, 11435)
    assert json.loads(raw)["clip_id"] == "A"
    assert len(seen["messages"][0]["images"]) == 5
    assert seen["messages"][1]["content"] == "\n\n"
    assert "think" not in seen and "format" not in seen
    assert seen["options"]["num_ctx"] >= 8192


def test_journal_resumes_done_and_reserves_before_request(tmp_path, monkeypatch):
    version = dict(dataset_key="version-key", model="qwen3-vl:8b", model_digest="digest",
                   options=stable.OPTIONS)
    entry = dict(clip="A", stratum="video")
    monkeypatch.setattr(stable, "read_images", lambda *a: dict(
        frame_count=21, frame_indices=[2, 6, 10, 14, 18], images=["image"] * 5,
        source_frame_sha256=["hash"] * 5, image_sha256=["hash"] * 5,
    ))
    monkeypatch.setattr(stable, "ollama_model_digest", lambda *a: "digest")

    def transport(prompt, images, bound_version, port):
        state = json.loads(stable.record_path(tmp_path, "A").read_text())
        assert state["status"] == "pending"
        assert state["attempts"][-1]["raw"] is None
        return response(), dict(eval_count=180)

    monkeypatch.setattr(stable, "request", transport)
    state = stable.run_clip(tmp_path, entry, tmp_path, version, 11435)
    assert state["status"] == "valid"
    assert len(state["targets"]) == 4
    before = stable.record_path(tmp_path, "A").read_bytes()
    monkeypatch.setattr(stable, "request", lambda *a: pytest.fail("done record requested again"))
    stable.run_clip(tmp_path, entry, tmp_path, version, 11435)
    assert stable.record_path(tmp_path, "A").read_bytes() == before
    with pytest.raises(ValueError, match="version"):
        stable.run_clip(tmp_path, entry, tmp_path, {**version, "dataset_key": "new"}, 11435)


def test_missing_video_is_retained_as_four_masked_targets(tmp_path):
    version = dict(dataset_key="v", model="qwen3-vl:8b", model_digest="d", options=stable.OPTIONS)
    state = stable.run_clip(tmp_path, dict(clip="missing"), tmp_path, version, 11435)
    assert state["status"] == "failed"
    assert len(state["targets"]) == 4
    assert all(t["text"] is None for t in state["targets"])


def test_completion_refuses_partial_stable_dataset(tmp_path):
    atomic_json(tmp_path / "sample.json", dict(clips=[dict(clip="A"), dict(clip="B")]))
    with pytest.raises(ValueError, match="unfinished"):
        stable.verify_complete(tmp_path)


def test_whole_frames_preserve_aspect_and_match_crop_source_hashes(tmp_path, monkeypatch):
    import base64
    import hashlib

    frame = np.full((720, 1280, 3), 80, np.uint8)
    indices = [2, 6, 10, 14, 18]
    atomic_json(tmp_path / "repair_clips/A.json", dict(frames=[
        dict(index=i, source_frame_sha256=hashlib.sha256(frame.tobytes()).hexdigest())
        for i in indices
    ]))
    monkeypatch.setattr(stable, "frame_count", lambda *a: 21)
    monkeypatch.setattr(stable, "read_frame", lambda *a: frame)
    inputs = stable.read_images(tmp_path, "A", tmp_path)
    assert inputs["frame_indices"] == indices
    assert len(inputs["images"]) == 5
    decoded = cv2.imdecode(np.frombuffer(base64.b64decode(inputs["images"][0]), np.uint8),
                           cv2.IMREAD_COLOR)
    assert decoded.shape == (360, 640, 3)  # full frame, aspect preserved
    changed = frame.copy()
    changed[0, 0] = 0
    monkeypatch.setattr(stable, "read_frame", lambda *a: changed)
    with pytest.raises(ValueError, match="pixels"):
        stable.read_images(tmp_path, "A", tmp_path)

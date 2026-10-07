import json

import cv2
import numpy as np
import pytest
from how2sign_probe import repair
from how2sign_probe.detector import Region
from how2sign_probe.labeling import atomic_json, journal_path, make_job


def test_repair_preserves_done_hands_but_never_imports_mouth_labels(tmp_path, monkeypatch):
    old, new = tmp_path / "old", tmp_path / "new"
    atomic_json(old / "sample.json", {"clips": [{"clip": "A", "stratum": "s"}]})
    atomic_json(old / "provenance.json", {"model": "qwen3-vl:8b"})
    box = [10, 10, 50, 50]
    atomic_json(old / "regions.json", {"A|0": [
        ["hand", "LEFT", box, 0.9], ["mouth", "MOUTH", box, 0.9],
    ]})
    old.mkdir(exist_ok=True)
    (old / "crops").mkdir()
    hand = make_job("hand", "LEFT", "A", 0, json.dumps(box))
    mouth = make_job("mouth", "MOUTH", "A", 0, json.dumps(box))
    label = dict(handshape_class="fist", finger_spread="together", palm_direction="up",
                 thumb_position="tucked", visibility="clear")
    for job in [hand, mouth]:
        cv2.imwrite(str(old / "crops" / f"{job.key}.png"), np.zeros((224, 224, 3), np.uint8))
        atomic_json(journal_path(old, job), dict(
            identity=json.dumps(job.identity), status="done", label=label, attempts=[],
        ))
    monkeypatch.setattr(repair, "frames_for_clip", lambda *a: (
        [0, 1], lambda i: np.full((100, 100, 3), 20, np.uint8),
    ))
    monkeypatch.setattr(repair, "mouth_region", lambda *a: Region("mouth", "MOUTH", tuple(box), .9))
    repair.prepare(old, new)
    assert json.loads(journal_path(new, hand).read_text())["status"] == "done"
    assert not journal_path(new, mouth).exists()  # even if the key matches!
    assert journal_path(old, mouth).exists()
    receipt = json.loads((new / "repair_clips" / "A.json").read_text())
    assert receipt["frames"][0]["old_mouths"][0]["status"] == "done"
    assert receipt["frames"][1]["old_mouths"] == []
    assert receipt["frames"][1]["new_mouth_key"] is not None
    assert receipt["hands_imported"] == 1
    before = (new / "repair_clips" / "A.json").read_bytes()
    monkeypatch.setattr(repair, "mouth_region", lambda *a: pytest.fail("resume re-detected clip"))
    repair.prepare(old, new)
    assert (new / "repair_clips" / "A.json").read_bytes() == before


def test_completion_rejects_incomplete_journals(tmp_path):
    from how2sign_probe.completion import validate_completion
    atomic_json(tmp_path / "sample.json", {"clips": [{"clip": "A"}]})
    atomic_json(tmp_path / "regions.json", {"A|0": [["mouth", "MOUTH", [0, 0, 40, 40], .9]]})
    atomic_json(tmp_path / "detector_report.json", {"gate": {"pass": True}})
    with pytest.raises(ValueError, match="unfinished"):
        validate_completion(tmp_path)


def test_completion_validates_identity_and_schema(tmp_path):
    from how2sign_probe.completion import validate_completion
    box = [0, 0, 40, 40]
    atomic_json(tmp_path / "sample.json", {"clips": [{"clip": "A"}]})
    atomic_json(tmp_path / "regions.json", {"A|0": [["mouth", "MOUTH", box, .9]]})
    atomic_json(tmp_path / "detector_report.json", {"gate": {"pass": True}})
    job = make_job("mouth", "MOUTH", "A", 0, json.dumps(box))
    state = dict(identity=json.dumps(job.identity), status="done", attempts=[], label=dict(
        openness="open", lip_configuration="neutral", teeth_or_tongue="none",
        cheek_state="neutral", visibility="clear",
    ))
    atomic_json(journal_path(tmp_path, job), state)
    assert validate_completion(tmp_path)["done"] == 1
    state["identity"] = json.dumps(["wrong"])
    atomic_json(journal_path(tmp_path, job), state)
    with pytest.raises(ValueError, match="identity"):
        validate_completion(tmp_path)

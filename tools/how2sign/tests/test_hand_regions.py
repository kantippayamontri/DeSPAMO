import how2sign_probe.detector as detector
import numpy as np
from how2sign_probe.detector import Region, hand_regions


class _Landmark:
    def __init__(self, x, y):
        self.x, self.y = x, y


class _Classification:
    def __init__(self, label, score):
        self.label, self.score = label, score


class _Handedness:
    def __init__(self, label, score):
        self.classification = [_Classification(label, score)]


class _HandLandmarks:
    def __init__(self):
        points = [_Landmark(0.4, 0.4)] * 21
        points[0] = _Landmark(0.6, 0.6)
        self.landmark = points


class _HandsResult:
    def __init__(self, labels):
        self.multi_hand_landmarks = [_HandLandmarks() for _ in labels] if labels else None
        self.multi_handedness = [_Handedness(label, 0.9) for label in labels] or None


class _FakeHands:
    def __init__(self, result):
        self._result = result

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def process(self, frame):
        return self._result


def test_hand_regions_flips_mirrored_convention_for_unmirrored_video(monkeypatch):
    monkeypatch.setattr(
        detector.mp.solutions.hands,
        "Hands",
        lambda **kwargs: _FakeHands(_HandsResult(["Left", "Right"])),
    )
    rgb = np.zeros((260, 210, 3), np.uint8)
    regions = hand_regions(rgb, 260, 210)
    assert [(r.kind, r.side) for r in regions] == [
        ("hand", "RIGHT"),  # MediaPipe "Left" => anatomical RIGHT
        ("hand", "LEFT"),
    ]
    assert all(isinstance(r, Region) for r in regions)


def test_hand_regions_empty_result(monkeypatch):
    monkeypatch.setattr(
        detector.mp.solutions.hands, "Hands", lambda **kwargs: _FakeHands(_HandsResult([]))
    )
    assert hand_regions(np.zeros((260, 210, 3), np.uint8), 260, 210) == []

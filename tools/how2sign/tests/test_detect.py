import pytest
from how2sign_probe.detect import MouthTarget, landmark_box, mouth_box, side_label

HEIGHT, WIDTH = 720, 1280


def _landmarks(points):
    return [type("L", (), {"x": x, "y": y}) for x, y in points]


def test_landmark_box_scales_to_pixels():
    box = landmark_box(_landmarks([(0.25, 0.5), (0.75, 0.25)]), (HEIGHT, WIDTH))
    assert box == (320, 180, 960, 360)


def test_landmark_box_rejects_empty():
    with pytest.raises(ValueError):
        landmark_box([], (HEIGHT, WIDTH))


def test_mouth_box_targets_lip_region():
    lips = _landmarks([(0.46, 0.44), (0.54, 0.44), (0.50, 0.42), (0.50, 0.47)])
    face = _landmarks([(0.50, 0.30), (0.50, 0.90)])
    target = mouth_box(lips, face)
    assert isinstance(target, MouthTarget)
    x1, y1, x2, y2 = target.box
    assert 0 <= x1 < x2 <= WIDTH and 0 <= y1 < y2 <= HEIGHT
    assert abs((x1 + x2) / 2 - 0.5 * WIDTH) < 30
    assert (x2 - x1, y2 - y1) == (118, 237)
    assert y1 < (0.42 + 0.47) / 2 * HEIGHT < y2


def test_mouth_box_floors_tiny_lip_rings():
    lips = _landmarks([(0.50, 0.45), (0.50, 0.45)])
    face = _landmarks([(0.50, 0.20), (0.50, 0.70)])
    target = mouth_box(lips, face)
    assert target.box[2] - target.box[0] >= 30
    assert target.box[3] - target.box[1] >= 30


def test_mouth_box_requires_landmarks():
    with pytest.raises(ValueError):
        mouth_box([], _landmarks([(0.5, 0.5)]))


def test_side_label_maps_mirrored_convention():
    assert side_label("Right", mirrored=True) == "RIGHT"
    assert side_label("Right", mirrored=False) == "LEFT"
    assert side_label("Left", mirrored=False) == "RIGHT"
    with pytest.raises(ValueError):
        side_label("right", mirrored=True)

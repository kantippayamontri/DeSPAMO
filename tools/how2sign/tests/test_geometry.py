import numpy as np
import pytest
from how2sign_probe.crops import crop_box, resize_to_224
from how2sign_probe.frames import sample_indices


def test_sample_indices_five_distinct_positions():
    assert sample_indices(21) == [2, 6, 10, 14, 18]
    indices = sample_indices(5)
    assert len(set(indices)) == 5
    assert min(indices) >= 0 and max(indices) < 5
    with pytest.raises(ValueError):
        sample_indices(4)


def _frame():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)
    return data


def test_crop_box_pads_and_clamps():
    box = crop_box((10, 10, 30, 30), (720, 1280), expand=2.0)
    assert box == (0, 0, 40, 40)
    box = crop_box((600, 300, 700, 400), (720, 1280), expand=2.0)
    assert box[0] >= 0 and box[1] >= 0 and box[2] <= 1280 and box[3] <= 720


def test_crop_box_rejects_inverted():
    with pytest.raises(ValueError):
        crop_box((50, 50, 10, 10), (720, 1280), expand=1.0)


def test_crop_box_rejects_box_entirely_outside_frame():
    with pytest.raises(ValueError):
        crop_box((1400, 10, 1500, 90), (720, 1280), expand=1.0)


def test_crop_box_clamps_negative_origin_from_edge_landmarks():
    box = crop_box((-6, 81, 29, 129), (260, 210), expand=1.0)
    assert box[0] == 0 and box[2] <= 210 and box[3] <= 260
    assert box[2] >= 29 and box[1] <= 81


def test_resize_to_224_is_square_upscales_blur_downscales():
    box = crop_box((600, 300, 700, 400), (720, 1280), expand=1.0)
    crop, factor = resize_to_224(_frame(), box)
    assert crop.shape[:2] == (224, 224)
    expected = 224 / (box[2] - box[0])
    assert abs(factor - expected) < 1e-6


def test_resize_rejects_degenerate_box():
    with pytest.raises(ValueError):
        resize_to_224(_frame(), (10, 10, 10, 10))

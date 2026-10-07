import numpy as np
import pytest
from how2sign_probe.crops import CROP_SIZE, resize_to_224


def _frame():
    data = np.zeros((720, 1280, 3), dtype=np.uint8)
    data[100:200, 600:700] = 255
    return data


def test_resize_square_box_fills_canvas():
    crop, factor = resize_to_224(_frame(), (600, 100, 700, 200))
    assert crop.shape[:2] == (CROP_SIZE, CROP_SIZE)
    assert abs(factor - 2.24) < 1e-6
    assert (crop > 0).all()


def test_resize_centers_wide_content_with_border():
    data = _frame()
    data[:, 500:580] = 0
    data[:, 680:700] = 0
    crop, factor = resize_to_224(data, (500, 100, 700, 120))
    assert abs(factor - 1.12) < 1e-6
    assert crop[0, 112].tolist() == [0, 0, 0]
    assert (crop[112, 0:111] == 0).all()
    assert (crop[112, 146:173] > 0).all()
    assert (crop[112, 210:224] == 0).all()


def test_resize_rejects_degenerate():
    with pytest.raises(ValueError):
        resize_to_224(_frame(), (10, 10, 10, 10))

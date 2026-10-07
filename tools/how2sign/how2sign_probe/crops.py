"""Square crop geometry shared by the detector and the labeler."""

import cv2
import numpy as np

CROP_SIZE = 224


def crop_box(
    box: tuple[int, int, int, int], frame_shape, expand: float = 1.0
) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = (int(round(value)) for value in box)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox {box}")
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"bbox outside frame: {box}")
    centre_x = (x1 + x2) / 2.0
    centre_y = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * expand
    half = side / 2.0
    left = int(round(max(0.0, centre_x - half)))
    top = int(round(max(0.0, centre_y - half)))
    right = int(round(min(float(width), centre_x + half)))
    bottom = int(round(min(float(height), centre_y + half)))
    if right <= left or bottom <= top:
        raise ValueError(f"clipped crop is empty: {box}")
    return left, top, right, bottom


def resize_to_224(frame: np.ndarray, box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    if width < 1 or height < 1:
        raise ValueError(f"degenerate crop box {box}")
    crop = frame[top:bottom, left:right]
    scale = CROP_SIZE / max(width, height)
    new_w, new_h = max(1, int(round(width * scale))), max(1, int(round(height * scale)))
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    x = (CROP_SIZE - new_w) // 2
    y = (CROP_SIZE - new_h) // 2
    canvas[y : y + new_h, x : x + new_w] = resized
    return canvas, float(scale)

"""Five-frame sampling and MP4 decoding for the How2Sign crop probe."""

import cv2
import numpy as np

POSITIONS = (0.1, 0.3, 0.5, 0.7, 0.9)


def sample_indices(count: int) -> list[int]:
    if type(count) is not int or count < 5:
        raise ValueError("at least five frames required")
    indices = [int((count - 1) * position + 0.5) for position in POSITIONS]
    if len(set(indices)) != 5:
        raise ValueError("five sampled source indices must be distinct")
    return indices


def frame_count(path) -> int:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if count < 5:
        raise ValueError(f"{path}: fewer than five frames ({count})")
    return count


def read_frame(path, index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise ValueError(f"{path}: cannot decode frame {index}")
    return frame

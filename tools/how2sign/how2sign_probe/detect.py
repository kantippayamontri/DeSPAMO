"""Hand and mouth region detection on unmirrored sign-language frames.

MediaPipe hands reports handedness under the mirrored-selfie convention, so for
unmirrored recordings (both How2Sign and PHOENIX14T) the anatomical side is the
opposite label; ``side_label(label, mirrored=False)`` performs that flip and
``hand_regions`` is its only call site.
"""

import dataclasses

MOUTH_LIP_INDICES = (17, 402, 61, 291, 13, 14)
FACE_ANCHOR_INDICES = (4, 152)
MOUTH_WIDTH_FACTOR = 1.15
MOUTH_HEIGHT_FACTOR = 0.55
MIN_MOUTH_SPAN_PX = 30


@dataclasses.dataclass(frozen=True)
class MouthTarget:
    box: tuple[int, int, int, int]
    landmarks: tuple


def landmark_box(landmarks, frame_shape) -> tuple[int, int, int, int]:
    height, width = frame_shape[:2]
    if not landmarks:
        raise ValueError("no landmarks")
    xs = [point.x * width for point in landmarks]
    ys = [point.y * height for point in landmarks]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def side_label(label: str, mirrored: bool) -> str:
    normalized = str(label).strip()
    if normalized == "Left":
        anatomical = "LEFT" if mirrored else "RIGHT"
    elif normalized == "Right":
        anatomical = "RIGHT" if mirrored else "LEFT"
    else:
        raise ValueError(f"unknown handedness label: {label!r}")
    return anatomical


def mouth_box(lips, face_anchors, frame_shape=(720, 1280)) -> MouthTarget:
    """Mouth crop centred on the lip ring, sized by the nose-tip→chin span.

    Both dimensions carry pixel floors so a sparse or motion-blurred ring still
    yields a readable mouth window instead of a degenerate box.
    """
    height, width = frame_shape[:2]
    if len(lips) < 2 or len(face_anchors) < 2:
        raise ValueError("mouth needs a lip ring and two face anchors")
    xs = [point.x * width for point in lips]
    ys = [point.y * height for point in lips]
    centre_x = (min(xs) + max(xs)) / 2.0
    centre_y = (min(ys) + max(ys)) / 2.0
    nose_y, chin_y = face_anchors[0].y * height, face_anchors[1].y * height
    width_px = max(MIN_MOUTH_SPAN_PX * MOUTH_WIDTH_FACTOR, (max(xs) - min(xs)) * MOUTH_WIDTH_FACTOR)
    height_px = max(
        MIN_MOUTH_SPAN_PX * MOUTH_HEIGHT_FACTOR, abs(chin_y - nose_y) * MOUTH_HEIGHT_FACTOR
    )
    half_w, half_h = width_px / 2.0, height_px / 2.0
    x1, y1 = centre_x - half_w, centre_y - half_h
    x2, y2 = centre_x + half_w, centre_y + half_h
    if x1 < 0:
        x2, x1 = x2 - x1, 0.0
    if y1 < 0:
        y2, y1 = y2 - y1, 0.0
    if x2 > width:
        x1, x2 = max(0.0, x1 - (x2 - width)), float(width)
    if y2 > height:
        y1, y2 = max(0.0, y1 - (y2 - height)), float(height)
    box = (int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2)))
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError("degenerate mouth box")
    return MouthTarget(box=box, landmarks=tuple((p.x, p.y) for p in lips))

"""Region detection: MediaPipe hands plus two-stage face→FaceMesh on How2Sign frames.

``detect_regions`` is the only function that touches MediaPipe; geometry helpers
are pure so crop math stays unit-testable without model weights.
"""

import dataclasses

import cv2
import mediapipe as mp

from how2sign_probe.crops import crop_box
from how2sign_probe.detect import (
    FACE_ANCHOR_INDICES,
    MOUTH_LIP_INDICES,
    MouthTarget,
    landmark_box,
    mouth_box,
    side_label,
)

HAND_EXPAND = 1.25
FACE_EXPAND = 1.4
MIN_SCORE = 0.2
DETECTION_FLOOR_PX = 24


@dataclasses.dataclass(frozen=True)
class Region:
    kind: str
    side: str
    box: tuple[int, int, int, int]
    score: float


def _to_pixels(landmarks, width, height, offset=(0, 0), *, frame_shape):
    """Lift face-local normalized points into full-frame normalized coordinates."""
    frame_height, frame_width = frame_shape[:2]
    class P:
        pass

    points = []
    for point in landmarks:
        item = P()
        item.x = (point.x * width + offset[0]) / frame_width
        item.y = (point.y * height + offset[1]) / frame_height
        points.append(item)
    return points


def hand_regions(rgb, height, width):
    regions = []
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=2, min_detection_confidence=MIN_SCORE
    ) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return regions
    for landmarks, handed in zip(result.multi_hand_landmarks, result.multi_handedness, strict=True):
        classification = handed.classification[0]
        box = crop_box(
            landmark_box(landmarks.landmark, (height, width)),
            (height, width),
            expand=HAND_EXPAND,
        )
        regions.append(
            Region(
                kind="hand",
                side=side_label(classification.label, mirrored=False),
                box=box,
                score=float(classification.score),
            )
        )
    return regions


def mouth_region(rgb, height, width):
    with mp.solutions.face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=MIN_SCORE
    ) as detector:
        found = detector.process(rgb)
    if not found.detections:
        return None
    best = max(found.detections, key=lambda item: item.score[0])
    relative = best.location_data.relative_bounding_box
    raw = (
        int(relative.xmin * width),
        int(relative.ymin * height),
        int((relative.xmin + relative.width) * width),
        int((relative.ymin + relative.height) * height),
    )
    try:
        face_box = crop_box(raw, (height, width), expand=FACE_EXPAND)
    except ValueError:
        return None
    sub = rgb[face_box[1] : face_box[3], face_box[0] : face_box[2]]
    sub_height, sub_width = sub.shape[:2]
    with mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True, max_num_faces=1, min_detection_confidence=MIN_SCORE
    ) as mesh:
        result = mesh.process(sub)
    if not result.multi_face_landmarks:
        return None
    points = result.multi_face_landmarks[0].landmark
    lips = _to_pixels(
        [points[index] for index in MOUTH_LIP_INDICES],
        sub_width,
        sub_height,
        (face_box[0], face_box[1]),
        frame_shape=(height, width),
    )
    anchors = _to_pixels(
        [points[index] for index in FACE_ANCHOR_INDICES],
        sub_width,
        sub_height,
        (face_box[0], face_box[1]),
        frame_shape=(height, width),
    )
    target: MouthTarget = mouth_box(lips, anchors, (height, width))
    return Region(kind="mouth", side="MOUTH", box=target.box, score=float(best.score[0]))


def detect_regions(image_bgr):
    height, width = image_bgr.shape[:2]
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    regions = hand_regions(rgb, height, width)
    mouth = mouth_region(rgb, height, width)
    if mouth is not None:
        regions.append(mouth)
    return regions


def below_floor(region: Region) -> bool:
    return (
        region.box[2] - region.box[0] < DETECTION_FLOOR_PX
        or region.box[3] - region.box[1] < DETECTION_FLOOR_PX
    )

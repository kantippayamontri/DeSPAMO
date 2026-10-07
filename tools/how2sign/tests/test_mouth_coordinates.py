from types import SimpleNamespace

import how2sign_probe.detector as detector
import numpy as np
import pytest
from how2sign_probe.detector import _to_pixels


@pytest.mark.parametrize("frame_width,frame_height", [(1280, 720), (210, 260)])
def test_face_local_landmarks_map_to_source_pixel_coordinates(frame_width, frame_height):
    points = _to_pixels(
        [SimpleNamespace(x=0.5, y=0.5)], 40, 60, (70, 90),
        frame_shape=(frame_height, frame_width),
    )
    assert points[0].x * frame_width == pytest.approx(90)
    assert points[0].y * frame_height == pytest.approx(120)


def test_mouth_region_keeps_crop_on_face_not_image_edge(monkeypatch):
    class Context:
        def __init__(self, result):
            self.result = result

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def process(self, image):
            return self.result

    rect = SimpleNamespace(xmin=500/1280, ymin=150/720, width=200/1280, height=200/720)
    found = SimpleNamespace(detections=[SimpleNamespace(
        score=[.99], location_data=SimpleNamespace(relative_bounding_box=rect),
    )])
    points = [SimpleNamespace(x=.5, y=.5) for _ in range(468)]
    points[61] = SimpleNamespace(x=.4, y=.5)
    points[291] = SimpleNamespace(x=.6, y=.5)
    points[4] = SimpleNamespace(x=.5, y=.2)
    points[152] = SimpleNamespace(x=.5, y=.8)
    mesh = SimpleNamespace(multi_face_landmarks=[SimpleNamespace(landmark=points)])
    monkeypatch.setattr(detector.mp.solutions.face_detection, "FaceDetection",
                        lambda **kwargs: Context(found))
    monkeypatch.setattr(detector.mp.solutions.face_mesh, "FaceMesh",
                        lambda **kwargs: Context(mesh))
    region = detector.mouth_region(np.zeros((720, 1280, 3), np.uint8), 720, 1280)
    x1, y1, x2, y2 = region.box
    assert (x1+x2)/2 == pytest.approx(600, abs=1)
    assert (y1+y2)/2 == pytest.approx(250, abs=1)

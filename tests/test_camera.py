from __future__ import annotations

from pathlib import Path

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.vision import camera
from oracle_report.vision.camera import _camera_candidate_indices, _configure_capture, draw_overlay
from oracle_report.vision.framing import build_capture_guide


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38


class FakeCapture:
    def __init__(self, backend_name: str) -> None:
        self._backend_name = backend_name
        self.set_calls: list[tuple[int, int]] = []
        self._opened = True
        self.released = False

    def getBackendName(self) -> str:
        result = self._backend_name
        return result

    def set(self, property_id: int, value: int) -> bool:
        self.set_calls.append((property_id, value))
        result = True
        return result

    def isOpened(self) -> bool:
        return self._opened

    def release(self) -> None:
        self.released = True


class FakeCv2Open:
    CAP_V4L2 = 200
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38

    def __init__(self, open_indices: set[int]) -> None:
        self.open_indices = open_indices
        self.calls: list[int] = []

    def VideoCapture(self, camera_index: int, backend=None):
        self.calls.append(camera_index)
        capture = FakeCapture("V4L2")
        capture._opened = camera_index in self.open_indices
        return capture


class FakeDrawCv2:
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16

    def __init__(self) -> None:
        self.rectangle_calls: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        self.line_calls: list[tuple[tuple[int, int], tuple[int, int], int]] = []
        self.circle_calls: list[tuple[tuple[int, int], int]] = []
        self.text_calls: list[str] = []

    def rectangle(
        self,
        frame,
        start: tuple[int, int],
        end: tuple[int, int],
        color,
        thickness: int,
    ) -> None:
        self.rectangle_calls.append((start, end, thickness))

    def line(
        self,
        frame,
        start: tuple[int, int],
        end: tuple[int, int],
        color,
        thickness: int,
    ) -> None:
        self.line_calls.append((start, end, thickness))

    def circle(self, frame, point: tuple[int, int], radius: int, color, thickness: int) -> None:
        self.circle_calls.append((point, radius))

    def putText(self, frame, message: str, *args) -> None:
        self.text_calls.append(message)


def test_configure_capture_skips_property_writes_for_gstreamer() -> None:
    capture = FakeCapture("GStreamer")

    _configure_capture(FakeCv2, capture, _capture_config())

    assert capture.set_calls == []


def test_configure_capture_applies_property_writes_for_v4l2() -> None:
    capture = FakeCapture("V4L2")

    _configure_capture(FakeCv2, capture, _capture_config())

    assert capture.set_calls == [
        (FakeCv2.CAP_PROP_FRAME_WIDTH, 640),
        (FakeCv2.CAP_PROP_FRAME_HEIGHT, 480),
        (FakeCv2.CAP_PROP_FPS, 15),
        (FakeCv2.CAP_PROP_BUFFERSIZE, 1),
    ]


def test_draw_overlay_shows_only_head_guide() -> None:
    cv2 = FakeDrawCv2()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    guide = build_capture_guide(frame.shape[1], frame.shape[0])

    draw_overlay(cv2, frame, "ready", [], False)

    assert len(cv2.rectangle_calls) == 2
    assert cv2.rectangle_calls[0][0] == (guide.head_box.x, guide.head_box.y)
    assert cv2.rectangle_calls[0][1] == (
        guide.head_box.x + guide.head_box.width,
        guide.head_box.y + guide.head_box.height,
    )
    assert len(cv2.line_calls) == 0


def test_draw_overlay_can_hide_head_guide_for_web_preview() -> None:
    cv2 = FakeDrawCv2()
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    draw_overlay(cv2, frame, "ready", [], False, (), False)

    assert len(cv2.rectangle_calls) == 1
    assert cv2.rectangle_calls[0][0] == (0, 0)


def test_camera_candidate_indices_prefer_configured_index() -> None:
    config = _capture_config()

    result = _camera_candidate_indices(config)

    assert result[:4] == (0, 1, 2, 3)


def test_camera_candidate_indices_can_disable_auto_detect() -> None:
    config = CaptureConfig(
        camera_index=2,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=2.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=Path("runs"),
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
        camera_auto_detect=False,
    )

    result = _camera_candidate_indices(config)

    assert result == (2,)


def test_open_camera_falls_back_to_next_device(monkeypatch) -> None:
    fake_cv2 = FakeCv2Open({1})

    monkeypatch.setattr(camera.os, "name", "posix")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)

    cv2, capture = camera.open_camera(_capture_config())

    assert cv2 is fake_cv2
    assert fake_cv2.calls[:2] == [0, 0]
    assert 1 in fake_cv2.calls
    assert capture.isOpened() is True


def test_open_camera_reports_permission_hint_for_inaccessible_video_devices(monkeypatch) -> None:
    fake_cv2 = FakeCv2Open(set())

    monkeypatch.setattr(camera.os, "name", "posix")
    monkeypatch.setattr(camera, "_import_cv2", lambda: fake_cv2)
    monkeypatch.setattr(camera, "_discover_video_device_paths", lambda: ["/dev/video0"])
    monkeypatch.setattr(camera.os, "access", lambda path, mode: False)

    try:
        camera.open_camera(_capture_config())
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected open_camera to fail")

    assert "attempted indices: 0, 1, 2, 3, 4, 5" in message
    assert "/dev/video0" in message
    assert "video group membership" in message


def _capture_config() -> CaptureConfig:
    result = CaptureConfig(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=2.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=Path("runs"),
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
    )
    return result

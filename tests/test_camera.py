from __future__ import annotations

from pathlib import Path

from oracle_report.config import CaptureConfig
from oracle_report.vision.camera import _configure_capture


class FakeCv2:
    CAP_PROP_FRAME_WIDTH = 3
    CAP_PROP_FRAME_HEIGHT = 4
    CAP_PROP_FPS = 5
    CAP_PROP_BUFFERSIZE = 38


class FakeCapture:
    def __init__(self, backend_name: str) -> None:
        self._backend_name = backend_name
        self.set_calls: list[tuple[int, int]] = []

    def getBackendName(self) -> str:
        result = self._backend_name
        return result

    def set(self, property_id: int, value: int) -> bool:
        self.set_calls.append((property_id, value))
        result = True
        return result


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

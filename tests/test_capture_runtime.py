from __future__ import annotations

from pathlib import Path

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import FaceQuality
from oracle_report.vision import runtime
from oracle_report.vision.framing import build_capture_guide


class FakeCapture:
    def __init__(self, frame: np.ndarray) -> None:
        self._frame = frame
        self.released = False

    def read(self):
        result = (True, self._frame.copy())
        return result

    def release(self) -> None:
        self.released = True


class FakeDetector:
    def detect(self, frame: np.ndarray):
        guide = build_capture_guide(frame.shape[1], frame.shape[0])
        result = [guide.head_box]
        return result


class FakeAnalyzer:
    def analyze(self, frame: np.ndarray, face):
        result = FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05)
        return result


def test_run_capture_saves_raw_frame_not_overlay(monkeypatch, tmp_path: Path) -> None:
    raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    capture = FakeCapture(raw_frame)
    saved: dict[str, int] = {}
    preview: dict[str, int] = {}

    def fake_open_camera(config: CaptureConfig):
        result = (FakeCv2(), capture)
        return result

    def fake_build_capture_processors(config: CaptureConfig):
        result = (FakeDetector(), FakeAnalyzer())
        return result

    def fake_draw_overlay(cv2, frame, *args) -> None:
        frame[:, :, :] = 255

    def fake_save_capture_artifact(frame, decision, output_dir):
        saved["max_pixel"] = int(frame.max())
        result = object()
        return result

    def frame_callback(cv2, frame) -> None:
        preview["max_pixel"] = int(frame.max())

    monkeypatch.setattr(runtime, "open_camera", fake_open_camera)
    monkeypatch.setattr(runtime, "build_capture_processors", fake_build_capture_processors)
    monkeypatch.setattr(runtime, "draw_overlay", fake_draw_overlay)
    monkeypatch.setattr(runtime, "save_capture_artifact", fake_save_capture_artifact)

    result = runtime.run_capture(_capture_config(tmp_path), frame_callback=frame_callback)

    assert result is not None
    assert preview["max_pixel"] == 255
    assert saved["max_pixel"] == 0
    assert capture.released is True


def test_run_capture_can_return_mock_landmark_artifact(tmp_path: Path) -> None:
    config = _capture_config(tmp_path)
    config = CaptureConfig(
        camera_index=config.camera_index,
        frame_width=config.frame_width,
        frame_height=config.frame_height,
        camera_fps=config.camera_fps,
        min_face_seconds=config.min_face_seconds,
        face_min_size_px=config.face_min_size_px,
        face_detection_scale=config.face_detection_scale,
        face_detection_interval=config.face_detection_interval,
        output_dir=config.output_dir,
        show_preview=config.show_preview,
        eye_min_count=config.eye_min_count,
        eyebrow_min_edge_density=config.eyebrow_min_edge_density,
        camera_auto_detect=config.camera_auto_detect,
        mock_capture_enabled=True,
        mock_landmark_metrics_json='{"eye_width_ratio": 0.19, "eye_spacing_ratio": 0.28}',
    )

    result = runtime.run_capture(config)

    assert result.image_path.exists() is True
    assert result.quality.ready is True
    assert "0.190" in result.quality.landmark_metrics_text
    assert "해석 힌트" in result.quality.landmark_rules_text
    assert "삼정" in result.quality.landmark_rules_text
    assert "랜드마크 룰 기반" in result.face_analysis
    assert result.face.width >= 96


class FakeCv2:
    pass


def _capture_config(tmp_path: Path) -> CaptureConfig:
    result = CaptureConfig(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=0.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=tmp_path,
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
    )
    return result

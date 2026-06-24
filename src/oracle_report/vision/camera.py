from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import FaceBox
from oracle_report.vision.detection import HaarFaceDetector, _import_cv2
from oracle_report.vision.quality import OpenCvFaceQualityAnalyzer


_GSTREAMER_BACKEND_NAME = "GSTREAMER"
_VIDEO_CAPTURE_BUFFER_SIZE = 1
_FACE_ANALYSIS_MODE_LLM_IMAGE = 1
_FACE_ANALYSIS_MODE_LANDMARK_RULE = 2


def open_camera(config: CaptureConfig) -> tuple[Any, Any]:
    cv2 = _import_cv2()
    capture = _open_video_capture(cv2, config.camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"failed to open camera index {config.camera_index}")
    _configure_capture(cv2, capture, config)
    result = (cv2, capture)
    return result


def _open_video_capture(cv2: Any, camera_index: int) -> Any:
    if os.name == "posix" and hasattr(cv2, "CAP_V4L2"):
        capture = cv2.VideoCapture(camera_index, cv2.CAP_V4L2)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(camera_index)
    else:
        capture = cv2.VideoCapture(camera_index)
    result = capture
    return result


def _configure_capture(cv2: Any, capture: Any, config: CaptureConfig) -> None:
    backend_name = _read_capture_backend_name(capture)
    if backend_name != _GSTREAMER_BACKEND_NAME:
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, config.frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, config.frame_height)
        capture.set(cv2.CAP_PROP_FPS, config.camera_fps)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, _VIDEO_CAPTURE_BUFFER_SIZE)


def _read_capture_backend_name(capture: Any) -> str:
    backend_name = ""
    if hasattr(capture, "getBackendName"):
        backend_name = capture.getBackendName().upper()
    result = backend_name
    return result


def build_default_face_detector(config: CaptureConfig) -> HaarFaceDetector:
    result = HaarFaceDetector(
        min_size_px=config.face_min_size_px,
        detection_scale=config.face_detection_scale,
        detection_interval=config.face_detection_interval,
    )
    return result


def build_default_quality_analyzer(
    config: CaptureConfig,
) -> OpenCvFaceQualityAnalyzer:
    result = OpenCvFaceQualityAnalyzer(
        eye_min_count=config.eye_min_count,
        eyebrow_min_edge_density=config.eyebrow_min_edge_density,
    )
    return result


def build_capture_processors(config: CaptureConfig):
    if config.face_analysis_mode == _FACE_ANALYSIS_MODE_LANDMARK_RULE:
        from oracle_report.vision.landmarks import (
            MediaPipeLandmarkFaceDetector,
            MediaPipeLandmarkQualityAnalyzer,
        )

        detector = MediaPipeLandmarkFaceDetector(
            min_size_px=config.face_min_size_px,
            detection_scale=config.face_detection_scale,
            detection_interval=config.face_detection_interval,
        )
        analyzer = MediaPipeLandmarkQualityAnalyzer(detector)
        result = (detector, analyzer)
    else:
        detector = build_default_face_detector(config)
        analyzer = build_default_quality_analyzer(config)
        result = (detector, analyzer)
    return result


def draw_overlay(
    cv2: Any,
    frame: np.ndarray,
    message: str,
    faces: Sequence[FaceBox],
    warning: bool,
    landmarks: Sequence[tuple[int, int]] = (),
) -> None:
    color = (0, 180, 0)
    if warning:
        color = (0, 0, 255)
    for face in faces:
        cv2.rectangle(
            frame,
            (face.x, face.y),
            (face.x + face.width, face.y + face.height),
            color,
            2,
        )
    for point in landmarks:
        cv2.circle(frame, point, 2, color, -1)
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 54), (0, 0, 0), -1)
    cv2.putText(
        frame,
        message,
        (24, 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

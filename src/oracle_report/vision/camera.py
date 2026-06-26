from __future__ import annotations

import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import FaceBox
from oracle_report.vision.detection import HaarFaceDetector, _import_cv2
from oracle_report.vision.framing import CaptureGuide, build_capture_guide
from oracle_report.vision.quality import OpenCvFaceQualityAnalyzer


_GSTREAMER_BACKEND_NAME = "GSTREAMER"
_VIDEO_CAPTURE_BUFFER_SIZE = 1
_FACE_ANALYSIS_MODE_LLM_IMAGE = 1
_FACE_ANALYSIS_MODE_LANDMARK_RULE = 2
_OVERLAY_HEIGHT_PX = 54
_OVERLAY_TEXT_POSITION = (24, 14)
_OVERLAY_TEXT_BASELINE_POSITION = (24, 36)
_OVERLAY_FONT_SIZE = 24
_GUIDE_HEAD_COLOR = (0, 210, 255)
_GUIDE_THICKNESS = 2
_KOREAN_FONT_PATHS = (
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("C:/Windows/Fonts/malgun.ttf"),
)


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
    guide = build_capture_guide(frame.shape[1], frame.shape[0])
    _draw_capture_guide(cv2, frame, guide)
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
    cv2.rectangle(
        frame,
        (0, 0),
        (frame.shape[1], _OVERLAY_HEIGHT_PX),
        (0, 0, 0),
        -1,
    )
    if not _draw_unicode_text(frame, message):
        _draw_cv2_text(cv2, frame, message)


def _draw_capture_guide(cv2: Any, frame: np.ndarray, guide: CaptureGuide) -> None:
    head = guide.head_box
    cv2.rectangle(
        frame,
        (head.x, head.y),
        (head.x + head.width, head.y + head.height),
        _GUIDE_HEAD_COLOR,
        _GUIDE_THICKNESS,
    )


def _draw_cv2_text(cv2: Any, frame: np.ndarray, message: str) -> None:
    cv2.putText(
        frame,
        message,
        _OVERLAY_TEXT_BASELINE_POSITION,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def _draw_unicode_text(frame: np.ndarray, message: str) -> bool:
    drawn = False
    if not message.isascii():
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            drawn = False
        else:
            font = _load_overlay_font()
            if font is not None:
                rgb_frame = np.ascontiguousarray(frame[:, :, ::-1])
                image = Image.fromarray(rgb_frame)
                draw = ImageDraw.Draw(image)
                draw.text(
                    _OVERLAY_TEXT_POSITION,
                    message,
                    font=font,
                    fill=(255, 255, 255),
                )
                frame[:, :, :] = np.asarray(image)[:, :, ::-1]
                drawn = True
    return drawn


@lru_cache(maxsize=1)
def _load_overlay_font() -> Any | None:
    font = None
    try:
        from PIL import ImageFont
    except ImportError:
        font = None
    else:
        for font_path in _KOREAN_FONT_PATHS:
            if font is None and font_path.exists():
                try:
                    font = ImageFont.truetype(str(font_path), _OVERLAY_FONT_SIZE)
                except OSError:
                    font = None
    result = font
    return result

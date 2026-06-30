from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from oracle_report.models import FaceBox


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        ...


_MIN_SCALED_FACE_SIZE_PX = 24
_HAAR_CASCADE_DIR_ENV = "ORACLE_HAAR_CASCADE_DIR"
_SYSTEM_HAAR_CASCADE_DIRS = (
    Path("/usr/share/opencv4/haarcascades"),
    Path("/usr/share/opencv/haarcascades"),
    Path("/usr/local/share/opencv4/haarcascades"),
    Path("/usr/local/share/opencv/haarcascades"),
)


class HaarFaceDetector:
    def __init__(
        self,
        min_size_px: int = 96,
        detection_scale: float = 0.5,
        detection_interval: int = 2,
    ) -> None:
        self._cv2 = _import_cv2()
        cascade_path = self._resolve_cascade_path()
        self._cascade = self._cv2.CascadeClassifier(str(cascade_path))
        self._min_size_px = min_size_px
        self._detection_scale = detection_scale
        self._detection_interval = detection_interval
        self._frame_index = 0
        self._cached_faces: list[FaceBox] = []
        if self._cascade.empty():
            raise RuntimeError(f"failed to load face cascade: {cascade_path}")
        if detection_scale <= 0.0 or detection_scale > 1.0:
            raise ValueError("detection_scale must be > 0.0 and <= 1.0.")
        if detection_interval <= 0:
            raise ValueError("detection_interval must be greater than 0.")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        should_run_detection = self._frame_index % self._detection_interval == 0
        self._frame_index = self._frame_index + 1
        if should_run_detection:
            self._cached_faces = self._detect_now(frame)
        result = list(self._cached_faces)
        return result

    def _detect_now(self, frame: np.ndarray) -> list[FaceBox]:
        gray = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2GRAY)
        detection_gray = self._resize_for_detection(gray)
        scaled_min_size = max(
            _MIN_SCALED_FACE_SIZE_PX,
            int(round(self._min_size_px * self._detection_scale)),
        )
        faces = self._cascade.detectMultiScale(
            detection_gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(scaled_min_size, scaled_min_size),
        )
        result = [
            _restore_face_box(x, y, w, h, self._detection_scale)
            for (x, y, w, h) in faces
        ]
        result.sort(key=lambda face: face.width * face.height, reverse=True)
        return result

    def _resize_for_detection(self, gray: np.ndarray) -> np.ndarray:
        result = gray
        if self._detection_scale < 1.0:
            result = self._cv2.resize(
                gray,
                (0, 0),
                fx=self._detection_scale,
                fy=self._detection_scale,
                interpolation=self._cv2.INTER_AREA,
            )
        return result

    def _resolve_cascade_path(self) -> Path:
        result = resolve_haar_cascade_path(
            self._cv2,
            "haarcascade_frontalface_default.xml",
        )
        return result


def resolve_haar_cascade_path(cv2_module: Any, filename: str) -> Path:
    result: Path | None = None
    candidates = _haar_cascade_candidates(cv2_module, filename)
    for candidate in candidates:
        if candidate.is_file():
            result = candidate
            break
    if result is None:
        searched = ", ".join(str(candidate) for candidate in candidates)
        raise RuntimeError(
            f"failed to find OpenCV Haar cascade {filename}. "
            f"Install opencv-data or set {_HAAR_CASCADE_DIR_ENV}. "
            f"Searched: {searched}",
        )
    return result


def _haar_cascade_candidates(cv2_module: Any, filename: str) -> tuple[Path, ...]:
    directories: list[Path] = []
    configured_dir = os.getenv(_HAAR_CASCADE_DIR_ENV)
    if configured_dir is not None and configured_dir.strip() != "":
        directories.append(Path(configured_dir.strip()))
    cv2_data = getattr(cv2_module, "data", None)
    cv2_haar_dir = getattr(cv2_data, "haarcascades", "")
    if isinstance(cv2_haar_dir, str) and cv2_haar_dir.strip() != "":
        directories.append(Path(cv2_haar_dir))
    directories.extend(_SYSTEM_HAAR_CASCADE_DIRS)
    result = tuple(directory / filename for directory in directories)
    return result


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "OpenCV is required for camera capture. Install python3-opencv on "
            "Raspberry Pi or pip install -e '.[camera]'.",
        ) from exc
    try:
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except Exception:
        pass
    result = cv2
    return result


def _restore_face_box(
    x: int,
    y: int,
    width: int,
    height: int,
    detection_scale: float,
) -> FaceBox:
    result = FaceBox(
        int(round(x / detection_scale)),
        int(round(y / detection_scale)),
        int(round(width / detection_scale)),
        int(round(height / detection_scale)),
        1.0,
    )
    return result

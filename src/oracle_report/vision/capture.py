from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from oracle_report.models import CaptureArtifact, CaptureDecision, FaceBox, FaceQuality
from oracle_report.vision.clock import Clock, SystemClock
from oracle_report.vision.detection import FaceDetector
from oracle_report.vision.framing import evaluate_face_framing
from oracle_report.vision.quality import FaceQualityAnalyzer


class FaceCaptureHarness:
    def __init__(
        self,
        detector: FaceDetector,
        quality_analyzer: FaceQualityAnalyzer,
        min_face_seconds: float = 2.0,
        face_min_size_px: int = 96,
        clock: Clock | None = None,
    ) -> None:
        self._detector = detector
        self._quality_analyzer = quality_analyzer
        self._min_face_seconds = min_face_seconds
        self._face_min_size_px = face_min_size_px
        self._clock = clock or SystemClock()
        self._tracking_since: float | None = None
        self._last_face: FaceBox | None = None

    def observe(self, frame: np.ndarray) -> CaptureDecision:
        now = self._clock.monotonic()
        faces = self._usable_faces(self._detector.detect(frame))
        state = "searching"
        elapsed = 0.0
        face: FaceBox | None = None
        quality: FaceQuality | None = None
        should_capture = False
        message = "정면 얼굴을 카메라 중앙에 맞춰 주세요."

        if len(faces) == 1:
            face = faces[0]
            framing = evaluate_face_framing(face, frame.shape[1], frame.shape[0])
            if not framing.ready:
                state = "warning"
                message = framing.warning
                elapsed = 0.0
                self._reset_tracking()
            else:
                self._start_or_continue_tracking(now, face)
                elapsed = self._tracked_elapsed(now)
                state = "tracking"
                quality = self._quality_analyzer.analyze(frame, face)
                if quality.ready:
                    message = (
                        f"correct - 촬영 조건이 좋습니다: "
                        f"{elapsed:.1f}/{self._min_face_seconds:.1f}s"
                    )
                    if elapsed >= self._min_face_seconds:
                        state = "captured"
                        should_capture = True
                        message = "correct - 캡처 조건을 만족했습니다."
                else:
                    state = "warning"
                    message = " ".join(quality.warnings)
                    elapsed = 0.0
                    self._reset_tracking()
        else:
            if len(faces) > 1:
                state = "warning"
                message = "한 명만 카메라 앞에 서 주세요."
            self._reset_tracking()

        result = CaptureDecision(
            state=state,
            elapsed_seconds=elapsed,
            face=face,
            quality=quality,
            should_capture=should_capture,
            message=message,
            landmark_points=() if quality is None else quality.landmark_points,
            face_analysis="" if quality is None else quality.face_analysis,
        )
        return result

    def _usable_faces(self, faces: list[FaceBox]) -> list[FaceBox]:
        result = [
            face
            for face in faces
            if face.width >= self._face_min_size_px
            and face.height >= self._face_min_size_px
        ]
        return result

    def _start_or_continue_tracking(self, now: float, face: FaceBox) -> None:
        if self._tracking_since is None or not _faces_overlap(self._last_face, face):
            self._tracking_since = now
        self._last_face = face

    def _tracked_elapsed(self, now: float) -> float:
        result = 0.0
        if self._tracking_since is not None:
            result = max(0.0, now - self._tracking_since)
        return result

    def _reset_tracking(self) -> None:
        self._tracking_since = None
        self._last_face = None


def save_capture_artifact(
    frame: np.ndarray,
    decision: CaptureDecision,
    output_dir: Path,
    filename: str = "capture.jpg",
) -> CaptureArtifact:
    cv2 = _import_cv2_for_capture()
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / filename
    ok = cv2.imwrite(str(image_path), frame)
    if not ok:
        raise RuntimeError(f"failed to write capture image: {image_path}")
    if decision.face is None or decision.quality is None:
        raise ValueError("capture decision must include face and quality")
    result = CaptureArtifact(
        image_path=image_path,
        face=decision.face,
        captured_at=datetime.now(),
        quality=decision.quality,
        landmark_points=decision.landmark_points,
        face_analysis=decision.face_analysis,
    )
    return result


def _faces_overlap(previous: FaceBox | None, current: FaceBox) -> bool:
    result = False
    if previous is not None:
        prev_area = previous.width * previous.height
        curr_area = current.width * current.height
        x0 = max(previous.x, current.x)
        y0 = max(previous.y, current.y)
        x1 = min(previous.x + previous.width, current.x + current.width)
        y1 = min(previous.y + previous.height, current.y + current.height)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        smaller_area = max(1, min(prev_area, curr_area))
        result = intersection / smaller_area >= 0.35
    return result


def _import_cv2_for_capture():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to save capture artifacts.") from exc
    result = cv2
    return result

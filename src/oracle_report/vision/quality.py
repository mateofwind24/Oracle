from __future__ import annotations

from typing import Any, Protocol

import numpy as np

from oracle_report.models import FaceBox, FaceQuality
from oracle_report.vision.detection import _import_cv2, resolve_haar_cascade_path


class FaceQualityAnalyzer(Protocol):
    def analyze(self, frame: np.ndarray, face: FaceBox) -> FaceQuality:
        ...


class OpenCvFaceQualityAnalyzer:
    def __init__(
        self,
        eye_min_count: int = 2,
        eyebrow_min_edge_density: float = 0.018,
    ) -> None:
        self._cv2 = _import_cv2()
        self._eye_min_count = eye_min_count
        self._eyebrow_min_edge_density = eyebrow_min_edge_density
        eye_cascade_path = resolve_haar_cascade_path(
            self._cv2,
            "haarcascade_eye_tree_eyeglasses.xml",
        )
        self._eye_cascade = self._cv2.CascadeClassifier(
            str(eye_cascade_path),
        )
        if self._eye_cascade.empty():
            raise RuntimeError(f"failed to load eye cascade: {eye_cascade_path}")

    def analyze(self, frame: np.ndarray, face: FaceBox) -> FaceQuality:
        face_roi = _crop(frame, face)
        gray = self._cv2.cvtColor(face_roi, self._cv2.COLOR_BGR2GRAY)
        eye_count = self._count_open_eye_candidates(gray)
        eyebrow_score = self._estimate_eyebrow_edge_density(gray)
        warnings: list[str] = []

        if eye_count < self._eye_min_count:
            warnings.append("눈을 감았거나 눈 영역이 충분히 보이지 않습니다.")
        if eyebrow_score < self._eyebrow_min_edge_density:
            warnings.append("눈썹이 가려졌거나 조명이 약해 눈썹 윤곽이 부족합니다.")

        result = FaceQuality(
            ready=len(warnings) == 0,
            warnings=tuple(warnings),
            eye_count=eye_count,
            eyebrow_score=eyebrow_score,
            frontality_score=1.0,
            occlusion_score=1.0,
        )
        return result

    def _count_open_eye_candidates(self, gray_face: np.ndarray) -> int:
        height = gray_face.shape[0]
        upper = gray_face[: int(height * 0.62), :]
        eyes = self._eye_cascade.detectMultiScale(
            upper,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(18, 18),
        )
        result = int(len(eyes))
        return result

    def _estimate_eyebrow_edge_density(self, gray_face: np.ndarray) -> float:
        height = gray_face.shape[0]
        width = gray_face.shape[1]
        y0 = max(0, int(height * 0.18))
        y1 = min(height, int(height * 0.42))
        x0 = max(0, int(width * 0.12))
        x1 = min(width, int(width * 0.88))
        brow_band = gray_face[y0:y1, x0:x1]
        edges = self._cv2.Canny(brow_band, 60, 140)
        result = float(np.count_nonzero(edges)) / float(edges.size)
        return result


class MediaPipeFaceQualityAnalyzer:
    def __init__(self, eye_closed_threshold: float = 0.20) -> None:
        self._cv2 = _import_cv2()
        self._mp = self._import_mediapipe()
        self._mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._eye_closed_threshold = eye_closed_threshold

    def analyze(self, frame: np.ndarray, face: FaceBox) -> FaceQuality:
        rgb = self._cv2.cvtColor(frame, self._cv2.COLOR_BGR2RGB)
        output = self._mesh.process(rgb)
        warnings: list[str] = []
        eye_count = 0
        eyebrow_score = 0.0
        if output.multi_face_landmarks:
            landmarks = output.multi_face_landmarks[0].landmark
            left_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
            right_ear = _eye_aspect_ratio(
                landmarks,
                (362, 385, 387, 263, 373, 380),
            )
            eye_count = int(left_ear >= self._eye_closed_threshold) + int(
                right_ear >= self._eye_closed_threshold,
            )
            eyebrow_score = _eyebrow_geometry_score(landmarks)
            if eye_count < 2:
                warnings.append("눈을 감았거나 눈꺼풀 간격이 너무 좁습니다.")
            if eyebrow_score < 0.01:
                warnings.append("눈썹 위치가 얼굴 랜드마크에서 안정적으로 보이지 않습니다.")
        else:
            warnings.append("얼굴 랜드마크를 안정적으로 찾지 못했습니다.")

        result = FaceQuality(
            ready=len(warnings) == 0,
            warnings=tuple(warnings),
            eye_count=eye_count,
            eyebrow_score=eyebrow_score,
            frontality_score=1.0 if len(warnings) == 0 else 0.0,
            occlusion_score=1.0 if len(warnings) == 0 else 0.0,
        )
        return result

    def _import_mediapipe(self) -> Any:
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError(
                "mediapipe is required for landmark quality analysis. "
                "Install pip install -e '.[quality]' or use the OpenCV backend.",
            ) from exc
        if not hasattr(mp, "solutions") or not hasattr(mp.solutions, "face_mesh"):
            version = getattr(mp, "__version__", "unknown")
            raise RuntimeError(
                "installed mediapipe package is missing solutions.face_mesh "
                f"(version: {version}). Reinstall a compatible mediapipe build "
                "or use the OpenCV backend."
            )
        result = mp
        return result


def _crop(frame: np.ndarray, face: FaceBox) -> np.ndarray:
    height = frame.shape[0]
    width = frame.shape[1]
    x0 = max(0, face.x)
    y0 = max(0, face.y)
    x1 = min(width, face.x + face.width)
    y1 = min(height, face.y + face.height)
    result = frame[y0:y1, x0:x1]
    return result


def _eye_aspect_ratio(landmarks: Any, indices: tuple[int, int, int, int, int, int]) -> float:
    p1 = landmarks[indices[0]]
    p2 = landmarks[indices[1]]
    p3 = landmarks[indices[2]]
    p4 = landmarks[indices[3]]
    p5 = landmarks[indices[4]]
    p6 = landmarks[indices[5]]
    vertical_one = _distance(p2, p6)
    vertical_two = _distance(p3, p5)
    horizontal = _distance(p1, p4)
    result = 0.0
    if horizontal > 0.0:
        result = (vertical_one + vertical_two) / (2.0 * horizontal)
    return result


def _eyebrow_geometry_score(landmarks: Any) -> float:
    left_brow = landmarks[105]
    left_eye = landmarks[159]
    right_brow = landmarks[334]
    right_eye = landmarks[386]
    left_gap = abs(left_eye.y - left_brow.y)
    right_gap = abs(right_eye.y - right_brow.y)
    result = (left_gap + right_gap) * 0.5
    return result


def _distance(a: Any, b: Any) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    result = float((dx * dx + dy * dy) ** 0.5)
    return result

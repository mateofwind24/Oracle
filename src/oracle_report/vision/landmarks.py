from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oracle_report.models import FaceBox, FaceQuality
from oracle_report.vision.detection import _import_cv2


_LANDMARK_MODE_NAME = "랜드마크 룰 기반"
_MIN_POSE_SCORE = 0.68
_MIN_OCCLUSION_SCORE = 0.82
_EYE_OPEN_THRESHOLD = 0.18
_KEY_LANDMARK_INDICES = (
    10,
    33,
    61,
    105,
    133,
    152,
    159,
    234,
    263,
    291,
    334,
    362,
    386,
    454,
)
_DRAW_LANDMARK_INDICES = (
    10,
    33,
    61,
    105,
    133,
    145,
    152,
    159,
    234,
    263,
    291,
    334,
    362,
    374,
    386,
    454,
)


@dataclass(frozen=True)
class NormalizedLandmark:
    x: float
    y: float
    z: float
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True)
class LandmarkDetection:
    face: FaceBox | None = None
    points: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    draw_points: tuple[tuple[int, int], ...] = field(default_factory=tuple)
    landmarks: tuple[NormalizedLandmark, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class LandmarkMetrics:
    frontality_score: float
    occlusion_score: float
    eye_count: int
    eyebrow_score: float
    face_aspect_ratio: float
    eye_spacing_ratio: float
    mouth_width_ratio: float
    lower_face_ratio: float
    mouth_corner_delta: float


class MediaPipeLandmarkFaceDetector:
    def __init__(
        self,
        min_size_px: int = 96,
        detection_scale: float = 0.5,
        detection_interval: int = 1,
    ) -> None:
        self._cv2 = _import_cv2()
        self._mp = _import_mediapipe()
        self._mesh = self._mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._min_size_px = min_size_px
        self._detection_scale = detection_scale
        self._detection_interval = detection_interval
        self._frame_index = 0
        self._latest = LandmarkDetection()
        if detection_scale <= 0.0 or detection_scale > 1.0:
            raise ValueError("detection_scale must be > 0.0 and <= 1.0.")
        if detection_interval <= 0:
            raise ValueError("detection_interval must be greater than 0.")

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        should_run_detection = self._frame_index % self._detection_interval == 0
        self._frame_index = self._frame_index + 1
        if should_run_detection:
            self._latest = self._detect_now(frame)
        result = [] if self._latest.face is None else [self._latest.face]
        return result

    def latest_detection(self) -> LandmarkDetection:
        result = self._latest
        return result

    def _detect_now(self, frame: np.ndarray) -> LandmarkDetection:
        height = frame.shape[0]
        width = frame.shape[1]
        model_frame = self._resize_for_detection(frame)
        rgb = self._cv2.cvtColor(model_frame, self._cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        output = self._mesh.process(rgb)
        result = LandmarkDetection()
        if output.multi_face_landmarks:
            landmarks = _normalize_landmarks(output.multi_face_landmarks[0].landmark)
            points = _landmarks_to_points(landmarks, width, height)
            face = _face_box_from_points(points, width, height)
            if face.width >= self._min_size_px and face.height >= self._min_size_px:
                result = LandmarkDetection(
                    face=face,
                    points=points,
                    draw_points=_select_draw_points(points),
                    landmarks=landmarks,
                )
        return result

    def _resize_for_detection(self, frame: np.ndarray) -> np.ndarray:
        result = frame
        if self._detection_scale < 1.0:
            result = self._cv2.resize(
                frame,
                (0, 0),
                fx=self._detection_scale,
                fy=self._detection_scale,
                interpolation=self._cv2.INTER_AREA,
            )
        return result


class MediaPipeLandmarkQualityAnalyzer:
    def __init__(self, detector: MediaPipeLandmarkFaceDetector) -> None:
        self._detector = detector

    def analyze(self, frame: np.ndarray, face: FaceBox) -> FaceQuality:
        detection = self._detector.latest_detection()
        warnings: list[str] = []
        metrics = _empty_metrics()
        face_analysis = ""
        if detection.face is None or not detection.landmarks:
            warnings.append("얼굴 랜드마크를 안정적으로 찾지 못했습니다.")
        else:
            metrics = _compute_landmark_metrics(detection.landmarks)
            if metrics.frontality_score < _MIN_POSE_SCORE:
                warnings.append("정면을 바라봐 주세요.")
            if metrics.occlusion_score < _MIN_OCCLUSION_SCORE:
                warnings.append("얼굴을 가리는 물체를 치워 주세요.")
            if metrics.eye_count < 2:
                warnings.append("눈을 뜨고 카메라를 정면으로 봐 주세요.")
            face_analysis = build_rule_based_face_analysis(metrics)

        result = FaceQuality(
            ready=len(warnings) == 0,
            warnings=tuple(warnings),
            eye_count=metrics.eye_count,
            eyebrow_score=metrics.eyebrow_score,
            frontality_score=metrics.frontality_score,
            occlusion_score=metrics.occlusion_score,
            landmark_points=detection.draw_points,
            face_analysis=face_analysis,
        )
        return result


def build_rule_based_face_analysis(metrics: LandmarkMetrics) -> str:
    face_shape = _face_shape_label(metrics.face_aspect_ratio)
    eye_spacing = _eye_spacing_label(metrics.eye_spacing_ratio)
    brow_label = _brow_label(metrics.eyebrow_score)
    mouth_label = _mouth_label(metrics.mouth_corner_delta)
    lower_face = _lower_face_label(metrics.lower_face_ratio)
    tags = ", ".join((face_shape, eye_spacing, brow_label, mouth_label))
    result = f"""
## 관상정보
- 분석 모드: {_LANDMARK_MODE_NAME}
- 얼굴 인상 태그: {tags}
- 비율 지표: 얼굴 세로/가로 {metrics.face_aspect_ratio:.2f}, 눈 사이 간격 {metrics.eye_spacing_ratio:.2f}, 입 너비 {metrics.mouth_width_ratio:.2f}, 하관 비율 {metrics.lower_face_ratio:.2f}
- 눈/눈썹 관찰: {eye_spacing}, {brow_label}
- 윤곽/표정 관찰: {face_shape}, {lower_face}, {mouth_label}
- 리포트에 넣을 보조 해석: 얼굴 비율이 {face_shape} 쪽으로 보이고, 눈썹과 눈 간격은 {brow_label} 인상을 줍니다. 표정은 {mouth_label} 흐름으로 관찰됩니다.
- 캡처 신뢰도: 정면 점수 {metrics.frontality_score:.2f}, 가림 추정 점수 {metrics.occlusion_score:.2f}
- 주의 문구: 랜드마크 비율 기반의 엔터테인먼트 보조 정보이며 실제 성격, 건강, 신원, 능력을 판단하지 않습니다.
""".strip()
    return result


def _compute_landmark_metrics(
    landmarks: tuple[NormalizedLandmark, ...],
) -> LandmarkMetrics:
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_eye_inner = landmarks[133]
    right_eye_inner = landmarks[362]
    nose_tip = landmarks[1]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    chin = landmarks[152]
    forehead = landmarks[10]
    left_face = landmarks[234]
    right_face = landmarks[454]
    face_width = max(0.001, _distance(left_face, right_face))
    face_height = max(0.001, _distance(forehead, chin))
    eye_line = max(0.001, _distance(left_eye, right_eye))
    eye_y_delta = abs(left_eye.y - right_eye.y) / face_height
    nose_center_delta = abs(nose_tip.x - ((left_eye.x + right_eye.x) * 0.5)) / eye_line
    mouth_y_delta = abs(left_mouth.y - right_mouth.y) / face_height
    frontality = (
        _score_from_delta(eye_y_delta, 0.045)
        + _score_from_delta(nose_center_delta, 0.22)
        + _score_from_delta(mouth_y_delta, 0.045)
    ) / 3.0
    occlusion = _landmark_visibility_score(landmarks)
    left_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
    right_ear = _eye_aspect_ratio(landmarks, (362, 385, 387, 263, 373, 380))
    eye_count = int(left_ear >= _EYE_OPEN_THRESHOLD) + int(
        right_ear >= _EYE_OPEN_THRESHOLD,
    )
    eyebrow_score = _eyebrow_geometry_score(landmarks)
    result = LandmarkMetrics(
        frontality_score=frontality,
        occlusion_score=occlusion,
        eye_count=eye_count,
        eyebrow_score=eyebrow_score,
        face_aspect_ratio=face_height / face_width,
        eye_spacing_ratio=_distance(left_eye_inner, right_eye_inner) / face_width,
        mouth_width_ratio=_distance(left_mouth, right_mouth) / face_width,
        lower_face_ratio=abs(chin.y - nose_tip.y) / face_height,
        mouth_corner_delta=right_mouth.y - left_mouth.y,
    )
    return result


def _empty_metrics() -> LandmarkMetrics:
    result = LandmarkMetrics(
        frontality_score=0.0,
        occlusion_score=0.0,
        eye_count=0,
        eyebrow_score=0.0,
        face_aspect_ratio=0.0,
        eye_spacing_ratio=0.0,
        mouth_width_ratio=0.0,
        lower_face_ratio=0.0,
        mouth_corner_delta=0.0,
    )
    return result


def _normalize_landmarks(landmarks: Any) -> tuple[NormalizedLandmark, ...]:
    result = tuple(
        NormalizedLandmark(
            x=float(item.x),
            y=float(item.y),
            z=float(item.z),
            visibility=float(getattr(item, "visibility", 1.0)),
            presence=float(getattr(item, "presence", 1.0)),
        )
        for item in landmarks
    )
    return result


def _landmarks_to_points(
    landmarks: tuple[NormalizedLandmark, ...],
    width: int,
    height: int,
) -> tuple[tuple[int, int], ...]:
    result = tuple(
        (
            int(round(_clamp(item.x, 0.0, 1.0) * float(width - 1))),
            int(round(_clamp(item.y, 0.0, 1.0) * float(height - 1))),
        )
        for item in landmarks
    )
    return result


def _face_box_from_points(
    points: tuple[tuple[int, int], ...],
    width: int,
    height: int,
) -> FaceBox:
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x0 = max(0, min(xs))
    y0 = max(0, min(ys))
    x1 = min(width - 1, max(xs))
    y1 = min(height - 1, max(ys))
    result = FaceBox(
        x=x0,
        y=y0,
        width=max(1, x1 - x0),
        height=max(1, y1 - y0),
        confidence=1.0,
    )
    return result


def _select_draw_points(points: tuple[tuple[int, int], ...]) -> tuple[tuple[int, int], ...]:
    result = tuple(points[index] for index in _DRAW_LANDMARK_INDICES if index < len(points))
    return result


def _landmark_visibility_score(landmarks: tuple[NormalizedLandmark, ...]) -> float:
    visible_count = 0
    total_count = 0
    for index in _KEY_LANDMARK_INDICES:
        if index < len(landmarks):
            total_count = total_count + 1
            item = landmarks[index]
            inside_frame = 0.0 <= item.x <= 1.0 and 0.0 <= item.y <= 1.0
            confident = item.visibility >= 0.35 and item.presence >= 0.35
            if inside_frame and confident:
                visible_count = visible_count + 1
    result = 0.0
    if total_count > 0:
        result = float(visible_count) / float(total_count)
    return result


def _eye_aspect_ratio(
    landmarks: tuple[NormalizedLandmark, ...],
    indices: tuple[int, int, int, int, int, int],
) -> float:
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


def _eyebrow_geometry_score(landmarks: tuple[NormalizedLandmark, ...]) -> float:
    left_brow = landmarks[105]
    left_eye = landmarks[159]
    right_brow = landmarks[334]
    right_eye = landmarks[386]
    left_gap = abs(left_eye.y - left_brow.y)
    right_gap = abs(right_eye.y - right_brow.y)
    result = (left_gap + right_gap) * 0.5
    return result


def _distance(a: NormalizedLandmark, b: NormalizedLandmark) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    result = float((dx * dx + dy * dy) ** 0.5)
    return result


def _score_from_delta(delta: float, tolerance: float) -> float:
    result = _clamp(1.0 - (delta / tolerance), 0.0, 1.0)
    return result


def _clamp(value: float, low: float, high: float) -> float:
    result = max(low, min(high, value))
    return result


def _face_shape_label(face_aspect_ratio: float) -> str:
    result = "균형형 윤곽"
    if face_aspect_ratio >= 1.45:
        result = "세로로 긴 윤곽"
    elif face_aspect_ratio <= 1.18:
        result = "가로 안정감이 있는 윤곽"
    return result


def _eye_spacing_label(eye_spacing_ratio: float) -> str:
    result = "눈 사이 간격이 균형적입니다"
    if eye_spacing_ratio >= 0.34:
        result = "눈 사이 간격이 넓은 편입니다"
    elif eye_spacing_ratio <= 0.24:
        result = "눈 사이 간격이 좁은 편입니다"
    return result


def _brow_label(eyebrow_score: float) -> str:
    result = "눈썹과 눈 사이가 안정적인 편입니다"
    if eyebrow_score >= 0.075:
        result = "눈썹과 눈 사이가 여유 있는 편입니다"
    elif eyebrow_score <= 0.045:
        result = "눈썹과 눈 사이가 가까운 편입니다"
    return result


def _mouth_label(mouth_corner_delta: float) -> str:
    result = "입꼬리 균형이 안정적입니다"
    if mouth_corner_delta >= 0.018:
        result = "입꼬리가 한쪽으로 기울어 보입니다"
    elif mouth_corner_delta <= -0.018:
        result = "입꼬리가 한쪽으로 기울어 보입니다"
    return result


def _lower_face_label(lower_face_ratio: float) -> str:
    result = "하관 비율이 균형적입니다"
    if lower_face_ratio >= 0.46:
        result = "하관 비율이 긴 편입니다"
    elif lower_face_ratio <= 0.34:
        result = "하관 비율이 짧은 편입니다"
    return result


def _import_mediapipe() -> Any:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError(
            "mediapipe is required for face analysis mode 2. "
            "Install pip install -e '.[quality]' or use mode 1.",
        ) from exc
    result = mp
    return result

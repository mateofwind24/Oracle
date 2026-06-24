from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

import numpy as np

from oracle_report.models import FaceBox, FaceQuality
from oracle_report.vision.detection import _import_cv2
from oracle_report.vision.physiognomy_rule_repository import (
    PhysiognomyRuleMatch,
    PhysiognomyRuleRepository,
)


_LANDMARK_MODE_NAME = "랜드마크 룰 기반"
_MIN_POSE_SCORE = 0.50
_MIN_OCCLUSION_SCORE = 0.70
_FRONT_EYE_LEVEL_TOLERANCE = 0.09
_FRONT_NOSE_CENTER_TOLERANCE = 0.35
_FRONT_MOUTH_LEVEL_TOLERANCE = 0.09
_EYE_OPEN_THRESHOLD = 0.18
_KEY_LANDMARK_INDICES = (
    2,
    10,
    13,
    33,
    61,
    70,
    98,
    105,
    133,
    152,
    172,
    159,
    234,
    263,
    291,
    300,
    327,
    334,
    362,
    386,
    397,
    454,
)
_DRAW_LANDMARK_INDICES = (
    2,
    10,
    13,
    33,
    61,
    70,
    98,
    105,
    133,
    145,
    152,
    159,
    172,
    234,
    263,
    291,
    300,
    327,
    334,
    362,
    374,
    386,
    397,
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
    upper_zone_ratio: float
    middle_zone_ratio: float
    lower_zone_ratio: float
    third_balance_error: float
    brow_eye_span_ratio: float
    brow_eye_gap_ratio: float
    nose_width_ratio: float
    philtrum_chin_ratio: float
    jaw_width_ratio: float
    mouth_balance_delta: float


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
    repository = _physiognomy_rule_repository()
    matches = _evaluate_physio_rules(metrics, repository)
    tags = _format_rule_tags(matches)
    detail_lines = _format_rule_details(matches)
    auxiliary_text = _format_auxiliary_interpretation(matches)
    unsupported = ", ".join(repository.unsupported_features())
    safety_note = repository.safety_note()
    result = f"""
## 관상정보
- 분석 모드: {_LANDMARK_MODE_NAME}
- 참고 기준: 삼정, 오관, 십이궁, 얼굴형, 하관 비율을 랜드마크로 측정 가능한 항목에 맞춰 변환
- 주요 태그: {tags}
- 비율 지표: 삼정 상/중/하 {metrics.upper_zone_ratio:.2f}/{metrics.middle_zone_ratio:.2f}/{metrics.lower_zone_ratio:.2f}, 삼정 편차 {metrics.third_balance_error:.2f}, 얼굴 세로/가로 {metrics.face_aspect_ratio:.2f}, 미간 {metrics.eye_spacing_ratio:.2f}, 눈썹/눈 폭 {metrics.brow_eye_span_ratio:.2f}, 코 폭 {metrics.nose_width_ratio:.2f}, 입 폭 {metrics.mouth_width_ratio:.2f}, 하관 폭 {metrics.jaw_width_ratio:.2f}
- 세부 관찰:
{detail_lines}
- 리포트에 넣을 보조 해석: {auxiliary_text}
- 적용 제외 기준: {unsupported}은 현재 랜드마크만으로 안정 측정하기 어려워 룰에 넣지 않았습니다.
- 캡처 신뢰도: 정면 점수 {metrics.frontality_score:.2f}, 가림 추정 점수 {metrics.occlusion_score:.2f}
- 주의 문구: {safety_note}
""".strip()
    return result


@lru_cache(maxsize=1)
def _physiognomy_rule_repository() -> PhysiognomyRuleRepository:
    result = PhysiognomyRuleRepository()
    return result


def _evaluate_physio_rules(
    metrics: LandmarkMetrics,
    repository: PhysiognomyRuleRepository,
) -> tuple[PhysiognomyRuleMatch, ...]:
    result = repository.lookup_many(
        {
            "third_balance_error": metrics.third_balance_error,
            "upper_zone_ratio": metrics.upper_zone_ratio,
            "middle_zone_ratio": metrics.middle_zone_ratio,
            "lower_zone_ratio": metrics.lower_zone_ratio,
            "face_aspect_ratio": metrics.face_aspect_ratio,
            "eye_spacing_ratio": metrics.eye_spacing_ratio,
            "brow_eye_span_ratio": metrics.brow_eye_span_ratio,
            "brow_eye_gap_ratio": metrics.brow_eye_gap_ratio,
            "nose_width_ratio": metrics.nose_width_ratio,
            "mouth_width_ratio": metrics.mouth_width_ratio,
            "philtrum_chin_ratio": metrics.philtrum_chin_ratio,
            "jaw_width_ratio": metrics.jaw_width_ratio,
            "mouth_balance_delta": metrics.mouth_balance_delta,
        },
    )
    return result


def _format_rule_tags(matches: tuple[PhysiognomyRuleMatch, ...]) -> str:
    result = ", ".join(match.tag for match in matches[:8])
    return result


def _format_rule_details(matches: tuple[PhysiognomyRuleMatch, ...]) -> str:
    result = "\n".join(
        (
            f"  - {match.title}: {match.observation} "
            f"({match.basis}, 측정값 {match.value:.2f})"
        )
        for match in matches
    )
    return result


def _format_auxiliary_interpretation(
    matches: tuple[PhysiognomyRuleMatch, ...],
) -> str:
    selected = matches[:7]
    result = " ".join(match.interpretation for match in selected)
    return result


def _compute_landmark_metrics(
    landmarks: tuple[NormalizedLandmark, ...],
) -> LandmarkMetrics:
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    left_eye_inner = landmarks[133]
    right_eye_inner = landmarks[362]
    nose_tip = landmarks[1]
    nose_base = landmarks[2]
    upper_lip = landmarks[13]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]
    chin = landmarks[152]
    forehead = landmarks[10]
    left_face = landmarks[234]
    right_face = landmarks[454]
    left_brow_outer = landmarks[70]
    right_brow_outer = landmarks[300]
    left_brow = landmarks[105]
    right_brow = landmarks[334]
    left_nose = landmarks[98]
    right_nose = landmarks[327]
    left_jaw = landmarks[172]
    right_jaw = landmarks[397]
    face_width = max(0.001, _distance(left_face, right_face))
    face_height = max(0.001, abs(chin.y - forehead.y))
    eye_line = max(0.001, _distance(left_eye, right_eye))
    eye_y_delta = abs(left_eye.y - right_eye.y) / face_height
    nose_center_delta = abs(nose_tip.x - ((left_eye.x + right_eye.x) * 0.5)) / eye_line
    mouth_y_delta = abs(left_mouth.y - right_mouth.y) / face_height
    brow_y = (left_brow.y + right_brow.y) * 0.5
    upper_zone_ratio = abs(brow_y - forehead.y) / face_height
    middle_zone_ratio = abs(nose_base.y - brow_y) / face_height
    lower_zone_ratio = abs(chin.y - nose_base.y) / face_height
    third_balance_error = max(
        abs(upper_zone_ratio - (1.0 / 3.0)),
        abs(middle_zone_ratio - (1.0 / 3.0)),
        abs(lower_zone_ratio - (1.0 / 3.0)),
    )
    eyebrow_gap_ratio = _eyebrow_geometry_score(landmarks) / face_height
    brow_span = max(0.001, abs(right_brow_outer.x - left_brow_outer.x))
    eye_span = max(0.001, abs(right_eye.x - left_eye.x))
    lower_height = max(0.001, abs(chin.y - nose_base.y))
    frontality = (
        _score_from_delta(eye_y_delta, _FRONT_EYE_LEVEL_TOLERANCE)
        + _score_from_delta(nose_center_delta, _FRONT_NOSE_CENTER_TOLERANCE)
        + _score_from_delta(mouth_y_delta, _FRONT_MOUTH_LEVEL_TOLERANCE)
    ) / 3.0
    occlusion = _landmark_visibility_score(landmarks)
    left_ear = _eye_aspect_ratio(landmarks, (33, 160, 158, 133, 153, 144))
    right_ear = _eye_aspect_ratio(landmarks, (362, 385, 387, 263, 373, 380))
    eye_count = int(left_ear >= _EYE_OPEN_THRESHOLD) + int(
        right_ear >= _EYE_OPEN_THRESHOLD,
    )
    result = LandmarkMetrics(
        frontality_score=frontality,
        occlusion_score=occlusion,
        eye_count=eye_count,
        eyebrow_score=eyebrow_gap_ratio,
        face_aspect_ratio=face_height / face_width,
        eye_spacing_ratio=_distance(left_eye_inner, right_eye_inner) / face_width,
        mouth_width_ratio=_distance(left_mouth, right_mouth) / face_width,
        lower_face_ratio=lower_zone_ratio,
        mouth_corner_delta=right_mouth.y - left_mouth.y,
        upper_zone_ratio=upper_zone_ratio,
        middle_zone_ratio=middle_zone_ratio,
        lower_zone_ratio=lower_zone_ratio,
        third_balance_error=third_balance_error,
        brow_eye_span_ratio=brow_span / eye_span,
        brow_eye_gap_ratio=eyebrow_gap_ratio,
        nose_width_ratio=_distance(left_nose, right_nose) / face_width,
        philtrum_chin_ratio=abs(upper_lip.y - nose_base.y) / lower_height,
        jaw_width_ratio=_distance(left_jaw, right_jaw) / face_width,
        mouth_balance_delta=abs(right_mouth.y - left_mouth.y) / face_height,
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
        upper_zone_ratio=0.0,
        middle_zone_ratio=0.0,
        lower_zone_ratio=0.0,
        third_balance_error=0.0,
        brow_eye_span_ratio=0.0,
        brow_eye_gap_ratio=0.0,
        nose_width_ratio=0.0,
        philtrum_chin_ratio=0.0,
        jaw_width_ratio=0.0,
        mouth_balance_delta=0.0,
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

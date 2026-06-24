from __future__ import annotations

import sqlite3
from pathlib import Path

from oracle_report.vision.landmarks import (
    _FRONT_EYE_LEVEL_TOLERANCE,
    _FRONT_MOUTH_LEVEL_TOLERANCE,
    _FRONT_NOSE_CENTER_TOLERANCE,
    _MIN_POSE_SCORE,
    _landmark_geometry_score,
    _score_from_delta,
    LandmarkMetrics,
    NormalizedLandmark,
    build_rule_based_face_analysis,
)
from oracle_report.vision.physiognomy_rule_data import (
    PHYSIOGNOMY_RULES,
    RULE_SOURCE_REFERENCES,
)
from oracle_report.vision.physiognomy_rule_repository import (
    PhysiognomyRuleRepository,
    build_physio_rule_database,
)


def test_rule_data_is_sourced() -> None:
    source_ids = {source.id for source in RULE_SOURCE_REFERENCES}

    assert len(PHYSIOGNOMY_RULES) >= 10
    assert "gimpo_three_zones" in source_ids
    assert "encykorea_face" in source_ids
    assert all(source.url.startswith("https://") for source in RULE_SOURCE_REFERENCES)
    assert all(rule.source_ids for rule in PHYSIOGNOMY_RULES)
    assert all(
        source_id in source_ids
        for rule in PHYSIOGNOMY_RULES
        for source_id in rule.source_ids
    )


def test_rule_database_queries_ranges_by_ratio(tmp_path: Path) -> None:
    db_path = tmp_path / "physiognomy_rules.sqlite"

    built = build_physio_rule_database(db_path)
    skipped = build_physio_rule_database(db_path)
    repository = PhysiognomyRuleRepository(db_path)
    match = repository.lookup("eye_spacing_ratio", 0.28)

    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT ranges.tag
            FROM physiognomy_rules AS rules
            JOIN physiognomy_rule_ranges AS ranges
              ON ranges.rule_id = rules.id
            WHERE rules.metric = ?
              AND (ranges.min_value IS NULL OR ? >= ranges.min_value)
              AND (ranges.max_value IS NULL OR ? < ranges.max_value)
            """,
            ("eye_spacing_ratio", 0.28, 0.28),
        ).fetchone()

    assert built is True
    assert skipped is False
    assert match is not None
    assert match.tag == "미간 균형형"
    assert row is not None
    assert row[0] == "미간 균형형"


def test_frontality_threshold_allows_slight_head_roll() -> None:
    eye_y_delta = 0.045
    nose_center_delta = 0.16
    mouth_y_delta = 0.045

    score = (
        _score_from_delta(eye_y_delta, _FRONT_EYE_LEVEL_TOLERANCE)
        + _score_from_delta(nose_center_delta, _FRONT_NOSE_CENTER_TOLERANCE)
        + _score_from_delta(mouth_y_delta, _FRONT_MOUTH_LEVEL_TOLERANCE)
    ) / 3.0

    assert score >= _MIN_POSE_SCORE


def test_landmark_geometry_score_ignores_missing_visibility_fields() -> None:
    landmarks = _build_centered_face_landmarks(visibility=0.0, presence=0.0)

    score = _landmark_geometry_score(landmarks)

    assert score > 0.99


def test_rule_based_face_analysis_includes_auxiliary_interpretation() -> None:
    metrics = LandmarkMetrics(
        frontality_score=0.91,
        occlusion_score=0.94,
        eye_count=2,
        eyebrow_score=0.08,
        face_aspect_ratio=1.32,
        eye_spacing_ratio=0.28,
        mouth_width_ratio=0.36,
        lower_face_ratio=0.33,
        mouth_corner_delta=0.0,
        upper_zone_ratio=0.33,
        middle_zone_ratio=0.34,
        lower_zone_ratio=0.33,
        third_balance_error=0.01,
        brow_eye_span_ratio=1.08,
        brow_eye_gap_ratio=0.08,
        nose_width_ratio=0.19,
        philtrum_chin_ratio=0.26,
        jaw_width_ratio=0.66,
        mouth_balance_delta=0.01,
    )

    result = build_rule_based_face_analysis(metrics)

    assert "랜드마크 룰 기반" in result
    assert "삼정" in result
    assert "비율 지표" in result
    assert "세부 관찰" in result
    assert "리포트에 넣을 보조 해석" in result
    assert "적용 제외 기준" in result
    assert "엔터테인먼트 보조 정보" in result


def _build_centered_face_landmarks(
    visibility: float = 1.0,
    presence: float = 1.0,
) -> tuple[NormalizedLandmark, ...]:
    items = [
        NormalizedLandmark(0.50, 0.50, 0.0, visibility, presence)
        for _ in range(468)
    ]
    points = {
        1: (0.50, 0.50),
        2: (0.50, 0.58),
        10: (0.50, 0.20),
        13: (0.50, 0.62),
        33: (0.36, 0.42),
        61: (0.42, 0.65),
        70: (0.34, 0.36),
        98: (0.45, 0.55),
        105: (0.40, 0.36),
        133: (0.45, 0.42),
        145: (0.40, 0.45),
        152: (0.50, 0.82),
        159: (0.40, 0.39),
        172: (0.34, 0.76),
        234: (0.28, 0.50),
        263: (0.64, 0.42),
        291: (0.58, 0.65),
        300: (0.66, 0.36),
        327: (0.55, 0.55),
        334: (0.60, 0.36),
        362: (0.55, 0.42),
        374: (0.60, 0.45),
        386: (0.60, 0.39),
        397: (0.66, 0.76),
        454: (0.72, 0.50),
    }
    for index, point in points.items():
        items[index] = NormalizedLandmark(
            point[0],
            point[1],
            0.0,
            visibility,
            presence,
        )
    result = tuple(items)
    return result

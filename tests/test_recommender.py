from __future__ import annotations

from datetime import datetime
from pathlib import Path

from oracle_report.recommender import recommend_faces
from oracle_report.saju.engine import build_saju_reading


def test_recommend_faces_normalizes_english_target_gender(tmp_path: Path) -> None:
    reading = build_saju_reading(datetime(1995, 3, 15, 14, 30))

    korean_results = recommend_faces(tmp_path / "faces.sqlite", "여성", reading)
    english_results = recommend_faces(tmp_path / "faces.sqlite", "female", reading)

    assert english_results
    assert english_results[0].target_gender == korean_results[0].target_gender
    assert english_results[0].score == korean_results[0].score

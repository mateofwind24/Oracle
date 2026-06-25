from __future__ import annotations

from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.report_html import (
    render_compatibility_report_html,
    render_personal_report_html,
)
from oracle_report.saju.repository import ManseRepository, build_manse_database


def test_report_html_fallback_face_blocks_explain_terms(tmp_path: Path) -> None:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    manse_lookup = ManseRepository(db_path).lookup(profile)

    html = render_personal_report_html(
        profile,
        manse_lookup,
        "not-json-face-analysis",
        (),
        "not-json",
    )

    assert "관상 정보는 보조 해석으로만 사용합니다" not in html
    assert "사주 결과와 충돌" not in html
    assert "표정과 눈썹 흐름" not in html
    assert "눈은 시선의 또렷함" in html
    assert "삼정은 얼굴을 위" in html


def test_compatibility_report_html_uses_structured_layout(tmp_path: Path) -> None:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1997)
    repository = ManseRepository(db_path)
    left = BirthProfile(
        name="left",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    right = BirthProfile(
        name="right",
        birth_datetime=datetime(1997, 5, 20, 12, 0),
        gender="여성",
    )

    html = render_compatibility_report_html(
        left,
        right,
        "연인",
        repository.lookup(left),
        repository.lookup(right),
        "얼굴 관찰 fixture",
        '{"essence":"궁합 핵심","action_title":"행동 제목","action_body":"행동 본문"}',
    )

    assert html.startswith("<!DOCTYPE html>")
    assert "oracle-report compatibility-report" in html
    assert "궁합 핵심" in html
    assert "행동 제목" in html
    assert "left 님과 right 님" in html

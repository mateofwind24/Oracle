from __future__ import annotations

import json
from datetime import datetime

from oracle_report.models import BirthProfile
from oracle_report.recommender import FaceRecommendation
from oracle_report.report_html import (
    render_compatibility_report_html,
    render_personal_report_html,
)
from oracle_report.saju.repository import ManseRepository


def test_report_html_fallback_face_blocks_explain_terms() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    manse_lookup = ManseRepository().lookup(profile)

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


def test_personal_saju_only_report_hides_face_recommendations() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    recommendation = FaceRecommendation(
        display_name="추천 후보",
        image_path=None,
        target_gender="여성",
        face_tags=("밝은 표정",),
        saju_tags=("목",),
        reason="보완 설명",
        score=9,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        (recommendation,),
        '{"essence":"사주 핵심","saju_blocks":[]}',
        skip_face=True,
    )

    assert "Oracle · 사주 리포트" in html
    assert "FACE MATCH" not in html
    assert "궁합 좋은 얼굴 추천" not in html
    assert "추천 후보" not in html
    assert "관상 — 얼굴이 말하는 것" not in html


def test_compatibility_report_html_uses_structured_layout() -> None:
    repository = ManseRepository()
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


def test_report_profile_shows_unknown_birth_time_without_helper_basis() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
        birth_time_known=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        (),
        '{"essence":"사주 핵심"}',
        skip_face=True,
    )

    assert "시간 미상" in html
    assert "오시(午時) 보조 기준" not in html


def test_report_html_uses_auto_wrapping_for_block_bodies() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    generated_text = json.dumps(
        {
            "essence": "사주 핵심",
            "saju_blocks": [
                {
                    "category": "자동 줄바꿈",
                    "title": "본문 줄바꿈",
                    "summary": "요약",
                    "body": "첫 번째 줄\n두 번째 줄\n세 번째 줄",
                },
            ],
        },
        ensure_ascii=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        (),
        generated_text,
        skip_face=True,
    )

    assert "첫 번째 줄 두 번째 줄 세 번째 줄" in html
    assert "첫 번째 줄<br>두 번째 줄" not in html


def test_report_html_collapses_literal_newline_markers_for_auto_wrapping() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="male",
    )
    generated_text = json.dumps(
        {
            "essence": "saju essence",
            "saju_blocks": [
                {
                    "category": "auto wrap",
                    "title": "literal markers",
                    "summary": "summary",
                    "body": "first line\\nsecond line\r\nthird line",
                },
            ],
        },
        ensure_ascii=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        (),
        generated_text,
        skip_face=True,
    )

    assert "first line second line third line" in html
    assert "first line\\nsecond line" not in html
    assert "second line<br>third line" not in html


def test_compatibility_report_html_keeps_six_saju_blocks() -> None:
    repository = ManseRepository()
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
    generated_text = json.dumps(
        {
            "essence": "궁합 핵심",
            "saju_blocks": [
                {
                    "category": f"사주 카테고리 {index}",
                    "title": f"사주 제목 {index}",
                    "summary": f"사주 요약 {index}",
                    "body": f"사주 본문 {index}",
                }
                for index in range(1, 7)
            ],
        },
        ensure_ascii=False,
    )

    html = render_compatibility_report_html(
        left,
        right,
        "연인",
        repository.lookup(left),
        repository.lookup(right),
        "얼굴 관찰 fixture",
        generated_text,
    )

    assert "사주 제목 6" in html


def test_profile_hanja_marks_use_readable_text_color() -> None:
    repository = ManseRepository()
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
        '{"essence":"궁합 핵심"}',
    )

    assert ".person-mark" in html
    assert ".person-day" in html
    assert (
        ".person-mark{display:flex;flex-direction:column;align-items:center;"
        "justify-content:center;width:116px;height:116px;border-radius:50%;"
        "color:#fff}"
    ) not in html
    assert (
        '.person-day{font-family:"Song Myung",serif;font-size:48px;'
        "line-height:1;color:#fff;"
    ) not in html


def test_water_profile_marks_use_white_inner_text() -> None:
    repository = ManseRepository()
    left = BirthProfile(
        name="left",
        birth_datetime=datetime(1995, 1, 1, 12, 0),
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
        '{"essence":"궁합 핵심"}',
    )

    assert ".person-mark.c-su,.person-day.c-su{color:#fff;background:var(--su);border-color:var(--su)}" in html
    assert ".person-mark.c-su .person-ko{color:#fff}" in html

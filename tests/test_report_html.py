from __future__ import annotations

import json
from datetime import datetime

from oracle_report.models import BirthProfile
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
        "not-json",
    )

    assert "관상 정보는 보조 해석으로만 사용합니다" not in html
    assert "사주 결과와 충돌" not in html
    assert "표정과 눈썹 흐름" not in html
    assert "눈은 시선의 또렷함" in html
    assert "삼정은 얼굴을 위" in html


def test_personal_report_hides_face_saju_synthesis_but_keeps_keywords() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    generated_text = json.dumps(
        {
            "essence": "사주 핵심",
            "synthesis_title": "전체 흐름을 정리하면",
            "synthesis_body": "사주 흐름을 중심으로 정리한 본문",
            "synthesis_summary": "강점은 살리고 부족한 리듬은 보완하세요.",
            "convergence": [
                {"face": "개인 관상 비교 근거", "saju": "개인 사주 비교 근거"},
            ],
            "tags": ["균형 키워드"],
        },
        ensure_ascii=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "얼굴 관찰 fixture",
        generated_text,
    )

    assert "전체 흐름을 정리하면" not in html
    assert "사주 흐름을 중심으로 정리한 본문" not in html
    assert "강점은 살리고 부족한 리듬은 보완하세요." not in html
    assert "개인 관상 비교 근거" not in html
    assert "개인 사주 비교 근거" not in html
    assert "균형 키워드" in html
    assert "cute-keyword-card" in html
    assert html.index("균형 키워드") < html.index("관상 - 얼굴이 말하는 인상")
    assert "관상과 사주가 만나는 지점" not in html


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
        json.dumps(
            {
                "essence": "궁합 핵심",
                "action_title": "행동 제목",
                "action_body": "행동 본문",
                "compatibility_score": 84,
                "compatibility_score_label": "찰떡 보완형",
                "compatibility_score_summary": "설렘과 안정감이 같이 살아나는 조합이에요.",
                "compatibility_saju_score": 82,
                "compatibility_face_score": 86,
                "compatibility_mode_bonus": 8,
            },
            ensure_ascii=False,
        ),
    )

    assert html.startswith("<!DOCTYPE html>")
    assert "cute-compatibility-report" in html
    assert "궁합 핵심" in html
    assert "left 님과 right 님" in html
    assert "궁합 점수" in html
    assert "compat-score-heart-card" in html
    assert ">84</div>" in html
    assert "/100" in html
    assert "한 줄 평가" in html
    assert "설렘과 안정감이 같이 살아나는 조합이에요." in html
    assert html.index("궁합 점수") < html.index("관계를 채워주는 키워드")


def test_compatibility_report_html_hides_synthesis_but_keeps_action_and_keywords() -> None:
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
            "synthesis_title": "사주 총정리",
            "synthesis_body": "사주 흐름만 정리한 본문",
            "convergence": [
                {"face": "궁합 관상 근거", "saju": "궁합 사주 근거"},
            ],
            "action_title": "액션 제목",
            "action_body": "액션 본문",
            "tags": ["궁합 키워드"],
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

    assert "사주 총정리" not in html
    assert "사주 흐름만 정리한 본문" not in html
    assert "궁합 관상 근거" not in html
    assert "궁합 사주 근거" not in html
    assert "궁합 키워드" in html
    assert "cute-keyword-card" in html
    assert "궁합 점수" in html
    assert html.index("궁합 키워드") < html.index("두 사람의 관계 분위기")


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
        '{"essence":"사주 핵심"}',
        skip_face=True,
    )

    assert "시간 미상" in html
    assert "오시(午時) 보조 기준" not in html


def test_saju_only_report_hides_synthesis_but_keeps_keywords() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )
    generated_text = json.dumps(
        {
            "essence": "사주 핵심",
            "synthesis_title": "전체 흐름을 정리하면",
            "synthesis_body": "사주-only 총정리 본문",
            "synthesis_summary": "사주-only 총정리 요약",
            "tags": ["사주 키워드"],
        },
        ensure_ascii=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        generated_text,
        skip_face=True,
    )

    assert "전체 흐름을 정리하면" not in html
    assert "사주-only 총정리 본문" not in html
    assert "사주-only 총정리 요약" not in html
    assert "나를 채워주는 키워드" in html
    assert "사주 키워드" in html
    assert "cute-keyword-card" in html
    assert html.index("사주 키워드") < html.index("사주 - 타고난 기운의 설계도")


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


def test_report_graphic_hanja_glyphs_use_black_text() -> None:
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

    assert '.person-mark .person-hanja{font-family:"Song Myung",serif;font-size:54px;line-height:1;color:#111}' in html
    assert '.person-day{font-family:"Song Myung",serif;font-size:48px;line-height:1;color:#111;' in html
    assert '.cell .ch{font-family:"Song Myung",serif;font-size:34px;line-height:1.1;color:#111}' in html


def test_report_block_text_uses_full_block_width() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        '{"essence":"사주 핵심"}',
        skip_face=True,
    )

    assert ".b-sum{font-size:15.5px;line-height:1.78;color:var(--mok);font-weight:400;margin-bottom:12px;padding-left:14px;border-left:3px solid var(--mok)}" in html
    assert ".b-body{font-size:15.5px;line-height:1.78;color:var(--ink)}" in html
    assert ".b-sum{" in html
    assert ".b-body{" in html
    assert "max-width:42ch}.part.saju .b-sum" not in html
    assert ".b-body{font-size:15.5px;line-height:1.78;color:var(--ink);max-width:" not in html


def test_report_renderer_preserves_short_llm_block_body_without_synthetic_padding() -> None:
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
                    "category": "종합 형국",
                    "title": "짧은 본문",
                    "summary": "핵심 요약입니다.",
                    "body": "첫 문장입니다. 두 번째 문장입니다.",
                },
            ],
        },
        ensure_ascii=False,
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        generated_text,
        skip_face=True,
    )

    assert "첫 문장입니다. 두 번째 문장입니다." in html
    assert "이 내용은 종합 형국 흐름" not in html
    assert "짧은 본문이라는 관점" not in html


def test_essence_uses_body_line_spacing_and_full_width() -> None:
    profile = BirthProfile(
        name="tester",
        birth_datetime=datetime(1995, 3, 15, 12, 0),
        gender="남성",
    )

    html = render_personal_report_html(
        profile,
        ManseRepository().lookup(profile),
        "",
        '{"essence":"사주 핵심"}',
        skip_face=True,
    )

    assert '.essence{font-family:"Gowun Batang",serif;font-size:19px;line-height:1.78;margin-top:24px;color:var(--ink)}' in html
    assert "max-width:30ch" not in html

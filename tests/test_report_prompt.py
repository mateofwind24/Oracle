from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import (
    build_compatibility_face_analysis_prompt,
    build_personal_face_analysis_prompt,
    build_personal_final_prompt,
)


def test_personal_face_analysis_prompt_contains_required_context() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_personal_face_analysis_prompt(profile, FaceReadingInput(None, None))

    assert "개인 리포트" in prompt
    assert "1995-03-15 14:30:00" in prompt
    assert "신원, 나이, 성별, 민족, 건강" in prompt


def test_compatibility_face_analysis_prompt_contains_pair_context() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_compatibility_face_analysis_prompt(
        profile,
        FaceReadingInput(None, None),
        "첫 번째 사람",
        "연인",
    )

    assert "두 사람 궁합 리포트" in prompt
    assert "궁합 모드: 연인" in prompt
    assert "현재 분석 대상: 첫 번째 사람" in prompt


def test_personal_final_prompt_contains_json_schema() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_personal_final_prompt(
        profile,
        "사주 입력",
        "관상 입력",
        "추천 입력",
    )

    assert "\"face_blocks\"" in prompt
    assert "\"saju_blocks\"" in prompt
    assert "사주 입력" in prompt
    assert "관상 입력" in prompt


def test_prompt_template_can_be_overridden_from_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps({"personal_face_analysis": "CUSTOM ${name} ${quality_text}"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_personal_face_analysis_prompt(profile, FaceReadingInput(None, None))

    assert prompt == "CUSTOM tester - 품질 정보 없음"

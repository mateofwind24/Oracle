from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from oracle_report import prompt_templates
from oracle_report.models import BirthProfile
from oracle_report.report import (
    build_couple_saju_reading_prompt,
    build_saju_reading_prompt,
)


_PROMPT_TEMPLATE_NAMES = (
    "saju_reading",
    "saju_reading_couple",
)


def test_runtime_prompts_define_explicit_cache_prefixes() -> None:
    prompt_path = Path("configs/prompts.json")
    root = json.loads(prompt_path.read_text(encoding="utf-8"))

    for prompt_name in _PROMPT_TEMPLATE_NAMES:
        prompt_config = root[prompt_name]

        assert isinstance(prompt_config, dict)
        assert isinstance(prompt_config["id_slot"], int)
        assert isinstance(prompt_config["prefix"], list)
        assert isinstance(prompt_config["body"], list)
        assert prompt_config["prefix"]
        assert prompt_config["body"]
        assert "${" not in "\n".join(prompt_config["prefix"])
        assert "${" in "\n".join(prompt_config["body"])


def test_saju_reading_prompt_omits_face_and_recommendation_schema() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(
        profile,
        "사주 입력",
    )

    assert "\"saju_blocks\"" in prompt
    assert "\"face_blocks\"" not in prompt
    assert "\"recommendation_title\"" not in prompt
    assert "사주 입력" in prompt
    assert "얼굴 관찰 메모" not in prompt
    assert "추천받고 싶은 얼굴" not in prompt
    assert prompt.name == "saju_reading"
    assert prompt.slot_id == 1
    assert prompt.prefix.strip() != ""
    assert "각 saju_blocks의 summary는 body의 핵심을 1~2개의 짧은 문장" in prompt.prefix
    assert (
        "각 saju_blocks의 body는 정확히 "
        f"{prompt_templates.REPORT_BLOCK_SENTENCE_COUNT}개의 완성된 문장"
        in prompt.prefix
    )
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body의 문장 수는 서로 같아야" not in prompt.prefix
    assert "자동 줄바꿈 기준" not in prompt.prefix
    assert "5~6줄" not in prompt.prefix
    assert "180~220자" not in prompt.prefix
    assert "줄바꿈 이스케이프" in prompt.prefix
    assert "줄바꿈은 \\n으로 표현" not in prompt.prefix
    assert "입력받은 이름 필드에 '님'을 붙여 사용" in prompt.prefix
    assert "상담가이자 스토리텔러" in prompt.prefix
    assert "실제 경험, 감정, 관계 장면과 연결" in prompt.prefix
    assert "성향 -> 사람들이 나를 어떻게 보는지 -> 강점 -> 주의할 점 -> 현재 시기의 흐름 -> 앞으로의 조언" in prompt.prefix
    assert "반드시/틀림없이" in prompt.prefix
    assert "좋은 내용은 약 80%, 주의하거나 보완할 내용은 약 20%" in prompt.prefix
    assert "좋은 내용 -> 안 좋은 내용 -> 좋은 내용" in prompt.prefix
    assert "사주 입력" not in prompt.prefix
    assert "사주 입력" in prompt.body
    assert "같은 근거와 같은 표현을 여러 블록에서 반복하지 않습니다" in prompt.prefix
    assert "[블록별 근거 배분]" in prompt.prefix
    assert "일간 하나로 모든 블록을 설명하지 않습니다" in prompt.prefix
    assert "'임수 일간', '갑목 일간'처럼 일간명을 직접 반복하지 않습니다" in prompt.prefix


def test_saju_reading_prompt_uses_input_name_for_honorifics() -> None:
    profile = BirthProfile(name="홍길동", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(
        profile,
        "일간은 임수입니다.",
    )

    assert "- 이름: 홍길동" in prompt.body
    assert "임수님" not in prompt.body
    assert "입력받은 이름 필드에 '님'을 붙여 사용" in prompt.prefix
    assert "이름과 님 사이를 띄어쓰지 않습니다" in prompt.prefix
    assert "일간이나 오행" in prompt.prefix
    assert "갑목님, 임수님, 계수님 같은 표현은 절대 쓰지 않습니다" in prompt.prefix
    assert "임수 일간은" not in prompt.prefix


def test_couple_saju_reading_prompt_uses_pair_saju_only() -> None:
    left = BirthProfile(name="left", birth_datetime=datetime(1995, 3, 15, 14, 30))
    right = BirthProfile(name="right", birth_datetime=datetime(1997, 5, 20, 9, 0))

    prompt = build_couple_saju_reading_prompt(
        left,
        right,
        "연인",
        "LEFT SAJU INPUT",
        "RIGHT SAJU INPUT",
    )

    assert "\"saju_blocks\"" in prompt
    assert "\"pair_blocks\"" not in prompt
    assert "LEFT SAJU INPUT" in prompt
    assert "RIGHT SAJU INPUT" in prompt
    assert "saju_blocks는 아래 스키마의 6개 카테고리를 빠짐없이 정확히 작성합니다" in prompt.prefix
    assert "각 saju_blocks의 summary는 body의 핵심을 1~2개의 짧은 문장" in prompt.prefix
    assert (
        "각 saju_blocks의 body는 정확히 "
        f"{prompt_templates.REPORT_BLOCK_SENTENCE_COUNT}개의 완성된 문장"
        in prompt.prefix
    )
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body의 문장 수는 서로 같아야" not in prompt.prefix
    assert "자동 줄바꿈 기준" not in prompt.prefix
    assert "5~6줄" not in prompt.prefix
    assert "180~220자" not in prompt.prefix
    assert "줄바꿈 이스케이프" in prompt.prefix
    assert "줄바꿈은 \\n으로 표현" not in prompt.prefix
    assert "입력받은 left_name/right_name 필드에 '님'을 붙여 사용" in prompt.prefix
    assert "이름과 님 사이를 띄어쓰지 않습니다" in prompt.prefix
    assert "'임수 일간', '갑목 일간'처럼 일간명을 직접 반복하지 않습니다" in prompt.prefix
    assert "상담가이자 스토리텔러" in prompt.prefix
    assert "실제 경험, 감정, 관계 장면과 연결" in prompt.prefix
    assert "각자의 성향 -> 서로가 상대를 어떻게 느끼는지 -> 관계의 강점 -> 주의할 점 -> 현재 관계 흐름 -> 앞으로의 조언" in prompt.prefix
    assert "반드시/틀림없이" in prompt.prefix
    assert "좋은 내용은 약 80%, 주의하거나 보완할 내용은 약 20%" in prompt.prefix
    assert "좋은 내용 -> 안 좋은 내용 -> 좋은 내용" in prompt.prefix


def test_report_sentence_count_guidance_uses_single_constant(monkeypatch) -> None:
    monkeypatch.setattr(prompt_templates, "REPORT_BLOCK_SENTENCE_COUNT", 7)
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(profile, "사주 입력")
    debug_prompt = prompt_templates.render_debug_prompt_template(
        "personal_final",
        {
            "name": "tester",
            "gender": "male",
            "birth_datetime": "1995-03-15 미시(未時)",
            "birth_time_text": "미시(未時)",
            "timezone": "Asia/Seoul",
            "saju_text": "사주 입력",
            "face_analysis": "얼굴 관찰",
            "recommendation_text": "추천 정보",
        },
    )

    assert "summary는 body의 핵심을 1~2개의 짧은 문장" in prompt.prefix
    assert "body는 정확히 7개의 완성된 문장" in prompt.prefix
    assert "body는 정확히 7개의 완성된 문장" in debug_prompt
    assert "summary와 body는 각각 정확히" not in prompt.prefix
    assert "summary와 body는 각각 정확히" not in debug_prompt
    assert "정확히 6개의 완성된 문장" not in prompt.prefix


def test_prompt_template_can_be_overridden_from_json(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prompt_path = tmp_path / "prompts.json"
    prompt_path.write_text(
        json.dumps({"saju_reading": "CUSTOM ${name} ${saju_text}"}),
        encoding="utf-8",
    )
    monkeypatch.setenv("ORACLE_PROMPTS_PATH", str(prompt_path))
    profile = BirthProfile(name="tester", birth_datetime=datetime(1995, 3, 15, 14, 30))

    prompt = build_saju_reading_prompt(profile, "SAJU")

    assert prompt == "CUSTOM tester SAJU"

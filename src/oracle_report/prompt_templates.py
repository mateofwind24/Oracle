from __future__ import annotations

import json
import os
from dataclasses import dataclass
from collections.abc import Mapping
from pathlib import Path
from string import Template
from typing import Any


_DEFAULT_PROMPTS_PATH = Path("configs/prompts.json")
_DEFAULT_DEBUG_PROMPTS_PATH = Path("configs/prompts_debug.json")
_PROMPTS_PATH_ENV_NAME = "ORACLE_PROMPTS_PATH"
_DEBUG_PROMPTS_PATH_ENV_NAME = "ORACLE_DEBUG_PROMPTS_PATH"
_DEFAULT_PROMPT_SLOTS = {
    "personal_face_analysis": 0,
    "saju_reading": 1,
    "face_analysis_copule": 2,
    "saju_reading_couple": 3,
    "compatibility_face_analysis": 4,
}


@dataclass(frozen=True)
class RenderedPrompt:
    name: str
    prefix: str
    body: str
    slot_id: int | None

    @property
    def text(self) -> str:
        result = self.body
        if self.prefix.strip() != "":
            result = f"{self.prefix.strip()}\n\n{self.body.strip()}".strip()
        return result

    def __str__(self) -> str:
        result = self.text
        return result

    def __contains__(self, item: str) -> bool:
        result = item in self.text
        return result

    def __eq__(self, other: object) -> bool:
        result = False
        if isinstance(other, str):
            result = self.text == other
        elif isinstance(other, RenderedPrompt):
            result = (
                self.name == other.name
                and self.prefix == other.prefix
                and self.body == other.body
                and self.slot_id == other.slot_id
            )
        return result


@dataclass(frozen=True)
class PromptTemplateInfo:
    name: str
    prefix: str
    body_template: str
    slot_id: int | None


def render_prompt_template(name: str, values: Mapping[str, object]) -> RenderedPrompt:
    templates = _load_prompt_templates(_prompt_templates_path())
    template_parts = _template_parts(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    for key, default_value in (
        ("landmark_metrics_text", "- 랜드마크 측정값 없음"),
        ("landmark_context_text", "- 구조화된 관찰 컨텍스트 없음"),
        ("landmark_rules_text", "- 랜드마크 규칙 해석 힌트 없음"),
    ):
        if key not in string_values:
            string_values[key] = default_value
    prefix = Template(template_parts.prefix).substitute(string_values).strip()
    body = Template(template_parts.body_template).substitute(string_values).strip()
    result = RenderedPrompt(
        name=name,
        prefix=prefix,
        body=body,
        slot_id=template_parts.slot_id,
    )
    return result


def render_debug_prompt_template(name: str, values: Mapping[str, object]) -> str:
    templates = _load_prompt_templates(_debug_prompt_templates_path())
    template_text = _template_text(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    result = Template(template_text).substitute(string_values).strip()
    return result


def list_prompt_template_info() -> tuple[PromptTemplateInfo, ...]:
    templates = _load_prompt_templates(_prompt_templates_path())
    result = tuple(_template_parts(templates, name) for name in templates)
    return result


def _prompt_templates_path() -> Path:
    result = _configured_prompt_path(_PROMPTS_PATH_ENV_NAME, _DEFAULT_PROMPTS_PATH)
    return result


def _debug_prompt_templates_path() -> Path:
    result = _configured_prompt_path(
        _DEBUG_PROMPTS_PATH_ENV_NAME,
        _DEFAULT_DEBUG_PROMPTS_PATH,
    )
    return result


def _configured_prompt_path(env_name: str, default_path: Path) -> Path:
    configured_path = os.getenv(env_name, "")
    result = _DEFAULT_PROMPTS_PATH
    if configured_path.strip() != "":
        result = Path(configured_path)
    else:
        result = default_path
    return result


def _load_prompt_templates(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as prompt_file:
        root = json.load(prompt_file)
    if not isinstance(root, dict):
        raise ValueError(f"prompt template file must contain a JSON object: {path}")
    result = root
    return result


def _template_text(templates: Mapping[str, Any], name: str) -> str:
    template_parts = _template_parts(templates, name)
    result = template_parts.prefix
    if template_parts.prefix.strip() != "":
        result = f"{template_parts.prefix.strip()}\n\n{template_parts.body_template.strip()}"
    elif template_parts.body_template.strip() != "":
        result = template_parts.body_template
    return result


def _template_parts(templates: Mapping[str, Any], name: str) -> PromptTemplateInfo:
    raw_value = templates.get(name)
    if isinstance(raw_value, dict):
        result = _dict_template_parts(name, raw_value)
    elif isinstance(raw_value, (str, list)):
        result = _legacy_template_parts(name, raw_value)
    else:
        raise ValueError(
            "prompt template "
            f"'{name}' must be a string, a list of strings, or an object.",
        )
    return result


def _dict_template_parts(name: str, raw_value: Mapping[str, Any]) -> PromptTemplateInfo:
    prefix = _template_fragment_text(raw_value.get("prefix", ""))
    body = _template_fragment_text(raw_value.get("body", raw_value.get("input", "")))
    slot_id = _template_slot_id(name, raw_value.get("id_slot", raw_value.get("slot_id")))
    result = PromptTemplateInfo(
        name=name,
        prefix=prefix,
        body_template=body,
        slot_id=slot_id,
    )
    return result


def _legacy_template_parts(name: str, raw_value: object) -> PromptTemplateInfo:
    template_text = _template_fragment_text(raw_value)
    lines = template_text.splitlines()
    first_dynamic_index = _first_dynamic_line_index(lines)
    prefix_lines = lines
    body_lines: list[str] = []
    if first_dynamic_index is not None:
        prefix_lines = lines[:first_dynamic_index]
        body_lines = lines[first_dynamic_index:]
    result = PromptTemplateInfo(
        name=name,
        prefix="\n".join(prefix_lines).strip(),
        body_template="\n".join(body_lines).strip(),
        slot_id=_template_slot_id(name, None),
    )
    return result


def _template_fragment_text(raw_value: object) -> str:
    result = ""
    if isinstance(raw_value, str):
        result = raw_value
    elif isinstance(raw_value, list) and all(isinstance(item, str) for item in raw_value):
        result = "\n".join(raw_value)
    else:
        raise ValueError("prompt template fragment must be a string or list of strings.")
    return result


def _first_dynamic_line_index(lines: list[str]) -> int | None:
    result = None
    for index, line in enumerate(lines):
        if "${" in line:
            result = index
            break
    return result


def _template_slot_id(name: str, configured_value: object) -> int | None:
    result = _DEFAULT_PROMPT_SLOTS.get(name)
    if configured_value is not None:
        result = int(configured_value)
    return result


def render_distributed_prompt_template(
    name: str,
    values: Mapping[str, object],
    target_category: str | None = None,
    is_metadata: bool = False,
) -> RenderedPrompt:
    rendered = render_prompt_template(name, values)
    if not (target_category or is_metadata):
        return rendered

    prefix = rendered.prefix
    schema_start = prefix.find("[출력 JSON 스키마]")
    
    # 1. Isolate the static system rules before the schema block
    prefix_before_schema = prefix[:schema_start] if schema_start != -1 else prefix
    
    # 2. Extract static common body (saju birth profile / data)
    common_body = rendered.body
    
    # 3. Unify static parts as the new prefix so that they are 100% identical in token sequence
    unified_common_prefix = f"{prefix_before_schema.strip()}\n\n{common_body.strip()}"
    
    # 4. Construct category-specific suffix instructions to go into the body segment
    suffix_instructions = ""
    if is_metadata:
        if name == "saju_reading":
            suffix_instructions = """[출력 JSON 스키마]
너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 단일 JSON 객체로 응답해야 한다. saju_blocks 필드는 절대 포함하지 마라.
{
  "essence": "이 사주의 전체적인 흐름과 삶의 방향성을 3~4문장의 풍부한 해설 단락(약 150~200자)으로 요약한 내용",
  "element_note": "[오행 분포]의 강한 기운과 보완이 필요한 기운이 생활 리듬에 어떻게 드러나는지 1-2문장으로 설명",
  "saju_subtitle": "사주 섹션 핵심을 8-16자 안팎의 짧은 문구",
  "tags": ["태그1", "태그2", "태그3", "태그4"],
  "disclaimer": "참고용 엔터테인먼트 리포트라는 짧은 고지"
}"""
        elif name == "personal_face_analysis":
            suffix_instructions = """[출력 JSON 스키마]
너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 단일 JSON 객체로 응답해야 한다. face_blocks 필드는 절대 포함하지 마라.
{
  "face_subtitle": "얼굴 관찰 섹션 오른쪽 짧은 키워드",
  "face_summary": "관상 관찰을 1문장으로 요약"
}"""
        elif name == "saju_reading_couple":
            suffix_instructions = """[출력 JSON 스키마]
너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 단일 JSON 객체로 응답해야 한다. saju_blocks 필드는 절대 포함하지 마라.
{
  "essence": "두 사람의 사주 궁합 핵심 요약",
  "saju_subtitle": "사주 섹션 짧은 부제",
  "synthesis_title": "사주 궁합 종합 제목",
  "synthesis_body": "두 사람의 사주 흐름을 종합한 설명",
  "action_title": "관계를 좋게 만드는 행동 제안 제목",
  "action_body": "실천 가능한 행동 제안",
  "tags": ["태그1", "태그2", "태그3", "태그4"],
  "disclaimer": "참고용 엔터테인먼트 리포트라는 짧은 고지"
}"""
        elif name == "face_analysis_copule":
            suffix_instructions = """[출력 JSON 스키마]
너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 단일 JSON 객체로 응답해야 한다. pair_blocks 필드는 절대 포함하지 마라.
{
  "pair_subtitle": "관상 기반 관계 분위기 부제",
  "face_summary": "두 사람의 관상 관찰을 1문장으로 요약"
}"""
        else:
            suffix_instructions = "[출력 JSON 스키마]\n오직 요약 정보와 메타데이터 필드들만 JSON으로 채워서 출력하고, 개별 상세 블록 필드는 포함하지 마십시오."
    else:
        suffix_instructions = f"""[출력 JSON 스키마]
당신은 오직 '{target_category}' 카테고리에 대한 분석만 수행합니다.
다른 메타데이터 필드나 다른 카테고리 블록은 절대 포함하지 말고, 오직 아래 포맷의 단일 JSON 객체 하나만 출력해야 합니다.
{{
  "category": "{target_category}",
  "title": "이 분석을 대표하는 호기심을 자극하면서도 핵심을 찌르는 제목",
  "summary": "쉬운 한국어 해요체의 짧은 요약 문장",
  "body": "이 분석에 대한 구체적이고 현실적인 설명 본문"
}}

[분석 대상 카테고리]
- 카테고리: {target_category}"""

    # We return the unified static content as the prefix, and the variable parts as the suffix body.
    # When concatenated by the LlamaCppChatClient, the prefix sequence remains 100% identical.
    return RenderedPrompt(
        name=f"{rendered.name}_split",
        prefix=unified_common_prefix.strip(),
        body=suffix_instructions.strip(),
        slot_id=None,
    )

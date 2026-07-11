from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from string import Template
from typing import Any, Mapping

_PROMPTS_PATH_ENV_NAME = "ORACLE_PROMPTS_PATH"

_DEFAULT_PROMPTS_PATH = Path("configs/prompts.json")

_DEFAULT_PROMPT_SLOTS = {
    "personal_face_analysis": 2,
    "face_analysis_couple": 2,
    "saju_reading": 3,
    "saju_reading_couple": 3,
}

REPORT_BLOCK_SENTENCE_COUNT = 10


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
    templates, sources = _load_prompt_templates_with_sources(_prompt_templates_path())
    template_parts = _template_parts(templates, name)
    string_values = {key: str(value) for key, value in values.items()}
    prefix = Template(template_parts.prefix).substitute(string_values).strip()
    body = Template(template_parts.body_template).substitute(string_values).strip()
    result = RenderedPrompt(
        name=name,
        prefix=prefix,
        body=body,
        slot_id=template_parts.slot_id,
    )
    _log_prompt_render(name, template_parts, sources.get(name), result)
    return result


def list_prompt_template_info() -> tuple[PromptTemplateInfo, ...]:
    templates = _load_prompt_templates(_prompt_templates_path())
    result = tuple(_template_parts(templates, name) for name in templates)
    return result


def _prompt_templates_path() -> Path:
    result = _configured_prompt_path(_PROMPTS_PATH_ENV_NAME, _DEFAULT_PROMPTS_PATH)
    return result


def _configured_prompt_path(env_name: str, default_path: Path) -> Path:
    configured_path = os.getenv(env_name, "")
    result = _DEFAULT_PROMPTS_PATH
    if configured_path.strip() != "":
        result = Path(configured_path)
    return result


def _load_prompt_templates(path: Path) -> dict[str, Any]:
    result, _sources = _load_prompt_templates_with_sources(path)
    return result


def _load_prompt_templates_with_sources(path: Path) -> tuple[dict[str, Any], dict[str, Path]]:
    with path.open(encoding="utf-8") as prompt_file:
        root = json.load(prompt_file)
    if not isinstance(root, dict):
        raise ValueError(f"prompt template file must contain a JSON object: {path}")
    result = _resolve_prompt_template_root(root, path)
    return result


def _resolve_prompt_template_root(
    root: Mapping[str, Any],
    path: Path,
) -> tuple[dict[str, Any], dict[str, Path]]:
    include_value = root.get("include")
    if include_value is None:
        result = dict(root)
        sources = {name: path for name in result}
        return result, sources
    if not isinstance(include_value, list) or not all(
        isinstance(item, str) for item in include_value
    ):
        raise ValueError(
            "prompt template include must be a list of file paths: "
            f"{path}",
        )
    result: dict[str, Any] = {}
    sources: dict[str, Path] = {}
    for include_path_text in include_value:
        include_path = Path(include_path_text)
        if not include_path.is_absolute():
            include_path = path.parent / include_path
        included_templates, included_sources = _load_prompt_templates_with_sources(
            include_path,
        )
        duplicate_names = set(result).intersection(included_templates)
        if duplicate_names:
            duplicate_list = ", ".join(sorted(duplicate_names))
            raise ValueError(
                "duplicate prompt template names across included files: "
                f"{duplicate_list}",
            )
        result.update(included_templates)
        sources.update(included_sources)
    inline_templates = {
        key: value
        for key, value in root.items()
        if key != "include"
    }
    duplicate_names = set(result).intersection(inline_templates)
    if duplicate_names:
        duplicate_list = ", ".join(sorted(duplicate_names))
        raise ValueError(
            "duplicate prompt template names between include and manifest: "
            f"{duplicate_list}",
        )
    result.update(inline_templates)
    sources.update({name: path for name in inline_templates})
    return result, sources


def _log_prompt_render(
    name: str,
    template_parts: PromptTemplateInfo,
    source_path: Path | None,
    rendered: RenderedPrompt,
) -> None:
    del template_parts
    source_message = ""
    if source_path is not None:
        source_message = f" source={source_path}"
    slot_message = ""
    if rendered.slot_id is not None:
        slot_message = f" id_slot={rendered.slot_id}"
    print(
        f"[PROMPT] name={name}{source_message}{slot_message} "
        f"prefix_chars={len(rendered.prefix)} body_chars={len(rendered.body)}",
        file=sys.stderr,
    )


import sys


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
    elif isinstance(raw_value, list):
        if not all(isinstance(item, str) for item in raw_value):
            raise ValueError(
                "prompt template fragments must be lists of strings",
            )
        result = "\n".join(raw_value)
    return result


def _first_dynamic_line_index(lines: list[str]) -> int | None:
    result = None
    for i, line in enumerate(lines):
        if "${" in line:
            result = i
            break
    return result


def _template_slot_id(name: str, configured_value: object) -> int | None:
    result = _DEFAULT_PROMPT_SLOTS.get(name)
    if configured_value is not None:
        result = int(configured_value)
    return result


# Category guides are now fully managed in configs/prompts_personal.json and configs/prompts_compatibility.json

_rules_cache = None
_system_instructions_cache = None


def _load_saju_rules() -> dict:
    global _rules_cache
    if _rules_cache is None:
        rules_path = Path("configs/saju_rules.json")
        if rules_path.exists():
            with rules_path.open(encoding="utf-8") as f:
                _rules_cache = json.load(f)
        else:
            _rules_cache = {"ten_gods": {}, "shinsals": {}, "relationships": {}}
    return _rules_cache


def _load_system_instructions() -> dict:
    global _system_instructions_cache
    if _system_instructions_cache is None:
        path = Path("configs/system_instructions.json")
        if path.exists():
            with path.open(encoding="utf-8") as f:
                _system_instructions_cache = json.load(f)
        else:
            _system_instructions_cache = {"personal": [], "compatibility": []}
    return _system_instructions_cache


def _build_saju_rules_section(saju_text: str) -> str:
    import re
    saju_words = set(re.findall(r'[가-힣a-zA-Z0-9]+', saju_text))

    spec_texts = []
    for match in re.finditer(r'\[만세력/사주명식\](.*?)(?=\[오행 분포\]|\[사주정보\]|$)', saju_text, re.DOTALL):
        spec_texts.append(match.group(1))
    if spec_texts:
        combined_spec_text = " ".join(spec_texts)
        saju_words_spec = set(re.findall(r'[가-힣a-zA-Z0-9]+', combined_spec_text))
    else:
        saju_words_spec = saju_words

    rules = _load_saju_rules()
    filtered_rules = []

    # 1. 십신(Ten Gods) 매칭
    ten_gods_lines = []
    for tg, desc in rules.get("ten_gods", {}).items():
        if tg in saju_words_spec:
            ten_gods_lines.append(f"  * {desc}")
    if ten_gods_lines:
        filtered_rules.append("- 십신(Ten Gods)의 10가지 내면 심리 본질 및 입체적 스토리텔링 가이드:")
        filtered_rules.extend(ten_gods_lines)
        filtered_rules.append("")

    # 2. 신살(Shinsals) 매칭
    shinsals_lines = []
    for ss, desc in rules.get("shinsals", {}).items():
        if ss in saju_words:
            shinsals_lines.append(f"  * {desc}")
    if shinsals_lines:
        filtered_rules.append("[4: 26종 특수 기운(Shinsal)의 장단점 분석 및 입체적 사주풀이 가이드]")
        filtered_rules.extend(shinsals_lines)
        filtered_rules.append("")

    # 3. 관계 흐름(Relationships) 매칭
    relationships_lines = []
    for rel, info in rules.get("relationships", {}).items():
        cond1 = info.get("cond1", [])
        cond2 = info.get("cond2", [])
        has_cond1 = any(c1 in saju_words for c1 in cond1)
        has_cond2 = any(c2 in saju_words for c2 in cond2)
        if has_cond1 and has_cond2:
            relationships_lines.append(f"  * {info.get('text')}")
    if relationships_lines:
        filtered_rules.append("- 구조적 역동성(Relationship)의 7대 흐름 및 스토리텔링 가이드:")
        filtered_rules.extend(relationships_lines)

    return "\n".join(filtered_rules)


def render_distributed_prompt_template(
    name: str,
    values: Mapping[str, object],
    target_category: str | None = None,
    is_metadata: bool = False,
) -> RenderedPrompt:
    # 1. saju_rules 동적 빌드 및 values 에 주입
    mutable_values = dict(values)
    if "saju" in name and "saju_rules" not in mutable_values:
        saju_text = ""
        if "saju_text" in mutable_values:
            saju_text = str(mutable_values["saju_text"])
        elif "left_saju_text" in mutable_values and "right_saju_text" in mutable_values:
            saju_text = f"{mutable_values['left_saju_text']} {mutable_values['right_saju_text']}"
        mutable_values["saju_rules"] = _build_saju_rules_section(saju_text)
    elif "saju_rules" not in mutable_values:
        mutable_values["saju_rules"] = ""

    # 2. 주입된 mutable_values 로 렌더링 호출
    rendered = render_prompt_template(name, mutable_values)
    if not (target_category or is_metadata):
        return rendered
    if name not in (
        "personal_face_analysis",
        "face_analysis_couple",
        "saju_reading",
        "saju_reading_couple",
    ):
        raise ValueError(f"unsupported distributed prompt template: {name}")

    # 3. templates 원본 맵 로드하여 categories/metadata 구조 데이터 꺼내기
    templates = _load_prompt_templates(_prompt_templates_path())
    prompt_config = templates.get(name, {})
    if not isinstance(prompt_config, dict):
        prompt_config = {}

    prefix = rendered.prefix
    common_body = rendered.body

    # 4. system_instructions.json 에서 정적 공통 가이드를 가져와 맨 앞에 접합
    static_system_text = ""
    sys_inst = _load_system_instructions()
    if name in ("saju_reading", "personal_face_analysis"):
        static_system_text = "\n".join(sys_inst.get("personal", []))
    elif name in ("saju_reading_couple", "face_analysis_couple"):
        static_system_text = "\n".join(sys_inst.get("compatibility", []))

    if static_system_text:
        prefix = f"{static_system_text.strip()}\n\n{prefix.strip()}"
    
    unified_common_prefix = f"{prefix.strip()}\n\n{common_body.strip()}"

    # 5. suffix_instructions (카테고리 지시문 또는 메타데이터 스키마) 동적 빌드
    suffix_instructions = ""
    if is_metadata:
        metadata_cfg = prompt_config.get("metadata", {})
        if name in ("saju_reading", "saju_reading_couple"):
            lines = [
                "[출력 형식]",
                "너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 형식으로 응답해야 한다. 카테고리 블록(=== CATEGORY ===)은 절대 작성하지 마라. 중괄호나 JSON 기호는 절대 사용하지 마세요.",
                "",
                "=== METADATA ==="
            ]
            for key, val in metadata_cfg.items():
                lines.append(f"### {key.upper()}: {val}")
            suffix_instructions = "\n".join(lines)
        else:
            lines = [
                "[출력 JSON 스키마]",
                "너는 오직 아래의 요약 정보와 메타데이터 필드들만 포함하는 단일 JSON 객체로 응답해야 한다. face_blocks 필드는 절대 포함하지 마라.",
                "{"
            ]
            fields = []
            for key, val in metadata_cfg.items():
                fields.append(f'  "{key}": "{val}"')
            lines.append(",\n".join(fields))
            lines.append("}")
            suffix_instructions = "\n".join(lines)
    else:
        categories_cfg = prompt_config.get("categories", {})
        cat_info = categories_cfg.get(target_category, {})
        if not isinstance(cat_info, dict):
            cat_info = {}

        if name in ("saju_reading", "saju_reading_couple"):
            guide_text = cat_info.get("guide", "")
            suffix_instructions = f"""[출력 형식]
당신은 오직 '{target_category}' 카테고리에 대한 분석만 수행합니다.
다른 메타데이터 필드나 다른 카테고리 블록은 절대 포함하지 말고, 오직 아래 포맷의 텍스트 형태로만 응답해야 합니다. 중괄호나 JSON 기호는 절대 사용하지 마세요.

=== CATEGORY: {target_category} ===
### TITLE: {cat_info.get('title', '제목')}
### SUMMARY: {cat_info.get('summary', '요약')}
### BODY: {cat_info.get('body', '본문')}

* 세부 가이드 지침: {guide_text}"""
        else:
            suffix_instructions = f"""[출력 JSON 스키마]
당신은 오직 '{target_category}' 카테고리에 대한 분석만 수행합니다.
다른 메타데이터 필드나 다른 카테고리 블록은 절대 포함하지 말고, 오직 아래 포맷의 단일 JSON 객체 하나만 출력해야 합니다.
{{
  "category": "{target_category}",
  "title": "{cat_info.get('title', '제목')}",
  "summary": "{cat_info.get('summary', '요약')}",
  "body": "{cat_info.get('body', '본문')}"
}}

* 세부 가이드 지침: {cat_info.get('guide', '')}"""

    result = RenderedPrompt(
        name=f"{rendered.name}_split",
        prefix=unified_common_prefix,
        body=suffix_instructions,
        slot_id=rendered.slot_id,
    )
    return result

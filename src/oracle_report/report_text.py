from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from oracle_report import prompt_templates


_REPORT_BLOCK_KEYS = ("face_blocks", "saju_blocks", "pair_blocks")
_REPORT_BLOCK_TEXT_KEYS = ("summary", "body")
_SENTENCE_ENDINGS = ".!?。！？"
_SUPPLEMENTAL_SENTENCE_TEMPLATES = (
    "이 내용은 {category} 흐름을 참고용으로 더 차분히 풀어 보는 설명이에요.",
    "{title}이라는 관점에서 강점과 보완점을 함께 살피면 이해가 쉬워요.",
    "{summary}라는 요약을 바탕으로 현재의 선택 기준을 점검해 볼 수 있어요.",
    "좋은 방향은 살리고 부담이 커지는 지점은 작게 조율하는 태도가 도움이 될 수 있어요.",
    "단정적인 결론보다 일상에서 반복되는 말투와 선택을 함께 보는 편이 좋아요.",
    "지금의 흐름은 한 번의 사건보다 꾸준히 나타나는 패턴으로 이해하면 더 자연스러워요.",
    "강하게 드러나는 면은 장점으로 쓰고 부족한 면은 생활 습관으로 보완할 수 있어요.",
    "관계나 일의 장면에서는 속도를 늦춰 확인하는 방식이 안정감을 높여 줄 수 있어요.",
    "마지막으로 이 해석은 참고용이므로 실제 선택에서는 자신의 상황을 함께 살피는 편이 좋아요.",
)


def normalize_report_payload_blocks(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    for key in _REPORT_BLOCK_KEYS:
        value = result.get(key)
        if isinstance(value, list):
            result[key] = _normalized_block_list(value)
    return result


def normalize_report_block_text_fields(block: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(block)
    context = _block_context(result)
    target_count = _target_sentence_count()
    for key in _REPORT_BLOCK_TEXT_KEYS:
        value = result.get(key)
        if isinstance(value, str) and value.strip() != "":
            result[key] = fit_report_text_sentence_count(value, context, target_count)
    return result


def fit_report_text_sentence_count(
    text: str,
    context: Mapping[str, str],
    target_count: int,
) -> str:
    normalized_target_count = max(1, int(target_count))
    sentences = _split_sentences(text)
    result_sentences = sentences[:normalized_target_count]
    original_sentence_count = len(result_sentences)
    while len(result_sentences) < normalized_target_count:
        supplemental_index = len(result_sentences) - original_sentence_count
        result_sentences.append(
            _supplemental_sentence(context, supplemental_index),
        )
    result = " ".join(result_sentences)
    return result


def _normalized_block_list(blocks: list[Any]) -> list[Any]:
    result: list[Any] = []
    for block in blocks:
        normalized_block = block
        if isinstance(block, Mapping):
            normalized_block = normalize_report_block_text_fields(block)
        result.append(normalized_block)
    return result


def _target_sentence_count() -> int:
    configured_count = int(prompt_templates.REPORT_BLOCK_SENTENCE_COUNT)
    result = max(1, configured_count)
    return result


def _split_sentences(text: str) -> list[str]:
    normalized_text = _normalize_inline_text(text)
    sentences: list[str] = []
    start_index = 0
    for index, character in enumerate(normalized_text):
        if character in _SENTENCE_ENDINGS:
            sentence = normalized_text[start_index : index + 1].strip()
            if sentence != "":
                sentences.append(sentence)
            start_index = index + 1
    tail = normalized_text[start_index:].strip()
    if tail != "":
        sentences.append(_ensure_sentence_ending(tail))
    result = sentences
    return result


def _normalize_inline_text(text: str) -> str:
    normalized_text = text.replace("\\r\\n", " ")
    normalized_text = normalized_text.replace("\\n", " ")
    normalized_text = normalized_text.replace("\\r", " ")
    normalized_text = normalized_text.replace("\r\n", " ")
    normalized_text = normalized_text.replace("\n", " ")
    normalized_text = normalized_text.replace("\r", " ")
    result = " ".join(normalized_text.split())
    return result


def _ensure_sentence_ending(text: str) -> str:
    result = text
    if result[-1] not in _SENTENCE_ENDINGS:
        result = f"{result}."
    return result


def _supplemental_sentence(context: Mapping[str, str], sentence_index: int) -> str:
    template = _SUPPLEMENTAL_SENTENCE_TEMPLATES[
        sentence_index % len(_SUPPLEMENTAL_SENTENCE_TEMPLATES)
    ]
    result = template.format(
        category=context["category"],
        title=context["title"],
        summary=context["summary"],
    )
    return result


def _block_context(block: Mapping[str, Any]) -> dict[str, str]:
    result = {
        "category": _context_text(block.get("category"), "이 항목"),
        "title": _context_text(block.get("title"), "이 제목"),
        "summary": _context_text(block.get("summary"), "이 요약"),
    }
    return result


def _context_text(value: Any, default: str) -> str:
    result = default
    if isinstance(value, str):
        candidate = _normalize_inline_text(value).strip()
        candidate = _strip_sentence_endings(candidate)
        if candidate != "":
            result = candidate
    return result


def _strip_sentence_endings(text: str) -> str:
    result = text
    while result != "" and result[-1] in _SENTENCE_ENDINGS:
        result = result[:-1].strip()
    return result

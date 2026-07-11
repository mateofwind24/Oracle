from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from oracle_report import prompt_templates
from oracle_report.config import CaptureConfig, LlmConfig, load_capture_config
from oracle_report.models import (
    CaptureArtifact,
    FaceBox,
    FaceQuality,
)
from oracle_report.workflow import (
    CompatibilityWorkflowInput,
    PersonalWorkflowInput,
    _load_json_payload_or_error,
    _normalize_payload_text,
    _run_sequential_pair_capture,
    run_compatibility_workflow,
    run_personal_workflow,
)
from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch
from oracle_report.vision.physiognomy_text_variations import build_pair_face_payload
from oracle_report.vision.runtime import run_capture


def _report_blocks(prefix: str, count: int) -> list[dict[str, str]]:
    result = []
    for index in range(count):
        number = index + 1
        result.append(
            {
                "category": f"{prefix} 카테고리 {number}",
                "title": f"{prefix} 제목 {number}",
                "summary": f"{prefix} 요약 {number}",
                "body": _block_body(prefix, number),
            },
        )
    return result


def _block_body(prefix: str, number: int) -> str:
    sentences = []
    for sentence_index in range(prompt_templates.REPORT_BLOCK_SENTENCE_COUNT):
        sentences.append(f"{prefix} 본문 {number}-{sentence_index + 1}입니다.")
    result = " ".join(sentences)
    return result


def test_load_json_payload_ignores_text_after_first_object() -> None:
    payload, error = _load_json_payload_or_error(
        '{"essence":"핵심"}\n\n{"extra":"ignored"}',
    )

    assert error == ""
    assert payload == {"essence": "핵심"}


def test_normalize_payload_removes_summary_prefix_from_body() -> None:
    payload = {
        "saju_blocks": [
            {
                "category": "종합 형국",
                "title": "테스트 제목",
                "summary": "전반적인 흐름은 유연한 사고가 돋보여요.",
                "body": "전반적인 흐름은 유연한 사고가 돋보여요. 생활 속에서는 기준을 세우는 습관이 도움이 될 수 있어요.",
            },
        ],
    }

    result = _normalize_payload_text(payload)

    assert (
        result["saju_blocks"][0]["body"]
        == "생활 속에서는 기준을 세우는 습관이 도움이 될 수 있어요."
    )


def test_normalize_payload_removes_emoji_from_llm_text() -> None:
    payload = {
        "essence": "현실적인 기준을 살짝 챙겨주면 좋아요. 💡",
        "saju_blocks": [
            {
                "category": "올해의 운세",
                "title": "에너지 분배",
                "summary": "활동성이 올라오는 흐름이에요. 🔥",
                "body": "에너지를 한곳에 몰아쓰기보다 나누어 쓰면 좋아요. 👍",
            },
        ],
    }

    result = _normalize_payload_text(payload)

    assert result["essence"] == "현실적인 기준을 살짝 챙겨주면 좋아요."
    assert result["saju_blocks"][0]["summary"] == "활동성이 올라오는 흐름이에요."
    assert (
        result["saju_blocks"][0]["body"]
        == "에너지를 한곳에 몰아쓰기보다 나누어 쓰면 좋아요."
    )


class FakeLlmClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        prompt_str = str(prompt)
        
        # 1. Face analysis branch
        if "face_blocks" in prompt_str or "personal_face_analysis" in prompt_str:
            return json.dumps(
                {
                    "face_subtitle": "테스트 관상 소제목",
                    "face_blocks": _report_blocks("관상", 5),
                    "face_summary": "관상 요약",
                },
                ensure_ascii=False,
            )

        # 2. Distributed Split Category branch
        import re
        match = re.search(r"오직\s*'([^']+)'\s*카테고리", prompt_str)
        if match:
            cat = match.group(1).strip()
            num = 1
            saju_cats = ["종합 형국 및 원국 해설", "오행 분포와 기운 개운법", "신살 및 십성 심리 해부", 
                         "성격 분석 및 후천적 변화", "직업 적성과 사회적 무기", "재물운과 자산 설계", 
                         "연애운과 이상형 인연", "부모 및 가족 관계", "대인관계와 인맥 다이어트", 
                         "공간 및 지리적 개운 처방", "올해의 운세 및 총평"]
            compat_cats = ["관계 구조", "상호 보완", "갈등 관리", "현재 관계 흐름", "실천 제안", "총평 및 조언"]
            
            for i, c in enumerate(saju_cats):
                if c.replace(" ", "") in cat.replace(" ", ""):
                    cat = c
                    num = i + 1
                    break
            for i, c in enumerate(compat_cats):
                if c.replace(" ", "") in cat.replace(" ", ""):
                    cat = c
                    num = i + 1
                    break
            return f"""=== CATEGORY: {cat} ===
### TITLE: 사주 제목 {num}
### SUMMARY: 사주 요약 {num}
### BODY: 사주 본문 {num} 10문장 이상 작성"""

        # 3. Distributed Split Metadata branch
        if "=== METADATA ===" in prompt_str and "카테고리 블록" in prompt_str:
            if "couple" in prompt_str.lower() or "compatibility" in prompt_str.lower() or "궁합" in prompt_str or "두 사람" in prompt_str:
                return """=== METADATA ===
### ESSENCE: 두 사람 궁합 핵심 문장
### SAJU_SUBTITLE: 궁합 사주 테스트
### ACTION_TITLE: 궁합 행동 제목
### ACTION_BODY: 궁합 행동 본문
### TAGS: 궁합 테스트 태그
### DISCLAIMER: 궁합 테스트 고지"""
            else:
                return """=== METADATA ===
### ESSENCE: 사주 핵심 문장
### ELEMENT_NOTE: 사주 오행 메모
### SAJU_SUBTITLE: 사주 소제목
### TAGS: 사주 태그
### DISCLAIMER: 사주 고지"""

        # 4. Integrated Saju (Non-distributed or full report)
        if "saju_reading_couple" in prompt_str or "couple" in prompt_str.lower() or "compatibility" in prompt_str.lower() or "궁합" in prompt_str:
            res = """=== METADATA ===
### ESSENCE: 두 사람 궁합 핵심 문장
### SAJU_SUBTITLE: 궁합 사주 테스트
### ACTION_TITLE: 궁합 행동 제목
### ACTION_BODY: 궁합 행동 본문
### TAGS: 궁합 테스트 태그
### DISCLAIMER: 궁합 테스트 고지

=== CATEGORY: 관계 구조 ===
### TITLE: 관계 제목
### SUMMARY: 관계 요약
### BODY: 관계 본문 10문장 이상 작성

=== CATEGORY: 상호 보완 ===
### TITLE: 상호 보완 제목
### SUMMARY: 상호 보완 요약
### BODY: 상호 보완 본문 10문장 이상 작성

=== CATEGORY: 갈등 관리 ===
### TITLE: 갈등 관리 제목
### SUMMARY: 갈등 관리 요약
### BODY: 갈등 관리 본문 10문장 이상 작성

=== CATEGORY: 현재 관계 흐름 ===
### TITLE: 현재 관계 흐름 제목
### SUMMARY: 현재 관계 흐름 요약
### BODY: 현재 관계 흐름 본문 10문장 이상 작성

=== CATEGORY: 실천 제안 ===
### TITLE: 실천 제안 제목
### SUMMARY: 실천 제안 요약
### BODY: 실천 제안 본문 10문장 이상 작성

=== CATEGORY: 총평 및 조언 ===
### TITLE: 총평 및 조언 제목
### SUMMARY: 총평 및 조언 요약
### BODY: 총평 및 조언 본문 10문장 이상 작성"""
            return res
        else:
            categories = [
                "종합 형국 및 원국 해설", "오행 분포와 기운 개운법", "신살 및 십성 심리 해부", 
                "성격 분석 및 후천적 변화", "직업 적성과 사회적 무기", "재물운과 자산 설계", 
                "연애운과 이상형 인연", "부모 및 가족 관계", "대인관계와 인맥 다이어트", 
                "공간 및 지리적 개운 처방", "올해의 운세 및 총평"
            ]
            res = """=== METADATA ===
### ESSENCE: 사주 핵심 문장
### ELEMENT_NOTE: 사주 오행 메모
### SAJU_SUBTITLE: 사주 소제목
### TAGS: 사주 태그
### DISCLAIMER: 사주 고지"""
            for i, cat in enumerate(categories):
                res += f"""\n\n=== CATEGORY: {cat} ===
### TITLE: 사주 제목 {i+1}
### SUMMARY: 사주 요약 {i+1}
### BODY: 사주 본문 {i+1} 10문장 이상 작성"""
            return res

        return "LLM 결과"


class RecordingFaceClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[Path | None] = []

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        result = json.dumps(
            {
                "face_subtitle": "크롭 관상 소제목",
                "face_blocks": _report_blocks("크롭 관상", 5),
                "face_summary": "크롭 관상 요약",
            },
            ensure_ascii=False,
        )
        return result


class RecordingSajuClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[Path | None] = []

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        result = json.dumps(
            {
                "essence": "분리 사주 핵심",
                "element_note": "분리 사주 오행",
                "saju_subtitle": "분리 사주 소제목",
                "saju_blocks": _report_blocks("분리 사주", 6),
                "tags": ["분리 사주 태그"],
                "disclaimer": "분리 사주 고지",
            },
            ensure_ascii=False,
        )
        return result


class RecordingPairFaceClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[Path | None] = []

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        result = json.dumps(
            {
                "pair_subtitle": "PAIR FACE SUBTITLE",
                "pair_blocks": _report_blocks("PAIR FACE", 4),
                "face_summary": "PAIR FACE SUMMARY",
            },
            ensure_ascii=False,
        )
        return result


class RecordingPairSajuClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[Path | None] = []

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        result = json.dumps(
            {
                "essence": "PAIR SAJU ESSENCE",
                "saju_subtitle": "PAIR SAJU SUBTITLE",
                "saju_blocks": _report_blocks("PAIR SAJU", 6),
                "action_title": "PAIR ACTION TITLE",
                "action_body": "PAIR ACTION BODY",
                "tags": ["PAIR TAG"],
                "disclaimer": "PAIR DISCLAIMER",
            },
            ensure_ascii=False,
        )
        return result


class FailsOnFacePromptClient(FakeLlmClient):
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        is_saju_prompt = "\"saju_blocks\"" in prompt
        if ("face_blocks" in prompt or "pair_blocks" in prompt) and not is_saju_prompt:
            raise AssertionError("face LLM must not run in landmark rule mode")
        result = super().generate(prompt, image_path)
        return result


class PartialFinalReportClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        prompt_str = str(prompt)
        if "METADATA" in prompt_str or "metadata" in prompt_str.lower():
            return """=== METADATA ===
### ESSENCE: 부분 결과
### SAJU_SUBTITLE: 부분 소제목
### TAGS: 부분, 테스트, 사주
### DISCLAIMER: 부분 고지"""
        import re
        match = re.search(r"오직\s*'([^']+)'\s*카테고리", prompt_str)
        if match:
            cat = match.group(1).strip()
            if "종합" in cat or "원국" in cat:
                return f"""=== CATEGORY: {cat} ===
### TITLE: 부분 제목
### SUMMARY: 부분 요약
### BODY: 부분 본문이 충분히 긴 내용으로 10문장 이상 작성되어 있습니다. 아주 꽉 차 있어요."""
            else:
                return "FAIL"
        result = json.dumps(
            {
                "essence": "부분 결과",
                "saju_blocks": [
                    {
                        "category": "부분",
                        "title": "부분 제목",
                        "summary": "부분 요약",
                        "body": "부분 본문",
                    },
                ],
            },
            ensure_ascii=False,
        )
        return result


class NewlineBodyReportClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        prompt_str = str(prompt)
        if "METADATA" in prompt_str or "metadata" in prompt_str.lower():
            return """=== METADATA ===
### ESSENCE: newline payload
### SAJU_SUBTITLE: newline subtitle
### TAGS: newline, test, tags
### DISCLAIMER: newline disclaimer"""
        import re
        match = re.search(r"오직\s*'([^']+)'\s*카테고리", prompt_str)
        if match:
            cat = match.group(1).strip()
            return f"""=== CATEGORY: {cat} ===
### TITLE: newline title
### SUMMARY: newline summary
### BODY: first line\\nsecond line\nthird line가 충분한 길이로 들어가 있습니다. 이 본문은 아주 깁니다."""
        result = json.dumps(
            {
                "essence": "newline payload",
                "saju_blocks": [
                    {
                        "category": "newline category",
                        "title": "newline title",
                        "summary": "newline summary",
                        "body": "first line\\nsecond line\nthird line",
                    },
                ],
            },
            ensure_ascii=False,
        )
        return result


class MalformedJsonReportClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        prompt_str = str(prompt)
        if "METADATA" in prompt_str or "metadata" in prompt_str.lower():
            return """```json
{
    “essence”: “보정된 결과예요.”,
    “tags”: [“보정”, “테스트”, “태그”],
    “disclaimer”: “보정 고지예요.”
}
```"""
        import re
        match = re.search(r"오직\s*'([^']+)'\s*카테고리", prompt_str)
        if match:
            cat = match.group(1).strip()
            if "종합" in cat or "원국" in cat:
                return f"""```json
{{
    “category”: “{cat}”
    “title”: “보정 제목”,
    “summary”: “보정 요약은 충분한 길이를 갖고 있어요.”,
    “body”: “보정 본문은 쉼표가 빠졌더라도 후처리로 복구되어 UI에 반영되어야 해요.”
}}
```"""
            else:
                return f"""=== CATEGORY: {cat} ===
### TITLE: 보정 제목
### SUMMARY: 보정 요약은 충분한 길이를 갖고 있어요.
### BODY: 보정 본문은 쉼표가 빠졌더라도 후처리로 복구되어 UI에 반영되어야 해요."""
        return """```json
{
    “essence”: “보정된 결과예요.”,
    “saju_blocks”: [
        {
            “category”: “보정 카테고리”
            “title”: “보정 제목”,
            “summary”: “보정 요약은 충분한 길이를 갖고 있어요.”,
            “body”: “보정 본문은 쉼표가 빠졌더라도 후처리로 복구되어 UI에 반영되어야 해요.”
        },
    ],
    “disclaimer”: “보정 고지예요.”
}
```"""


def test_llm_json_repair_handles_misclosed_saju_blocks_and_missing_comma() -> None:
    text = """```json
{
  "essence": "궁합 요약",
  "saju_blocks": [
    {
      "category": "관계 구조",
      "title": "관계 제목",
      "summary": "관계 요약",
      "body": "관계 본문"
    }
  },
  "action_title": "행동 제목",
  "action_body": "행동 본문"
  "tags": ["협업", "조율", "성장", "존중"],
  "disclaimer": "참고용 엔터테인먼트 리포트입니다."
}
```"""

    payload, error = _load_json_payload_or_error(text, label="test_repair")

    assert error == ""
    assert payload["saju_blocks"][0]["category"] == "관계 구조"
    assert payload["action_body"] == "행동 본문"
    assert payload["tags"] == ["협업", "조율", "성장", "존중"]


class DayMasterHonorificReportClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        prompt_str = str(prompt)
        if "METADATA" in prompt_str or "metadata" in prompt_str.lower():
            return """=== METADATA ===
### ESSENCE: 임수님은 넓은 시야가 돋보여요.
### ELEMENT_NOTE: 임수님에게는 현실 감각을 보완하는 흐름이 필요해요.
### SAJU_SUBTITLE: 임수님의 균형
### TAGS: 임수, 변화
### DISCLAIMER: 참고용 해석이에요."""
        import re
        match = re.search(r"오직\s*'([^']+)'\s*카테고리", prompt_str)
        if match:
            cat = match.group(1).strip()
            return f"""=== CATEGORY: {cat} ===
### TITLE: 임수님이 잡아야 할 중심
### SUMMARY: 님은 변화에 강한 흐름을 보여요.
### BODY: 임수님은 큰 흐름을 보는 힘이 있어요. 임수님은 변화를 잘 받아들일 수 있어요. 이 문장은 충분히 깁니다."""
        result = json.dumps(
            {
                "essence": "임수님은 넓은 시야가 돋보여요.",
                "element_note": "임수님에게는 현실 감각을 보완하는 흐름이 필요해요.",
                "saju_subtitle": "임수님의 균형",
                "saju_blocks": [
                    {
                        "category": "종합 형국",
                        "title": "임수님이 잡아야 할 중심",
                        "summary": "님은 변화에 강한 흐름을 보여요.",
                        "body": "임수님은 큰 흐름을 보는 힘이 있어요. 임수님은 변화를 잘 받아들일 수 있어요.",
                    },
                ],
                "tags": ["임수", "변화"],
                "disclaimer": "참고용 해석이에요.",
            },
            ensure_ascii=False,
        )
        return result


def test_personal_workflow_runs_without_real_camera_or_llm(
    tmp_path: Path,
    capsys,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
    )

    assert result.output_path.exists()
    assert result.timing_log_path is not None
    assert result.timing_log_path.exists()
    assert "capture" in result.timing_log_path.read_text(encoding="utf-8")
    assert "[timing] capture:" in capsys.readouterr().out
    assert result.output_path.name == "personal_report.html"
    assert result.report_html.startswith("<!DOCTYPE html>")
    assert "oracle-report" in result.report_fragment_html
    assert "face_blocks" in result.face_analysis
    assert "타고난 인상과 기본 상" in result.report_html


def test_personal_workflow_uses_rule_based_face_by_default(tmp_path: Path) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="모름",
        gender="남성",
        target_gender="여성",
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
    )

    assert "face_blocks" in result.face_analysis
    assert result.output_path.suffix == ".html"
    assert "oracle-report" in result.report_html
    assert "시간 미상" in result.report_html


def test_personal_workflow_rulebase_mode_skips_face_llm(
    tmp_path: Path,
) -> None:
    capture_config = replace(_capture_config(tmp_path), mock_capture_enabled=True)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="모름",
        gender="남성",
        target_gender="여성",
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FailsOnFacePromptClient(),
        capture_runner=run_capture,
    )
    payload = json.loads(result.face_analysis)

    assert payload["face_subtitle"]
    assert len(payload["face_blocks"]) == 5
    assert "타고난 인상과 기본 상" in result.report_html
    assert "랜드마크 룰 기반" not in result.face_analysis


def test_personal_workflow_uses_rule_based_face_by_default(
    tmp_path: Path,
) -> None:
    capture_config = replace(_capture_config(tmp_path), mock_capture_enabled=True)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="tester",
        birth_date="1995-03-15",
        birth_time="",
        gender="male",
        target_gender="female",
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FailsOnFacePromptClient(),
        capture_runner=run_capture,
    )
    payload = json.loads(result.face_analysis)

    assert payload["face_subtitle"]
    assert len(payload["face_blocks"]) == 5


def test_compatibility_workflow_runs_without_real_camera_or_llm(tmp_path: Path) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = CompatibilityWorkflowInput(
        left_name="갑",
        left_birth_date="1995-03-15",
        left_birth_time="14:30",
        left_gender="남성",
        right_name="을",
        right_birth_date="1997-05-20",
        right_birth_time="",
        right_gender="여성",
        mode="연인",
    )

    result = run_compatibility_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
        inter_capture_delay_seconds=0.0,
    )

    assert result.output_path.exists()
    assert result.timing_log_path is not None
    assert result.timing_log_path.exists()
    assert "compatibility_workflow" in result.timing_log_path.read_text(
        encoding="utf-8",
    )
    assert result.output_path.name == "compatibility_report.html"
    assert result.report_html.startswith("<!DOCTYPE html>")
    assert "oracle-report" in result.report_fragment_html
    assert "두 사람 궁합 핵심 문장" in result.report_html
    assert "궁합 점수" in result.report_html
    assert result.left_capture_path.parent.name == "person_1"
    assert result.right_capture_path.parent.name == "person_2"


def test_compatibility_workflow_rulebase_mode_skips_face_llm(
    tmp_path: Path,
) -> None:
    capture_config = replace(_capture_config(tmp_path), mock_capture_enabled=True)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = CompatibilityWorkflowInput(
        left_name="갑",
        left_birth_date="1995-03-15",
        left_birth_time="14:30",
        left_gender="남성",
        right_name="을",
        right_birth_date="1997-05-20",
        right_birth_time="",
        right_gender="여성",
        mode="연인",
    )

    result = run_compatibility_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FailsOnFacePromptClient(),
        capture_runner=run_capture,
        inter_capture_delay_seconds=0.0,
    )
    payload = json.loads(result.face_analysis)

    assert payload["pair_subtitle"]
    assert len(payload["pair_blocks"]) == 4
    assert "연인 관계에서는" in payload["pair_blocks"][0]["summary"]
    assert "두 사람의 관계 분위기" in result.report_html
    assert "궁합 점수" in result.report_html


def test_pair_rulebase_payload_reflects_compatibility_mode() -> None:
    matches = (
        PhysiognomyRuleMatch(
            rule_id="balance",
            metric="third_balance_error",
            title="삼정 균형",
            basis="test",
            tag="삼정 균형형",
            observation="삼정 비율이 안정적으로 관찰됩니다.",
            interpretation="균형 잡힌 인상으로 읽힙니다.",
            value=0.01,
        ),
        PhysiognomyRuleMatch(
            rule_id="mouth",
            metric="mouth_width_ratio",
            title="입매",
            basis="test",
            tag="입매 균형형",
            observation="입매가 자연스럽게 균형을 이룹니다.",
            interpretation="편안한 표현으로 읽힙니다.",
            value=0.35,
        ),
    )

    lover = build_pair_face_payload(matches, matches, "갑", "을", "same-seed", mode="연인")
    friend = build_pair_face_payload(matches, matches, "갑", "을", "same-seed", mode="친구")
    coworker = build_pair_face_payload(
        matches,
        matches,
        "갑",
        "을",
        "same-seed",
        mode="직장동료",
    )

    assert "감정" in lover["face_summary"]
    assert "친구 관계에서는" in friend["pair_blocks"][0]["summary"]
    assert "직장동료 관계에서는" in coworker["pair_blocks"][0]["summary"]
    assert lover["pair_blocks"][0]["body"] != coworker["pair_blocks"][0]["body"]
    for block in lover["pair_blocks"]:
        assert "갑님은" in block["body"]
        assert "을님은" in block["body"]


def test_pair_mock_capture_can_use_different_landmark_metrics(
    tmp_path: Path,
) -> None:
    capture_config = replace(
        _capture_config(tmp_path),
        mock_capture_enabled=True,
        mock_pair_left_landmark_metrics_json='{"eye_width_ratio": 0.19}',
        mock_pair_right_landmark_metrics_json='{"mouth_width_ratio": 0.43}',
    )

    result = _run_sequential_pair_capture(
        run_capture,
        capture_config,
        tmp_path / "pair-runs",
        0.0,
    )

    assert "눈 가로폭/얼굴 폭: 0.190" in result.left.quality.landmark_metrics_text
    assert "입 폭/얼굴 폭: 0.430" in result.right.quality.landmark_metrics_text
    assert result.left.image_path.parent.name == "person_1"
    assert result.right.image_path.parent.name == "person_2"


def test_mock_capture_env_enables_default_landmark_metrics(monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_MOCK_CAPTURE_ENABLED", "1")

    result = load_capture_config()

    assert result.mock_capture_enabled is True
    assert "eye_width_ratio" in result.mock_landmark_metrics_json
    assert "eye_width_ratio" in result.mock_pair_left_landmark_metrics_json
    assert "mouth_width_ratio" in result.mock_pair_right_landmark_metrics_json


def _build_test_manse_db(tmp_path: Path) -> Path:
    result = tmp_path / "unused-manse.sqlite"
    return result


def _capture_config(tmp_path: Path) -> CaptureConfig:
    result = CaptureConfig(
        camera_index=0,
        frame_width=640,
        frame_height=480,
        camera_fps=15,
        min_face_seconds=2.0,
        face_min_size_px=96,
        face_detection_scale=0.5,
        face_detection_interval=2,
        output_dir=tmp_path / "runs",
        show_preview=False,
        eye_min_count=2,
        eyebrow_min_edge_density=0.018,
    )
    return result


def _llm_config() -> LlmConfig:
    result = LlmConfig(
        model="fake",
        base_url="http://127.0.0.1:8080/v1",
        timeout_seconds=1.0,
        max_output_tokens=128,
        temperature=0.1,
    )
    return result


def _fake_single_capture(
    config: CaptureConfig,
    output_dir: Path | None = None,
) -> CaptureArtifact:
    destination = output_dir or config.output_dir
    destination.mkdir(parents=True, exist_ok=True)
    image_path = destination / "capture.jpg"
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[:, :] = (32, 48, 64)
    ok = cv2.imwrite(str(image_path), image)
    if not ok:
        raise RuntimeError(f"failed to write fake capture image: {image_path}")
    face_payload_json = json.dumps(
        {
            "face_subtitle": "rule subtitle",
            "face_blocks": _report_blocks("rule face", 5),
            "face_summary": "rule summary",
        },
        ensure_ascii=False,
    )
    result = CaptureArtifact(
        image_path=image_path,
        face=FaceBox(10, 10, 120, 120),
        captured_at=datetime(2026, 1, 1, 12, 0),
        quality=FaceQuality(
            ready=True,
            eye_count=2,
            eyebrow_score=0.05,
            face_analysis="## 관상정보\n- 분석 모드: 랜드마크 룰 기반",
            landmark_matches_json=json.dumps(
                [
                    {
                        "rule_id": "balance_mid",
                        "metric": "third_balance_error",
                        "title": "균형 잡힌 삼정",
                        "basis": "삼정 균형",
                        "tag": "balance",
                        "observation": "얼굴의 세로 균형이 안정적으로 읽혀요.",
                        "interpretation": "전체 인상에서 안정감과 차분함이 드러날 수 있어요.",
                        "value": 0.08,
                    },
                    {
                        "rule_id": "eyes_mid",
                        "metric": "eye_spacing_ratio",
                        "title": "차분한 눈매",
                        "basis": "눈 간격",
                        "tag": "eyes",
                        "observation": "눈 주변의 간격이 자연스럽게 잡혀 있어요.",
                        "interpretation": "상황을 관찰하고 반응하는 리듬이 차분하게 보일 수 있어요.",
                        "value": 0.31,
                    },
                    {
                        "rule_id": "nose_mid",
                        "metric": "nose_width_ratio",
                        "title": "중심 잡힌 코",
                        "basis": "코 폭",
                        "tag": "nose",
                        "observation": "코의 중심감이 얼굴 폭과 무난하게 어울려요.",
                        "interpretation": "중요한 선택에서 균형을 잡으려는 경향이 나타날 수 있어요.",
                        "value": 0.19,
                    },
                    {
                        "rule_id": "mouth_mid",
                        "metric": "mouth_width_ratio",
                        "title": "자연스러운 입매",
                        "basis": "입 폭",
                        "tag": "mouth",
                        "observation": "입매가 얼굴 폭과 자연스럽게 조화를 이뤄요.",
                        "interpretation": "말과 감정을 편안하게 조율하는 인상으로 이어질 수 있어요.",
                        "value": 0.39,
                    },
                    {
                        "rule_id": "jaw_mid",
                        "metric": "jaw_width_ratio",
                        "title": "안정적인 하관",
                        "basis": "하관 폭",
                        "tag": "jaw",
                        "observation": "하관의 폭이 전체 얼굴과 안정적으로 맞아요.",
                        "interpretation": "마무리와 지속력에서 차분한 장점이 드러날 수 있어요.",
                        "value": 0.72,
                    },
                ],
                ensure_ascii=False,
            ),
        ),
        face_analysis="## 관상정보\n- 분석 모드: 랜드마크 룰 기반",
    )
    return result


def test_personal_workflow_skips_face(tmp_path: Path) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=None,
    )

    assert result.output_path.exists()
    assert result.capture_path is None
    assert result.face_analysis == ""
    assert "saju_analysis" in result.timing_log_path.read_text(encoding="utf-8")
    assert "사주 핵심 문장" in result.report_html
    assert "사주 제목 1" in result.report_html
    assert "FACE MATCH" not in result.report_html
    assert "궁합 좋은 얼굴 추천" not in result.report_html
    assert "관상" not in result.report_html


def test_personal_workflow_normalizes_newline_markers_in_output_body(
    tmp_path: Path,
    capsys,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="tester",
        birth_date="1995-03-15",
        birth_time="",
        gender="male",
        target_gender="female",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=NewlineBodyReportClient(),
        capture_runner=None,
    )
    saved_markdown = (result.output_path.parent / "personal_report.md").read_text(
        encoding="utf-8",
    )
    captured = capsys.readouterr().out

    assert "first line second line third line" in result.markdown
    assert "first line second line third line" in saved_markdown
    assert "first line second line third line" in result.report_html
    assert "\\n" not in result.markdown
    assert "\\n" not in saved_markdown
    assert "\\n" not in result.report_html
    assert "\\n" not in captured


def test_json_payload_loader_repairs_common_llm_format_errors(capsys) -> None:
    payload, error = _load_json_payload_or_error(
        """```json
{
    “face_subtitle”: “보정된 관상 소제목”,
    “face_blocks”: [
        {
            “category”: “타고난 인상과 기본 상”
            “title”: “보정 제목”,
            “summary”: “보정 요약은 충분한 길이를 갖고 있어요.”,
            “body”: “보정 본문은 쉼표가 빠졌더라도 후처리로 복구되어 UI에 반영되어야 해요.”
        },
    ],
    “face_summary”: “보정된 요약이에요.”
}
```""",
        label="face_analysis",
    )

    captured = capsys.readouterr()
    assert error == ""
    assert "[LLM JSON REPAIR:face_analysis] applied repairs:" in captured.out
    assert "normalize_quotes" in captured.out
    assert "insert_missing_commas" in captured.out
    assert "remove_trailing_commas" in captured.out
    assert payload["face_subtitle"] == "보정된 관상 소제목"
    assert payload["face_blocks"][0]["title"] == "보정 제목"


def test_personal_workflow_uses_repaired_saju_json_output(tmp_path: Path) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=MalformedJsonReportClient(),
        capture_runner=None,
    )

    assert "보정 제목" in result.report_html
    assert "보정된 결과예요." in result.report_html
    assert "보정 고지예요." in result.report_html


def test_personal_workflow_replaces_day_master_honorific_with_name(
    tmp_path: Path,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=DayMasterHonorificReportClient(),
        capture_runner=None,
    )

    assert "임수님" not in result.markdown
    assert "임수님" not in result.report_html
    assert "홍길동님" not in result.markdown
    assert "홍길동님" not in result.report_html
    assert "넓은 시야" in result.markdown
    assert "변화에 강한 흐름" in result.markdown
    assert "균형" in result.report_html
    assert "님은" not in result.markdown
    assert '"임수"' not in result.markdown
    assert '"물 기운"' in result.markdown


def test_personal_workflow_keeps_partial_saju_json_without_full_ui_fallback(
    tmp_path: Path,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=PartialFinalReportClient(),
        capture_runner=None,
    )

    assert "부분 제목" in result.report_html
    assert "데이터 일시적 유실" in result.report_html
    assert "전체 흐름을 정리하면" not in result.report_html
    assert "사주/만세력 데이터에 나타난 오행 분포와 생활 리듬" not in result.report_html
    assert "나를 채워주는 키워드" in result.report_html
    assert "final report JSON field saju_blocks has 1 blocks" not in result.markdown
def test_personal_saju_follows_main_when_distributed_split_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORACLE_DISTRIBUTED_ROLE", "master")
    monkeypatch.setenv("ORACLE_DISTRIBUTED_SPLIT", "1")
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        skip_face=True,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=None,
    )

    assert "사주 핵심 문장" in result.report_html
    assert "사주 제목 1" in result.report_html
    assert "사주정보를 생성하지 못했습니다" not in result.report_html


def test_compatibility_saju_follows_main_when_distributed_split_enabled(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORACLE_DISTRIBUTED_ROLE", "master")
    monkeypatch.setenv("ORACLE_DISTRIBUTED_SPLIT", "1")
    monkeypatch.setenv("ORACLE_FACE_ANALYSIS_MODE", "2")
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = CompatibilityWorkflowInput(
        left_name="갑",
        left_birth_date="1995-03-15",
        left_birth_time="14:30",
        left_gender="남성",
        right_name="을",
        right_birth_date="1997-05-20",
        right_birth_time="",
        right_gender="여성",
        mode="연인",
    )

    result = run_compatibility_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
        inter_capture_delay_seconds=0.0,
    )

    assert "두 사람 궁합 핵심 문장" in result.report_html
    assert "궁합 사주정보를 생성하지 못했습니다" not in result.report_html


def test_speculative_work_stealing_from_local() -> None:
    class MockScheduler:
        def __init__(self):
            self.slave_metadata = {
                "local": {"compute_score": 5.0},
                "http://slave:7000": {"compute_score": 15.0}
            }

    scheduler = MockScheduler()
    active_assignments = {
        "local": {"is_metadata": False, "target_category": "종합 형국"}
    }
    completed_tasks = set()

    def is_task_done(task):
        key = (task["is_metadata"], task["target_category"])
        return key in completed_tasks

    def find_unfinished_speculative_task(my_url, is_my_local):
        import copy
        for worker_url, assigned_task in active_assignments.items():
            if assigned_task is None:
                continue
            is_other_local = (worker_url == "local")
            if is_my_local and not is_other_local:
                if not is_task_done(assigned_task):
                    return copy.deepcopy(assigned_task)
            elif not is_my_local and worker_url != my_url:
                my_score = scheduler.slave_metadata.get(my_url, {}).get("compute_score", 5.0)
                other_score = scheduler.slave_metadata.get(worker_url, {}).get("compute_score", 5.0)
                if my_score > other_score:
                    if not is_task_done(assigned_task):
                        return copy.deepcopy(assigned_task)
        return None

    stolen_task = find_unfinished_speculative_task("http://slave:7000", is_my_local=False)
    assert stolen_task is not None
    assert stolen_task["target_category"] == "종합 형국"

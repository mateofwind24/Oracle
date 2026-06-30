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


class FakeLlmClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        result = "LLM 결과"
        if "\"face_blocks\"" in prompt and image_path is not None:
            result = json.dumps(
                {
                    "face_subtitle": "테스트 관상 소제목",
                    "face_blocks": _report_blocks("관상", 5),
                    "face_summary": "관상 요약",
                },
                ensure_ascii=False,
            )
        elif "pair_blocks" in prompt:
            result = json.dumps(
                {
                    "essence": "두 사람 궁합 핵심 문장",
                    "pair_subtitle": "관계 분위기 테스트",
                    "pair_blocks": [
                        {
                            "category": "관계 카테고리",
                            "title": "관계 제목",
                            "summary": "관계 요약",
                            "body": _block_body("관계", 1),
                        },
                    ],
                    "saju_subtitle": "궁합 사주 테스트",
                    "saju_blocks": [
                        {
                            "category": "궁합 사주 카테고리",
                            "title": "궁합 사주 제목",
                            "summary": "궁합 사주 요약",
                            "body": _block_body("궁합 사주", 1),
                        },
                    ],
                    "synthesis_title": "궁합 종합 제목",
                    "synthesis_body": "궁합 종합 본문",
                    "convergence": [
                        {"face": "궁합 관상 근거", "saju": "궁합 사주 근거"},
                    ],
                    "action_title": "궁합 행동 제목",
                    "action_body": "궁합 행동 본문",
                    "tags": ["궁합 테스트 태그"],
                    "disclaimer": "궁합 테스트 고지",
                },
                ensure_ascii=False,
            )
        elif "\"saju_blocks\"" in prompt:
            result = json.dumps(
                {
                    "essence": "사주 핵심 문장",
                    "element_note": "사주 오행 메모",
                    "saju_subtitle": "사주 소제목",
                    "saju_blocks": _report_blocks("사주", 6),
                    "tags": ["사주 태그"],
                    "disclaimer": "사주 고지",
                },
                ensure_ascii=False,
            )
        elif "face_blocks" in prompt:
            result = json.dumps(
                {
                    "essence": "테스트 핵심 문장",
                    "element_note": "테스트 오행 메모",
                    "face_subtitle": "테스트 관상 소제목",
                    "face_blocks": _report_blocks("관상", 5),
                    "saju_subtitle": "테스트 사주 소제목",
                    "saju_blocks": _report_blocks("사주", 6),
                    "synthesis_title": "종합 제목",
                    "synthesis_body": "종합 본문",
                    "convergence": [
                        {"face": "관상 수렴", "saju": "사주 수렴"},
                    ],
                    "synthesis_summary": "종합 요약",
                    "tags": ["테스트 태그"],
                    "recommendation_title": "추천 제목",
                    "recommendation_lead": "추천 리드",
                    "disclaimer": "테스트 고지",
                },
                ensure_ascii=False,
            )
        return result


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
                "synthesis_title": "PAIR SYNTHESIS TITLE",
                "synthesis_body": "PAIR SYNTHESIS BODY",
                "action_title": "PAIR ACTION TITLE",
                "action_body": "PAIR ACTION BODY",
                "tags": ["PAIR TAG"],
                "disclaimer": "PAIR DISCLAIMER",
            },
            ensure_ascii=False,
        )
        return result


class FailingFaceClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        raise AssertionError("face LLM must not run in landmark rule mode")


class FailsOnFacePromptClient(FakeLlmClient):
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        is_saju_prompt = "\"saju_blocks\"" in prompt
        if ("face_blocks" in prompt or "pair_blocks" in prompt) and not is_saju_prompt:
            raise AssertionError("face LLM must not run in landmark rule mode")
        result = super().generate(prompt, image_path)
        return result


class PartialFinalReportClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        del prompt
        del image_path
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
        del prompt
        del image_path
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


class RepairingShortBodyReportClient:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.image_paths: list[Path | None] = []

    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        self.prompts.append(prompt)
        self.image_paths.append(image_path)
        body = "첫 문장입니다. 두 번째 문장입니다."
        if "[리포트 본문 재작성]" in prompt:
            body = (
                "첫 문장입니다. 두 번째 문장입니다. "
                "세 번째 문장입니다. 네 번째 문장입니다."
            )
        result = json.dumps(
            {
                "essence": "짧은 본문 핵심",
                "saju_subtitle": "짧은 본문 소제목",
                "saju_blocks": [
                    {
                        "category": "종합 형국",
                        "title": "짧은 본문",
                        "summary": "핵심 요약입니다.",
                        "body": body,
                    },
                ],
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
        recommendation_db_path=tmp_path / "faces.sqlite",
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
    assert "관상 제목 1" in result.report_html
    assert len(result.recommendations) > 0


def test_personal_workflow_uses_rule_based_face(tmp_path: Path) -> None:
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
        recommendation_db_path=tmp_path / "faces.sqlite",
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
    )

    assert "face_blocks" in result.face_analysis
    assert result.output_path.suffix == ".html"
    assert "oracle-report" in result.report_html
    assert "시간 미상" in result.report_html


def test_personal_workflow_rulebase_mode_skips_face_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORACLE_FACE_ANALYSIS_MODE", "2")
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
        recommendation_db_path=tmp_path / "faces.sqlite",
        report_client=FailsOnFacePromptClient(),
        capture_runner=run_capture,
    )
    payload = json.loads(result.face_analysis)

    assert payload["face_subtitle"]
    assert len(payload["face_blocks"]) == 5
    assert "타고난 인상과 기본 상" in result.report_html
    assert "랜드마크 룰 기반" not in result.face_analysis


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
    assert "궁합 행동 제목" in result.report_html
    assert result.left_capture_path.parent.name == "person_1"
    assert result.right_capture_path.parent.name == "person_2"


def test_compatibility_workflow_rulebase_mode_skips_face_llm(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("ORACLE_FACE_ANALYSIS_MODE", "2")
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
        send_image=False,
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
    result = CaptureArtifact(
        image_path=image_path,
        face=FaceBox(10, 10, 120, 120),
        captured_at=datetime(2026, 1, 1, 12, 0),
        quality=FaceQuality(
            ready=True,
            eye_count=2,
            eyebrow_score=0.05,
            face_analysis="## 관상정보\n- 분석 모드: 랜드마크 룰 기반",
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
        recommendation_db_path=tmp_path / "faces.sqlite",
        report_client=FakeLlmClient(),
        capture_runner=None,
    )

    assert result.output_path.exists()
    assert result.capture_path is None
    assert result.face_analysis == ""
    assert result.recommendations == ()
    assert "recommend_faces" not in result.timing_log_path.read_text(encoding="utf-8")
    assert "saju_analysis" in result.timing_log_path.read_text(encoding="utf-8")
    assert "사주 핵심 문장" in result.report_html
    assert "사주 제목 1" in result.report_html
    assert "FACE MATCH" not in result.report_html
    assert "궁합 좋은 얼굴 추천" not in result.report_html
    assert "관상" not in result.report_html


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
        recommendation_db_path=tmp_path / "faces.sqlite",
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
    assert "궁합 행동 제목" in result.report_html
    assert "궁합 사주정보를 생성하지 못했습니다" not in result.report_html


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
        face_llm_config=_llm_config(),
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        recommendation_db_path=tmp_path / "faces.sqlite",
        face_client=FailingFaceClient(),
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


def test_personal_workflow_rewrites_short_block_body_with_llm(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(prompt_templates, "REPORT_BLOCK_SENTENCE_COUNT", 4)
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
    report_client = RepairingShortBodyReportClient()

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        face_llm_config=_llm_config(),
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        recommendation_db_path=tmp_path / "faces.sqlite",
        face_client=FailingFaceClient(),
        report_client=report_client,
        capture_runner=None,
    )
    payload = json.loads(result.markdown)
    body = payload["saju_blocks"][0]["body"]

    assert len(report_client.prompts) == 2
    assert report_client.image_paths == [None, None]
    assert "[리포트 본문 재작성]" in report_client.prompts[1]
    assert "summary는 body의 핵심을 1~2개의 짧은 문장" in report_client.prompts[1]
    assert "body는 정확히 4개의 완성된 문장" in report_client.prompts[1]
    assert body.count(".") == 4
    assert "첫 문장입니다. 두 번째 문장입니다." in body
    assert "이 내용은 종합 형국 흐름" not in body
    assert body in result.report_html


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
        recommendation_db_path=tmp_path / "faces.sqlite",
        report_client=PartialFinalReportClient(),
        capture_runner=None,
    )

    assert "부분 제목" in result.report_html
    assert "오행 분포는" in result.report_html
    assert "사주 데이터는 강점과 보완점을 함께 보여주는 참고 지도입니다." in result.report_html
    assert "final report JSON field saju_blocks has 1 blocks" not in result.markdown

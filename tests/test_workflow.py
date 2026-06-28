from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.models import (
    CaptureArtifact,
    FaceBox,
    FaceQuality,
)
from oracle_report.workflow import (
    CompatibilityWorkflowInput,
    PersonalWorkflowInput,
    run_compatibility_workflow,
    run_personal_workflow,
)


def _report_blocks(prefix: str, count: int) -> list[dict[str, str]]:
    result = []
    for index in range(count):
        number = index + 1
        result.append(
            {
                "category": f"{prefix} 카테고리 {number}",
                "title": f"{prefix} 제목 {number}",
                "summary": f"{prefix} 요약 {number}",
                "body": f"{prefix} 본문 {number}",
            },
        )
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
                            "body": "관계 본문",
                        },
                    ],
                    "saju_subtitle": "궁합 사주 테스트",
                    "saju_blocks": [
                        {
                            "category": "궁합 사주 카테고리",
                            "title": "궁합 사주 제목",
                            "summary": "궁합 사주 요약",
                            "body": "궁합 사주 본문",
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
                "saju_blocks": _report_blocks("PAIR SAJU", 4),
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
    assert "랜드마크 룰 기반" in result.face_analysis
    assert "얼굴 관찰에서 보이는 표현 리듬" in result.report_html
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

    assert "랜드마크 룰 기반" in result.face_analysis
    assert result.output_path.suffix == ".html"
    assert "oracle-report" in result.report_html
    assert "시간 미상" in result.report_html


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


def test_personal_workflow_status_callback_progressive(
    tmp_path: Path,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="14:30",
        gender="남성",
        target_gender="여성",
    )

    callback_calls = []
    def status_callback(phase: str, message: str, html: str = "") -> None:
        callback_calls.append((phase, message, html))

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        recommendation_db_path=tmp_path / "faces.sqlite",
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
        status_callback=status_callback,
    )

    assert len(callback_calls) > 0
    phase, message, html = callback_calls[0]
    assert phase == "generating"
    assert "사주" in message
    assert html != ""
    assert "사주 핵심 문장" in html


def test_compatibility_workflow_status_callback_progressive(
    tmp_path: Path,
) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = CompatibilityWorkflowInput(
        left_name="홍길동",
        left_birth_date="1995-03-15",
        left_birth_time="14:30",
        left_gender="남성",
        right_name="성춘향",
        right_birth_date="1996-04-20",
        right_birth_time="10:00",
        right_gender="여성",
        mode="연인",
    )

    callback_calls = []
    def status_callback(phase: str, message: str, html: str = "") -> None:
        callback_calls.append((phase, message, html))

    result = run_compatibility_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
        inter_capture_delay_seconds=0.0,
        status_callback=status_callback,
    )

    assert len(callback_calls) > 0
    phase, message, html = callback_calls[0]
    assert phase == "generating"
    assert "궁합" in message
    assert html != ""


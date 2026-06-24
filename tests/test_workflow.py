from __future__ import annotations

from datetime import datetime
from pathlib import Path

from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.models import (
    CaptureArtifact,
    FaceBox,
    FaceQuality,
)
from oracle_report.saju.repository import build_manse_database
from oracle_report.workflow import (
    CompatibilityWorkflowInput,
    PersonalWorkflowInput,
    run_compatibility_workflow,
    run_personal_workflow,
)


class FakeLlmClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        result = "LLM 결과"
        if "출력 형식" in prompt and image_path is not None:
            result = "## 관상정보\n- 얼굴 인상 태그: 차분함"
        elif "Oracle 종합 리포트" in prompt:
            result = "# 개인 리포트\n## 내 관상과 궁합 좋은 이성 얼굴 추천\n추천 포함"
        elif "궁합 리포트" in prompt:
            result = "# 궁합 리포트\n## 관계를 좋게 만드는 행동 제안\n대화"
        return result


class FailingFaceClient:
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        raise AssertionError("face LLM must not run in landmark rule mode")


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
        face_llm_config=_llm_config(),
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        recommendation_db_path=tmp_path / "faces.sqlite",
        face_client=FakeLlmClient(),
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
    )

    assert result.output_path.exists()
    assert result.timing_log_path is not None
    assert result.timing_log_path.exists()
    assert "capture" in result.timing_log_path.read_text(encoding="utf-8")
    assert "[timing] capture:" in capsys.readouterr().out
    assert "개인 리포트" in result.markdown
    assert len(result.recommendations) > 0


def test_personal_workflow_uses_rule_based_face_mode(tmp_path: Path) -> None:
    capture_config = _capture_config(tmp_path)
    manse_db_path = _build_test_manse_db(tmp_path)
    workflow_input = PersonalWorkflowInput(
        name="홍길동",
        birth_date="1995-03-15",
        birth_time="",
        gender="남성",
        target_gender="여성",
        face_analysis_mode=2,
    )

    result = run_personal_workflow(
        workflow_input=workflow_input,
        capture_config=capture_config,
        face_llm_config=_llm_config(),
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        recommendation_db_path=tmp_path / "faces.sqlite",
        face_client=FailingFaceClient(),
        report_client=FakeLlmClient(),
        capture_runner=_fake_single_capture,
    )

    assert "랜드마크 룰 기반" in result.face_analysis
    assert "개인 리포트" in result.markdown


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
        face_llm_config=_llm_config(),
        report_llm_config=_llm_config(),
        manse_db_path=manse_db_path,
        face_client=FakeLlmClient(),
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
    assert "궁합 리포트" in result.markdown
    assert result.left_capture_path.parent.name == "person_1"
    assert result.right_capture_path.parent.name == "person_2"


def _build_test_manse_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1997)
    result = db_path
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
    image_path.write_bytes(b"fake")
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

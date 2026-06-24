from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import time
from pathlib import Path
from typing import Callable, Generic, Protocol, TypeVar

from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import (
    BirthProfile,
    CaptureArtifact,
    SequentialPairCaptureArtifact,
)
from oracle_report.recommender import (
    FaceRecommendation,
    format_recommendations,
    recommend_faces,
)
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import (
    build_compatibility_final_prompt,
    build_compatibility_face_analysis_prompt,
    build_personal_face_analysis_prompt,
    build_personal_final_prompt,
)
from oracle_report.report_html import render_personal_report_html
from oracle_report.saju.repository import ManseLookupResult, ManseRepository
from oracle_report.vision.runtime import run_capture


COMPATIBILITY_MODES = ("연인", "친구", "직장동료")
FACE_ANALYSIS_MODE_LLM_IMAGE = 1
FACE_ANALYSIS_MODE_LANDMARK_RULE = 2
FACE_ANALYSIS_MODES = (FACE_ANALYSIS_MODE_LLM_IMAGE, FACE_ANALYSIS_MODE_LANDMARK_RULE)
_T = TypeVar("_T")


class TextGenerator(Protocol):
    def generate(self, prompt: str, image_path: Path | None = None) -> str:
        ...


@dataclass(frozen=True)
class PersonalWorkflowInput:
    name: str
    birth_date: str
    birth_time: str
    gender: str
    target_gender: str
    face_analysis_mode: int = FACE_ANALYSIS_MODE_LLM_IMAGE


@dataclass(frozen=True)
class CompatibilityWorkflowInput:
    left_name: str
    left_birth_date: str
    left_birth_time: str
    left_gender: str
    right_name: str
    right_birth_date: str
    right_birth_time: str
    right_gender: str
    mode: str
    face_analysis_mode: int = FACE_ANALYSIS_MODE_LLM_IMAGE


@dataclass(frozen=True)
class PersonalWorkflowResult:
    markdown: str
    report_html: str
    report_fragment_html: str
    output_path: Path
    capture_path: Path
    recommendations: tuple[FaceRecommendation, ...]
    face_analysis: str
    manse_status: str
    timing_log_path: Path | None = None


@dataclass(frozen=True)
class CompatibilityWorkflowResult:
    markdown: str
    output_path: Path
    left_capture_path: Path
    right_capture_path: Path
    face_analysis: str
    left_manse_status: str
    right_manse_status: str
    timing_log_path: Path | None = None


@dataclass(frozen=True)
class _GeneratedText:
    text: str
    error: str


@dataclass(frozen=True)
class _WorkflowTimingEntry:
    label: str
    elapsed_seconds: float
    started_at: datetime
    finished_at: datetime


@dataclass(frozen=True)
class _TimedCallResult(Generic[_T]):
    value: _T
    timing: _WorkflowTimingEntry


@dataclass
class _WorkflowTimingRecorder:
    workflow_name: str
    started_at: datetime = field(default_factory=datetime.now)
    started_counter: float = field(default_factory=time.perf_counter)
    entries: list[_WorkflowTimingEntry] = field(default_factory=list)

    def run(
        self,
        label: str,
        function: Callable[..., _T],
        *args: object,
        **kwargs: object,
    ) -> _T:
        timed_result = _timed_call(label, function, *args, **kwargs)
        self.add(timed_result.timing)
        result = timed_result.value
        return result

    def add(self, timing: _WorkflowTimingEntry) -> None:
        self.entries.append(timing)
        print(_format_timing_line(timing))

    def finish_total(self) -> None:
        finished_at = datetime.now()
        timing = _WorkflowTimingEntry(
            label=self.workflow_name,
            elapsed_seconds=time.perf_counter() - self.started_counter,
            started_at=self.started_at,
            finished_at=finished_at,
        )
        self.add(timing)

    def write_log(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            _format_timing_log(self.workflow_name, self.entries),
            encoding="utf-8",
        )
        print(f"[timing] log saved: {path}")
        result = path
        return result


def run_personal_workflow(
    workflow_input: PersonalWorkflowInput,
    capture_config: CaptureConfig,
    face_llm_config: LlmConfig,
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    recommendation_db_path: Path,
    face_client: TextGenerator | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
) -> PersonalWorkflowResult:
    face_analysis_mode = _validate_face_analysis_mode(
        workflow_input.face_analysis_mode,
    )
    active_capture_config = replace(
        capture_config,
        face_analysis_mode=face_analysis_mode,
    )
    profile = _build_birth_profile(
        workflow_input.name,
        workflow_input.birth_date,
        workflow_input.birth_time,
        workflow_input.gender,
    )
    output_dir = _new_session_dir(active_capture_config.output_dir, "personal")
    timing_recorder = _WorkflowTimingRecorder("personal_workflow")
    repository = ManseRepository(manse_db_path)
    active_face_client = face_client
    if face_analysis_mode == FACE_ANALYSIS_MODE_LLM_IMAGE:
        active_face_client = face_client or LlamaCppChatClient(face_llm_config)
    active_report_client = report_client or LlamaCppChatClient(report_llm_config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        manse_future = executor.submit(
            _timed_call,
            "manse_lookup",
            repository.lookup,
            profile,
        )
        capture_future = executor.submit(
            _timed_call,
            "capture",
            capture_runner,
            active_capture_config,
            output_dir,
        )
        capture_timed = capture_future.result()
        timing_recorder.add(capture_timed.timing)
        capture_artifact = capture_timed.value
        manse_timed = manse_future.result()
        timing_recorder.add(manse_timed.timing)
        manse_lookup = manse_timed.value

    face_analysis = timing_recorder.run(
        "face_analysis",
        _build_single_face_analysis,
        active_face_client,
        profile,
        capture_artifact,
        face_analysis_mode,
    )
    recommendations = timing_recorder.run(
        "recommend_faces",
        recommend_faces,
        recommendation_db_path,
        workflow_input.target_gender,
        manse_lookup.reading,
    )
    markdown = timing_recorder.run(
        "final_report",
        _build_personal_markdown,
        active_report_client,
        profile,
        manse_lookup,
        face_analysis.text,
        recommendations,
    )
    report_html = timing_recorder.run(
        "render_report_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis.text,
        recommendations,
        markdown,
    )
    report_fragment_html = timing_recorder.run(
        "render_report_fragment_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis.text,
        recommendations,
        markdown,
        False,
    )
    output_path = output_dir / "personal_report.html"
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    timing_recorder.finish_total()
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = PersonalWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
        output_path=output_path,
        capture_path=capture_artifact.image_path,
        recommendations=recommendations,
        face_analysis=face_analysis.text,
        manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def run_compatibility_workflow(
    workflow_input: CompatibilityWorkflowInput,
    capture_config: CaptureConfig,
    face_llm_config: LlmConfig,
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    face_client: TextGenerator | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    inter_capture_delay_seconds: float = 3.0,
) -> CompatibilityWorkflowResult:
    mode = _validate_mode(workflow_input.mode)
    face_analysis_mode = _validate_face_analysis_mode(
        workflow_input.face_analysis_mode,
    )
    active_capture_config = replace(
        capture_config,
        face_analysis_mode=face_analysis_mode,
    )
    left_profile = _build_birth_profile(
        workflow_input.left_name,
        workflow_input.left_birth_date,
        workflow_input.left_birth_time,
        workflow_input.left_gender,
    )
    right_profile = _build_birth_profile(
        workflow_input.right_name,
        workflow_input.right_birth_date,
        workflow_input.right_birth_time,
        workflow_input.right_gender,
    )
    output_dir = _new_session_dir(active_capture_config.output_dir, "compatibility")
    timing_recorder = _WorkflowTimingRecorder("compatibility_workflow")
    repository = ManseRepository(manse_db_path)
    active_face_client = face_client
    if face_analysis_mode == FACE_ANALYSIS_MODE_LLM_IMAGE:
        active_face_client = face_client or LlamaCppChatClient(face_llm_config)
    active_report_client = report_client or LlamaCppChatClient(report_llm_config)

    with ThreadPoolExecutor(max_workers=2) as executor:
        manse_future = executor.submit(
            _timed_call,
            "manse_lookup_pair",
            _lookup_pair_manse,
            repository,
            left_profile,
            right_profile,
        )
        capture_future = executor.submit(
            _timed_call,
            "capture_pair",
            _run_sequential_pair_capture,
            capture_runner,
            active_capture_config,
            output_dir,
            inter_capture_delay_seconds,
        )
        capture_timed = capture_future.result()
        timing_recorder.add(capture_timed.timing)
        capture_artifact = capture_timed.value
        manse_timed = manse_future.result()
        timing_recorder.add(manse_timed.timing)
        left_manse, right_manse = manse_timed.value

    face_analysis = timing_recorder.run(
        "face_analysis_pair",
        _build_pair_face_analysis,
        active_face_client,
        left_profile,
        right_profile,
        capture_artifact,
        mode,
        face_analysis_mode,
    )
    markdown = timing_recorder.run(
        "final_report",
        _build_compatibility_markdown,
        active_report_client,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis.text,
    )
    output_path = output_dir / "compatibility_report.md"
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        markdown,
        encoding="utf-8",
    )
    timing_recorder.finish_total()
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = CompatibilityWorkflowResult(
        markdown=markdown,
        output_path=output_path,
        left_capture_path=capture_artifact.left.image_path,
        right_capture_path=capture_artifact.right.image_path,
        face_analysis=face_analysis.text,
        left_manse_status="조회 완료",
        right_manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def _timed_call(
    label: str,
    function: Callable[..., _T],
    *args: object,
    **kwargs: object,
) -> _TimedCallResult[_T]:
    started_at = datetime.now()
    started_counter = time.perf_counter()
    value = function(*args, **kwargs)
    finished_at = datetime.now()
    timing = _WorkflowTimingEntry(
        label=label,
        elapsed_seconds=time.perf_counter() - started_counter,
        started_at=started_at,
        finished_at=finished_at,
    )
    result = _TimedCallResult(value=value, timing=timing)
    return result


def _format_timing_line(timing: _WorkflowTimingEntry) -> str:
    result = f"[timing] {timing.label}: {timing.elapsed_seconds:.3f}s"
    return result


def _format_timing_log(
    workflow_name: str,
    entries: list[_WorkflowTimingEntry],
) -> str:
    lines = [
        "# Oracle workflow timing log",
        f"workflow={workflow_name}",
        "",
    ]
    for entry in entries:
        lines.append(
            "\t".join(
                (
                    entry.started_at.isoformat(timespec="milliseconds"),
                    entry.finished_at.isoformat(timespec="milliseconds"),
                    entry.label,
                    f"{entry.elapsed_seconds:.3f}s",
                ),
            ),
        )
    result = "\n".join(lines) + "\n"
    return result


def _build_single_face_analysis(
    client: TextGenerator | None,
    profile: BirthProfile,
    artifact: CaptureArtifact,
    face_analysis_mode: int = FACE_ANALYSIS_MODE_LLM_IMAGE,
) -> _GeneratedText:
    if face_analysis_mode == FACE_ANALYSIS_MODE_LANDMARK_RULE:
        text = artifact.face_analysis or artifact.quality.face_analysis
        if text == "":
            text = "## 관상정보\n- 랜드마크 룰 기반 관상정보를 생성하지 못했습니다."
        result = _GeneratedText(text=text, error="")
    else:
        if client is None:
            raise ValueError("face analysis client is required for mode 1.")
        face_input = FaceReadingInput(
            image_path=artifact.image_path,
            quality=artifact.quality,
        )
        prompt = build_personal_face_analysis_prompt(profile, face_input)
        result = _safe_generate(
            client,
            prompt,
            artifact.image_path,
            "관상정보를 생성하지 못했습니다.",
        )
    return result


def _build_pair_face_analysis(
    client: TextGenerator | None,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
    face_analysis_mode: int = FACE_ANALYSIS_MODE_LLM_IMAGE,
) -> _GeneratedText:
    if face_analysis_mode == FACE_ANALYSIS_MODE_LANDMARK_RULE:
        left_analysis = _build_single_face_analysis(
            client,
            left_profile,
            artifact.left,
            face_analysis_mode,
        )
        right_analysis = _build_single_face_analysis(
            client,
            right_profile,
            artifact.right,
            face_analysis_mode,
        )
    else:
        left_analysis = _build_compatibility_face_analysis(
            client,
            left_profile,
            artifact.left,
            "첫 번째 사람",
            mode,
        )
        right_analysis = _build_compatibility_face_analysis(
            client,
            right_profile,
            artifact.right,
            "두 번째 사람",
            mode,
        )
    error = ""
    if left_analysis.error or right_analysis.error:
        error = " / ".join(
            item for item in (left_analysis.error, right_analysis.error) if item
        )
    text = "\n\n".join(
        (
            f"## 첫 번째 사람 관상정보\n{left_analysis.text}",
            f"## 두 번째 사람 관상정보\n{right_analysis.text}",
        ),
    )
    result = _GeneratedText(text=text, error=error)
    return result


def _build_compatibility_face_analysis(
    client: TextGenerator | None,
    profile: BirthProfile,
    artifact: CaptureArtifact,
    person_label: str,
    mode: str,
) -> _GeneratedText:
    if client is None:
        raise ValueError("face analysis client is required for mode 1.")
    face_input = FaceReadingInput(
        image_path=artifact.image_path,
        quality=artifact.quality,
    )
    prompt = build_compatibility_face_analysis_prompt(
        profile,
        face_input,
        person_label,
        mode,
    )
    result = _safe_generate(
        client,
        prompt,
        artifact.image_path,
        "관상정보를 생성하지 못했습니다.",
    )
    return result


def _build_personal_markdown(
    client: TextGenerator,
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
) -> str:
    recommendation_text = format_recommendations(recommendations)
    prompt = build_personal_final_prompt(
        profile,
        manse_lookup.formatted_text,
        face_analysis,
        recommendation_text,
    )
    generated = _safe_generate(
        client,
        prompt,
        None,
        "최종 리포트를 생성하지 못했습니다.",
    )
    markdown = generated.text
    if generated.error:
        markdown = _fallback_personal_markdown(
            profile,
            manse_lookup.formatted_text,
            face_analysis,
            recommendation_text,
            generated.error,
        )
    result = markdown
    return result


def _build_compatibility_markdown(
    client: TextGenerator,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    face_analysis: str,
) -> str:
    prompt = build_compatibility_final_prompt(
        left_profile,
        right_profile,
        mode,
        left_manse.formatted_text,
        right_manse.formatted_text,
        face_analysis,
    )
    generated = _safe_generate(
        client,
        prompt,
        None,
        "궁합 리포트를 생성하지 못했습니다.",
    )
    markdown = generated.text
    if generated.error:
        markdown = _fallback_compatibility_markdown(
            left_profile,
            right_profile,
            mode,
            left_manse.formatted_text,
            right_manse.formatted_text,
            face_analysis,
            generated.error,
        )
    result = markdown
    return result


def _lookup_pair_manse(
    repository: ManseRepository,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
) -> tuple[ManseLookupResult, ManseLookupResult]:
    left_result = repository.lookup(left_profile)
    right_result = repository.lookup(right_profile)
    result = (left_result, right_result)
    return result


def _run_sequential_pair_capture(
    capture_runner,
    capture_config: CaptureConfig,
    output_dir: Path,
    inter_capture_delay_seconds: float,
) -> SequentialPairCaptureArtifact:
    left_dir = output_dir / "person_1"
    right_dir = output_dir / "person_2"
    left_artifact = capture_runner(capture_config, left_dir)
    if inter_capture_delay_seconds > 0.0:
        time.sleep(inter_capture_delay_seconds)
    right_artifact = capture_runner(capture_config, right_dir)
    result = SequentialPairCaptureArtifact(
        left=left_artifact,
        right=right_artifact,
    )
    return result


def _safe_generate(
    client: TextGenerator,
    prompt: str,
    image_path: Path | None,
    fallback: str,
) -> _GeneratedText:
    text = fallback
    error = ""
    try:
        text = client.generate(prompt, image_path=image_path)
    except Exception as exc:
        error = str(exc)
        text = f"{fallback}\n\n오류: {error}"
    result = _GeneratedText(text=text, error=error)
    return result


def _build_birth_profile(
    name: str,
    birth_date: str,
    birth_time: str,
    gender: str,
) -> BirthProfile:
    cleaned_gender = gender.strip()
    if cleaned_gender == "":
        raise ValueError("성별은 남성 또는 여성으로 입력해야 합니다.")
    cleaned_time = birth_time.strip()
    birth_time_known = cleaned_time != ""
    time_text = cleaned_time
    if not birth_time_known:
        time_text = "12:00"
    birth_datetime = datetime.strptime(
        f"{birth_date.strip()} {time_text}",
        "%Y-%m-%d %H:%M",
    )
    result = BirthProfile(
        name=name.strip(),
        birth_datetime=birth_datetime,
        gender=cleaned_gender,
        birth_time_known=birth_time_known,
    )
    return result


def _validate_mode(mode: str) -> str:
    cleaned = mode.strip()
    if cleaned not in COMPATIBILITY_MODES:
        raise ValueError("궁합 모드는 연인, 친구, 직장동료 중 하나여야 합니다.")
    result = cleaned
    return result


def _validate_face_analysis_mode(mode: int | str) -> int:
    result = int(mode)
    if result not in FACE_ANALYSIS_MODES:
        raise ValueError("관상 분석 모드는 1 또는 2여야 합니다.")
    return result


def _new_session_dir(base_dir: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = base_dir / f"{prefix}_{stamp}"
    result.mkdir(parents=True, exist_ok=True)
    return result


def _fallback_personal_markdown(
    profile: BirthProfile,
    saju_text: str,
    face_analysis: str,
    recommendation_text: str,
    error: str,
) -> str:
    result = f"""
# {profile.name} 님 Oracle 종합 리포트
## 한 줄 요약
로컬 LLM 응답에 문제가 있어 룰 기반 정보로 기본 리포트를 생성했습니다.

## 사주팔자 핵심
{saju_text}

## 관상 보조 풀이
{face_analysis}

## 추천받고 싶은 얼굴 성별 기준 추천
{recommendation_text}

## 참고 문구
이 리포트는 엔터테인먼트 목적의 참고 자료입니다.

## 시스템 참고
최종 LLM 오류: {error}
""".strip()
    return result


def _fallback_compatibility_markdown(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_saju_text: str,
    right_saju_text: str,
    face_analysis: str,
    error: str,
) -> str:
    result = f"""
# {left_profile.name} 님과 {right_profile.name} 님의 {mode} 궁합 리포트
## 한 줄 요약
로컬 LLM 응답에 문제가 있어 룰 기반 정보로 기본 궁합 리포트를 생성했습니다.

## 첫 번째 사람 사주팔자
{left_saju_text}

## 두 번째 사람 사주팔자
{right_saju_text}

## 관상 보조 인상 궁합
{face_analysis}

## 참고 문구
이 리포트는 엔터테인먼트 목적의 참고 자료입니다.

## 시스템 참고
최종 LLM 오류: {error}
""".strip()
    return result

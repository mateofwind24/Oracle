# ==============================================================================
# WARNING: Do NOT modify or break the distributed inference (Priority Queue, 
# speculative work stealing, dynamic compute score feedback, and relative 
# yielding thresholds) implemented in this file. 
# These mechanisms are highly sensitive to modification.
# ==============================================================================
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import re
import time
import threading
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

from oracle_report import prompt_templates
from oracle_report.compatibility_score import build_compatibility_score_payload
from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import (
    BirthProfile,
    CaptureArtifact,
    FaceBox,
    SequentialPairCaptureArtifact,
)
from oracle_report.report_html import (
    render_compatibility_report_html,
    render_personal_report_html,
)
from oracle_report.saju.repository import (
    ManseLookupResult,
    ManseRepository,
    UNKNOWN_BIRTH_TIME_REPRESENTATIVE,
    representative_time_from_time_branch,
)
from oracle_report.vision.runtime import run_capture


COMPATIBILITY_MODES = ("연인", "친구", "직장동료")
_UNKNOWN_BIRTH_TIME_VALUES = frozenset(("", "모름", "미상", "unknown", "none"))
_SAJU_DAY_MASTER_HONORIFICS = (
    "갑목님",
    "을목님",
    "병화님",
    "정화님",
    "무토님",
    "기토님",
    "경금님",
    "신금님",
    "임수님",
    "계수님",
)
_SAJU_DAY_MASTER_TERM_REPLACEMENTS = (
    ("갑목", "나무 기운"),
    ("을목", "나무 기운"),
    ("병화", "불 기운"),
    ("정화", "불 기운"),
    ("무토", "흙 기운"),
    ("기토", "흙 기운"),
    ("경금", "금 기운"),
    ("신금", "금 기운"),
    ("임수", "물 기운"),
    ("계수", "물 기운"),
)
_EMOJI_PATTERN = re.compile(
    "["
    "\U0001F1E6-\U0001F1FF"
    "\U0001F300-\U0001FAFF"
    "\U00002600-\U000027BF"
    "\U0001F900-\U0001F9FF"
    "]+"
)
_T = TypeVar("_T")
_LAST_STATUS_CHECK_TIMES: dict[str, float] = {}
_LAST_COMPLETED_TIMES: dict[str, float] = {}
_COMPLETED_TIMES_LOCK = threading.Lock()


def get_virtual_score(addr: str, base_score: float) -> float:
    with _COMPLETED_TIMES_LOCK:
        last_time = _LAST_COMPLETED_TIMES.get(addr, 0.0)
    elapsed = time.time() - last_time
    if elapsed < 5.0:
        return base_score - 10.0
    return base_score


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
    skip_face: bool = False


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


@dataclass(frozen=True)
class PersonalWorkflowResult:
    markdown: str
    report_html: str
    report_fragment_html: str
    output_path: Path
    capture_path: Path | None
    face_analysis: str
    manse_status: str
    timing_log_path: Path | None = None


@dataclass(frozen=True)
class CompatibilityWorkflowResult:
    markdown: str
    report_html: str
    report_fragment_html: str
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
    report_llm_config: LlmConfig | None = None,
    manse_db_path: Path | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    progress_callback=None,
) -> PersonalWorkflowResult:
    del manse_db_path
    if report_llm_config is None:
        raise ValueError("report_llm_config is required.")
        
    if progress_callback:
        progress_callback(5, "얼굴 인식을 위해 카메라를 준비하는 중...")
        
    profile = _build_birth_profile(
        workflow_input.name,
        workflow_input.birth_date,
        workflow_input.birth_time,
        workflow_input.gender,
    )
    output_dir = _new_session_dir(capture_config.output_dir, "personal")
    timing_recorder = _WorkflowTimingRecorder("personal_workflow")
    repository = ManseRepository()
    active_report_client = report_client or LlamaCppChatClient(report_llm_config)

    capture_artifact = None
    if not workflow_input.skip_face:
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
                capture_config,
                output_dir,
            )
            capture_timed = capture_future.result()
            timing_recorder.add(capture_timed.timing)
            capture_artifact = capture_timed.value
            
            if progress_callback:
                progress_callback(20, "생년월일시 만세력 정보를 조회하는 중...")
                
            manse_timed = manse_future.result()
            timing_recorder.add(manse_timed.timing)
            manse_lookup = manse_timed.value
    else:
        if progress_callback:
            progress_callback(20, "생년월일시 만세력 정보를 조회하는 중...")
            
        manse_timed = _timed_call(
            "manse_lookup",
            repository.lookup,
            profile,
        )
        timing_recorder.add(manse_timed.timing)
        manse_lookup = manse_timed.value

    if not workflow_input.skip_face and capture_artifact is not None:
        face_analysis = timing_recorder.run(
            "face_analysis",
            _build_single_face_analysis,
            profile,
            capture_artifact,
        )
        face_analysis_text = face_analysis.text
    else:
        face_analysis_text = ""

    saju_analysis = timing_recorder.run(
        "saju_analysis",
        _build_saju_analysis,
        active_report_client,
        profile,
        manse_lookup,
        progress_callback=progress_callback,
    )
    saju_analysis_text = saju_analysis.text
    
    if progress_callback:
        progress_callback(92, "분석 리포트를 정제하고 요약 문장을 조립하는 중...")
        
    markdown = timing_recorder.run(
        "assemble_report",
        _build_personal_report_json,
        manse_lookup,
        face_analysis_text,
        saju_analysis_text,
        workflow_input.skip_face,
    )
    
    if progress_callback:
        progress_callback(95, "리포트 HTML 테마와 디자인 페이지를 구성하는 중...")
        
    report_html = timing_recorder.run(
        "render_report_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis_text,
        markdown,
        True,
        workflow_input.skip_face,
    )
    report_fragment_html = timing_recorder.run(
        "render_report_fragment_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis_text,
        markdown,
        False,
        workflow_input.skip_face,
    )
    output_path = output_dir / "personal_report.html"
    
    if progress_callback:
        progress_callback(98, "리포트 파일을 로컬 디스크에 저장하는 중...")
        
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    (output_dir / "personal_report.md").write_text(markdown, encoding="utf-8")
    timing_recorder.finish_total()
    
    if progress_callback:
        progress_callback(100, "완료!")
        
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = PersonalWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
        output_path=output_path,
        capture_path=capture_artifact.image_path if capture_artifact is not None else None,
        face_analysis=face_analysis_text,
        manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def run_compatibility_workflow(
    workflow_input: CompatibilityWorkflowInput,
    capture_config: CaptureConfig,
    report_llm_config: LlmConfig | None = None,
    manse_db_path: Path | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    inter_capture_delay_seconds: float = 3.0,
    progress_callback=None,
) -> CompatibilityWorkflowResult:
    del manse_db_path
    if report_llm_config is None:
        raise ValueError("report_llm_config is required.")
        
    if progress_callback:
        progress_callback(5, "카메라 연결 및 얼굴 촬영을 준비하는 중...")
        
    mode = _validate_mode(workflow_input.mode)
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
    output_dir = _new_session_dir(capture_config.output_dir, "compatibility")
    timing_recorder = _WorkflowTimingRecorder("compatibility_workflow")
    repository = ManseRepository()
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
            capture_config,
            output_dir,
            inter_capture_delay_seconds,
        )
        capture_timed = capture_future.result()
        timing_recorder.add(capture_timed.timing)
        capture_artifact = capture_timed.value
        
        if progress_callback:
            progress_callback(20, "두 사람의 생년월일 명조 정보를 분석하는 중...")
            
        manse_timed = manse_future.result()
        timing_recorder.add(manse_timed.timing)
        left_manse, right_manse = manse_timed.value

    face_analysis = timing_recorder.run(
        "face_analysis_pair",
        _build_pair_face_analysis,
        left_profile,
        right_profile,
        capture_artifact,
        mode,
    )
    saju_analysis = timing_recorder.run(
        "saju_analysis_pair",
        _build_compatibility_saju_analysis,
        active_report_client,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        progress_callback=progress_callback,
    )
    compatibility_score = timing_recorder.run(
        "compatibility_score",
        _build_compatibility_score_payload,
        mode,
        left_manse,
        right_manse,
        capture_artifact,
    )
    
    if progress_callback:
        progress_callback(92, "두 사람의 관계 시너지를 최종 분석하는 중...")
        
    markdown = timing_recorder.run(
        "final_report",
        _build_compatibility_report_json,
        face_analysis.text,
        saju_analysis.text,
        compatibility_score,
    )
    
    if progress_callback:
        progress_callback(95, "궁합 리포트 템플릿 페이지를 렌더링하는 중...")
        
    report_html = timing_recorder.run(
        "render_report_html",
        render_compatibility_report_html,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis.text,
        markdown,
    )
    report_fragment_html = timing_recorder.run(
        "render_report_fragment_html",
        render_compatibility_report_html,
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis.text,
        markdown,
        False,
    )
    output_path = output_dir / "compatibility_report.html"
    
    if progress_callback:
        progress_callback(98, "리포트 파일을 로컬 디스크에 저장하는 중...")
        
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    timing_recorder.finish_total()
    
    if progress_callback:
        progress_callback(100, "완료!")
        
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = CompatibilityWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
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
    profile: BirthProfile,
    artifact: CaptureArtifact,
) -> _GeneratedText:
    result = _build_single_rule_based_face_analysis(profile, artifact)
    return result


def _build_single_rule_based_face_analysis(
    profile: BirthProfile,
    artifact: CaptureArtifact,
) -> _GeneratedText:
    from oracle_report.vision.physiognomy_text_variations import build_personal_face_payload

    matches = _quality_rule_matches(artifact.quality)
    text = ""
    error = ""
    if matches:
        payload = build_personal_face_payload(
            matches,
            _single_face_seed(profile, artifact, matches),
        )
        text = json.dumps(payload, ensure_ascii=False)
    else:
        text = artifact.quality.face_payload_json.strip()
    if text == "":
        text = artifact.face_analysis.strip()
        error = "rule-based face payload is unavailable; using capture face analysis memo"
    result = _GeneratedText(text=text, error=error)
    return result


def _build_pair_face_analysis(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
) -> _GeneratedText:
    result = _build_pair_rule_based_face_analysis(
        left_profile,
        right_profile,
        artifact,
        mode,
    )
    return result

def _build_pair_rule_based_face_analysis(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
) -> _GeneratedText:
    from oracle_report.vision.physiognomy_text_variations import build_pair_face_payload

    left_matches = _quality_rule_matches(artifact.left.quality)
    right_matches = _quality_rule_matches(artifact.right.quality)
    payload = build_pair_face_payload(
        left_matches,
        right_matches,
        left_profile.name,
        right_profile.name,
        _pair_face_seed(
            left_profile,
            right_profile,
            artifact,
            left_matches,
            right_matches,
        ),
        mode=mode,
    )
    text = json.dumps(payload, ensure_ascii=False)
    result = _GeneratedText(text=text, error="")
    return result


def _build_compatibility_score_payload(
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    artifact: SequentialPairCaptureArtifact,
) -> dict[str, object]:
    result = build_compatibility_score_payload(
        mode,
        left_manse,
        right_manse,
        _quality_rule_matches(artifact.left.quality),
        _quality_rule_matches(artifact.right.quality),
    )
    return result


def _quality_rule_matches(quality) -> tuple[Any, ...]:
    from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch

    raw_text = getattr(quality, "landmark_matches_json", "").strip()
    rows: list[Any] = []
    if raw_text != "":
        try:
            loaded = json.loads(raw_text)
        except json.JSONDecodeError:
            loaded = []
        if isinstance(loaded, list):
            rows = loaded
    result = tuple(
        PhysiognomyRuleMatch(
            rule_id=str(row.get("rule_id", "")),
            metric=str(row.get("metric", "")),
            title=str(row.get("title", "")),
            basis=str(row.get("basis", "")),
            tag=str(row.get("tag", "")),
            observation=str(row.get("observation", "")),
            interpretation=str(row.get("interpretation", "")),
            value=float(row.get("value", 0.0)),
        )
        for row in rows
        if isinstance(row, dict)
    )
    return result


def _pair_face_seed(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    left_matches: tuple[Any, ...],
    right_matches: tuple[Any, ...],
) -> str:
    left_tags = ",".join(getattr(match, "tag", "") for match in left_matches[:6])
    right_tags = ",".join(getattr(match, "tag", "") for match in right_matches[:6])
    result = (
        f"{left_profile.name}:{right_profile.name}:"
        f"{artifact.left.captured_at.isoformat()}:{artifact.right.captured_at.isoformat()}:"
        f"{left_tags}:{right_tags}"
    )
    return result


def _single_face_seed(
    profile: BirthProfile,
    artifact: CaptureArtifact,
    matches: tuple[Any, ...],
) -> str:
    tags = ",".join(getattr(match, "tag", "") for match in matches[:8])
    result = f"{profile.name}:{artifact.captured_at.isoformat()}:{tags}"
    return result


def _build_saju_analysis(
    client: TextGenerator,
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    progress_callback=None,
) -> _GeneratedText:
    distributed_app_config = _load_active_distributed_app_config()
    categories = (
        "종합 형국 및 원국 해설",
        "오행 분포와 기운 개운법",
        "신살 및 십성 심리 해부",
        "성격 분석 및 후천적 변화",
        "직업 적성과 사회적 무기",
        "재물운과 자산 설계",
        "연애운과 이상형 인연",
        "부모 및 가족 관계",
        "대인관계와 인맥 다이어트",
        "공간 및 지리적 개운 처방",
        "올해의 운세 및 총평",
    )
    values = {
        "name": profile.name,
        "gender": manse_lookup.gender,
        "timezone": "KST",
        "saju_text": manse_lookup.formatted_text,
    }
    result = _safe_generate_distributed(
        "saju_reading",
        values,
        categories,
        None,
        distributed_app_config,
        client,
        "사주정보를 생성하지 못했습니다.",
        "saju_analysis",
        progress_callback=progress_callback,
    )
    result = _repair_saju_terms(result, profile.name, "saju_analysis")
    return result


def _build_compatibility_saju_analysis(
    client: TextGenerator,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    progress_callback=None,
) -> _GeneratedText:
    distributed_app_config = _load_active_distributed_app_config()
    from oracle_report.saju.repository import birth_time_display_from_profile
    categories = ("관계 구조", "상호 보완", "갈등 관리", "현재 관계 흐름", "실천 제안", "총평 및 조언")
    values = {
        "mode": mode,
        "left_name": left_profile.name,
        "left_gender": left_profile.gender,
        "left_birth_datetime": left_profile.birth_datetime.isoformat(),
        "left_birth_time_text": birth_time_display_from_profile(left_profile),
        "right_name": right_profile.name,
        "right_gender": right_profile.gender,
        "right_birth_datetime": right_profile.birth_datetime.isoformat(),
        "right_birth_time_text": birth_time_display_from_profile(right_profile),
        "left_saju_text": left_manse.formatted_text,
        "right_saju_text": right_manse.formatted_text,
    }
    result = _safe_generate_distributed(
        "saju_reading_couple",
        values,
        categories,
        None,
        distributed_app_config,
        client,
        "궁합 사주정보를 생성하지 못했습니다.",
        "saju_analysis_couple",
        progress_callback=progress_callback,
    )
    result = _repair_saju_terms(result, "", "saju_analysis_couple")
    return result



def _build_personal_report_json(
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    saju_analysis: str,
    skip_face: bool = False,
) -> str:
    face_payload, face_error = ({}, "")
    saju_payload = parse_oracle_text_response(saju_analysis)
    saju_error = ""
    if not skip_face:
        face_payload, face_error = _load_json_payload_or_error(
            face_analysis,
            label="face_analysis",
        )
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    if face_error:
        print(
            "[UI FALLBACK:face_analysis] invalid LLM output; "
            f"renderer will fill missing face fields. reason={face_error}",
        )
    payload = _merge_personal_payloads(
        manse_lookup,
        face_payload,
        saju_payload,
        skip_face,
    )
    payload = _normalize_payload_text(payload)
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _build_compatibility_report_json(
    face_analysis: str,
    saju_analysis: str,
    compatibility_score: dict[str, object] | None = None,
) -> str:
    face_payload, face_error = _load_json_payload_or_error(
        face_analysis,
        label="pair_face_analysis",
    )
    saju_payload = parse_oracle_text_response(saju_analysis)
    saju_error = ""
    if face_error:
        print(
            "[UI FALLBACK:pair_face_analysis] invalid face output; "
            f"renderer will fill missing pair fields. reason={face_error}",
        )
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis_couple] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    payload = _merge_compatibility_payloads(face_payload, saju_payload)
    if compatibility_score is not None:
        payload.update(compatibility_score)
    payload = _normalize_payload_text(payload)
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _merge_compatibility_payloads(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key in (
        "essence",
        "saju_subtitle",
        "saju_blocks",
        "action_title",
        "action_body",
        "tags",
        "disclaimer",
    ):
        value = saju_payload.get(key)
        if value:
            payload[key] = value
    for key in ("pair_subtitle", "pair_blocks"):
        value = face_payload.get(key)
        if value:
            payload[key] = value
    for key in ("essence", "action_title", "action_body"):
        if key not in payload:
            value = face_payload.get(key)
            if value:
                payload[key] = value
    result = payload
    return result


def _merge_personal_payloads(
    manse_lookup: ManseLookupResult,
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
    skip_face: bool,
) -> dict[str, Any]:
    del manse_lookup
    payload: dict[str, Any] = {}
    for key in (
        "essence",
        "element_note",
        "saju_subtitle",
        "saju_blocks",
        "tags",
        "disclaimer",
    ):
        value = saju_payload.get(key)
        if value:
            payload[key] = value
    if not skip_face:
        for key in ("face_subtitle", "face_blocks"):
            value = face_payload.get(key)
            if value:
                payload[key] = value
    result = payload
    return result


def _normalize_payload_text(value: Any) -> Any:
    result = value
    if isinstance(value, str):
        result = _normalize_inline_text(value)
    elif isinstance(value, list):
        result = [_normalize_payload_text(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _normalize_payload_text(item) for key, item in value.items()}
        result = _remove_repeated_summary_from_block(result)
    return result


def _remove_repeated_summary_from_block(payload: dict[str, Any]) -> dict[str, Any]:
    summary = payload.get("summary")
    body = payload.get("body")
    if isinstance(summary, str) and isinstance(body, str):
        payload["body"] = _strip_summary_prefix(summary, body)
    return payload


def _strip_summary_prefix(summary: str, body: str) -> str:
    summary_text = _normalize_inline_text(summary)
    body_text = _normalize_inline_text(body)
    candidates = [
        summary_text,
        _first_sentence(summary_text),
    ]
    result = body_text
    for candidate in candidates:
        stripped = _strip_text_prefix(candidate, result)
        if stripped != result and stripped:
            result = stripped
            break
    return result


def _first_sentence(text: str) -> str:
    match = re.match(r"^.+?[.!?](?:\s|$)", text)
    result = match.group(0).strip() if match else text
    return result


def _strip_text_prefix(prefix: str, text: str) -> str:
    result = text
    normalized_prefix = prefix.strip()
    if normalized_prefix and text.startswith(normalized_prefix):
        result = text[len(normalized_prefix) :].lstrip()
        result = re.sub(r"^[.!?]+\s*", "", result)
    return result


def _normalize_inline_text(text: str) -> str:
    normalized_text = text.replace("\\r\\n", " ")
    normalized_text = normalized_text.replace("\\n", " ")
    normalized_text = normalized_text.replace("\\r", " ")
    normalized_text = normalized_text.replace("\r\n", " ")
    normalized_text = normalized_text.replace("\n", " ")
    normalized_text = normalized_text.replace("\r", " ")
    normalized_text = _EMOJI_PATTERN.sub(" ", normalized_text)
    normalized_text = normalized_text.replace("\ufe0f", " ")
    normalized_text = normalized_text.replace("\u200d", " ")
    result = " ".join(normalized_text.split())
    return result



def _dominant_element(counts: dict[str, int], strongest: bool) -> str:
    elements = tuple(counts.keys())
    selector = max
    score = lambda element: (counts[element], -elements.index(element))
    if not strongest:
        selector = min
        score = lambda element: (counts[element], elements.index(element))
    result = selector(elements, key=score)
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
    left_config = _pair_capture_config(capture_config, "left")
    right_config = _pair_capture_config(capture_config, "right")
    left_artifact = capture_runner(left_config, left_dir)
    if inter_capture_delay_seconds > 0.0:
        time.sleep(inter_capture_delay_seconds)
    right_artifact = capture_runner(right_config, right_dir)
    result = SequentialPairCaptureArtifact(
        left=left_artifact,
        right=right_artifact,
    )
    return result


def _pair_capture_config(capture_config: CaptureConfig, side: str) -> CaptureConfig:
    result = capture_config
    if capture_config.mock_capture_enabled:
        metrics_json = ""
        if side == "left":
            metrics_json = capture_config.mock_pair_left_landmark_metrics_json
        elif side == "right":
            metrics_json = capture_config.mock_pair_right_landmark_metrics_json
        if metrics_json.strip():
            result = replace(capture_config, mock_landmark_metrics_json=metrics_json)
    return result



def _normalize_generated_output_text(text: str, label: str = "llm_json") -> str:
    payload, error = _load_json_payload_or_error(text, label=label)
    result = text
    if not error and payload:
        result = json.dumps(_normalize_payload_text(payload), ensure_ascii=False)
    return result


def _repair_saju_terms(
    generated: _GeneratedText,
    name: str,
    label: str,
) -> _GeneratedText:
    cleaned_name = name.strip()
    result = generated
    if generated.error != "":
        return result
    payload, error = _load_json_payload_or_error(generated.text, label=label)
    if error:
        return result
    fixed_payload = payload
    if cleaned_name != "":
        fixed_payload = _replace_day_master_honorifics_in_value(
            fixed_payload,
            f"{cleaned_name}님",
        )
        fixed_payload = _remove_personal_name_mentions_in_value(
            fixed_payload,
            cleaned_name,
        )
    fixed_payload = _replace_day_master_terms_in_value(fixed_payload)
    if cleaned_name != "":
        fixed_payload = _remove_personal_name_mentions_in_value(
            fixed_payload,
            cleaned_name,
        )
    if fixed_payload != payload:
        print(f"[LLM JSON REPAIR:{label}] softened day-master terms.")
        result = _GeneratedText(
            text=json.dumps(_normalize_payload_text(fixed_payload), ensure_ascii=False),
            error="",
        )
    return result


def _replace_day_master_honorifics_in_value(value: Any, replacement: str) -> Any:
    result = value
    if isinstance(value, str):
        result = value
        for honorific in _SAJU_DAY_MASTER_HONORIFICS:
            result = result.replace(honorific, replacement)
    elif isinstance(value, list):
        result = [
            _replace_day_master_honorifics_in_value(item, replacement)
            for item in value
        ]
    elif isinstance(value, dict):
        result = {
            key: _replace_day_master_honorifics_in_value(item, replacement)
            for key, item in value.items()
        }
    return result


def _replace_day_master_terms_in_value(value: Any) -> Any:
    result = value
    if isinstance(value, str):
        result = _replace_day_master_terms_in_text(value)
    elif isinstance(value, list):
        result = [_replace_day_master_terms_in_value(item) for item in value]
    elif isinstance(value, dict):
        result = {
            key: _replace_day_master_terms_in_value(item)
            for key, item in value.items()
        }
    return result


def _replace_day_master_terms_in_text(text: str) -> str:
    result = text
    for term, replacement in _SAJU_DAY_MASTER_TERM_REPLACEMENTS:
        result = result.replace(f"{term}님", replacement)
    for term, replacement in _SAJU_DAY_MASTER_TERM_REPLACEMENTS:
        result = re.sub(rf"{term}\s*일간", replacement, result)
    for term, replacement in _SAJU_DAY_MASTER_TERM_REPLACEMENTS:
        result = result.replace(term, replacement)
    result = result.replace(" 일간", " 중심 기운")
    result = result.replace("일간", "중심 기운")
    return result


def _remove_personal_name_mentions_in_value(value: Any, name: str) -> Any:
    result = value
    if isinstance(value, str):
        result = _remove_personal_name_mentions_in_text(value, name)
    elif isinstance(value, list):
        result = [_remove_personal_name_mentions_in_value(item, name) for item in value]
    elif isinstance(value, dict):
        result = {
            key: _remove_personal_name_mentions_in_value(item, name)
            for key, item in value.items()
        }
    return result


def _remove_personal_name_mentions_in_text(text: str, name: str) -> str:
    result = text
    escaped_name = re.escape(name)
    result = re.sub(rf"{escaped_name}\s*님(?:은|이|에게는|에게|의)?\s*", "", result)
    result = re.sub(r"(?<![가-힣A-Za-z0-9])님(?:은|이|에게는|에게|의)?\s*", "", result)
    result = re.sub(r"\s+([,.!?])", r"\1", result)
    result = " ".join(result.split())
    return result


def parse_oracle_text_response(text: str) -> dict:
    # 0. JSON format check first for fallback compatibility
    trimmed = text.strip()
    if (trimmed.startswith("{") and trimmed.endswith("}")) or (trimmed.startswith("```") and "{" in trimmed):
        json_candidate = trimmed
        if json_candidate.startswith("```"):
            lines = json_candidate.splitlines()
            if lines and (lines[0].startswith("```json") or lines[0] == "```"):
                json_candidate = "\n".join(lines[1:-1]).strip()
        try:
            parsed, error = _load_json_payload_or_error(json_candidate, label="parse_oracle")
            if not error and isinstance(parsed, dict):
                normalized = {}
                for k, v in parsed.items():
                    normalized[k.lower()] = v
                if "tags" in normalized and isinstance(normalized["tags"], str):
                    raw_tags = normalized["tags"]
                    normalized["tags"] = [t.strip() for t in re.split(r'[,，\s]+', raw_tags) if t.strip()]
                for blocks_key in ("saju_blocks", "pair_blocks", "face_blocks"):
                    if blocks_key in normalized and isinstance(normalized[blocks_key], list):
                        new_blocks = []
                        for b in normalized[blocks_key]:
                            if isinstance(b, dict):
                                new_b = {}
                                for bk, bv in b.items():
                                    new_b[bk.lower()] = bv
                                new_blocks.append(new_b)
                        normalized[blocks_key] = new_blocks
                return normalized
        except Exception:
            pass

    # 1. Plain text parser
    meta_output = {}
    saju_blocks = []
    
    # Parse METADATA block
    metadata_match = re.search(
        r'===\s*METADATA\s*===(.*?)(?====\s*CATEGORY|===|$)', 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    meta_text = None
    if metadata_match:
        meta_text = metadata_match.group(1)
    elif "### essence" in text.lower():
        cat_start = re.search(r'===\s*CATEGORY', text, re.IGNORECASE)
        if cat_start:
            meta_text = text[:cat_start.start()]
        else:
            meta_text = text
            
    if meta_text:
        essence_m = re.search(r'###\s*ESSENCE:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        elem_m = re.search(r'###\s*ELEMENT_NOTE:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        sub_m = re.search(r'###\s*SAJU_SUBTITLE:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        action_t_m = re.search(r'###\s*ACTION_TITLE:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        action_b_m = re.search(r'###\s*ACTION_BODY:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        tags_m = re.search(r'###\s*TAGS:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        disc_m = re.search(r'###\s*DISCLAIMER:\s*(.*?)(?=\s*###|\Z)', meta_text, re.DOTALL | re.IGNORECASE)
        
        if essence_m:
            meta_output["essence"] = essence_m.group(1).strip()
        if elem_m:
            meta_output["element_note"] = elem_m.group(1).strip()
        if sub_m:
            meta_output["saju_subtitle"] = sub_m.group(1).strip()
        if action_t_m:
            meta_output["action_title"] = action_t_m.group(1).strip()
        if action_b_m:
            meta_output["action_body"] = action_b_m.group(1).strip()
        if tags_m:
            raw_tags = tags_m.group(1).strip()
            tags = [t.strip() for t in re.split(r'[,，\s]+', raw_tags) if t.strip()]
            meta_output["tags"] = tags
        if disc_m:
            meta_output["disclaimer"] = disc_m.group(1).strip()
            
    # Parse CATEGORY blocks
    category_blocks = re.findall(
        r'===\s*CATEGORY:\s*(.*?)\s*===(.*?)(?====\s*CATEGORY|===\s*METADATA|$)', 
        text, 
        re.DOTALL | re.IGNORECASE
    )
    for cat_name, cat_body in category_blocks:
        cat_name = cat_name.strip()
        title_m = re.search(r'###\s*TITLE:\s*(.*?)(?=\s*###|\Z)', cat_body, re.DOTALL | re.IGNORECASE)
        sum_m = re.search(r'###\s*SUMMARY:\s*(.*?)(?=\s*###|\Z)', cat_body, re.DOTALL | re.IGNORECASE)
        body_m = re.search(r'###\s*BODY:\s*(.*?)(?=\s*###|\Z)', cat_body, re.DOTALL | re.IGNORECASE)
        
        block = {"category": cat_name}
        if title_m:
            block["title"] = title_m.group(1).strip()
        if sum_m:
            block["summary"] = sum_m.group(1).strip()
        if body_m:
            block["body"] = body_m.group(1).strip()
            
        saju_blocks.append(block)
        
    result = meta_output
    if saju_blocks:
        result["saju_blocks"] = saju_blocks
    return result


def _is_parsed_output_valid(parsed: dict, is_metadata: bool) -> bool:
    if not parsed or not isinstance(parsed, dict):
        return False
    if is_metadata:
        if "essence" not in parsed or not parsed["essence"].strip():
            return False
        if "tags" not in parsed or not parsed["tags"]:
            return False
        return True
    else:
        # For individual category tasks, they might be nested in 'saju_blocks' or direct
        # Check sub-blocks first
        blocks = parsed.get("saju_blocks", [])
        if not blocks and "category" in parsed:
            blocks = [parsed]
        if not blocks:
            # Check if direct title/summary/body keys are present (fallback)
            if "title" in parsed and "summary" in parsed and "body" in parsed:
                blocks = [parsed]
                
        if not blocks:
            return False
            
        for b in blocks:
            if not isinstance(b, dict):
                return False
            if "title" not in b or not b["title"].strip():
                return False
            if "summary" not in b or not b["summary"].strip():
                return False
            if "body" not in b or len(b["body"].strip()) < 10:
                return False
        return True


def _load_json_payload_or_error(
    text: str,
    label: str = "llm_json",
) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence_text(cleaned)
    start = cleaned.find("{")
    if start >= 0:
        cleaned = cleaned[start:]
    payload: dict[str, Any] = {}
    error = ""
    try:
        loaded, repair_steps = _load_json_with_repairs(cleaned)
        if repair_steps:
            print(
                f"[LLM JSON REPAIR:{label}] applied repairs: "
                + ", ".join(repair_steps),
            )
        if isinstance(loaded, dict):
            payload = loaded
        else:
            error = "LLM JSON must be an object"
    except json.JSONDecodeError as exc:
        error = (
            "LLM JSON parse failed: "
            f"{exc.msg} at line {exc.lineno}, column {exc.colno}"
        )
    result = (payload, error)
    return result


def _load_json_with_repairs(text: str) -> tuple[Any, tuple[str, ...]]:
    candidates = [text]
    candidate_steps = [tuple()]
    normalized_quotes = _normalize_json_quotes(text)
    if normalized_quotes != text:
        candidates.append(normalized_quotes)
        candidate_steps.append(("normalize_quotes",))
    repaired = normalized_quotes
    repaired_steps = list(candidate_steps[-1])
    for _ in range(3):
        next_repaired, step_names = _repair_json_text(repaired)
        if next_repaired == repaired:
            break
        repaired = next_repaired
        repaired_steps.extend(step_names)
        candidates.append(repaired)
        candidate_steps.append(tuple(repaired_steps))
    last_error: json.JSONDecodeError | None = None
    decoder = json.JSONDecoder()
    for candidate, steps in zip(candidates, candidate_steps):
        try:
            loaded, _ = decoder.raw_decode(candidate)
            return loaded, steps
        except json.JSONDecodeError as exc:
            last_error = exc
    if last_error is None:
        raise json.JSONDecodeError("empty JSON", text, 0)
    raise last_error


def _normalize_json_quotes(text: str) -> str:
    replacements = {
        "“": '"',
        "”": '"',
        "„": '"',
        "‟": '"',
        "’": "'",
        "‘": "'",
    }
    result = text
    for old_text, new_text in replacements.items():
        result = result.replace(old_text, new_text)
    return result


def _repair_json_text(text: str) -> tuple[str, tuple[str, ...]]:
    result = text
    applied_steps: list[str] = []
    with_fixed_block_arrays = _repair_misclosed_block_arrays(result)
    if with_fixed_block_arrays != result:
        applied_steps.append("fix_misclosed_block_arrays")
        result = with_fixed_block_arrays
    without_trailing_commas = re.sub(r",\s*([}\]])", r"\1", result)
    if without_trailing_commas != result:
        applied_steps.append("remove_trailing_commas")
        result = without_trailing_commas
    with_missing_commas = re.sub(
        (
            r'("(?:[^"\\]|\\.)*"|\btrue\b|\bfalse\b|\bnull\b|'
            r'-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|[}\]])'
            r'(\s*)(?=(?:"|[{\[]|\btrue\b|\bfalse\b|\bnull\b|-?\d))'
        ),
        _insert_missing_comma,
        result,
    )
    if with_missing_commas != result:
        applied_steps.append("insert_missing_commas")
        result = with_missing_commas
    return result, tuple(applied_steps)


def _repair_misclosed_block_arrays(text: str) -> str:
    result = text
    block_keys = ("saju_blocks", "face_blocks", "pair_blocks")
    following_keys = (
        "action_title",
        "action_body",
        "tags",
        "disclaimer",
        "face_summary",
        "face_subtitle",
        "pair_subtitle",
        "essence",
    )
    following_key_pattern = "|".join(re.escape(key) for key in following_keys)
    for block_key in block_keys:
        pattern = (
            rf'("{re.escape(block_key)}"\s*:\s*\[.*?\n)'
            rf'(?P<indent>\s*)}}\s*,'
            rf'(?=\s*\n\s*"({following_key_pattern})"\s*:)'
        )
        result = re.sub(
            pattern,
            lambda match: f"{match.group(1)}{match.group('indent')}],",
            result,
            flags=re.DOTALL,
        )
    return result


def _insert_missing_comma(match: re.Match[str]) -> str:
    token = match.group(1)
    whitespace = match.group(2)
    if "," in whitespace:
        return match.group(0)
    return f"{token},{whitespace}"


def _strip_markdown_fence_text(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    result = "\n".join(lines)
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
    time_text, birth_time_known = _normalize_birth_time(birth_time)
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


def _normalize_birth_time(birth_time: str) -> tuple[str, bool]:
    cleaned_time = birth_time.strip()
    birth_time_known = cleaned_time.lower() not in _UNKNOWN_BIRTH_TIME_VALUES
    time_text = cleaned_time
    if not birth_time_known:
        time_text = UNKNOWN_BIRTH_TIME_REPRESENTATIVE
    else:
        time_branch_time = representative_time_from_time_branch(cleaned_time)
        if time_branch_time is not None:
            time_text = time_branch_time
    result = (time_text, birth_time_known)
    return result


def _validate_mode(mode: str) -> str:
    cleaned = mode.strip()
    if cleaned not in COMPATIBILITY_MODES:
        raise ValueError("궁합 모드는 연인, 친구, 직장동료 중 하나여야 합니다.")
    result = cleaned
    return result



def _new_session_dir(base_dir: Path, prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    result = base_dir / f"{prefix}_{stamp}"
    result.mkdir(parents=True, exist_ok=True)
    return result
# --- _load_active_distributed_app_config ---
def _load_active_distributed_app_config():
    from oracle_report.config import load_app_config

    app_config = load_app_config()
    result = app_config
    return result



# --- _safe_generate_distributed ---
def _safe_generate_distributed(
    prompt_name: str,
    values: dict[str, object],
    categories: tuple[str, ...],
    image_path: Path | None,
    app_config,
    client: TextGenerator | None,
    fallback: str,
    debug_label: str,
    progress_callback=None,
) -> _GeneratedText:
    text = fallback
    error = ""
    try:
        text = _generate_distributed(
            prompt_name,
            values,
            categories,
            image_path,
            app_config,
            client,
            progress_callback=progress_callback,
        )
        text = _normalize_generated_output_text(text, debug_label)
        print(
            f"\n[LLM RAW:{debug_label}:BEGIN]\n"
            f"{text}\n"
            f"[LLM RAW:{debug_label}:END]\n",
        )
    except Exception as exc:
        error = str(exc)
        text = f"{fallback}\n\n오류: {error}"
        print(f"\n[LLM RAW:{debug_label}:ERROR] {error}\n")
    result = _GeneratedText(text=text, error=error)
    return result



# --- DistributedTaskScheduler ---

class DistributedTaskScheduler:
    def __init__(self, slave_addrs: list[str]) -> None:
        self.slave_addrs = slave_addrs
        self.slave_metadata = {}
        self._next_index = 0
        self._update_slave_statuses()

    def _update_slave_statuses(self) -> None:
        import requests
        from concurrent.futures import ThreadPoolExecutor
        from oracle_report.config import load_llm_config
        from oracle_report.llm import is_local_llm_running, LlamaCppChatClient

        def get_status(addr: str) -> tuple[str, dict[str, object]]:
            if addr == "local":
                is_busy = is_local_llm_running()
                tps = 1.0
                score = 2.0
                try:
                    llm_config = load_llm_config()
                    client = LlamaCppChatClient(llm_config)
                    tps = client.get_or_measure_tps()
                    score = client.get_compute_score()
                except Exception:
                    pass
                return addr, {
                    "status": "busy" if is_busy else "idle",
                    "compute_score": score,
                    "tps": tps,
                }
            else:
                try:
                    response = requests.get(
                        f"{addr.rstrip('/')}/api/distributed/status",
                        timeout=2.0,
                    )
                    if response.status_code == 200:
                        data = response.json()
                        return addr, {
                            "status": data.get("status", "idle"),
                            "compute_score": float(data.get("compute_score", 5.0)),
                            "tps": float(data.get("tps", 1.0)),
                        }
                except Exception:
                    pass
                return addr, {
                    "status": "busy",
                    "compute_score": 0.1,
                    "tps": 0.1,
                }

        with ThreadPoolExecutor(max_workers=max(1, len(self.slave_addrs))) as executor:
            results = executor.map(get_status, self.slave_addrs)
            for addr, meta in results:
                self.slave_metadata[addr] = meta

    def select_slave(self, task_name: str) -> str:
        del task_name
        if not self.slave_addrs:
            raise RuntimeError(
                "No slave addresses available for distributed task execution.",
            )
        
        # 1. idle한 워커 중 compute_score가 가장 높은 순으로 정렬
        idle_workers = [
            addr
            for addr, meta in self.slave_metadata.items()
            if meta.get("status") == "idle"
        ]
        if idle_workers:
            idle_workers.sort(
                key=lambda addr: self.slave_metadata[addr].get("compute_score", 1.0),
                reverse=True,
            )
            selected = idle_workers[0]
            # 한 워커로 쏠림 방지를 위해 선택된 워커의 임시 상태 변경
            self.slave_metadata[selected]["status"] = "busy"
            return selected
        
        # 2. 모든 워커가 busy하다면 전체 워커 중 compute_score가 가장 높은 순으로 라운드 로빈 선택
        all_workers = list(self.slave_addrs)
        all_workers.sort(
            key=lambda addr: self.slave_metadata.get(addr, {}).get("compute_score", 1.0),
            reverse=True,
        )
        selected = all_workers[self._next_index % len(all_workers)]
        self._next_index += 1
        return selected

# --- _generate_distributed ---
def _generate_distributed(
    prompt_name: str,
    values: dict[str, object],
    categories: tuple[str, ...],
    image_path: Path | None,
    app_config,
    client: TextGenerator | None = None,
    progress_callback=None,
) -> str:
    import queue
    import threading
    import requests
    import time
    import base64
    import copy
    from urllib.parse import urlparse
    import socket
    from oracle_report.config import load_report_llm_config
    from oracle_report.llm import LlamaCppChatClient, is_local_llm_running

    face_llm_config = load_report_llm_config()
    report_llm_config = load_report_llm_config()

    # Clear stale global state from previous runs
    _LAST_STATUS_CHECK_TIMES.clear()
    _LAST_COMPLETED_TIMES.clear()

    # Build the task list
    tasks = [{"is_metadata": True, "target_category": None, "retries": 0}]
    for cat in categories:
        tasks.append({"is_metadata": False, "target_category": cat, "retries": 0})

    image_base64 = None
    if image_path and image_path.exists():
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    task_queue = queue.PriorityQueue()
    task_id = 0
    task_id_lock = threading.Lock()

    def put_task(task):
        nonlocal task_id
        is_meta = task.get("is_metadata", False)
        cat = task.get("target_category")
        retries = task.get("retries", 0)
        if is_meta:
            priority = 0
        elif retries > 0:
            priority = 1
        else:
            try:
                priority = 2 + categories.index(cat)
            except ValueError:
                priority = 100
        with task_id_lock:
            task_queue.put((priority, task_id, task))
            task_id += 1

    for task in tasks:
        put_task(task)

    scheduler = DistributedTaskScheduler(app_config.slave_addrs)

    local_score = 5.0
    if client is not None:
        try:
            local_score = client.get_compute_score()
        except Exception:
            pass
    else:
        try:
            from oracle_report.llm import LlamaCppChatClient
            temp_client = LlamaCppChatClient(report_llm_config)
            local_score = temp_client.get_compute_score()
        except Exception:
            pass
    scheduler.slave_metadata["local"] = {
        "status": "idle",
        "compute_score": local_score,
        "tps": 1.0,
    }

    # Pre-fetch and synchronize initial compute scores and status of all slaves
    for slave_url in app_config.slave_addrs:
        scheduler.slave_metadata[slave_url] = {
            "status": "idle",
            "compute_score": 50.0,
            "tps": 1.0,
        }
        try:
            status_url = f"{slave_url.rstrip('/')}/api/distributed/status"
            res = requests.get(status_url, timeout=2.0)
            if res.status_code == 200:
                status_data = res.json()
                score = status_data.get("compute_score")
                if score is not None:
                    scheduler.slave_metadata[slave_url]["compute_score"] = float(score)
                status_val = status_data.get("status")
                if status_val:
                    scheduler.slave_metadata[slave_url]["status"] = status_val
        except Exception:
            pass

    results = []
    results_lock = threading.Lock()
    completed_tasks = set()  # Set of (is_metadata, target_category) that have succeeded
    completed_lock = threading.Lock()

    # Track currently active tasks per worker
    active_assignments = {}
    assignments_lock = threading.Lock()

    def is_task_done(task):
        with completed_lock:
            key = (task["is_metadata"], task["target_category"])
            return key in completed_tasks

    def is_metadata_completed():
        for t in tasks:
            if t.get("is_metadata", False):
                if not is_task_done(t):
                    return False
        return True

    def mark_task_done(task):
        """Mark task as done. Returns True if this was the first completion, False if already done."""
        with completed_lock:
            key = (task["is_metadata"], task["target_category"])
            if key not in completed_tasks:
                completed_tasks.add(key)
                task_queue.task_done()  # Increment queue completion safely exactly once
                return True
            return False

    def find_unfinished_speculative_task(my_url, is_my_local):
        if not app_config.distributed_speculative:
            return None
        # Pre-fetch own score OUTSIDE the lock to avoid blocking all workers
        my_raw_score = scheduler.slave_metadata.get(my_url, {}).get("compute_score", 5.0)
        if not is_my_local and my_raw_score == 5.0:
            now = time.time()
            last_check = _LAST_STATUS_CHECK_TIMES.get(my_url, 0.0)
            if now - last_check >= 5.0:
                _LAST_STATUS_CHECK_TIMES[my_url] = now
                try:
                    status_url = f"{my_url.rstrip('/')}/api/distributed/status"
                    res = requests.get(status_url, timeout=2.0)
                    if res.status_code == 200:
                        status_data = res.json()
                        score = status_data.get("compute_score")
                        if score is not None:
                            scheduler.slave_metadata[my_url]["compute_score"] = float(score)
                            my_raw_score = float(score)
                except Exception:
                    pass

        # Speculative work stealing: find any task currently assigned to another slower node 
        # (or just any task) that has NOT completed yet.
        with assignments_lock:
            for worker_url, assigned_task in active_assignments.items():
                if assigned_task is None:
                    continue
                is_other_local = (worker_url == "local" or worker_url.startswith("local_"))
                if is_my_local and not is_other_local:
                    if not is_task_done(assigned_task):
                        return copy.deepcopy(assigned_task)
                elif not is_my_local and is_other_local:
                    if not is_task_done(assigned_task):
                        return copy.deepcopy(assigned_task)
                elif not is_my_local and not is_other_local and worker_url != my_url:
                    my_score = get_virtual_score(my_url, my_raw_score)
                    other_score = get_virtual_score(worker_url, scheduler.slave_metadata.get(worker_url, {}).get("compute_score", 5.0))
                    if my_score > other_score:
                        if not is_task_done(assigned_task):
                            return copy.deepcopy(assigned_task)
        return None

    def worker_loop(slave_url: str) -> None:
        consecutive_failures = 0
        max_consecutive_failures = 5

        is_local = False
        try:
            parsed = urlparse(slave_url)
            hostname = parsed.hostname or ""
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            if hostname in ("localhost", "127.0.0.1", "0.0.0.0"):
                is_local = True
            elif port == app_config.port:
                try:
                    ip = socket.gethostbyname(hostname)
                    if ip.startswith("127."):
                        is_local = True
                    else:
                        my_hostname = socket.gethostname()
                        if hostname == my_hostname:
                            is_local = True
                        else:
                            my_ips = socket.gethostbyname_ex(my_hostname)[2]
                            if ip in my_ips:
                                 is_local = True
                except Exception:
                    if hostname == "192.168.0.13":
                        is_local = True
        except Exception:
            pass

        local_client = client
        if is_local and local_client is None:
            is_face = "face" in prompt_name
            llm_config = face_llm_config if is_face else report_llm_config
            local_client = LlamaCppChatClient(llm_config)
            print(f"[Distributed] URL {slave_url} detected as local. Bypassing HTTP to run directly via LlamaCppChatClient.")

        while True:
            if consecutive_failures >= max_consecutive_failures:
                print(f"[Distributed][Offline] Slave {slave_url} failed consecutively {max_consecutive_failures} times. Stopping worker thread.")
                break

            task = None
            speculative = False

            try:
                _, _, task = task_queue.get(block=True, timeout=0.5)
                if is_task_done(task):
                    task_queue.task_done()
                    continue
            except queue.Empty:
                task = find_unfinished_speculative_task(slave_url, is_local)
                if task is None:
                    break
                speculative = True

            with assignments_lock:
                active_assignments[slave_url] = task

            is_meta = task["is_metadata"]
            cat = task["target_category"]

            # If it's a remote slave, check its availability status first (Hybrid Mode Support)
            compute_score = 5.0
            if not is_local:
                now = time.time()
                last_check = _LAST_STATUS_CHECK_TIMES.get(slave_url, 0.0)
                cached_status = scheduler.slave_metadata.get(slave_url, {}).get("status", "idle")
                if now - last_check >= 3.0 or cached_status == "busy":
                    _LAST_STATUS_CHECK_TIMES[slave_url] = now
                    try:
                        status_url = f"{slave_url.rstrip('/')}/api/distributed/status"
                        res = requests.get(status_url, timeout=2.0)
                        if res.status_code == 200:
                            status_data = res.json()
                            score = status_data.get("compute_score")
                            if score is not None:
                                scheduler.slave_metadata[slave_url]["compute_score"] = float(score)
                            status_val = status_data.get("status")
                            if status_val:
                                scheduler.slave_metadata[slave_url]["status"] = status_val
                    except Exception:
                        pass
                
                meta_info = scheduler.slave_metadata.get(slave_url, {})
                compute_score = meta_info.get("compute_score", 5.0)
            else:
                try:
                    compute_score = local_client.get_compute_score()
                except Exception:
                    compute_score = 5.0

            # Direct processing without yielding

            rendered = None
            if is_local or app_config.debug:
                from oracle_report.prompt_templates import render_distributed_prompt_template
                try:
                    rendered = render_distributed_prompt_template(
                        name=prompt_name,
                        values=values,
                        target_category=cat,
                        is_metadata=is_meta,
                    )
                except Exception as e:
                    rendered = f"[Failed to render prompt for debug/local]: {e}"

            if app_config.debug:
                prefix_tag = "SPECULATIVE" if speculative else "NORMAL"
                print(f"\n--- [DEBUG: Distributed {prefix_tag} Request to {slave_url}] ---")
                print(f"Category: {cat or 'metadata'}")
                print(f"Prompt:\n{rendered}")
                print("-" * 50 + "\n", flush=True)

            # Get current score of target worker and master
            my_curr_score = scheduler.slave_metadata.get(slave_url, {}).get("compute_score", 5.0)
            local_curr_score = scheduler.slave_metadata.get("local", {}).get("compute_score", 5.0)
            from urllib.parse import urlparse
            device_name = "local (127.0.0.1)" if is_local else f"{slave_url} ({urlparse(slave_url).hostname or ''})"

            prefix_tag = "SPECULATIVE" if speculative else "NORMAL"
            print(f"[Distributed][Start] Task '{cat or 'metadata'}' dispatched to {device_name} (Worker Score: {my_curr_score:.2f}, Master Score: {local_curr_score:.2f})")

            success = False
            output = None
            error_msg = ""
            start_time = time.perf_counter()

            if is_local:
                try:
                    output = local_client.generate(rendered, image_path=image_path)
                    if app_config.debug:
                        print(f"\n--- [DEBUG: Distributed Response from {device_name}] ---")
                        print(f"Category: {cat or 'metadata'}")
                        print(f"Output:\n{output}")
                        print("-" * 50 + "\n", flush=True)
                    parsed = parse_oracle_text_response(output)
                    if _is_parsed_output_valid(parsed, is_meta):
                        success = True
                    else:
                        success = False
                        error_msg = "Direct local generation parsed output is invalid"
                except Exception as e:
                    error_msg = f"Direct local generation failed: {e}"
            else:
                api_url = f"{slave_url.rstrip('/')}/api/distributed/generate"
                payload = {
                    "prompt_name": prompt_name,
                    "target_category": cat,
                    "is_metadata": is_meta,
                    "values": values,
                    "image_base64": image_base64
                }
                try:
                    res = requests.post(api_url, json=payload, timeout=300.0)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("status") == "success":
                            candidate_output = data.get("output")
                            if app_config.debug:
                                print(f"\n--- [DEBUG: Distributed Response from {device_name}] ---")
                                print(f"Category: {cat or 'metadata'}")
                                print(f"Output:\n{candidate_output}")
                                print("-" * 50 + "\n", flush=True)
                            parsed = parse_oracle_text_response(candidate_output)
                            if _is_parsed_output_valid(parsed, is_meta):
                                success = True
                                output = candidate_output
                                score = data.get("compute_score")
                                if score is not None:
                                    scheduler.slave_metadata[slave_url]["compute_score"] = float(score)
                            else:
                                success = False
                                error_msg = "Remote slave returned invalid parsed output"
                        else:
                            error_msg = data.get("error", "Unknown slave error")
                    else:
                        error_msg = f"HTTP status {res.status_code}"
                except Exception as e:
                    error_msg = str(e)

            elapsed = time.perf_counter() - start_time

            with assignments_lock:
                active_assignments[slave_url] = None

            # Fetch updated scores for logging
            my_updated_score = scheduler.slave_metadata.get(slave_url, {}).get("compute_score", 5.0)
            local_updated_score = scheduler.slave_metadata.get("local", {}).get("compute_score", 5.0)

            if success:
                consecutive_failures = 0
                with _COMPLETED_TIMES_LOCK:
                    _LAST_COMPLETED_TIMES[slave_url] = time.time()
                print(f"[Distributed] Task '{cat or 'metadata'}' completed on {device_name} in {elapsed:.2f}s (Worker Score: {my_updated_score:.2f}, Master Score: {local_updated_score:.2f})")
                # Atomic check-mark-append: mark_task_done returns True only for the first completer
                if mark_task_done(task):
                    with results_lock:
                        results.append({"task": task, "success": True, "output": output})
            else:
                # Only count non-speculative failures for consecutive failure tracking (#2)
                if not speculative:
                    consecutive_failures += 1
                print(f"[Distributed] Task '{cat or 'metadata'}' failed on {device_name} in {elapsed:.2f}s. Error: {error_msg} (Worker Score: {my_updated_score:.2f}, Master Score: {local_updated_score:.2f})")
                if not speculative:
                    if is_task_done(task):
                        task_queue.task_done()  # Balance the original get() (#1)
                        continue
                    task["retries"] += 1
                    if task["retries"] <= 3:
                        print(f"[Distributed][Retry] Task {cat or 'metadata'} failed on {slave_url} (Error: {error_msg}). Retrying ({task['retries']}/3)...")
                        put_task(task)
                        task_queue.task_done()  # Balance the original get()
                        time.sleep(1.0)
                    else:
                        print(f"[Distributed][Error] Task {cat or 'metadata'} failed on {slave_url} after 3 retries. Error: {error_msg}")
                        mark_task_done(task)  # mark_task_done already calls task_done()
                        with results_lock:
                            results.append({"task": task, "success": False, "error": error_msg})

    def local_worker_loop() -> None:
        from oracle_report.prompt_templates import render_distributed_prompt_template
        is_face = "face" in prompt_name
        llm_config = face_llm_config if is_face else report_llm_config
        local_client = client
        if local_client is None:
            local_client = LlamaCppChatClient(llm_config)

        try:
            local_score = local_client.get_compute_score()
        except Exception:
            local_score = 5.0

        consecutive_failures = 0
        max_consecutive_failures = 10

        while True:
            if consecutive_failures >= max_consecutive_failures:
                print(f"[Distributed][Offline] Local worker failed consecutively {max_consecutive_failures} times. Stopping worker thread.")
                break

            task = None
            speculative = False

            try:
                _, _, task = task_queue.get(block=True, timeout=0.5)
                if is_task_done(task):
                    task_queue.task_done()
                    continue
            except queue.Empty:
                task = find_unfinished_speculative_task("local", True)
                if task is None:
                    break
                speculative = True

            with assignments_lock:
                active_assignments[threading.current_thread().name] = task

            is_meta = task["is_metadata"]
            cat = task["target_category"]

            # Local worker directly processes its own retrieved tasks without yielding to remote slaves.

            # Wrap template rendering in try/except (#10)
            try:
                rendered = render_distributed_prompt_template(
                    name=prompt_name,
                    values=values,
                    target_category=cat,
                    is_metadata=is_meta,
                )
            except Exception as e:
                rendered = f"[Failed to render prompt for local]: {e}"

            if app_config.debug:
                prefix_tag = "SPECULATIVE" if speculative else "NORMAL"
                print(f"\n--- [DEBUG: Distributed Local {prefix_tag} Request] ---")
                print(f"Category: {cat or 'metadata'}")
                print(f"Prompt:\n{rendered}")
                print("-" * 50 + "\n", flush=True)

            local_curr_score = scheduler.slave_metadata.get("local", {}).get("compute_score", 5.0)
            prefix_tag = "SPECULATIVE" if speculative else "NORMAL"
            print(f"[Distributed][Start] Local Task '{cat or 'metadata'}' dispatched to local (127.0.0.1) (Local Score: {local_curr_score:.2f})")

            success = False
            output = None
            error_msg = ""
            start_time = time.perf_counter()
            try:
                output = local_client.generate(rendered, image_path=image_path)
                if app_config.debug:
                    print(f"\n--- [DEBUG: Distributed Local Response] ---")
                    print(f"Category: {cat or 'metadata'}")
                    print(f"Output:\n{output}")
                    print("-" * 50 + "\n", flush=True)
                parsed = parse_oracle_text_response(output)
                if _is_parsed_output_valid(parsed, is_meta):
                    success = True
                else:
                    success = False
                    error_msg = "Local generation parsed output is invalid"
            except Exception as e:
                error_msg = str(e)

            elapsed = time.perf_counter() - start_time

            with assignments_lock:
                active_assignments[threading.current_thread().name] = None

            local_updated_score = scheduler.slave_metadata.get("local", {}).get("compute_score", 5.0)

            if success:
                consecutive_failures = 0
                with _COMPLETED_TIMES_LOCK:
                    _LAST_COMPLETED_TIMES["local"] = time.time()
                print(f"[Distributed] Task '{cat or 'metadata'}' completed on local (127.0.0.1) in {elapsed:.2f}s (Local Score: {local_updated_score:.2f})")
                # Atomic check-mark-append (#3)
                if mark_task_done(task):
                    with results_lock:
                        results.append({"task": task, "success": True, "output": output})
            else:
                # Only count non-speculative failures (#2)
                if not speculative:
                    consecutive_failures += 1
                print(f"[Distributed] Task '{cat or 'metadata'}' failed on local (127.0.0.1) in {elapsed:.2f}s. Error: {error_msg} (Local Score: {local_updated_score:.2f})")
                if not speculative:
                    if is_task_done(task):
                        task_queue.task_done()  # Balance the original get() (#1)
                        continue
                    task["retries"] += 1
                    if task["retries"] <= 3:
                        print(f"[Distributed][Retry] Local task {cat or 'metadata'} failed (Error: {error_msg}). Retrying ({task['retries']}/3)...")
                        put_task(task)
                        task_queue.task_done()  # Balance the original get()
                        time.sleep(1.0)
                    else:
                        print(f"[Distributed][Error] Local task {cat or 'metadata'} failed after 3 retries. Error: {error_msg}")
                        mark_task_done(task)
                        with results_lock:
                            results.append({"task": task, "success": False, "error": error_msg})

    threads = []
    worker_urls = list(app_config.slave_addrs)
    if app_config.distributed_local_fallback:
        if app_config.distributed_role in ("master", "hybrid") or not worker_urls:
            if "local" not in worker_urls:
                worker_urls.append("local")

    for url in worker_urls:
        if url == "local":
            for i in range(1):
                t = threading.Thread(
                    target=local_worker_loop,
                    name=f"local_{i}",
                    daemon=True
                )
                t.start()
                threads.append(t)
        else:
            t = threading.Thread(target=worker_loop, args=(url,), daemon=True)
            t.start()
            threads.append(t)

    global_start_time = time.perf_counter()
    global_timeout = 1800  # 30 minutes max

    last_reported_count = -1
    while task_queue.unfinished_tasks > 0:
        elapsed = time.perf_counter() - global_start_time
        if elapsed > global_timeout:
            print(f"[Distributed][Timeout] Global timeout of {global_timeout}s reached. Breaking.")
            break
        active_workers = any(t.is_alive() for t in threads)
        if not active_workers:
            print("[Distributed][Fatal] All worker threads have terminated, but some tasks are still unfinished. Breaking to avoid deadlock.")
            break
            
        if progress_callback:
            with results_lock:
                completed_count = len(results)
            total_count = len(tasks)
            if completed_count != last_reported_count:
                last_reported_count = completed_count
                progress_val = int(30 + (completed_count / total_count) * 60)
                progress_callback(progress_val, f"사주 리포트를 해석하는 중... ({completed_count}/{total_count})")
                
        time.sleep(0.5)

    # Wait for worker threads to finish before processing results (#9)
    for t in threads:
        t.join(timeout=5.0)

    meta_output = {}
    blocks_by_category = {}

    def _safe_clean(text: str | None) -> str:
        if not text:
            return ""
        return re.sub(r"[^가-힣a-zA-Z0-9]", "", str(text)).lower()

    # 1. Parse async worker outputs using parse_oracle_text_response
    for r in results:
        if not r["success"]:
            continue

        output_str = r["output"]
        parsed_json = parse_oracle_text_response(output_str)

        if not parsed_json or not isinstance(parsed_json, dict):
            continue

        # Handle metadata task
        if r["task"].get("is_metadata"):
            meta_output = parsed_json
        else:
            # First match: by targeted category name from system task config
            req_cat = r["task"].get("target_category")
            if req_cat:
                blocks_by_category[_safe_clean(req_cat)] = parsed_json
            
            # Second match: backup using LLM's own category field
            llm_cat = parsed_json.get("category")
            if llm_cat:
                blocks_by_category[_safe_clean(llm_cat)] = parsed_json

    # 2. Metadata Fallback to prevent complete loss
    if not meta_output or "essence" not in meta_output:
        meta_output = {
            "essence": "사주 분석 결과를 종합적으로 분석 중입니다. 잠시 후 상세 리포트를 확인해 주세요.",
            "element_note": "오행 기운의 조화와 균형을 바탕으로 후천적인 개운법을 제안합니다.",
            "saju_subtitle": "나를 찾아가는 사주 명리 분석",
            "tags": ["사주명리", "운세분석", "개운처방", "평생사주"],
            "disclaimer": "본 리포트는 참고용 엔터테인먼트 콘텐츠입니다."
        }

    final_dict = meta_output
    block_key = "saju_blocks"
    if "face" in prompt_name:
        block_key = "pair_blocks" if "couple" in prompt_name else "face_blocks"

    # 3. Align blocks strictly by defined category order
    ordered_blocks = []
    for cat in categories:
        cleaned_target = _safe_clean(cat)
        
        if cleaned_target in blocks_by_category:
            matched = blocks_by_category[cleaned_target]
            matched["category"] = cat  # Enforce standardized category name
            
            # Sub-block checks in case response is full report payload
            if "saju_blocks" in matched and isinstance(matched["saju_blocks"], list):
                for b in matched["saju_blocks"]:
                    if _safe_clean(b.get("category")) == cleaned_target:
                        matched = b
                        matched["category"] = cat
                        break
            elif "pair_blocks" in matched and isinstance(matched["pair_blocks"], list):
                for b in matched["pair_blocks"]:
                    if _safe_clean(b.get("category")) == cleaned_target:
                        matched = b
                        matched["category"] = cat
                        break
            elif "face_blocks" in matched and isinstance(matched["face_blocks"], list):
                for b in matched["face_blocks"]:
                    if _safe_clean(b.get("category")) == cleaned_target:
                        matched = b
                        matched["category"] = cat
                        break

            # Field fallback checks
            if "title" not in matched or not matched["title"]:
                matched["title"] = f"{cat} 분석"
            if "summary" not in matched or not matched["summary"]:
                matched["summary"] = "상세 요약 내용을 불러오는 중입니다."
            if "body" not in matched or not matched["body"]:
                matched["body"] = "상세 본문 내용을 생성하지 못했습니다."
                
            ordered_blocks.append(matched)
        else:
            # Isolation block to prevent block shifting on worker failure
            ordered_blocks.append({
                "category": cat,
                "title": f"{cat} 분석 안내",
                "summary": "데이터 일시적 유실",
                "body": "해당 카테고리의 생성 결과가 올바른 JSON 규격을 벗어났거나 유실되었습니다. 다시 생성해 주세요."
            })

    final_dict[block_key] = ordered_blocks
    result = json.dumps(final_dict, ensure_ascii=False)
    return result



# --- Distributed Helpers ---
def _combine_distributed_outputs(
    prompt_name: str,
    categories: tuple[str, ...],
    results: list[dict[str, object]],
) -> dict[str, object]:
    metadata: dict[str, object] = {}
    blocks_by_category: dict[str, dict[str, object]] = {}
    for item in results:
        task = item["task"]
        output = str(item["output"])
        payload = _parse_distributed_json(output)
        if bool(task["is_metadata"]):
            metadata.update(payload)
        else:
            category = str(task["target_category"])
            if payload:
                blocks_by_category[category] = payload
    block_key = _distributed_block_key(prompt_name)
    metadata[block_key] = [
        blocks_by_category.get(category, _default_distributed_block(category))
        for category in categories
    ]
    result = metadata
    return result


def _parse_distributed_json(text: str) -> dict[str, object]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence_text(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    payload: dict[str, object] = {}
    if start >= 0 and end >= start:
        try:
            loaded, _ = _load_json_with_repairs(cleaned[start : end + 1])
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            try:
                loaded = json.loads(cleaned[start : end + 1])
                if isinstance(loaded, dict):
                    payload = loaded
            except Exception:
                payload = {}
    result = payload
    return result


def _distributed_block_key(prompt_name: str) -> str:
    if prompt_name == "personal_face_analysis":
        result = "face_blocks"
    elif prompt_name == "face_analysis_couple":
        result = "pair_blocks"
    else:
        raise ValueError(f"unsupported distributed prompt: {prompt_name}")
    return result


def _default_distributed_block(category: str) -> dict[str, object]:
    result = {
        "category": category,
        "title": "분석 오류",
        "summary": "분산 작업 결과를 만들지 못했어요.",
        "body": "해당 카테고리의 분석을 생성하지 못했습니다.",
    }
    return result


def _encode_distributed_image(image_path: Path | None) -> str | None:
    import base64

    result = None
    if image_path is not None and image_path.exists():
        result = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return result


def _is_local_distributed_worker(worker_url: str, app_config) -> bool:
    from urllib.parse import urlparse

    result = worker_url == "local"
    if not result:
        parsed = urlparse(worker_url)
        host = parsed.hostname or ""
        port = parsed.port
        result = host in ("localhost", "127.0.0.1", "0.0.0.0")
        if not result and port == app_config.port:
            result = host == ""
    return result

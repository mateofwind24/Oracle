from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

from oracle_report.config import CaptureConfig, LlmConfig
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import (
    BirthProfile,
    CaptureArtifact,
    FaceBox,
    SequentialPairCaptureArtifact,
)
from oracle_report.recommender import (
    FaceRecommendation,
    recommend_faces,
)
from oracle_report.report import (
    build_couple_saju_reading_prompt,
    build_saju_reading_prompt,
)
from oracle_report.report_html import (
    render_compatibility_report_html,
    render_personal_report_html,
    _DEFAULT_FACE_BLOCKS,
)
from oracle_report.saju.repository import (
    ManseLookupResult,
    ManseRepository,
    UNKNOWN_BIRTH_TIME_REPRESENTATIVE,
    representative_time_from_time_branch,
)
from oracle_report.vision.runtime import run_capture


COMPATIBILITY_MODES = ("연인", "친구", "직장동료")
FACE_ANALYSIS_MODE_LLM_IMAGE = 1
FACE_ANALYSIS_MODE_LANDMARK_RULE = 2
FACE_ANALYSIS_MODES = (FACE_ANALYSIS_MODE_LLM_IMAGE, FACE_ANALYSIS_MODE_LANDMARK_RULE)
_UNKNOWN_BIRTH_TIME_VALUES = frozenset(("", "모름", "미상", "unknown", "none"))
_FACE_CROP_MARGIN_RATIO = 0.2
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
    recommendations: tuple[FaceRecommendation, ...]
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
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    recommendation_db_path: Path,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    status_callback: Callable[[str, str, str, str], None] | None = None,
) -> PersonalWorkflowResult:
    del manse_db_path
    active_capture_config = capture_config
    profile = _build_birth_profile(
        workflow_input.name,
        workflow_input.birth_date,
        workflow_input.birth_time,
        workflow_input.gender,
    )
    output_dir = _new_session_dir(active_capture_config.output_dir, "personal")
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
                active_capture_config,
                output_dir,
            )
            capture_timed = capture_future.result()
            timing_recorder.add(capture_timed.timing)
            capture_artifact = capture_timed.value
            manse_timed = manse_future.result()
            timing_recorder.add(manse_timed.timing)
            manse_lookup = manse_timed.value
    else:
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
    )
    saju_analysis_text = saju_analysis.text

    if status_callback is not None:
        mid_recommendations: tuple[FaceRecommendation, ...] = ()
        if not workflow_input.skip_face:
            mid_recommendations = recommend_faces(
                recommendation_db_path,
                workflow_input.target_gender,
                manse_lookup.reading,
            )
        mid_markdown = _build_personal_report_json(
            manse_lookup,
            face_analysis_text,
            saju_analysis_text,
            mid_recommendations,
            workflow_input.skip_face,
        )
        mid_html = render_personal_report_html(
            profile,
            manse_lookup,
            face_analysis_text,
            mid_recommendations,
            mid_markdown,
            True,
            workflow_input.skip_face,
        )
        status_callback(
            phase="generating",
            message="사주 풀이 완료! 관상 및 얼굴 추천 리포트를 추가로 분석하고 있습니다...",
            html=mid_html,
        )

    recommendations: tuple[FaceRecommendation, ...] = ()
    if not workflow_input.skip_face:
        recommendations = timing_recorder.run(
            "recommend_faces",
            recommend_faces,
            recommendation_db_path,
            workflow_input.target_gender,
            manse_lookup.reading,
        )
    markdown = timing_recorder.run(
        "assemble_report",
        _build_personal_report_json,
        manse_lookup,
        face_analysis_text,
        saju_analysis_text,
        recommendations,
        workflow_input.skip_face,
    )
    report_html = timing_recorder.run(
        "render_report_html",
        render_personal_report_html,
        profile,
        manse_lookup,
        face_analysis_text,
        recommendations,
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
        recommendations,
        markdown,
        False,
        workflow_input.skip_face,
    )
    output_path = output_dir / "personal_report.html"
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    (output_dir / "personal_report.md").write_text(markdown, encoding="utf-8")
    timing_recorder.finish_total()
    timing_log_path = timing_recorder.write_log(output_dir / "timings.log")
    result = PersonalWorkflowResult(
        markdown=markdown,
        report_html=report_html,
        report_fragment_html=report_fragment_html,
        output_path=output_path,
        capture_path=capture_artifact.image_path if capture_artifact is not None else None,
        recommendations=recommendations,
        face_analysis=face_analysis_text,
        manse_status="조회 완료",
        timing_log_path=timing_log_path,
    )
    return result


def run_compatibility_workflow(
    workflow_input: CompatibilityWorkflowInput,
    capture_config: CaptureConfig,
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    inter_capture_delay_seconds: float = 3.0,
    status_callback: Callable[[str, str, str, str], None] | None = None,
) -> CompatibilityWorkflowResult:
    del manse_db_path
    mode = _validate_mode(workflow_input.mode)
    active_capture_config = capture_config
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
        left_profile,
        right_profile,
        capture_artifact,
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
    )
    saju_analysis_text = saju_analysis.text

    if status_callback is not None:
        mid_markdown = _build_compatibility_report_json(
            face_analysis.text,
            saju_analysis_text,
        )
        mid_html = render_compatibility_report_html(
            left_profile,
            right_profile,
            mode,
            left_manse,
            right_manse,
            face_analysis.text,
            mid_markdown,
            True,
        )
        status_callback(
            phase="generating",
            message="사주 궁합 풀이 완료! 관상 궁합을 추가로 분석하고 있습니다...",
            html=mid_html,
        )

    markdown = timing_recorder.run(
        "final_report",
        _build_compatibility_report_json,
        face_analysis.text,
        saju_analysis_text,
    )
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
    timing_recorder.run(
        "save_report",
        output_path.write_text,
        report_html,
        encoding="utf-8",
    )
    timing_recorder.finish_total()
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
    text = artifact.face_analysis or artifact.quality.face_analysis
    if text == "":
        text = "## 관상정보\n- 랜드마크 룰 기반 관상정보를 생성하지 못했습니다."
    result = _GeneratedText(text=text, error="")
    return result


def _build_pair_face_analysis(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
) -> _GeneratedText:
    left_analysis = _build_single_face_analysis(
        left_profile,
        artifact.left,
    )
    right_analysis = _build_single_face_analysis(
        right_profile,
        artifact.right,
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




def _face_llm_image_path(artifact: CaptureArtifact) -> Path:
    result = artifact.image_path
    try:
        cv2 = _import_cv2_for_face_crop()
    except RuntimeError as exc:
        print(
            "[FACE CROP] OpenCV is unavailable; "
            f"using original image: {artifact.image_path}. reason={exc}",
        )
    else:
        frame = cv2.imread(str(artifact.image_path))
        if frame is None:
            print(
                "[FACE CROP] failed to read captured image; "
                f"using original image: {artifact.image_path}",
            )
        else:
            image_height, image_width = frame.shape[:2]
            x0, y0, x1, y1 = _expanded_face_bounds(
                artifact.face,
                image_width,
                image_height,
            )
            cropped = frame[y0:y1, x0:x1]
            crop_path = artifact.image_path.with_name(
                f"{artifact.image_path.stem}_face_crop{artifact.image_path.suffix}",
            )
            ok = bool(cropped.size) and cv2.imwrite(str(crop_path), cropped)
            if ok:
                result = crop_path
            else:
                print(
                    "[FACE CROP] failed to write cropped face image; "
                    f"using original image: {artifact.image_path}",
                )
    return result


def _pair_face_llm_image_path(artifact: SequentialPairCaptureArtifact) -> Path:
    left_path = _face_llm_image_path(artifact.left)
    right_path = _face_llm_image_path(artifact.right)
    result = left_path
    try:
        cv2 = _import_cv2_for_face_crop()
    except RuntimeError as exc:
        print(
            "[FACE CROP] OpenCV is unavailable; "
            f"using first face image for pair analysis: {left_path}. reason={exc}",
        )
    else:
        left_image = cv2.imread(str(left_path))
        right_image = cv2.imread(str(right_path))
        if left_image is None or right_image is None:
            print(
                "[FACE CROP] failed to read pair face crops; "
                f"using first face image: {left_path}",
            )
        else:
            target_height = max(left_image.shape[0], right_image.shape[0])
            left_resized = _resize_image_to_height(cv2, left_image, target_height)
            right_resized = _resize_image_to_height(cv2, right_image, target_height)
            combined = cv2.hconcat((left_resized, right_resized))
            pair_path = _pair_face_image_path(left_path, right_path)
            ok = cv2.imwrite(str(pair_path), combined)
            if ok:
                result = pair_path
            else:
                print(
                    "[FACE CROP] failed to write pair face image; "
                    f"using first face image: {left_path}",
                )
    return result


def _resize_image_to_height(cv2, image, target_height: int):
    image_height, image_width = image.shape[:2]
    result = image
    if image_height > 0 and image_height != target_height:
        target_width = max(1, int(image_width * target_height / image_height))
        result = cv2.resize(image, (target_width, target_height))
    return result


def _pair_face_image_path(left_path: Path, right_path: Path) -> Path:
    result_dir = left_path.parent
    if left_path.parent.parent == right_path.parent.parent:
        result_dir = left_path.parent.parent
    result = result_dir / "pair_face_crop.jpg"
    return result


def _expanded_face_bounds(
    face: FaceBox,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    margin_x = int(face.width * _FACE_CROP_MARGIN_RATIO)
    margin_y = int(face.height * _FACE_CROP_MARGIN_RATIO)
    x0 = max(0, face.x - margin_x)
    y0 = max(0, face.y - margin_y)
    x1 = min(image_width, face.x + face.width + margin_x)
    y1 = min(image_height, face.y + face.height + margin_y)
    result = (x0, y0, x1, y1)
    return result


def _import_cv2_for_face_crop():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required to crop face images.") from exc
    result = cv2
    return result


def _build_saju_analysis(
    client: TextGenerator,
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
) -> _GeneratedText:
    from oracle_report.config import load_app_config, load_face_llm_config, load_report_llm_config
    app_config = load_app_config()
    if app_config.distributed_split and app_config.distributed_role == "master":
        categories = ["종합 형국", "타고난 성향과 심리 패턴", "재물운과 적성", "연애운과 인간관계", "올해의 운세", "총평 및 인생의 조언"]
        values = {
            "name": profile.name,
            "gender": manse_lookup.gender,
            "timezone": "KST",
            "saju_text": manse_lookup.formatted_text
        }
        try:
            text = _generate_distributed(
                "saju_reading",
                values,
                categories,
                None,
                app_config,
                load_face_llm_config(),
                load_report_llm_config(),
            )
            return _GeneratedText(text=text, error="")
        except Exception as exc:
            return _GeneratedText(text="", error=str(exc))

    prompt = build_saju_reading_prompt(profile, manse_lookup.formatted_text)
    result = _safe_generate(
        client,
        prompt,
        None,
        "사주정보를 생성하지 못했습니다.",
        debug_label="saju_analysis",
    )
    return result


def _build_compatibility_saju_analysis(
    client: TextGenerator,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
) -> _GeneratedText:
    from oracle_report.config import load_app_config, load_face_llm_config, load_report_llm_config
    app_config = load_app_config()
    if app_config.distributed_split and app_config.distributed_role == "master":
        categories = ["관계 구조", "상호 보완", "갈등 관리", "실천 제안"]
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
            "right_saju_text": right_manse.formatted_text
        }
        try:
            text = _generate_distributed(
                "saju_reading_couple",
                values,
                categories,
                None,
                app_config,
                load_face_llm_config(),
                load_report_llm_config(),
            )
            return _GeneratedText(text=text, error="")
        except Exception as exc:
            return _GeneratedText(text="", error=str(exc))

    prompt = build_couple_saju_reading_prompt(
        left_profile,
        right_profile,
        mode,
        left_manse.formatted_text,
        right_manse.formatted_text,
    )
    result = _safe_generate(
        client,
        prompt,
        None,
        "궁합 사주정보를 생성하지 못했습니다.",
        debug_label="saju_analysis_couple",
    )
    return result


def _parse_face_markdown_to_payload(face_analysis: str, prefix: str = "face") -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if not face_analysis.strip():
        return payload

    lines = face_analysis.splitlines()
    tags = ""
    details = []
    
    system_keywords = (
        "분석 모드",
        "참고 기준",
        "비율 지표",
        "적용 제외 기준",
        "캡처 신뢰도",
        "참고 고지",
        "리포트에 넣을 설명 문장",
        "주요 태그"
    )

    for line in lines:
        line_strip = line.strip()
        if line_strip.startswith("- 주요 태그:"):
            tags = line_strip.replace("- 주요 태그:", "").strip()
        elif line_strip.startswith("- ") and ":" in line_strip:
            content = line_strip[2:].strip()
            parts = content.split(":", 1)
            if len(parts) == 2:
                title = parts[0].strip()
                desc = parts[1].strip()
                
                if title in system_keywords:
                    continue
                    
                if "(" in desc:
                    desc = desc.split("(", 1)[0].strip()
                
                if desc and not desc.endswith("."):
                    desc += "."
                
                details.append({
                    "title": title,
                    "summary": desc,
                    "body": desc
                })

    subtitle_key = f"{prefix}_subtitle"
    blocks_key = f"{prefix}_blocks"

    if tags:
        payload[subtitle_key] = f"얼굴 비율 · {tags}"
    else:
        payload[subtitle_key] = "얼굴 비율 · 인상 관찰"

    categories = ["기본 구조와 첫인상", "강점으로 읽히는 복과 기세", "관계와 대인운", "앞으로 살릴 운의 방향", "조심할 점과 생활 조언"]
    face_blocks = []
    for i, detail in enumerate(details):
        cat = categories[i % len(categories)]
        face_blocks.append({
            "category": cat,
            "title": detail["title"],
            "summary": detail["summary"],
            "body": detail["body"]
        })
        
    if not face_blocks:
        face_blocks = list(_DEFAULT_FACE_BLOCKS)

    payload[blocks_key] = face_blocks
    return payload


def _build_personal_report_json(
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    saju_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
    skip_face: bool = False,
) -> str:
    face_payload = {}
    if not skip_face:
        face_payload = _parse_face_markdown_to_payload(face_analysis, prefix="face")
    saju_payload, saju_error = _load_json_payload_or_error(saju_analysis)
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    payload = _merge_personal_payloads(
        manse_lookup,
        face_payload,
        saju_payload,
        recommendations,
        skip_face,
    )
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _build_compatibility_report_json(
    face_analysis: str,
    saju_analysis: str,
) -> str:
    face_payload = _parse_face_markdown_to_payload(face_analysis, prefix="pair")
    saju_payload, saju_error = _load_json_payload_or_error(saju_analysis)
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis_couple] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    payload = _merge_compatibility_payloads(face_payload, saju_payload)
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
        "synthesis_title",
        "synthesis_body",
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
    for key in ("essence", "synthesis_title", "synthesis_body", "action_title", "action_body"):
        if key not in payload:
            value = face_payload.get(key)
            if value:
                payload[key] = value
    payload["convergence"] = _combined_pair_convergence(face_payload, saju_payload)
    result = payload
    return result


def _combined_pair_convergence(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> list[dict[str, str]]:
    existing = face_payload.get("convergence") or saju_payload.get("convergence")
    result = []
    if isinstance(existing, list) and existing:
        result = existing
    else:
        for index in range(3):
            result.append(
                {
                    "face": _block_summary(
                        face_payload,
                        "pair_blocks",
                        index,
                        "두 사람의 얼굴 관찰에서 보이는 관계 분위기",
                    ),
                    "saju": _block_summary(
                        saju_payload,
                        "saju_blocks",
                        index,
                        "두 사람의 사주 흐름에서 보이는 상호 보완점",
                    ),
                },
            )
    return result


def _merge_personal_payloads(
    manse_lookup: ManseLookupResult,
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
    recommendations: tuple[FaceRecommendation, ...],
    skip_face: bool,
) -> dict[str, Any]:
    reading = manse_lookup.reading
    strongest = _dominant_element(reading.element_counts, strongest=True)
    weakest = _dominant_element(reading.element_counts, strongest=False)
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
        payload["convergence"] = _combined_convergence(face_payload, saju_payload)
        payload["recommendation_title"] = f"{weakest} 기운을 보완해 줄 얼굴"
        payload["recommendation_lead"] = _recommendation_lead(recommendations)
    payload["synthesis_title"] = _synthesis_title(skip_face)
    payload["synthesis_body"] = _synthesis_body(
        face_payload,
        saju_payload,
        strongest,
        weakest,
        skip_face,
    )
    payload["synthesis_summary"] = (
        "결론은 단정이 아니라 참고입니다. 강점은 살리고 부족한 리듬은 생활에서 "
        "보완하세요."
    )
    result = payload
    return result


def _combined_convergence(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
) -> list[dict[str, str]]:
    result = []
    for index in range(3):
        result.append(
            {
                "face": _block_summary(
                    face_payload,
                    "face_blocks",
                    index,
                    "얼굴 관찰에서 보이는 표현 리듬",
                ),
                "saju": _block_summary(
                    saju_payload,
                    "saju_blocks",
                    index,
                    "사주 데이터에서 보이는 생활 리듬",
                ),
            },
        )
    return result


def _block_summary(
    payload: dict[str, Any],
    key: str,
    index: int,
    default: str,
) -> str:
    blocks = payload.get(key)
    result = default
    if isinstance(blocks, list) and index < len(blocks):
        block = blocks[index]
        if isinstance(block, dict):
            result = _text_from_payload(
                block,
                "summary",
                _text_from_payload(block, "title", default),
            )
    return result


def _text_from_payload(payload: dict[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    result = default
    if isinstance(value, str) and value.strip():
        result = value.strip()
    return result


def _recommendation_lead(recommendations: tuple[FaceRecommendation, ...]) -> str:
    result = (
        "사주에서 보완이 필요한 리듬을 기준으로, 얼굴 추천 후보를 참고용으로 "
        "정리했어요."
    )
    if recommendations:
        result = recommendations[0].reason
    return result


def _synthesis_title(skip_face: bool) -> str:
    result = "사주와 얼굴 관찰이 만나는 지점"
    if skip_face:
        result = "사주 흐름을 정리하면"
    return result


def _synthesis_body(
    face_payload: dict[str, Any],
    saju_payload: dict[str, Any],
    strongest: str,
    weakest: str,
    skip_face: bool,
) -> str:
    saju_line = _text_from_payload(
        saju_payload,
        "essence",
        f"{strongest} 기운을 살리고 {weakest} 기운을 보완하는 흐름이 보여요.",
    )
    face_line = _text_from_payload(
        face_payload,
        "face_summary",
        "얼굴 관찰은 표현 방식과 대화 분위기를 보조적으로 보여줘요.",
    )
    result = saju_line
    if not skip_face:
        result = f"{saju_line} {face_line}"
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
    debug_label: str = "llm",
) -> _GeneratedText:
    text = fallback
    error = ""
    try:
        text = client.generate(prompt, image_path=image_path)
        print(f"\n[LLM RAW:{debug_label}:BEGIN]\n{text}\n[LLM RAW:{debug_label}:END]\n")
    except Exception as exc:
        error = str(exc)
        text = f"{fallback}\n\n오류: {error}"
        print(f"\n[LLM RAW:{debug_label}:ERROR] {error}\n")
    result = _GeneratedText(text=text, error=error)
    return result


def _load_json_payload_or_error(text: str) -> tuple[dict[str, Any], str]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence_text(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    payload: dict[str, Any] = {}
    error = ""
    try:
        loaded = json.loads(cleaned)
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


class DistributedTaskScheduler:
    def __init__(self, slave_addrs: list[str]) -> None:
        self.slave_addrs = slave_addrs
        self.slave_metadata = {
            addr: {"cuda": False, "weight": 1.0, "compute_score": 5.0}
            for addr in slave_addrs
        }
        self._next_index = 0

    def select_slave(self, task_name: str) -> str:
        if not self.slave_addrs:
            raise RuntimeError("No slave addresses available for distributed task execution.")
        addr = self.slave_addrs[self._next_index % len(self.slave_addrs)]
        self._next_index += 1
        return addr


def _generate_distributed(
    prompt_name: str,
    values: dict[str, Any],
    categories: list[str],
    image_path: Path | None,
    app_config,
    face_llm_config,
    report_llm_config,
) -> str:
    import base64
    import queue
    import threading
    import requests

    tasks = [{"is_metadata": True, "target_category": None, "retries": 0}]
    for cat in categories:
        tasks.append({"is_metadata": False, "target_category": cat, "retries": 0})

    image_base64 = None
    if image_path and image_path.exists():
        image_base64 = base64.b64encode(image_path.read_bytes()).decode("ascii")

    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    scheduler = DistributedTaskScheduler(app_config.slave_addrs)

    results = []
    results_lock = threading.Lock()

    def worker_loop(slave_url: str) -> None:
        consecutive_failures = 0
        max_consecutive_failures = 3

        # Determine if the target slave URL is actually the localhost/master itself to bypass HTTP
        is_local = False
        from urllib.parse import urlparse
        import socket
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
                    # Fallback check for the user's specific IP
                    if hostname == "192.168.0.13":
                        is_local = True
        except Exception:
            pass

        local_client = None
        if is_local:
            from oracle_report.prompt_templates import render_distributed_prompt_template
            from oracle_report.llm import LlamaCppChatClient
            is_face = "face" in prompt_name
            llm_config = face_llm_config if is_face else report_llm_config
            local_client = LlamaCppChatClient(llm_config)
            print(f"[Distributed] URL {slave_url} detected as local. Bypassing HTTP to run directly via LlamaCppChatClient.")

        while True:
            if consecutive_failures >= max_consecutive_failures:
                print(f"[Distributed][Offline] Slave {slave_url} failed consecutively {max_consecutive_failures} times. Stopping worker thread.")
                break
            try:
                task = task_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                break

            is_meta = task["is_metadata"]
            cat = task["target_category"]

            # If it's a remote slave, check its availability status first (Hybrid Mode Support)
            compute_score = 5.0
            if not is_local:
                try:
                    status_url = f"{slave_url.rstrip('/')}/api/distributed/status"
                    res = requests.get(status_url, timeout=2.0)
                    if res.status_code == 200:
                        status_data = res.json()
                        if status_data.get("status") == "busy":
                            task_queue.put(task)
                            task_queue.task_done()  # Keep unfinished_tasks counter in sync
                            time.sleep(1.0)
                            continue
                        score = status_data.get("compute_score")
                        if score is not None:
                            scheduler.slave_metadata[slave_url]["compute_score"] = float(score)
                            compute_score = float(score)
                except Exception:
                    compute_score = scheduler.slave_metadata.get(slave_url, {}).get("compute_score", 5.0)
            else:
                try:
                    compute_score = local_client.get_compute_score()
                except Exception:
                    compute_score = 5.0

            # 중요도 기반 태스크 라우팅 (Task-to-Model Routing)
            is_core_category = cat in ("종합 형국", "타고난 성향과 심리 패턴", "총평 및 인생의 조언")
            if is_core_category and compute_score < 20.0:
                other_high_perf_exists = any(
                    meta.get("compute_score", 0.0) >= 20.0 
                    for meta in scheduler.slave_metadata.values()
                )
                if other_high_perf_exists:
                    task_queue.put(task)
                    task_queue.task_done()
                    time.sleep(0.5)
                    continue

            # Render prompt for local generation or debugging
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

            # Debug logging: Request
            if app_config.debug:
                print(f"\n--- [DEBUG: Distributed Request to {slave_url}] ---")
                print(f"Category: {cat or 'metadata'}")
                print(f"Prompt:\n{rendered}")
                print("-" * 50 + "\n", flush=True)

            success = False
            output = None
            error_msg = ""

            if is_local:
                try:
                    output = local_client.generate(rendered, image_path=image_path)
                    success = True
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
                    # Increase timeout to 180s to handle resource-constrained edge devices gracefully
                    res = requests.post(api_url, json=payload, timeout=180.0)
                    if res.status_code == 200:
                        data = res.json()
                        if data.get("status") == "success":
                            success = True
                            output = data.get("output")
                        else:
                            error_msg = data.get("error", "Unknown slave error")
                    else:
                        error_msg = f"HTTP status {res.status_code}"
                except Exception as e:
                    error_msg = str(e)

            # Debug logging: Response
            if success and app_config.debug:
                print(f"\n--- [DEBUG: Distributed Response from {slave_url}] ---")
                print(f"Category: {cat or 'metadata'}")
                print(f"Output:\n{output}")
                print("-" * 50 + "\n", flush=True)

            if success:
                consecutive_failures = 0
                with results_lock:
                    results.append({"task": task, "success": True, "output": output})
                task_queue.task_done()
            else:
                consecutive_failures += 1
                task["retries"] += 1
                if task["retries"] <= 3:
                    print(f"[Distributed][Retry] Task {cat or 'metadata'} failed on {slave_url} (Error: {error_msg}). Retrying ({task['retries']}/3)...")
                    task_queue.put(task)
                    time.sleep(1.0)
                else:
                    print(f"[Distributed][Error] Task {cat or 'metadata'} failed on {slave_url} after 3 retries. Error: {error_msg}")
                    with results_lock:
                        results.append({"task": task, "success": False, "error": error_msg})
                    task_queue.task_done()

    def local_worker_loop() -> None:
        from oracle_report.prompt_templates import render_distributed_prompt_template
        from oracle_report.llm import LlamaCppChatClient

        is_face = "face" in prompt_name
        llm_config = face_llm_config if is_face else report_llm_config
        client = LlamaCppChatClient(llm_config)

        try:
            local_score = client.get_compute_score()
        except Exception:
            local_score = 5.0

        while True:
            try:
                task = task_queue.get(block=True, timeout=1.0)
            except queue.Empty:
                break

            is_meta = task["is_metadata"]
            cat = task["target_category"]

            # 중요도 기반 태스크 라우팅 (Task-to-Model Routing)
            is_core_category = cat in ("종합 형국", "타고난 성향과 심리 패턴", "총평 및 인생의 조언")
            if is_core_category and local_score < 20.0:
                other_high_perf_exists = any(
                    meta.get("compute_score", 0.0) >= 20.0 
                    for meta in scheduler.slave_metadata.values()
                )
                if other_high_perf_exists:
                    task_queue.put(task)
                    task_queue.task_done()
                    time.sleep(0.5)
                    continue
            rendered = render_distributed_prompt_template(
                name=prompt_name,
                values=values,
                target_category=cat,
                is_metadata=is_meta,
            )

            # Debug logging: Request
            if app_config.debug:
                print(f"\n--- [DEBUG: Distributed Local Request] ---")
                print(f"Category: {cat or 'metadata'}")
                print(f"Prompt:\n{rendered}")
                print("-" * 50 + "\n", flush=True)

            success = False
            output = None
            error_msg = ""
            try:
                output = client.generate(rendered, image_path=image_path)
                success = True
            except Exception as e:
                error_msg = str(e)

            if success:
                # Debug logging: Response
                if app_config.debug:
                    print(f"\n--- [DEBUG: Distributed Local Response] ---")
                    print(f"Category: {cat or 'metadata'}")
                    print(f"Output:\n{output}")
                    print("-" * 50 + "\n", flush=True)
                with results_lock:
                    results.append({"task": task, "success": True, "output": output})
                task_queue.task_done()
            else:
                task["retries"] += 1
                if task["retries"] <= 3:
                    print(f"[Distributed][Retry] Local task {cat or 'metadata'} failed (Error: {error_msg}). Retrying ({task['retries']}/3)...")
                    task_queue.put(task)
                    time.sleep(1.0)
                else:
                    print(f"[Distributed][Error] Local task {cat or 'metadata'} failed after 3 retries. Error: {error_msg}")
                    with results_lock:
                        results.append({"task": task, "success": False, "error": error_msg})
                    task_queue.task_done()

    threads = []
    if app_config.slave_addrs:
        for slave_url in app_config.slave_addrs:
            t = threading.Thread(target=worker_loop, args=(slave_url,), daemon=True)
            t.start()
            threads.append(t)
    else:
        # Fallback to local execution threads
        for _ in range(2):
            t = threading.Thread(target=local_worker_loop, daemon=True)
            t.start()
            threads.append(t)

    # Monitor queue completion and guard against thread crashes (deadlock avoidance)
    while task_queue.unfinished_tasks > 0:
        active_workers = any(t.is_alive() for t in threads)
        if not active_workers:
            print("[Distributed][Fatal] All worker threads have terminated, but some tasks are still unfinished. Breaking to avoid deadlock.")
            break
        time.sleep(0.5)

    meta_output = {}
    blocks_outputs = []

    for r in results:
        if not r["success"]:
            print(f"[Distributed] Task failed: {r.get('error')}")
            continue

        output_str = r["output"]
        cleaned = output_str.strip()
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if lines and (lines[0].startswith("```json") or lines[0] == "```"):
                cleaned = "\n".join(lines[1:-1])
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end >= start:
            cleaned = cleaned[start : end + 1]

        try:
            parsed = json.loads(cleaned)
        except Exception as e:
            print(f"[Distributed] JSON parsing failed: {e}")
            parsed = {}

        if r["task"]["is_metadata"]:
            meta_output = parsed
        else:
            blocks_outputs.append(parsed)

    final_dict = meta_output
    block_key = "saju_blocks"
    if "face" in prompt_name:
        block_key = "face_blocks" if "couple" not in prompt_name else "pair_blocks"

    ordered_blocks = []
    for cat in categories:
        matched = None
        for b in blocks_outputs:
            if b.get("category") == cat:
                matched = b
                break
        if matched:
            ordered_blocks.append(matched)
        else:
            ordered_blocks.append({"category": cat, "title": "분석 오류", "summary": "연산 실패", "body": "해당 카테고리의 분석을 생성하지 못했습니다."})

    final_dict[block_key] = ordered_blocks
    return json.dumps(final_dict, ensure_ascii=False)

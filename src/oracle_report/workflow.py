from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime
import json
import time
from pathlib import Path
from typing import Any, Callable, Generic, Protocol, TypeVar

from oracle_report import prompt_templates
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
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import (
    build_couple_face_analysis_prompt,
    build_couple_saju_reading_prompt,
    build_personal_face_analysis_prompt,
    build_saju_reading_prompt,
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
FACE_ANALYSIS_MODE_LLM_IMAGE = 1
FACE_ANALYSIS_MODE_LANDMARK_RULE = 2
FACE_ANALYSIS_MODES = (FACE_ANALYSIS_MODE_LLM_IMAGE, FACE_ANALYSIS_MODE_LANDMARK_RULE)
_UNKNOWN_BIRTH_TIME_VALUES = frozenset(("", "모름", "미상", "unknown", "none"))
_FACE_CROP_MARGIN_RATIO = 0.2
_REPORT_BLOCK_KEYS = ("face_blocks", "saju_blocks", "pair_blocks")
_BODY_REWRITE_PROMPT_MARKER = "[리포트 본문 재작성]"
_SENTENCE_ENDINGS = ".!?。！？"
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
    face_analysis_mode: int = FACE_ANALYSIS_MODE_LLM_IMAGE


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
    face_llm_config: LlmConfig,
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    recommendation_db_path: Path,
    face_client: TextGenerator | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
) -> PersonalWorkflowResult:
    del manse_db_path
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
    repository = ManseRepository()
    active_face_client = face_client
    if face_analysis_mode == FACE_ANALYSIS_MODE_LLM_IMAGE:
        active_face_client = face_client or LlamaCppChatClient(face_llm_config)
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
            active_face_client,
            profile,
            capture_artifact,
            face_analysis_mode,
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
    face_llm_config: LlmConfig,
    report_llm_config: LlmConfig,
    manse_db_path: Path,
    face_client: TextGenerator | None = None,
    report_client: TextGenerator | None = None,
    capture_runner=run_capture,
    inter_capture_delay_seconds: float = 3.0,
) -> CompatibilityWorkflowResult:
    del manse_db_path
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
    repository = ManseRepository()
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
    markdown = timing_recorder.run(
        "final_report",
        _build_compatibility_report_json,
        face_analysis.text,
        saju_analysis.text,
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
        image_path = _face_llm_image_path(artifact)
        face_input = FaceReadingInput(
            image_path=image_path,
            quality=artifact.quality,
        )
        prompt = build_personal_face_analysis_prompt(profile, face_input)
        result = _safe_generate(
            client,
            prompt,
            image_path,
            "관상정보를 생성하지 못했습니다.",
            debug_label="personal_face_analysis",
        )
        result = _repair_short_report_block_bodies(
            client,
            result,
            "personal_face_analysis",
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
    else:
        result = _build_couple_face_analysis(
            client,
            left_profile,
            right_profile,
            artifact,
            mode,
        )
    return result


def _build_couple_face_analysis(
    client: TextGenerator | None,
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    artifact: SequentialPairCaptureArtifact,
    mode: str,
) -> _GeneratedText:
    if client is None:
        raise ValueError("face analysis client is required for mode 1.")
    image_path = _pair_face_llm_image_path(artifact)
    left_input = FaceReadingInput(
        image_path=image_path,
        quality=artifact.left.quality,
    )
    right_input = FaceReadingInput(
        image_path=image_path,
        quality=artifact.right.quality,
    )
    prompt = build_couple_face_analysis_prompt(
        left_profile,
        right_profile,
        mode,
        left_input,
        right_input,
    )
    result = _safe_generate(
        client,
        prompt,
        image_path,
        "궁합 관상정보를 생성하지 못했습니다.",
        debug_label="face_analysis_copule",
    )
    result = _repair_short_report_block_bodies(
        client,
        result,
        "face_analysis_copule",
    )
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
    prompt = build_saju_reading_prompt(profile, manse_lookup.formatted_text)
    result = _safe_generate(
        client,
        prompt,
        None,
        "사주정보를 생성하지 못했습니다.",
        debug_label="saju_analysis",
    )
    result = _repair_short_report_block_bodies(
        client,
        result,
        "saju_analysis",
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
    result = _repair_short_report_block_bodies(
        client,
        result,
        "saju_analysis_couple",
    )
    return result


def _build_personal_report_json(
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    saju_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
    skip_face: bool = False,
) -> str:
    face_payload, face_error = ({}, "")
    saju_payload, saju_error = _load_json_payload_or_error(saju_analysis)
    if not skip_face:
        face_payload, face_error = _load_json_payload_or_error(face_analysis)
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
        recommendations,
        skip_face,
    )
    payload = _normalize_payload_text(payload)
    result = json.dumps(payload, ensure_ascii=False)
    return result


def _build_compatibility_report_json(
    face_analysis: str,
    saju_analysis: str,
) -> str:
    face_payload, face_error = _load_json_payload_or_error(face_analysis)
    saju_payload, saju_error = _load_json_payload_or_error(saju_analysis)
    if face_error:
        print(
            "[UI FALLBACK:face_analysis_copule] invalid LLM output; "
            f"renderer will fill missing pair fields. reason={face_error}",
        )
    if saju_error:
        print(
            "[UI FALLBACK:saju_analysis_couple] invalid LLM output; "
            f"renderer will fill missing saju fields. reason={saju_error}",
        )
    payload = _merge_compatibility_payloads(face_payload, saju_payload)
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


def _normalize_payload_text(value: Any) -> Any:
    result = value
    if isinstance(value, str):
        result = _normalize_inline_text(value)
    elif isinstance(value, list):
        result = [_normalize_payload_text(item) for item in value]
    elif isinstance(value, dict):
        result = {key: _normalize_payload_text(item) for key, item in value.items()}
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
        text = _normalize_generated_output_text(text)
        print(f"\n[LLM RAW:{debug_label}:BEGIN]\n{text}\n[LLM RAW:{debug_label}:END]\n")
    except Exception as exc:
        error = str(exc)
        text = f"{fallback}\n\n오류: {error}"
        print(f"\n[LLM RAW:{debug_label}:ERROR] {error}\n")
    result = _GeneratedText(text=text, error=error)
    return result


def _repair_short_report_block_bodies(
    client: TextGenerator,
    generated: _GeneratedText,
    debug_label: str,
) -> _GeneratedText:
    result = generated
    if generated.error == "":
        payload, error = _load_json_payload_or_error(generated.text)
        if error == "" and _payload_has_short_report_body(payload):
            repair_prompt = _build_report_body_rewrite_prompt(payload)
            repaired = _safe_generate(
                client,
                repair_prompt,
                None,
                generated.text,
                debug_label=f"{debug_label}_body_rewrite",
            )
            repaired_payload, repaired_error = _load_json_payload_or_error(
                repaired.text,
            )
            if repaired.error == "" and repaired_error == "" and repaired_payload:
                result = repaired
            else:
                print(
                    f"[LLM REPAIR:{debug_label}] body rewrite failed; "
                    "keeping original LLM output.",
                )
    return result


def _payload_has_short_report_body(payload: dict[str, Any]) -> bool:
    target_count = _target_report_body_sentence_count()
    result = False
    for block in _iter_report_blocks(payload):
        body = block.get("body")
        if isinstance(body, str):
            sentence_count = _count_sentences(body)
            if sentence_count < target_count:
                result = True
                break
    return result


def _iter_report_blocks(payload: dict[str, Any]) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for key in _REPORT_BLOCK_KEYS:
        value = payload.get(key)
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    blocks.append(item)
    result = blocks
    return result


def _target_report_body_sentence_count() -> int:
    configured_count = int(prompt_templates.REPORT_BLOCK_SENTENCE_COUNT)
    result = max(1, configured_count)
    return result


def _count_sentences(text: str) -> int:
    normalized_text = _normalize_inline_text(text)
    count = 0
    previous_was_ending = False
    for character in normalized_text:
        if character in _SENTENCE_ENDINGS:
            if not previous_was_ending:
                count += 1
            previous_was_ending = True
        elif not character.isspace():
            previous_was_ending = False
    if normalized_text != "" and count == 0:
        count = 1
    result = count
    return result


def _build_report_body_rewrite_prompt(payload: dict[str, Any]) -> str:
    target_count = _target_report_body_sentence_count()
    payload_text = json.dumps(
        _normalize_payload_text(payload),
        ensure_ascii=False,
    )
    lines = (
        _BODY_REWRITE_PROMPT_MARKER,
        "너는 리포트 JSON의 블록 body를 LLM 출력으로 다시 작성하는 API입니다.",
        "출력은 JSON 객체 하나만 작성합니다. Markdown 코드블록, 설명문, 주석을 붙이지 않습니다.",
        "기존 JSON의 키, 배열 순서, category, title은 유지합니다.",
        (
            "각 face_blocks/saju_blocks/pair_blocks의 summary는 body의 핵심을 "
            "1~2개의 짧은 문장으로 요약합니다. summary에는 아래 문장 수를 "
            "적용하지 않습니다."
        ),
        (
            "각 face_blocks/saju_blocks/pair_blocks의 body는 정확히 "
            f"{target_count}개의 완성된 문장으로 작성합니다."
        ),
        (
            "부족한 문장을 코드처럼 덧붙이지 말고, category/title/summary와 "
            "맞는 자연스러운 해설을 LLM이 새로 씁니다."
        ),
        (
            "JSON 문자열 안에 수동 줄바꿈이나 줄바꿈 이스케이프를 넣지 않고, "
            "문장 사이에는 공백만 사용합니다."
        ),
        "",
        "[원본 JSON]",
        payload_text,
    )
    result = "\n".join(lines)
    return result


def _normalize_generated_output_text(text: str) -> str:
    payload, error = _load_json_payload_or_error(text)
    result = text
    if not error and payload:
        result = json.dumps(_normalize_payload_text(payload), ensure_ascii=False)
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

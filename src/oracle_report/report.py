from __future__ import annotations

from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput, format_face_quality
from oracle_report.prompt_templates import render_prompt_template
from oracle_report.saju.repository import (
    birth_datetime_display_from_profile,
    birth_time_display_from_profile,
)


def build_personal_face_analysis_prompt(
    birth_profile: BirthProfile,
    face_input: FaceReadingInput,
) -> str:
    result = _build_face_analysis_prompt(
        "personal_face_analysis",
        birth_profile,
        face_input,
        {
            "person_label": "개인 리포트 대상",
            "mode": "개인",
        },
    )
    return result


def build_compatibility_face_analysis_prompt(
    birth_profile: BirthProfile,
    face_input: FaceReadingInput,
    person_label: str,
    mode: str,
) -> str:
    result = _build_face_analysis_prompt(
        "compatibility_face_analysis",
        birth_profile,
        face_input,
        {
            "person_label": person_label,
            "mode": mode,
        },
    )
    return result


def build_saju_reading_prompt(
    birth_profile: BirthProfile,
    saju_text: str,
) -> str:
    result = render_prompt_template(
        "saju_reading",
        {
            "name": birth_profile.name,
            "gender": _gender_text(birth_profile),
            "birth_datetime": birth_datetime_display_from_profile(birth_profile),
            "birth_time_text": _birth_time_text(birth_profile),
            "timezone": birth_profile.timezone,
            "saju_text": saju_text,
        },
    )
    return result


def build_couple_saju_reading_prompt(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_saju_text: str,
    right_saju_text: str,
) -> str:
    result = render_prompt_template(
        "saju_reading_couple",
        {
            "left_name": left_profile.name,
            "left_gender": _gender_text(left_profile),
            "left_birth_datetime": birth_datetime_display_from_profile(left_profile),
            "left_birth_time_text": _birth_time_text(left_profile),
            "right_name": right_profile.name,
            "right_gender": _gender_text(right_profile),
            "right_birth_datetime": birth_datetime_display_from_profile(right_profile),
            "right_birth_time_text": _birth_time_text(right_profile),
            "mode": mode,
            "left_saju_text": left_saju_text,
            "right_saju_text": right_saju_text,
        },
    )
    return result


def build_couple_face_analysis_prompt(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_face_input: FaceReadingInput,
    right_face_input: FaceReadingInput,
) -> str:
    result = render_prompt_template(
        "face_analysis_copule",
        {
            "left_name": left_profile.name,
            "left_gender": _gender_text(left_profile),
            "left_birth_datetime": birth_datetime_display_from_profile(left_profile),
            "left_birth_time_text": _birth_time_text(left_profile),
            "left_quality_text": format_face_quality(left_face_input.quality),
            "right_name": right_profile.name,
            "right_gender": _gender_text(right_profile),
            "right_birth_datetime": birth_datetime_display_from_profile(right_profile),
            "right_birth_time_text": _birth_time_text(right_profile),
            "right_quality_text": format_face_quality(right_face_input.quality),
            "mode": mode,
        },
    )
    return result


def _build_face_analysis_prompt(
    template_name: str,
    birth_profile: BirthProfile,
    face_input: FaceReadingInput,
    extra_values: dict[str, str],
) -> str:
    quality_text = format_face_quality(face_input.quality)
    landmark_metrics_text = "- 랜드마크 측정값 없음"
    landmark_context_text = "- 구조화된 관찰 컨텍스트 없음"
    landmark_rules_text = "- 랜드마크 규칙 해석 힌트 없음"
    if face_input.quality is not None:
        if face_input.quality.landmark_metrics_text.strip() != "":
            landmark_metrics_text = face_input.quality.landmark_metrics_text
        if face_input.quality.landmark_context_text.strip() != "":
            landmark_context_text = face_input.quality.landmark_context_text
        if face_input.quality.landmark_rules_text.strip() != "":
            landmark_rules_text = face_input.quality.landmark_rules_text
    values = {
        "name": birth_profile.name,
        "gender": _gender_text(birth_profile),
        "birth_datetime": birth_datetime_display_from_profile(birth_profile),
        "birth_time_text": _birth_time_text(birth_profile),
        "quality_text": quality_text,
        "landmark_metrics_text": landmark_metrics_text,
        "landmark_context_text": landmark_context_text,
        "landmark_rules_text": landmark_rules_text,
    }
    values.update(extra_values)
    result = render_prompt_template(template_name, values)
    return result


def _gender_text(profile: BirthProfile) -> str:
    result = profile.gender
    if result == "":
        result = "미입력"
    return result


def _birth_time_text(profile: BirthProfile) -> str:
    result = birth_time_display_from_profile(profile)
    return result

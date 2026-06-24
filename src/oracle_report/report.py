from __future__ import annotations

from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput, format_face_quality
from oracle_report.prompt_templates import render_prompt_template


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


def build_personal_final_prompt(
    birth_profile: BirthProfile,
    saju_text: str,
    face_analysis: str,
    recommendation_text: str,
) -> str:
    result = render_prompt_template(
        "personal_final",
        {
            "name": birth_profile.name,
            "gender": _gender_text(birth_profile),
            "birth_datetime": birth_profile.birth_datetime.isoformat(sep=" "),
            "birth_time_text": _birth_time_text(birth_profile),
            "timezone": birth_profile.timezone,
            "saju_text": saju_text,
            "face_analysis": face_analysis,
            "recommendation_text": recommendation_text,
        },
    )
    return result


def build_compatibility_final_prompt(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_saju_text: str,
    right_saju_text: str,
    face_analysis: str,
) -> str:
    result = render_prompt_template(
        "compatibility_final",
        {
            "left_name": left_profile.name,
            "left_gender": _gender_text(left_profile),
            "left_birth_datetime": left_profile.birth_datetime.isoformat(sep=" "),
            "left_birth_time_text": _birth_time_text(left_profile),
            "right_name": right_profile.name,
            "right_gender": _gender_text(right_profile),
            "right_birth_datetime": right_profile.birth_datetime.isoformat(sep=" "),
            "right_birth_time_text": _birth_time_text(right_profile),
            "mode": mode,
            "left_saju_text": left_saju_text,
            "right_saju_text": right_saju_text,
            "face_analysis": face_analysis,
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
    values = {
        "name": birth_profile.name,
        "gender": _gender_text(birth_profile),
        "birth_datetime": birth_profile.birth_datetime.isoformat(sep=" "),
        "birth_time_text": _birth_time_text(birth_profile),
        "quality_text": quality_text,
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
    result = "입력됨"
    if not profile.birth_time_known:
        result = "미입력, 정오 기준으로 사주를 보조 계산함"
    return result

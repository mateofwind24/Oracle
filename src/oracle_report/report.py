from __future__ import annotations

from oracle_report.models import BirthProfile
from oracle_report.prompt_templates import render_prompt_template
from oracle_report.saju.repository import (
    birth_datetime_display_from_profile,
    birth_time_display_from_profile,
)


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


def _gender_text(profile: BirthProfile) -> str:
    result = profile.gender
    if result == "":
        result = "미입력"
    return result


def _birth_time_text(profile: BirthProfile) -> str:
    result = birth_time_display_from_profile(profile)
    return result

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.saju.calendar import (
    time_branch_index_from_datetime as _chart_time_branch_index_from_datetime,
)
from oracle_report.saju.engine import ELEMENTS, SajuReading, build_saju_reading


MANSE_GENDERS = ("남성", "여성")
MANSE_TIME_BRANCH_LABELS = (
    "자시",
    "축시",
    "인시",
    "묘시",
    "진시",
    "사시",
    "오시",
    "미시",
    "신시",
    "유시",
    "술시",
    "해시",
)
MANSE_TIME_BRANCH_HANJA = (
    "子時",
    "丑時",
    "寅時",
    "卯時",
    "辰時",
    "巳時",
    "午時",
    "未時",
    "申時",
    "酉時",
    "戌時",
    "亥時",
)
MANSE_TIME_BRANCH_RANGES = (
    "23:30-01:30",
    "01:30-03:30",
    "03:30-05:30",
    "05:30-07:30",
    "07:30-09:30",
    "09:30-11:30",
    "11:30-13:30",
    "13:30-15:30",
    "15:30-17:30",
    "17:30-19:30",
    "19:30-21:30",
    "21:30-23:30",
)
_REPRESENTATIVE_TIMES = (
    "00:30",
    "02:30",
    "04:30",
    "06:30",
    "08:30",
    "10:30",
    "12:30",
    "14:30",
    "16:30",
    "18:30",
    "20:30",
    "22:30",
)
UNKNOWN_BIRTH_TIME_REPRESENTATIVE = "12:30"
_UNKNOWN_BIRTH_TIME_DISPLAY = "시간 미상 (오시(午時) 보조 기준)"


@dataclass(frozen=True)
class ManseLookupResult:
    reading: SajuReading
    formatted_text: str
    gender: str
    time_branch_label: str
    daeun_direction: str


class ManseRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def lookup(self, profile: BirthProfile) -> ManseLookupResult:
        result = build_manse_lookup(profile)
        return result


def build_manse_lookup(profile: BirthProfile) -> ManseLookupResult:
    gender = normalize_gender(profile.gender)
    reading = build_saju_reading(profile.birth_datetime)
    time_branch_index = time_branch_index_from_datetime(profile.birth_datetime)
    time_branch_label = MANSE_TIME_BRANCH_LABELS[time_branch_index]
    daeun_direction = _daeun_direction(reading.chart.year.stem_index, gender)
    formatted_text = _format_manse_lookup(
        profile=profile,
        reading=reading,
        gender=gender,
        daeun_direction=daeun_direction,
    )
    result = ManseLookupResult(
        reading=reading,
        formatted_text=formatted_text,
        gender=gender,
        time_branch_label=time_branch_label,
        daeun_direction=daeun_direction,
    )
    return result


def normalize_gender(gender: str) -> str:
    cleaned = gender.strip()
    mapping = {
        "남": "남성",
        "남자": "남성",
        "남성": "남성",
        "male": "남성",
        "m": "남성",
        "여": "여성",
        "여자": "여성",
        "여성": "여성",
        "female": "여성",
        "f": "여성",
    }
    result = mapping.get(cleaned.lower(), cleaned)
    if result not in MANSE_GENDERS:
        raise ValueError("성별은 남성 또는 여성으로 입력해야 합니다.")
    return result


def time_branch_index_from_datetime(birth_datetime: datetime) -> int:
    result = _chart_time_branch_index_from_datetime(birth_datetime)
    return result


def time_branch_label_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = MANSE_TIME_BRANCH_LABELS[index]
    return result


def time_branch_display_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = f"{MANSE_TIME_BRANCH_LABELS[index]}({MANSE_TIME_BRANCH_HANJA[index]})"
    return result


def time_branch_range_display_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = (
        f"{time_branch_display_from_index(index)} "
        f"{MANSE_TIME_BRANCH_RANGES[index]}"
    )
    return result


def time_branch_display_from_datetime(birth_datetime: datetime) -> str:
    index = time_branch_index_from_datetime(birth_datetime)
    result = time_branch_display_from_index(index)
    return result


def birth_time_display_from_profile(profile: BirthProfile) -> str:
    result = time_branch_display_from_datetime(profile.birth_datetime)
    if not profile.birth_time_known:
        result = _UNKNOWN_BIRTH_TIME_DISPLAY
    return result


def birth_datetime_display_from_profile(profile: BirthProfile) -> str:
    birth_date = profile.birth_datetime.date().isoformat()
    birth_time = birth_time_display_from_profile(profile)
    result = f"{birth_date} {birth_time}"
    return result


def representative_time_from_time_branch(value: str) -> str | None:
    cleaned = _normalize_time_branch_token(value)
    result = None
    for index, label in enumerate(MANSE_TIME_BRANCH_LABELS):
        hanja = MANSE_TIME_BRANCH_HANJA[index]
        names = {
            _normalize_time_branch_token(label),
            _normalize_time_branch_token(hanja),
            _normalize_time_branch_token(f"{label}({hanja})"),
            _normalize_time_branch_token(f"{label} {hanja}"),
        }
        if cleaned in names:
            result = _REPRESENTATIVE_TIMES[index]
    return result


def _format_manse_lookup(
    profile: BirthProfile,
    reading: SajuReading,
    gender: str,
    daeun_direction: str,
) -> str:
    chart = reading.chart
    counts = ", ".join(
        f"{element} {reading.element_counts[element]}" for element in ELEMENTS
    )
    birth_date = profile.birth_datetime.date().isoformat()
    time_branch_label = birth_time_display_from_profile(profile)
    result = "\n".join(
        (
            "[만세력/사주명식]",
            f"- 기준일: {birth_date} {time_branch_label} ({gender})",
            f"- 년주: {chart.year.label}",
            f"- 월주: {chart.month.label}",
            f"- 일주: {chart.day.label}",
            f"- 시주: {chart.hour.label}",
            f"- 대운 방향: {daeun_direction}",
            "",
            "[오행 분포]",
            f"- {counts}",
            "",
            "[사주정보]",
            reading.interpretation,
        ),
    )
    return result


def _daeun_direction(year_stem_index: int, gender: str) -> str:
    year_polarity = "양" if year_stem_index % 2 == 0 else "음"
    result = "역행"
    if (gender == "남성" and year_polarity == "양") or (
        gender == "여성" and year_polarity == "음"
    ):
        result = "순행"
    return result


def _validate_time_branch_index(index: int) -> None:
    if not 0 <= index < len(MANSE_TIME_BRANCH_LABELS):
        raise ValueError("시간 지지 인덱스가 범위를 벗어났습니다.")


def _normalize_time_branch_token(value: str) -> str:
    result = (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
    )
    return result

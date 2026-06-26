from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

from oracle_report.saju.solar_terms import (
    saju_month_number_for_datetime,
    saju_year_for_datetime,
)
from oracle_report.saju.solar_terms_data import (
    SOLAR_TERM_DATA_MAX_YEAR,
    SOLAR_TERM_DATA_MIN_YEAR,
)


HEAVENLY_STEMS = ("갑", "을", "병", "정", "무", "기", "경", "신", "임", "계")
EARTHLY_BRANCHES = ("자", "축", "인", "묘", "진", "사", "오", "미", "신", "유", "술", "해")
STEM_ELEMENTS = ("목", "목", "화", "화", "토", "토", "금", "금", "수", "수")
STEM_POLARITIES = ("양", "음", "양", "음", "양", "음", "양", "음", "양", "음")
BRANCH_ELEMENTS = ("수", "토", "목", "목", "토", "화", "화", "토", "금", "금", "토", "수")
BRANCH_ANIMALS = ("쥐", "소", "호랑이", "토끼", "용", "뱀", "말", "양", "원숭이", "닭", "개", "돼지")
DAY_PILLAR_ANCHOR = date(1992, 10, 24)
DAY_PILLAR_ANCHOR_GANJI_INDEX = 9
TIME_BRANCH_SHIFT_MINUTES = 30


@dataclass(frozen=True)
class StemBranch:
    stem_index: int
    branch_index: int

    @property
    def stem(self) -> str:
        result = HEAVENLY_STEMS[self.stem_index]
        return result

    @property
    def branch(self) -> str:
        result = EARTHLY_BRANCHES[self.branch_index]
        return result

    @property
    def element(self) -> str:
        result = STEM_ELEMENTS[self.stem_index]
        return result

    @property
    def polarity(self) -> str:
        result = STEM_POLARITIES[self.stem_index]
        return result

    @property
    def label(self) -> str:
        result = f"{self.stem}{self.branch}"
        return result


@dataclass(frozen=True)
class SajuChart:
    birth_datetime: datetime
    year: StemBranch
    month: StemBranch
    day: StemBranch
    hour: StemBranch


def build_saju_chart(birth_datetime: datetime) -> SajuChart:
    _validate_supported_year(birth_datetime.year)
    saju_year = saju_year_for_datetime(birth_datetime)
    month_number = saju_month_number_for_datetime(birth_datetime)
    year = _year_pillar(saju_year)
    month = _month_pillar(month_number, year.stem_index)
    day = _day_pillar(birth_datetime.date())
    hour = _hour_pillar(birth_datetime, day.stem_index)
    result = SajuChart(
        birth_datetime=birth_datetime,
        year=year,
        month=month,
        day=day,
        hour=hour,
    )
    return result


def julian_day_number(day: date) -> int:
    result = day.toordinal() + 1721425
    return result


def time_branch_index_from_datetime(birth_datetime: datetime) -> int:
    total_minutes = birth_datetime.hour * 60 + birth_datetime.minute
    shifted_minutes = (total_minutes + TIME_BRANCH_SHIFT_MINUTES) % (24 * 60)
    result = shifted_minutes // 120
    return result


def _year_pillar(year: int) -> StemBranch:
    cycle_index = (year - 4) % 60
    result = _sexagenary_from_cycle_index(cycle_index)
    return result


def _month_pillar(month_number: int, year_stem_index: int) -> StemBranch:
    branch_index = (month_number + 1) % 12
    stem_index = ((year_stem_index % 5) * 2 + month_number + 1) % 10
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _day_pillar(day: date) -> StemBranch:
    days_diff = (day - DAY_PILLAR_ANCHOR).days
    ganji_index = (DAY_PILLAR_ANCHOR_GANJI_INDEX + days_diff) % 60
    stem_index = ganji_index % 10
    branch_index = ganji_index % 12
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _hour_pillar(birth_datetime: datetime, day_stem_index: int) -> StemBranch:
    branch_index = time_branch_index_from_datetime(birth_datetime)
    zi_hour_stem = (day_stem_index % 5) * 2
    stem_index = (zi_hour_stem + branch_index) % 10
    result = StemBranch(stem_index=stem_index, branch_index=branch_index)
    return result


def _sexagenary_from_cycle_index(cycle_index: int) -> StemBranch:
    result = StemBranch(stem_index=cycle_index % 10, branch_index=cycle_index % 12)
    return result


def _validate_supported_year(year: int) -> None:
    if not SOLAR_TERM_DATA_MIN_YEAR <= year <= SOLAR_TERM_DATA_MAX_YEAR:
        raise ValueError(
            "만세력 계산은 정밀 절입표가 있는 "
            f"{SOLAR_TERM_DATA_MIN_YEAR}~{SOLAR_TERM_DATA_MAX_YEAR}년만 지원합니다.",
        )

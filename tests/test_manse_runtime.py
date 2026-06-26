from __future__ import annotations

from datetime import datetime

import pytest

from oracle_report.saju.calendar import build_saju_chart


def _pillar_labels(birth_datetime: datetime) -> tuple[str, str, str, str]:
    chart = build_saju_chart(birth_datetime)
    result = (
        chart.year.label,
        chart.month.label,
        chart.day.label,
        chart.hour.label,
    )
    return result


def test_runtime_calculation_matches_manseryeok_golden_case() -> None:
    result = _pillar_labels(datetime(1992, 10, 24, 5, 30))

    assert result == ("임신", "경술", "계유", "을묘")


def test_runtime_calculation_uses_lichun_for_year_pillar() -> None:
    before = build_saju_chart(datetime(2024, 1, 14, 22, 30))
    after = build_saju_chart(datetime(2024, 2, 10, 12, 0))

    assert before.year.label == "계묘"
    assert after.year.label == "갑진"


def test_runtime_calculation_uses_precise_solar_month_boundary() -> None:
    result = _pillar_labels(datetime(1999, 10, 20, 10, 25))

    assert result == ("기묘", "갑술", "을사", "신사")


def test_runtime_calculation_uses_midnight_day_boundary_default() -> None:
    chart = build_saju_chart(datetime(2024, 3, 10, 23, 30))

    assert chart.day.label == "계유"
    assert chart.hour.label == "임자"


def test_runtime_hour_pillar_uses_korean_shifted_time_branch() -> None:
    before_zi = build_saju_chart(datetime(2024, 3, 10, 23, 29))
    start_zi = build_saju_chart(datetime(2024, 3, 10, 23, 30))
    end_zi = build_saju_chart(datetime(2024, 3, 10, 1, 29))
    start_chuk = build_saju_chart(datetime(2024, 3, 10, 1, 30))

    assert before_zi.hour.label == "계해"
    assert start_zi.hour.label == "임자"
    assert end_zi.hour.label == "임자"
    assert start_chuk.hour.label == "계축"


def test_runtime_calculation_limits_precise_solar_term_years() -> None:
    build_saju_chart(datetime(2300, 1, 1, 0, 0))

    with pytest.raises(ValueError):
        build_saju_chart(datetime(1799, 12, 31, 23, 59))

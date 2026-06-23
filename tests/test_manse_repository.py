from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from oracle_report.models import BirthProfile
from oracle_report.saju.repository import (
    MANSE_GENDERS,
    MANSE_TIME_BRANCH_LABELS,
    ManseDataNotFoundError,
    ManseRepository,
    build_manse_database,
    normalize_gender,
)


def test_builds_complete_manse_rows_for_year_range(tmp_path: Path) -> None:
    db_path = tmp_path / "manse.sqlite"

    built = build_manse_database(db_path, start_year=1995, end_year=1995)
    skipped = build_manse_database(db_path, start_year=1995, end_year=1995)

    assert built is True
    assert skipped is False
    assert db_path.exists()


def test_manse_lookup_returns_prebuilt_record(tmp_path: Path) -> None:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    repository = ManseRepository(db_path)
    profile = BirthProfile(
        name="홍길동",
        birth_datetime=datetime(1995, 3, 15, 14, 30),
        gender="남성",
    )

    result = repository.lookup(profile)

    assert "만세력/사주명식" in result.formatted_text
    assert result.gender in MANSE_GENDERS
    assert result.time_branch_label in MANSE_TIME_BRANCH_LABELS


def test_manse_lookup_does_not_calculate_missing_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "manse.sqlite"
    build_manse_database(db_path, start_year=1995, end_year=1995)
    repository = ManseRepository(db_path)
    profile = BirthProfile(
        name="미래",
        birth_datetime=datetime(1996, 1, 1, 0, 0),
        gender="여성",
    )

    with pytest.raises(ManseDataNotFoundError):
        repository.lookup(profile)


def test_normalizes_supported_gender_aliases() -> None:
    assert normalize_gender("남자") == "남성"
    assert normalize_gender("female") == "여성"

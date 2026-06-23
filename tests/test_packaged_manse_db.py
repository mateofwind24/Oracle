from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from oracle_report.models import BirthProfile
from oracle_report.saju.repository import MANSE_TIME_BRANCH_LABELS, ManseRepository


ROOT_DIR = Path(__file__).resolve().parents[1]
PACKAGED_DB_PATH = ROOT_DIR / "data" / "manse.sqlite"
START_YEAR = 1900
END_YEAR = 2100


def test_packaged_manse_db_covers_default_range() -> None:
    start = date(START_YEAR, 1, 1)
    end = date(END_YEAR, 12, 31)
    expected_rows = ((end - start).days + 1) * len(MANSE_TIME_BRANCH_LABELS)

    assert PACKAGED_DB_PATH.exists()
    with sqlite3.connect(PACKAGED_DB_PATH) as connection:
        metadata = dict(connection.execute("SELECT key, value FROM manse_metadata"))
        row_count = int(
            connection.execute("SELECT COUNT(*) FROM manse_entries").fetchone()[0],
        )
        boundary_count = int(
            connection.execute(
                """
                SELECT COUNT(*)
                FROM manse_entries
                WHERE (birth_ordinal = ? AND time_branch_index = 0)
                   OR (birth_ordinal = ? AND time_branch_index = 11)
                """,
                (start.toordinal(), end.toordinal()),
            ).fetchone()[0],
        )

    assert metadata["schema_version"] == "1"
    assert metadata["start_year"] == str(START_YEAR)
    assert metadata["end_year"] == str(END_YEAR)
    assert metadata["genders"] == "남성,여성"
    assert metadata["expected_rows"] == str(expected_rows)
    assert row_count == expected_rows
    assert boundary_count == 2


@pytest.mark.parametrize(
    ("birth_datetime", "gender"),
    (
        (datetime(1900, 1, 1, 0, 0), "남성"),
        (datetime(2000, 2, 29, 14, 30), "여성"),
        (datetime(2100, 12, 31, 23, 30), "남자"),
        (datetime(1995, 3, 15, 8, 20), "female"),
    ),
)
def test_packaged_manse_db_queries_birthdate_time_and_gender(
    birth_datetime: datetime,
    gender: str,
) -> None:
    repository = ManseRepository(PACKAGED_DB_PATH)
    profile = BirthProfile(
        name="sample",
        birth_datetime=birth_datetime,
        gender=gender,
    )

    result = repository.lookup(profile)

    assert result.daeun_direction in {"순행", "역행"}
    assert result.time_branch_label in MANSE_TIME_BRANCH_LABELS
    assert "사주명식" in result.formatted_text

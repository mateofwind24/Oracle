from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.saju.calendar import (
    BRANCH_ANIMALS,
    EARTHLY_BRANCHES,
    HEAVENLY_STEMS,
    STEM_ELEMENTS,
    SajuChart,
    StemBranch,
)
from oracle_report.saju.engine import ELEMENTS, SajuReading, build_saju_reading
from oracle_report.saju.rules import BALANCE_RULES, DAY_MASTER_RULES, DOMINANT_RULES


MANSE_SCHEMA_VERSION = "1"
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
_REPRESENTATIVE_HOURS = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22)
_GENDER_DAEUN_COLUMNS = {
    "남성": "daeun_direction_male",
    "여성": "daeun_direction_female",
}


class ManseDataNotFoundError(LookupError):
    pass


@dataclass(frozen=True)
class ManseLookupResult:
    reading: SajuReading
    formatted_text: str
    gender: str
    time_branch_label: str
    daeun_direction: str


class ManseRepository:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def lookup(self, profile: BirthProfile) -> ManseLookupResult:
        _ensure_existing_manse_db(self._db_path)
        gender = normalize_gender(profile.gender)
        time_branch_index = time_branch_index_from_datetime(profile.birth_datetime)
        birth_ordinal = profile.birth_datetime.date().toordinal()
        row = self._read_row(birth_ordinal, time_branch_index)
        if row is None:
            raise ManseDataNotFoundError(
                "만세력 DB에 해당 생년월일/태어난 시간/성별 데이터가 없습니다. "
                "scripts/build_manse_db.py 또는 ./build.sh로 DB를 먼저 생성하세요.",
            )
        result = _lookup_result_from_row(profile, row, gender)
        return result

    def _read_row(
        self,
        birth_ordinal: int,
        time_branch_index: int,
    ) -> sqlite3.Row | None:
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute(
                """
                SELECT *
                FROM manse_entries
                WHERE birth_ordinal = ?
                  AND time_branch_index = ?
                """,
                (birth_ordinal, time_branch_index),
            ).fetchone()
        result = row
        return result


def build_manse_database(
    db_path: Path,
    start_year: int,
    end_year: int,
    force: bool = False,
) -> bool:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    expected_rows = _expected_row_count(start_year, end_year)
    result = False
    is_ready = _database_is_ready(db_path, start_year, end_year, expected_rows)
    if force or not is_ready:
        with sqlite3.connect(db_path) as connection:
            _create_schema(connection)
            connection.execute("DELETE FROM manse_entries")
            connection.execute("DELETE FROM manse_metadata")
            _insert_manse_rows(connection, start_year, end_year)
            _write_metadata(connection, start_year, end_year, expected_rows)
            connection.commit()
        result = True
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
    result = ((birth_datetime.hour + 1) // 2) % 12
    return result


def _ensure_existing_manse_db(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"만세력 DB가 없습니다: {db_path}. ./build.sh로 먼저 생성하세요.",
        )
    with sqlite3.connect(db_path) as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'manse_entries'
            """,
        ).fetchone()
    if table is None:
        raise ManseDataNotFoundError(
            f"만세력 DB 스키마가 없습니다: {db_path}. ./build.sh로 다시 생성하세요.",
        )


def _lookup_result_from_row(
    profile: BirthProfile,
    row: sqlite3.Row,
    gender: str,
) -> ManseLookupResult:
    chart = _chart_from_row(profile.birth_datetime, row)
    element_counts = {
        "목": int(row["element_wood"]),
        "화": int(row["element_fire"]),
        "토": int(row["element_earth"]),
        "금": int(row["element_metal"]),
        "수": int(row["element_water"]),
    }
    reading = SajuReading(
        chart=chart,
        element_counts=element_counts,
        day_master=chart.day.stem,
        summary_lines=(
            _summary_line_1(chart),
            _summary_line_2(row),
            _summary_line_3(chart),
        ),
        interpretation=_interpretation_from_row(row, chart, element_counts),
    )
    daeun_direction = _daeun_direction_from_row(row, gender)
    formatted_text = _format_manse_lookup(row, reading, gender, daeun_direction)
    result = ManseLookupResult(
        reading=reading,
        formatted_text=formatted_text,
        gender=gender,
        time_branch_label=MANSE_TIME_BRANCH_LABELS[int(row["time_branch_index"])],
        daeun_direction=daeun_direction,
    )
    return result


def _chart_from_row(birth_datetime: datetime, row: sqlite3.Row) -> SajuChart:
    result = SajuChart(
        birth_datetime=birth_datetime,
        year=StemBranch(int(row["year_stem_index"]), int(row["year_branch_index"])),
        month=StemBranch(int(row["month_stem_index"]), int(row["month_branch_index"])),
        day=StemBranch(int(row["day_stem_index"]), int(row["day_branch_index"])),
        hour=StemBranch(int(row["hour_stem_index"]), int(row["hour_branch_index"])),
    )
    return result


def _format_manse_lookup(
    row: sqlite3.Row,
    reading: SajuReading,
    gender: str,
    daeun_direction: str,
) -> str:
    chart = reading.chart
    counts = ", ".join(
        f"{element} {reading.element_counts[element]}" for element in ELEMENTS
    )
    birth_date = date.fromordinal(int(row["birth_ordinal"])).isoformat()
    time_branch_label = MANSE_TIME_BRANCH_LABELS[int(row["time_branch_index"])]
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


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS manse_entries (
            birth_ordinal INTEGER NOT NULL,
            time_branch_index INTEGER NOT NULL,
            year_stem_index INTEGER NOT NULL,
            year_branch_index INTEGER NOT NULL,
            month_stem_index INTEGER NOT NULL,
            month_branch_index INTEGER NOT NULL,
            day_stem_index INTEGER NOT NULL,
            day_branch_index INTEGER NOT NULL,
            hour_stem_index INTEGER NOT NULL,
            hour_branch_index INTEGER NOT NULL,
            element_wood INTEGER NOT NULL,
            element_fire INTEGER NOT NULL,
            element_earth INTEGER NOT NULL,
            element_metal INTEGER NOT NULL,
            element_water INTEGER NOT NULL,
            strongest_element_index INTEGER NOT NULL,
            weakest_element_index INTEGER NOT NULL,
            daeun_direction_male TEXT NOT NULL,
            daeun_direction_female TEXT NOT NULL,
            PRIMARY KEY (birth_ordinal, time_branch_index)
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS manse_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """,
    )


def _insert_manse_rows(
    connection: sqlite3.Connection,
    start_year: int,
    end_year: int,
) -> None:
    rows: list[tuple[object, ...]] = []
    current = date(start_year, 1, 1)
    final = date(end_year, 12, 31)
    while current <= final:
        for time_branch_index, hour in enumerate(_REPRESENTATIVE_HOURS):
            birth_datetime = datetime(current.year, current.month, current.day, hour, 0)
            reading = build_saju_reading(birth_datetime)
            rows.append(
                _entry_tuple(
                    current,
                    time_branch_index,
                    reading,
                )
            )
        if len(rows) >= 5000:
            _flush_rows(connection, rows)
            rows.clear()
        current = current + timedelta(days=1)
    if rows:
        _flush_rows(connection, rows)


def _entry_tuple(
    birth_date: date,
    time_branch_index: int,
    reading: SajuReading,
) -> tuple[object, ...]:
    chart = reading.chart
    strongest = _strongest_element(reading.element_counts)
    weakest = _weakest_element(reading.element_counts)
    result = (
        birth_date.toordinal(),
        time_branch_index,
        chart.year.stem_index,
        chart.year.branch_index,
        chart.month.stem_index,
        chart.month.branch_index,
        chart.day.stem_index,
        chart.day.branch_index,
        chart.hour.stem_index,
        chart.hour.branch_index,
        reading.element_counts["목"],
        reading.element_counts["화"],
        reading.element_counts["토"],
        reading.element_counts["금"],
        reading.element_counts["수"],
        ELEMENTS.index(strongest),
        ELEMENTS.index(weakest),
        _daeun_direction(chart.year.stem_index, "남성"),
        _daeun_direction(chart.year.stem_index, "여성"),
    )
    return result


def _flush_rows(connection: sqlite3.Connection, rows: list[tuple[object, ...]]) -> None:
    connection.executemany(
        """
        INSERT INTO manse_entries (
            birth_ordinal,
            time_branch_index,
            year_stem_index,
            year_branch_index,
            month_stem_index,
            month_branch_index,
            day_stem_index,
            day_branch_index,
            hour_stem_index,
            hour_branch_index,
            element_wood,
            element_fire,
            element_earth,
            element_metal,
            element_water,
            strongest_element_index,
            weakest_element_index,
            daeun_direction_male,
            daeun_direction_female
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def _write_metadata(
    connection: sqlite3.Connection,
    start_year: int,
    end_year: int,
    expected_rows: int,
) -> None:
    metadata = (
        ("schema_version", MANSE_SCHEMA_VERSION),
        ("start_year", str(start_year)),
        ("end_year", str(end_year)),
        ("genders", ",".join(MANSE_GENDERS)),
        ("time_branches", str(len(MANSE_TIME_BRANCH_LABELS))),
        ("expected_rows", str(expected_rows)),
    )
    connection.executemany(
        "INSERT OR REPLACE INTO manse_metadata (key, value) VALUES (?, ?)",
        metadata,
    )


def _database_is_ready(
    db_path: Path,
    start_year: int,
    end_year: int,
    expected_rows: int,
) -> bool:
    result = False
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                metadata = dict(
                    connection.execute("SELECT key, value FROM manse_metadata"),
                )
                row_count = int(
                    connection.execute("SELECT COUNT(*) FROM manse_entries").fetchone()[0],
                )
            result = (
                metadata.get("schema_version") == MANSE_SCHEMA_VERSION
                and metadata.get("start_year") == str(start_year)
                and metadata.get("end_year") == str(end_year)
                and metadata.get("expected_rows") == str(expected_rows)
                and row_count == expected_rows
            )
        except sqlite3.Error:
            result = False
    return result


def _expected_row_count(start_year: int, end_year: int) -> int:
    start = date(start_year, 1, 1)
    end = date(end_year, 12, 31)
    day_count = (end - start).days + 1
    result = day_count * len(MANSE_TIME_BRANCH_LABELS)
    return result


def _strongest_element(counts: dict[str, int]) -> str:
    result = max(
        ELEMENTS,
        key=lambda element: (counts[element], -ELEMENTS.index(element)),
    )
    return result


def _weakest_element(counts: dict[str, int]) -> str:
    result = min(ELEMENTS, key=lambda element: (counts[element], ELEMENTS.index(element)))
    return result


def _daeun_direction(year_stem_index: int, gender: str) -> str:
    year_polarity = "양" if year_stem_index % 2 == 0 else "음"
    result = "역행"
    if (gender == "남성" and year_polarity == "양") or (
        gender == "여성" and year_polarity == "음"
    ):
        result = "순행"
    return result


def _daeun_direction_from_row(row: sqlite3.Row, gender: str) -> str:
    column = _GENDER_DAEUN_COLUMNS[gender]
    result = str(row[column])
    return result


def _summary_line_1(chart: SajuChart) -> str:
    result = (
        f"일간은 {chart.day.stem}{STEM_ELEMENTS[chart.day.stem_index]}"
        f"({chart.day.polarity})입니다."
    )
    return result


def _summary_line_2(row: sqlite3.Row) -> str:
    strongest = ELEMENTS[int(row["strongest_element_index"])]
    weakest = ELEMENTS[int(row["weakest_element_index"])]
    result = f"가장 강한 오행은 {strongest}, 보완하면 좋은 오행은 {weakest}입니다."
    return result


def _summary_line_3(chart: SajuChart) -> str:
    result = f"연지는 {BRANCH_ANIMALS[chart.year.branch_index]}의 흐름으로 사회적 첫인상을 참고합니다."
    return result


def _interpretation_from_row(
    row: sqlite3.Row,
    chart: SajuChart,
    element_counts: dict[str, int],
) -> str:
    strongest = ELEMENTS[int(row["strongest_element_index"])]
    weakest = ELEMENTS[int(row["weakest_element_index"])]
    month_branch = EARTHLY_BRANCHES[chart.month.branch_index]
    month_stem = HEAVENLY_STEMS[chart.month.stem_index]
    count_text = ", ".join(f"{key}:{value}" for key, value in element_counts.items())
    result = "\n".join(
        (
            DAY_MASTER_RULES[chart.day.stem],
            f"월주는 {month_stem}{month_branch}라서 계절감과 사회적 리듬을 함께 봅니다.",
            f"오행 카운트는 {count_text}입니다.",
            DOMINANT_RULES[strongest],
            BALANCE_RULES[weakest],
        ),
    )
    return result

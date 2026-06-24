from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from oracle_report.vision.physiognomy_rule_data import (
    PHYSIOGNOMY_RULES,
    PHYSIOGNOMY_SAFETY_NOTE,
    RULE_SOURCE_REFERENCES,
    UNSUPPORTED_PHYSIOGNOMY_FEATURES,
)


PHYSIOGNOMY_RULE_SCHEMA_VERSION = "1"
DEFAULT_PHYSIOGNOMY_RULE_DB_PATH = Path("data/physiognomy_rules.sqlite")


@dataclass(frozen=True)
class PhysiognomyRuleMatch:
    rule_id: str
    metric: str
    title: str
    basis: str
    tag: str
    observation: str
    interpretation: str
    value: float


class PhysiognomyRuleDatabaseError(LookupError):
    pass


class PhysiognomyRuleRepository:
    def __init__(self, db_path: Path = DEFAULT_PHYSIOGNOMY_RULE_DB_PATH) -> None:
        self._db_path = db_path

    def lookup(self, metric: str, value: float) -> PhysiognomyRuleMatch | None:
        _ensure_existing_rule_db(self._db_path)
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            result = _lookup_rule_match(connection, metric, value)
        return result

    def lookup_many(
        self,
        metric_values: Mapping[str, float],
    ) -> tuple[PhysiognomyRuleMatch, ...]:
        _ensure_existing_rule_db(self._db_path)
        matches: list[PhysiognomyRuleMatch] = []
        with sqlite3.connect(self._db_path) as connection:
            connection.row_factory = sqlite3.Row
            for metric, value in metric_values.items():
                match = _lookup_rule_match(connection, metric, value)
                if match is not None:
                    matches.append(match)
        result = tuple(matches)
        return result

    def unsupported_features(self) -> tuple[str, ...]:
        _ensure_existing_rule_db(self._db_path)
        with sqlite3.connect(self._db_path) as connection:
            rows = connection.execute(
                """
                SELECT name
                FROM physiognomy_unsupported_features
                ORDER BY display_order
                """,
            ).fetchall()
        result = tuple(str(row[0]) for row in rows)
        return result

    def safety_note(self) -> str:
        _ensure_existing_rule_db(self._db_path)
        with sqlite3.connect(self._db_path) as connection:
            row = connection.execute(
                """
                SELECT value
                FROM physiognomy_metadata
                WHERE key = 'safety_note'
                """,
            ).fetchone()
        result = PHYSIOGNOMY_SAFETY_NOTE if row is None else str(row[0])
        return result


def build_physio_rule_database(db_path: Path, force: bool = False) -> bool:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    result = False
    is_ready = _database_is_ready(db_path)
    if force or not is_ready:
        with sqlite3.connect(db_path) as connection:
            _create_schema(connection)
            _clear_tables(connection)
            _insert_sources(connection)
            _insert_rules(connection)
            _insert_unsupported_features(connection)
            _write_metadata(connection)
            connection.commit()
        result = True
    return result


def _lookup_rule_match(
    connection: sqlite3.Connection,
    metric: str,
    value: float,
) -> PhysiognomyRuleMatch | None:
    row = connection.execute(
        """
        SELECT
            rules.id AS rule_id,
            rules.metric AS metric,
            rules.title AS title,
            rules.basis AS basis,
            ranges.tag AS tag,
            ranges.observation AS observation,
            ranges.interpretation AS interpretation
        FROM physiognomy_rules AS rules
        JOIN physiognomy_rule_ranges AS ranges
          ON ranges.rule_id = rules.id
        WHERE rules.metric = ?
          AND (ranges.min_value IS NULL OR ? >= ranges.min_value)
          AND (ranges.max_value IS NULL OR ? < ranges.max_value)
        ORDER BY rules.display_order, ranges.display_order
        LIMIT 1
        """,
        (metric, value, value),
    ).fetchone()
    result = None
    if row is not None:
        result = PhysiognomyRuleMatch(
            rule_id=str(row["rule_id"]),
            metric=str(row["metric"]),
            title=str(row["title"]),
            basis=str(row["basis"]),
            tag=str(row["tag"]),
            observation=str(row["observation"]),
            interpretation=str(row["interpretation"]),
            value=value,
        )
    return result


def _ensure_existing_rule_db(db_path: Path) -> None:
    if not db_path.exists():
        raise FileNotFoundError(
            f"관상 룰 DB가 없습니다: {db_path}. oracle-build-physiognomy-db로 생성하세요.",
        )
    with sqlite3.connect(db_path) as connection:
        table = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = 'physiognomy_rules'
            """,
        ).fetchone()
    if table is None:
        raise PhysiognomyRuleDatabaseError(
            f"관상 룰 DB 스키마가 없습니다: {db_path}. DB를 다시 생성하세요.",
        )


def _database_is_ready(db_path: Path) -> bool:
    result = False
    if db_path.exists():
        try:
            with sqlite3.connect(db_path) as connection:
                metadata = dict(
                    connection.execute(
                        "SELECT key, value FROM physiognomy_metadata",
                    ),
                )
                rule_count = connection.execute(
                    "SELECT COUNT(*) FROM physiognomy_rules",
                ).fetchone()[0]
                range_count = connection.execute(
                    "SELECT COUNT(*) FROM physiognomy_rule_ranges",
                ).fetchone()[0]
            expected_range_count = sum(len(rule.ranges) for rule in PHYSIOGNOMY_RULES)
            result = (
                metadata.get("schema_version") == PHYSIOGNOMY_RULE_SCHEMA_VERSION
                and int(rule_count) == len(PHYSIOGNOMY_RULES)
                and int(range_count) == expected_range_count
            )
        except sqlite3.Error:
            result = False
    return result


def _create_schema(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_sources (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            note TEXT NOT NULL
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_rules (
            id TEXT PRIMARY KEY,
            metric TEXT NOT NULL UNIQUE,
            title TEXT NOT NULL,
            basis TEXT NOT NULL,
            display_order INTEGER NOT NULL
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_rule_ranges (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            rule_id TEXT NOT NULL,
            min_value REAL,
            max_value REAL,
            tag TEXT NOT NULL,
            observation TEXT NOT NULL,
            interpretation TEXT NOT NULL,
            display_order INTEGER NOT NULL,
            FOREIGN KEY (rule_id) REFERENCES physiognomy_rules(id)
        )
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_rule_sources (
            rule_id TEXT NOT NULL,
            source_id TEXT NOT NULL,
            PRIMARY KEY (rule_id, source_id),
            FOREIGN KEY (rule_id) REFERENCES physiognomy_rules(id),
            FOREIGN KEY (source_id) REFERENCES physiognomy_sources(id)
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_unsupported_features (
            name TEXT PRIMARY KEY,
            display_order INTEGER NOT NULL
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS physiognomy_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        ) WITHOUT ROWID
        """,
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_physio_rule_ranges_lookup
        ON physiognomy_rule_ranges(rule_id, min_value, max_value)
        """,
    )
    connection.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_physio_rules_metric
        ON physiognomy_rules(metric)
        """,
    )


def _clear_tables(connection: sqlite3.Connection) -> None:
    connection.execute("DELETE FROM physiognomy_rule_sources")
    connection.execute("DELETE FROM physiognomy_rule_ranges")
    connection.execute("DELETE FROM physiognomy_rules")
    connection.execute("DELETE FROM physiognomy_sources")
    connection.execute("DELETE FROM physiognomy_unsupported_features")
    connection.execute("DELETE FROM physiognomy_metadata")


def _insert_sources(connection: sqlite3.Connection) -> None:
    rows = tuple(
        (source.id, source.title, source.url, source.note)
        for source in RULE_SOURCE_REFERENCES
    )
    connection.executemany(
        """
        INSERT INTO physiognomy_sources (id, title, url, note)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def _insert_rules(connection: sqlite3.Connection) -> None:
    rule_rows = []
    range_rows = []
    source_rows = []
    for rule_order, rule in enumerate(PHYSIOGNOMY_RULES):
        rule_rows.append((rule.id, rule.metric, rule.title, rule.basis, rule_order))
        for range_order, rule_range in enumerate(rule.ranges):
            range_rows.append(
                (
                    rule.id,
                    rule_range.min_value,
                    rule_range.max_value,
                    rule_range.tag,
                    rule_range.observation,
                    rule_range.interpretation,
                    range_order,
                ),
            )
        for source_id in rule.source_ids:
            source_rows.append((rule.id, source_id))
    connection.executemany(
        """
        INSERT INTO physiognomy_rules (id, metric, title, basis, display_order)
        VALUES (?, ?, ?, ?, ?)
        """,
        rule_rows,
    )
    connection.executemany(
        """
        INSERT INTO physiognomy_rule_ranges (
            rule_id,
            min_value,
            max_value,
            tag,
            observation,
            interpretation,
            display_order
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        range_rows,
    )
    connection.executemany(
        """
        INSERT INTO physiognomy_rule_sources (rule_id, source_id)
        VALUES (?, ?)
        """,
        source_rows,
    )


def _insert_unsupported_features(connection: sqlite3.Connection) -> None:
    rows = tuple(
        (feature, display_order)
        for display_order, feature in enumerate(UNSUPPORTED_PHYSIOGNOMY_FEATURES)
    )
    connection.executemany(
        """
        INSERT INTO physiognomy_unsupported_features (name, display_order)
        VALUES (?, ?)
        """,
        rows,
    )


def _write_metadata(connection: sqlite3.Connection) -> None:
    range_count = sum(len(rule.ranges) for rule in PHYSIOGNOMY_RULES)
    rows = (
        ("schema_version", PHYSIOGNOMY_RULE_SCHEMA_VERSION),
        ("rule_count", str(len(PHYSIOGNOMY_RULES))),
        ("range_count", str(range_count)),
        ("source_count", str(len(RULE_SOURCE_REFERENCES))),
        ("safety_note", PHYSIOGNOMY_SAFETY_NOTE),
    )
    connection.executemany(
        """
        INSERT OR REPLACE INTO physiognomy_metadata (key, value)
        VALUES (?, ?)
        """,
        rows,
    )

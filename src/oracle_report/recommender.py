from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from oracle_report.saju.engine import ELEMENTS, SajuReading
from oracle_report.saju.repository import normalize_gender


@dataclass(frozen=True)
class FaceRecommendation:
    display_name: str
    image_path: Path | None
    target_gender: str
    face_tags: tuple[str, ...]
    saju_tags: tuple[str, ...]
    reason: str
    score: int


def recommend_faces(
    db_path: Path,
    target_gender: str,
    reading: SajuReading,
    limit: int = 3,
) -> tuple[FaceRecommendation, ...]:
    _ensure_face_db(db_path)
    weak_element = _weakest_element(reading)
    target = _normalize_target_gender(target_gender)
    rows = _read_candidates(db_path)
    scored = [
        _score_candidate(row, target, weak_element)
        for row in rows
        if _candidate_is_visible(row)
    ]
    filtered = [item for item in scored if item.score > 0]
    filtered.sort(key=lambda item: item.score, reverse=True)
    result = tuple(filtered[:limit])
    return result


def _normalize_target_gender(target_gender: str) -> str:
    cleaned = target_gender.strip()
    result = ""
    if cleaned != "":
        result = normalize_gender(cleaned)
    return result


def format_recommendations(recommendations: tuple[FaceRecommendation, ...]) -> str:
    lines: list[str] = []
    if recommendations:
        for index, item in enumerate(recommendations, start=1):
            tags = ", ".join(item.face_tags)
            saju_tags = ", ".join(item.saju_tags)
            lines.append(
                f"{index}. {item.display_name}: 얼굴상({tags}), "
                f"사주 보완({saju_tags}) - {item.reason}",
            )
    else:
        lines.append("추천 DB에 맞는 후보가 없어 추천 섹션은 안내 문구로 대체합니다.")
    result = "\n".join(lines)
    return result


def _ensure_face_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS face_recommendations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                display_name TEXT NOT NULL,
                image_path TEXT NOT NULL,
                target_gender TEXT NOT NULL,
                face_tags TEXT NOT NULL,
                saju_tags TEXT NOT NULL,
                reason TEXT NOT NULL,
                visible INTEGER NOT NULL DEFAULT 1
            )
            """,
        )
        count = connection.execute(
            "SELECT COUNT(*) FROM face_recommendations",
        ).fetchone()[0]
        if int(count) == 0:
            connection.executemany(
                """
                INSERT INTO face_recommendations (
                    display_name,
                    image_path,
                    target_gender,
                    face_tags,
                    saju_tags,
                    reason,
                    visible
                ) VALUES (?, ?, ?, ?, ?, ?, 1)
                """,
                _sample_rows(),
            )
        connection.commit()


def _sample_rows() -> tuple[tuple[str, str, str, str, str, str], ...]:
    result = (
        (
            "부드러운 균형형",
            "",
            "여성",
            "부드러운 인상,둥근 윤곽,차분한 표정",
            "토,수",
            "안정감과 회복 리듬을 보완하는 얼굴상입니다.",
        ),
        (
            "선명한 추진형",
            "",
            "남성",
            "선명한 눈매,곧은 윤곽,밝은 표정",
            "화,목",
            "표현력과 시작 에너지를 보완하는 얼굴상입니다.",
        ),
        (
            "정돈된 신뢰형",
            "",
            "여성",
            "정돈된 눈썹,단정한 윤곽,안정된 시선",
            "금,토",
            "판단 기준과 현실감을 보완하는 얼굴상입니다.",
        ),
        (
            "유연한 대화형",
            "",
            "남성",
            "편안한 눈매,부드러운 입꼬리,열린 표정",
            "수,목",
            "소통과 유연성을 보완하는 얼굴상입니다.",
        ),
    )
    return result


def _read_candidates(db_path: Path) -> list[sqlite3.Row]:
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = list(
            connection.execute(
                """
                SELECT
                    display_name,
                    image_path,
                    target_gender,
                    face_tags,
                    saju_tags,
                    reason,
                    visible
                FROM face_recommendations
                """,
            ),
        )
    result = rows
    return result


def _score_candidate(
    row: sqlite3.Row,
    target_gender: str,
    weak_element: str,
) -> FaceRecommendation:
    score = 1
    row_gender = str(row["target_gender"])
    saju_tags = _split_tags(str(row["saju_tags"]))
    face_tags = _split_tags(str(row["face_tags"]))
    if target_gender != "" and row_gender == target_gender:
        score = score + 3
    if weak_element in saju_tags:
        score = score + 5
    image_text = str(row["image_path"]).strip()
    image_path: Path | None = None
    if image_text != "":
        image_path = Path(image_text)
    result = FaceRecommendation(
        display_name=str(row["display_name"]),
        image_path=image_path,
        target_gender=row_gender,
        face_tags=face_tags,
        saju_tags=saju_tags,
        reason=str(row["reason"]),
        score=score,
    )
    return result


def _candidate_is_visible(row: sqlite3.Row) -> bool:
    result = int(row["visible"]) == 1
    return result


def _split_tags(raw_value: str) -> tuple[str, ...]:
    result = tuple(item.strip() for item in raw_value.split(",") if item.strip())
    return result


def _weakest_element(reading: SajuReading) -> str:
    result = min(
        ELEMENTS,
        key=lambda element: (
            reading.element_counts[element],
            ELEMENTS.index(element),
        ),
    )
    return result

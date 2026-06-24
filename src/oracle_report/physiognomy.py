from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_report.models import FaceQuality


@dataclass(frozen=True)
class FaceReadingInput:
    image_path: Path | None
    quality: FaceQuality | None


def format_face_quality(quality: FaceQuality | None) -> str:
    result = "- 품질 정보 없음"
    if quality is not None:
        warnings = ", ".join(quality.warnings) if quality.warnings else "경고 없음"
        result = (
            f"- 눈 개수: {quality.eye_count}, "
            f"눈썹 점수: {quality.eyebrow_score:.3f}, "
            f"정면 점수: {quality.frontality_score:.2f}, "
            f"랜드마크 배치 점수: {quality.occlusion_score:.2f}, "
            f"경고: {warnings}"
        )
    return result

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oracle_report.models import FaceQuality


@dataclass(frozen=True)
class FaceReadingInput:
    image_path: Path | None
    quality: FaceQuality | None


def build_face_prompt(face_input: FaceReadingInput) -> str:
    quality_text = format_face_quality(face_input.quality)
    result = f"""
사진을 관상 리포트의 보조 입력으로만 사용해 주세요.

[사진 사용 원칙]
- 엔터테인먼트성 관상 풀이로 작성합니다.
- 신원, 나이, 성별, 민족, 건강상태, 직업, 경제력은 추정하지 않습니다.
- 외모 평가나 매력 점수처럼 사람을 서열화하는 표현은 쓰지 않습니다.
- 눈, 눈썹, 얼굴 윤곽, 표정처럼 사진에서 보이는 요소를 단정이 아닌 경향으로만 말합니다.
- 사주 룰 해석과 충돌하면 사주 룰 해석을 우선하고 사진은 보조 설명으로만 씁니다.

[캡처 품질]
{quality_text}
""".strip()
    return result


def format_face_quality(quality: FaceQuality | None) -> str:
    result = "- 품질 정보 없음"
    if quality is not None:
        warnings = ", ".join(quality.warnings) if quality.warnings else "경고 없음"
        result = (
            f"- 눈 후보: {quality.eye_count}, "
            f"눈썹 점수: {quality.eyebrow_score:.3f}, "
            f"정면 점수: {quality.frontality_score:.2f}, "
            f"가림 추정 점수: {quality.occlusion_score:.2f}, "
            f"경고: {warnings}"
        )
    return result

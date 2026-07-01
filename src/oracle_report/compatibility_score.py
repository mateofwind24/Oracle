from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from oracle_report.saju.calendar import SajuChart
from oracle_report.saju.engine import ELEMENTS
from oracle_report.saju.repository import ManseLookupResult
from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch


_FACE_SIMILARITY_TOLERANCES: Mapping[str, float] = {
    "third_balance_error": 0.08,
    "upper_zone_ratio": 0.08,
    "middle_zone_ratio": 0.08,
    "lower_zone_ratio": 0.08,
    "face_aspect_ratio": 0.25,
    "eye_width_ratio": 0.07,
    "eye_aspect_ratio": 0.07,
    "eye_spacing_ratio": 0.08,
    "nose_width_ratio": 0.06,
    "nose_length_width_ratio": 0.45,
    "mouth_width_ratio": 0.08,
    "jaw_width_ratio": 0.16,
}
_INTENSITY_RANGES: Mapping[str, tuple[float, float]] = {
    "eye_width_ratio": (0.17, 0.25),
    "mouth_width_ratio": (0.30, 0.48),
    "nose_width_ratio": (0.14, 0.24),
    "jaw_width_ratio": (0.62, 0.82),
}
_STEM_COMBINATIONS = frozenset(
    {
        frozenset((0, 5)),
        frozenset((1, 6)),
        frozenset((2, 7)),
        frozenset((3, 8)),
        frozenset((4, 9)),
    },
)
_BRANCH_HARMONIES = frozenset(
    {
        frozenset((0, 1)),
        frozenset((2, 11)),
        frozenset((3, 10)),
        frozenset((4, 9)),
        frozenset((5, 8)),
        frozenset((6, 7)),
    },
)
_BRANCH_CLASHES = frozenset(
    {
        frozenset((0, 6)),
        frozenset((1, 7)),
        frozenset((2, 8)),
        frozenset((3, 9)),
        frozenset((4, 10)),
        frozenset((5, 11)),
    },
)
_BRANCH_TRINES = (
    frozenset((8, 0, 4)),
    frozenset((11, 3, 7)),
    frozenset((2, 6, 10)),
    frozenset((5, 9, 1)),
)
_MONTH_TEMPERATURES: Mapping[int, float] = {
    0: -1.0,
    1: -0.7,
    2: -0.2,
    3: 0.1,
    4: 0.3,
    5: 0.8,
    6: 1.0,
    7: 0.7,
    8: 0.2,
    9: -0.1,
    10: -0.3,
    11: -1.0,
}


@dataclass(frozen=True)
class _SajuScoreParts:
    score: int
    element_complement: float
    temperature_complement: float
    clash_penalty: int
    relation_bonus: int


@dataclass(frozen=True)
class _FaceScoreParts:
    score: int
    similarity: float
    complement: float
    stability: float


@dataclass(frozen=True)
class CompatibilityScore:
    total: int
    saju: int
    face: int
    mode_bonus: int
    label: str
    summary: str


def build_compatibility_score_payload(
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    left_matches: Sequence[PhysiognomyRuleMatch],
    right_matches: Sequence[PhysiognomyRuleMatch],
) -> dict[str, object]:
    score = build_compatibility_score(
        mode,
        left_manse,
        right_manse,
        left_matches,
        right_matches,
    )
    result: dict[str, object] = {
        "compatibility_score": score.total,
        "compatibility_score_label": score.label,
        "compatibility_score_summary": score.summary,
        "compatibility_saju_score": score.saju,
        "compatibility_face_score": score.face,
        "compatibility_mode_bonus": score.mode_bonus,
    }
    return result


def build_compatibility_score(
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    left_matches: Sequence[PhysiognomyRuleMatch],
    right_matches: Sequence[PhysiognomyRuleMatch],
) -> CompatibilityScore:
    saju_parts = _score_saju(left_manse, right_manse)
    face_parts = _score_face(left_matches, right_matches)
    mode_bonus = _score_mode_bonus(mode, saju_parts, face_parts)
    fingerprint = _compatibility_fingerprint(
        mode,
        left_manse,
        right_manse,
        left_matches,
        right_matches,
    )
    texture_bonus = _score_texture(fingerprint)
    total = _clamp_int(
        round(
            (saju_parts.score * 0.58)
            + (face_parts.score * 0.42)
            + mode_bonus
            + texture_bonus,
        ),
        66,
        99,
    )
    label = _score_label(total)
    summary = _mode_summary(mode, total, fingerprint)
    result = CompatibilityScore(
        total=total,
        saju=saju_parts.score,
        face=face_parts.score,
        mode_bonus=mode_bonus,
        label=label,
        summary=summary,
    )
    return result


def _score_saju(
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
) -> _SajuScoreParts:
    left_counts = left_manse.reading.element_counts
    right_counts = right_manse.reading.element_counts
    element_complement = _element_complement(left_counts, right_counts)
    relation_bonus, clash_penalty = _relationship_score(
        left_manse.reading.chart,
        right_manse.reading.chart,
    )
    temperature_complement = _temperature_complement(
        left_manse.reading.chart,
        right_manse.reading.chart,
    )
    over_penalty = _overconcentration_penalty(left_counts, right_counts)
    score = _clamp_int(
        round(
            58
            + (element_complement * 28)
            + (relation_bonus * 1.1)
            + (temperature_complement * 13)
            - (clash_penalty * 1.0)
            - (over_penalty * 1.4),
        ),
        60,
        99,
    )
    result = _SajuScoreParts(
        score=score,
        element_complement=element_complement,
        temperature_complement=temperature_complement,
        clash_penalty=clash_penalty,
        relation_bonus=relation_bonus,
    )
    return result


def _score_face(
    left_matches: Sequence[PhysiognomyRuleMatch],
    right_matches: Sequence[PhysiognomyRuleMatch],
) -> _FaceScoreParts:
    left_values = _metric_values(left_matches)
    right_values = _metric_values(right_matches)
    similarity = _face_similarity(left_values, right_values)
    complement = _face_complement(left_values, right_values)
    stability = _face_stability(left_values, right_values)
    feature_contrast = _face_feature_contrast(left_values, right_values)
    score = _clamp_int(
        round(
            55
            + (similarity * 19)
            + (complement * 17)
            + (stability * 12)
            + (feature_contrast * 5),
        ),
        55,
        98,
    )
    result = _FaceScoreParts(
        score=score,
        similarity=similarity,
        complement=complement,
        stability=stability,
    )
    return result


def _element_complement(
    left_counts: Mapping[str, int],
    right_counts: Mapping[str, int],
) -> float:
    deficit_scores = []
    contrast_scores = []
    for element in ELEMENTS:
        left_count = left_counts.get(element, 0)
        right_count = right_counts.get(element, 0)
        left_deficit = max(0.0, 2.0 - left_count) / 2.0
        right_deficit = max(0.0, 2.0 - right_count) / 2.0
        left_supply = min(left_count / 3.0, 1.0)
        right_supply = min(right_count / 3.0, 1.0)
        deficit_scores.append(
            ((left_deficit * right_supply) + (right_deficit * left_supply)) / 2.0,
        )
        contrast_scores.append(min(abs(left_count - right_count) / 4.0, 1.0))
    deficit_fill = _average(deficit_scores, 0.42)
    contrast = _average(contrast_scores, 0.35)
    result = _clamp((deficit_fill * 0.65) + (contrast * 0.35), 0.0, 1.0)
    return result


def _relationship_score(left_chart: SajuChart, right_chart: SajuChart) -> tuple[int, int]:
    left_pillars = _pillars(left_chart)
    right_pillars = _pillars(right_chart)
    stem_bonus = 0
    branch_bonus = 0
    clash_penalty = 0
    for left in left_pillars:
        for right in right_pillars:
            stem_pair = frozenset((left.stem_index, right.stem_index))
            branch_pair = frozenset((left.branch_index, right.branch_index))
            if stem_pair in _STEM_COMBINATIONS:
                stem_bonus += 2
            if branch_pair in _BRANCH_HARMONIES:
                branch_bonus += 3
            if branch_pair in _BRANCH_CLASHES:
                clash_penalty += 2
    combined_branches = {
        pillar.branch_index
        for pillar in (*left_pillars, *right_pillars)
    }
    trine_bonus = 0
    for trine in _BRANCH_TRINES:
        if trine.issubset(combined_branches):
            trine_bonus += 5
    relation_bonus = min(stem_bonus, 6) + min(branch_bonus, 9) + min(trine_bonus, 5)
    result = (relation_bonus, min(clash_penalty, 8))
    return result


def _temperature_complement(left_chart: SajuChart, right_chart: SajuChart) -> float:
    left_temperature = _MONTH_TEMPERATURES.get(left_chart.month.branch_index, 0.0)
    right_temperature = _MONTH_TEMPERATURES.get(right_chart.month.branch_index, 0.0)
    result = _clamp(abs(left_temperature - right_temperature) / 1.8, 0.0, 1.0)
    return result


def _overconcentration_penalty(
    left_counts: Mapping[str, int],
    right_counts: Mapping[str, int],
) -> int:
    overlap_count = sum(
        1
        for element in ELEMENTS
        if left_counts.get(element, 0) >= 3 and right_counts.get(element, 0) >= 3
    )
    result = min(overlap_count, 4)
    return result


def _metric_values(matches: Sequence[PhysiognomyRuleMatch]) -> dict[str, float]:
    result = {match.metric: match.value for match in matches}
    return result


def _face_similarity(
    left_values: Mapping[str, float],
    right_values: Mapping[str, float],
) -> float:
    scores = []
    for metric, tolerance in _FACE_SIMILARITY_TOLERANCES.items():
        if metric not in left_values or metric not in right_values:
            continue
        difference = abs(left_values[metric] - right_values[metric])
        scores.append(1.0 - min(difference / tolerance, 1.0))
    result = _average(scores, 0.55)
    return result


def _face_complement(
    left_values: Mapping[str, float],
    right_values: Mapping[str, float],
) -> float:
    left_intensity = _face_intensity(left_values)
    right_intensity = _face_intensity(right_values)
    if left_intensity is None or right_intensity is None:
        result = 0.50
    else:
        difference = abs(left_intensity - right_intensity)
        result = 1.0 - min(abs(difference - 0.24) / 0.24, 1.0)
    return result


def _face_intensity(values: Mapping[str, float]) -> float | None:
    normalized_values = []
    for metric, (low, high) in _INTENSITY_RANGES.items():
        if metric not in values:
            continue
        normalized_values.append(_clamp((values[metric] - low) / (high - low), 0.0, 1.0))
    result = None if not normalized_values else _average(normalized_values, 0.5)
    return result


def _face_stability(
    left_values: Mapping[str, float],
    right_values: Mapping[str, float],
) -> float:
    scores = []
    for values in (left_values, right_values):
        if "third_balance_error" in values:
            scores.append(1.0 - min(abs(values["third_balance_error"]) / 0.12, 1.0))
        if "mouth_balance_delta" in values:
            scores.append(1.0 - min(abs(values["mouth_balance_delta"]) / 0.08, 1.0))
    result = _average(scores, 0.60)
    return result


def _face_feature_contrast(
    left_values: Mapping[str, float],
    right_values: Mapping[str, float],
) -> float:
    scores = []
    for metric, tolerance in _FACE_SIMILARITY_TOLERANCES.items():
        if metric not in left_values or metric not in right_values:
            continue
        difference = abs(left_values[metric] - right_values[metric])
        scores.append(min(difference / (tolerance * 1.35), 1.0))
    result = _average(scores, 0.50)
    return result


def _score_mode_bonus(
    mode: str,
    saju_parts: _SajuScoreParts,
    face_parts: _FaceScoreParts,
) -> int:
    if mode == "연인":
        raw = 3 + (saju_parts.temperature_complement * 5) + (face_parts.complement * 4)
    elif mode == "직장동료":
        raw = 3 + (saju_parts.element_complement * 5) + (face_parts.stability * 3)
    else:
        no_clash = 1.0 - min(saju_parts.clash_penalty / 8.0, 1.0)
        raw = 3 + (face_parts.similarity * 4) + (no_clash * 3)
    result = _clamp_int(round(raw), 0, 10)
    return result


def _score_label(total: int) -> str:
    if total >= 91:
        result = "강한 시너지형"
    elif total >= 83:
        result = "찰떡 보완형"
    elif total >= 74:
        result = "편안한 조율형"
    else:
        result = "천천히 맞춰가는 조합"
    return result


def _mode_summary(mode: str, total: int, fingerprint: int) -> str:
    summaries = {
        "연인": (
            (
                91,
                (
                    "서로의 빈칸을 다정하게 채워주는 찰떡 로맨스 조합이에요.",
                    "눈빛만 맞아도 분위기가 살아나는 고득점 로맨스 케미예요.",
                    "설렘과 안정감이 같이 올라와서 오래 보기 좋은 로맨스 조합이에요.",
                ),
            ),
            (
                83,
                (
                    "설렘과 안정감이 같이 살아나서, 맞춰갈수록 달달해지는 조합이에요.",
                    "서로 다른 매력이 잘 섞여서 관계 온도가 포근하게 올라가는 조합이에요.",
                    "한쪽이 흔들릴 때 다른 쪽이 살짝 잡아주는 달달한 보완형이에요.",
                ),
            ),
            (
                74,
                (
                    "다른 리듬을 조금만 맞추면 편안한 온기가 살아나는 로맨스 조합이에요.",
                    "처음엔 속도 차이가 보여도, 맞춰갈수록 매력이 살아나는 연애 조합이에요.",
                    "표현 방식을 천천히 맞추면 설렘이 안정감으로 이어질 수 있어요.",
                ),
            ),
            (
                0,
                (
                    "속도를 천천히 맞추면 서로의 매력을 더 잘 발견할 수 있는 조합이에요.",
                    "서두르기보다 작은 약속부터 맞추면 관계가 훨씬 부드러워져요.",
                    "차이를 바로 결론 내리지 않으면 의외의 설렘 포인트가 보일 수 있어요.",
                ),
            ),
        ),
        "친구": (
            (
                91,
                (
                    "취향과 텐션이 잘 맞아 같이 있을수록 즐거움이 커지는 친구 조합이에요.",
                    "만나면 자연스럽게 웃음 포인트가 맞는 찰떡 친구 케미예요.",
                    "가볍게 놀아도 깊게 기대도 되는 든든한 친구 조합이에요.",
                ),
            ),
            (
                83,
                (
                    "서로의 장점을 자연스럽게 받아주며 편하게 오래 가기 좋은 친구 조합이에요.",
                    "취향은 달라도 대화가 잘 풀려서 오래 보기 좋은 친구 조합이에요.",
                    "서로의 텐션을 받아주는 힘이 좋아 편안한 친구 케미가 보여요.",
                ),
            ),
            (
                74,
                (
                    "대화 리듬만 조금 맞추면 더 가볍고 즐겁게 붙는 친구 조합이에요.",
                    "서로의 생활 리듬을 존중하면 편안함이 쌓이는 친구 조합이에요.",
                    "농담의 선과 쉬는 타이밍만 맞추면 더 즐거워지는 친구 케미예요.",
                ),
            ),
            (
                0,
                (
                    "서로의 다른 습관을 존중하면 천천히 편해지는 친구 조합이에요.",
                    "처음엔 텐션 차이가 보여도, 거리감을 맞추면 편해지는 친구 조합이에요.",
                    "취향 차이를 인정하면 오히려 새로운 재미가 생기는 친구 케미예요.",
                ),
            ),
        ),
        "직장동료": (
            (
                91,
                (
                    "한 명은 방향을 잡고 한 명은 실행력을 보태는, 업무 시너지가 기대되는 콤비예요.",
                    "기획과 실행이 착착 맞물려 결과물을 빠르게 만들 수 있는 협업 조합이에요.",
                    "역할만 선명하면 서로의 장점이 바로 성과로 이어지는 업무 콤비예요.",
                ),
            ),
            (
                83,
                (
                    "역할을 잘 나누면 아이디어와 실행이 착착 맞물리는 협업 조합이에요.",
                    "서로의 업무 스타일을 인정하면 속도와 완성도가 같이 올라가는 조합이에요.",
                    "한쪽의 추진력과 다른 쪽의 조율력이 만나기 좋은 협업 케미예요.",
                ),
            ),
            (
                74,
                (
                    "업무 속도와 피드백 기준을 맞추면 안정적으로 성과를 만들 수 있는 조합이에요.",
                    "회의 방식과 마감 기준만 맞추면 꽤 단단하게 굴러가는 업무 조합이에요.",
                    "서로의 우선순위를 먼저 확인하면 협업 효율이 살아나는 콤비예요.",
                ),
            ),
            (
                0,
                (
                    "역할과 마감 기준을 선명하게 잡으면 실수가 줄어드는 협업 조합이에요.",
                    "업무 방식 차이를 룰로 정리하면 생각보다 안정적으로 맞춰갈 수 있어요.",
                    "피드백 타이밍만 맞추면 충돌보다 보완이 더 커지는 업무 조합이에요.",
                ),
            ),
        ),
    }
    candidates = summaries.get(mode, summaries["친구"])
    result_options = candidates[-1][1]
    for threshold, options in candidates:
        if total >= threshold:
            result_options = options
            break
    result = result_options[fingerprint % len(result_options)]
    return result


def _score_texture(fingerprint: int) -> int:
    result = (fingerprint % 11) - 3
    return result


def _compatibility_fingerprint(
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    left_matches: Sequence[PhysiognomyRuleMatch],
    right_matches: Sequence[PhysiognomyRuleMatch],
) -> int:
    tokens = [
        mode,
        _chart_signature(left_manse.reading.chart),
        _chart_signature(right_manse.reading.chart),
        _matches_signature(left_matches),
        _matches_signature(right_matches),
    ]
    text = "|".join(tokens)
    result = sum((index + 17) * ord(char) for index, char in enumerate(text))
    return result


def _chart_signature(chart: SajuChart) -> str:
    result = ",".join(
        f"{pillar.stem_index}:{pillar.branch_index}"
        for pillar in _pillars(chart)
    )
    return result


def _matches_signature(matches: Sequence[PhysiognomyRuleMatch]) -> str:
    result = ",".join(
        f"{match.metric}:{match.value:.4f}"
        for match in sorted(matches, key=lambda item: item.metric)
    )
    return result


def _pillars(chart: SajuChart):
    result = (chart.year, chart.month, chart.day, chart.hour)
    return result


def _average(values: Sequence[float], fallback: float) -> float:
    if len(values) == 0:
        result = fallback
    else:
        result = sum(values) / len(values)
    return result


def _clamp(value: float, minimum: float, maximum: float) -> float:
    result = max(minimum, min(value, maximum))
    return result


def _clamp_int(value: int, minimum: int, maximum: int) -> int:
    result = max(minimum, min(value, maximum))
    return result

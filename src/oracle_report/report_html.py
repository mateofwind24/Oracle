from __future__ import annotations

import json
from dataclasses import dataclass
from html import escape
from typing import Any, Mapping

from oracle_report.models import BirthProfile
from oracle_report.recommender import FaceRecommendation
from oracle_report.saju.calendar import BRANCH_ELEMENTS, STEM_ELEMENTS
from oracle_report.saju.engine import ELEMENTS
from oracle_report.saju.repository import ManseLookupResult


_STEM_HANJA = ("甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸")
_BRANCH_HANJA = ("子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥")
_ELEMENT_HANJA = {
    "목": "木",
    "화": "火",
    "토": "土",
    "금": "金",
    "수": "水",
}
_POLARITY_HANJA = {
    "양": "陽",
    "음": "陰",
}
_ELEMENT_CLASS = {
    "목": "c-mok",
    "화": "c-hwa",
    "토": "c-to",
    "금": "c-geum",
    "수": "c-su",
}
_DEFAULT_FACE_BLOCKS = (
    {
        "category": "종합 형국",
        "title": "얼굴 비율에서 보이는 첫인상",
        "summary": "관상 정보는 보조 해석으로만 사용합니다.",
        "body": "촬영된 얼굴의 랜드마크와 품질 정보를 바탕으로 보이는 인상만 정리했습니다.",
    },
    {
        "category": "타고난 성향과 심리",
        "title": "표정과 눈썹 흐름",
        "summary": "눈, 눈썹, 표정은 현재 보이는 인상입니다.",
        "body": "얼굴 사진으로 실제 성격을 단정하지 않고, 리포트의 분위기를 보조하는 관찰로만 사용합니다.",
    },
    {
        "category": "재물운과 적성",
        "title": "중정과 하관의 균형",
        "summary": "코와 하관 비율은 전통 관상 기준의 보조 소재입니다.",
        "body": "사주 결과와 충돌할 때는 사주 데이터가 우선이며, 얼굴 해석은 표현을 돕는 보조 설명입니다.",
    },
    {
        "category": "연애운과 인간관계",
        "title": "눈매와 입꼬리의 분위기",
        "summary": "관계 해석은 실제 미래를 예측하지 않습니다.",
        "body": "사진에서 보이는 표정 흐름을 바탕으로 편안한 소통 방향만 제안합니다.",
    },
    {
        "category": "인생의 흐름 · 삼정",
        "title": "상정, 중정, 하정의 균형",
        "summary": "삼정 비율은 전체 인상의 균형을 보는 참고값입니다.",
        "body": "이마, 얼굴 중심부, 하관의 상대 비율을 참고해 리포트의 보조 문맥을 구성합니다.",
    },
)
_DEFAULT_SAJU_BLOCKS = (
    {
        "category": "종합 형국",
        "title": "사주 데이터가 보여주는 큰 흐름",
        "summary": "일간과 오행 분포를 중심으로 해석합니다.",
        "body": "만세력 DB에서 조회한 사주명식과 오행 균형을 바탕으로 전체 흐름을 정리합니다.",
    },
    {
        "category": "타고난 성향과 심리 패턴",
        "title": "일간 중심 성향",
        "summary": "일간은 리포트의 중심 기준입니다.",
        "body": "일간, 강한 오행, 약한 오행을 함께 보며 성향을 단정하지 않고 경향으로만 설명합니다.",
    },
    {
        "category": "재물운과 적성",
        "title": "현실 감각과 역할",
        "summary": "오행 균형을 직업과 생활 조언으로 연결합니다.",
        "body": "중대한 진로 결정을 예측하지 않고, 사주 데이터에서 보이는 보완 방향만 제안합니다.",
    },
    {
        "category": "연애운과 인간관계",
        "title": "관계에서 필요한 균형",
        "summary": "관계는 예언이 아니라 참고 조언입니다.",
        "body": "사주 흐름에서 보이는 소통 방식과 보완점을 중심으로 관계 조언을 구성합니다.",
    },
    {
        "category": "올해의 운세",
        "title": "올해의 리듬",
        "summary": "현재 시점에서 참고할 생활 리듬입니다.",
        "body": "올해의 운세는 재미용 참고이며, 실제 미래를 단정하지 않습니다.",
    },
)


@dataclass(frozen=True)
class _ReportBlock:
    category: str
    title: str
    summary: str
    body: str


@dataclass(frozen=True)
class _Convergence:
    face: str
    saju: str


@dataclass(frozen=True)
class _RecommendationCard:
    name: str
    score: int
    reason: str
    tag: str


@dataclass(frozen=True)
class _PersonalReportView:
    name: str
    meta: str
    day_master_hanja: str
    day_master_label: str
    day_master_class: str
    essence: str
    element_counts: Mapping[str, int]
    element_note: str
    pillars: tuple[tuple[str, str, str, str, str], ...]
    face_subtitle: str
    face_blocks: tuple[_ReportBlock, ...]
    saju_subtitle: str
    saju_blocks: tuple[_ReportBlock, ...]
    synth_title: str
    synth_body: str
    convergence: tuple[_Convergence, ...]
    synth_summary: str
    tags: tuple[str, ...]
    recommendation_title: str
    recommendation_lead: str
    recommendation_cards: tuple[_RecommendationCard, ...]
    disclaimer: str


def render_personal_report_html(
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
    generated_text: str,
    full_document: bool = True,
) -> str:
    view = _build_personal_report_view(
        profile,
        manse_lookup,
        face_analysis,
        recommendations,
        generated_text,
    )
    body = _render_report_body(view)
    style = _report_style()
    if full_document:
        result = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Oracle 리포트 · {escape(view.name)}</title>
{_font_links()}
<style>
{style}
</style>
</head>
<body>
{body}
</body>
</html>
"""
    else:
        result = f"""{_font_links()}
<style>
{style}
</style>
{body}
"""
    return result


def _build_personal_report_view(
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    recommendations: tuple[FaceRecommendation, ...],
    generated_text: str,
) -> _PersonalReportView:
    payload = _load_generated_payload(generated_text)
    reading = manse_lookup.reading
    chart = reading.chart
    weakest = _weakest_element(reading.element_counts)
    strongest = _strongest_element(reading.element_counts)
    day_element = STEM_ELEMENTS[chart.day.stem_index]
    day_master_hanja = _STEM_HANJA[chart.day.stem_index]
    day_master_label = (
        f"{chart.day.stem}{day_element} · "
        f"{_POLARITY_HANJA.get(chart.day.polarity, chart.day.polarity)}"
        f"{_ELEMENT_HANJA[day_element]}"
    )
    birth_text = profile.birth_datetime.strftime("%Y. %m. %d · %H:%M")
    meta = f"{birth_text} · {profile.gender or '성별 미입력'} · 양력"
    essence = _payload_text(
        payload,
        "essence",
        f"{strongest} 기운을 살리고 {weakest} 기운을 보완하면 균형이 살아나는 사람",
    )
    element_note = _payload_text(
        payload,
        "element_note",
        f"가장 강한 오행은 {strongest}, 보완하면 좋은 오행은 {weakest}입니다.",
    )
    result = _PersonalReportView(
        name=profile.name,
        meta=meta,
        day_master_hanja=day_master_hanja,
        day_master_label=day_master_label,
        day_master_class=_ELEMENT_CLASS[day_element],
        essence=essence,
        element_counts=reading.element_counts,
        element_note=element_note,
        pillars=_pillar_views(manse_lookup),
        face_subtitle=_payload_text(payload, "face_subtitle", "랜드마크 · 관상 보조"),
        face_blocks=_payload_blocks(payload, "face_blocks", _DEFAULT_FACE_BLOCKS),
        saju_subtitle=_payload_text(
            payload,
            "saju_subtitle",
            f"{chart.day.stem}{day_element} · {strongest}강 · {weakest}보완",
        ),
        saju_blocks=_payload_blocks(payload, "saju_blocks", _default_saju_blocks(reading)),
        synth_title=_payload_text(
            payload,
            "synthesis_title",
            "관상과 사주를 함께 보면 더 선명해지는 흐름",
        ),
        synth_body=_payload_text(
            payload,
            "synthesis_body",
            "사주 데이터와 관상 보조 정보가 겹치는 지점을 중심으로 해석합니다.",
        ),
        convergence=_payload_convergence(payload, face_analysis, reading.interpretation),
        synth_summary=_payload_text(
            payload,
            "synthesis_summary",
            "결론은 단정이 아니라 참고입니다. 강점은 살리고 부족한 리듬은 생활에서 보완하세요.",
        ),
        tags=_payload_tags(payload, (f"{weakest} 보완", "균형", "휴식", "표현")),
        recommendation_title=_payload_text(
            payload,
            "recommendation_title",
            f"{weakest} 기운을 보완해 줄 얼굴",
        ),
        recommendation_lead=_payload_text(
            payload,
            "recommendation_lead",
            "내 얼굴과 닮은 사람보다 사주와 인상에서 부족한 리듬을 보완하는 후보를 우선 추천합니다.",
        ),
        recommendation_cards=_recommendation_cards(recommendations),
        disclaimer=_payload_text(
            payload,
            "disclaimer",
            "이 리포트는 관상 보조 분석과 사주/만세력 데이터를 바탕으로 생성된 재미용 콘텐츠입니다. 운명을 단정하지 않으며 참고로만 즐겨 주세요.",
        ),
    )
    return result


def _load_generated_payload(generated_text: str) -> dict[str, Any]:
    cleaned = generated_text.strip()
    if cleaned.startswith("```"):
        cleaned = _strip_markdown_fence(cleaned)
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start : end + 1]
    result: dict[str, Any] = {}
    try:
        loaded = json.loads(cleaned)
        if isinstance(loaded, dict):
            result = loaded
    except json.JSONDecodeError:
        result = {}
    return result


def _strip_markdown_fence(text: str) -> str:
    lines = text.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    result = "\n".join(lines)
    return result


def _payload_text(payload: Mapping[str, Any], key: str, default: str) -> str:
    value = payload.get(key)
    result = default
    if isinstance(value, str) and value.strip() != "":
        result = value.strip()
    return result


def _payload_blocks(
    payload: Mapping[str, Any],
    key: str,
    defaults: tuple[Mapping[str, str], ...],
) -> tuple[_ReportBlock, ...]:
    raw_blocks = payload.get(key)
    blocks: list[_ReportBlock] = []
    if isinstance(raw_blocks, list):
        for index, raw_block in enumerate(raw_blocks):
            default = defaults[min(index, len(defaults) - 1)]
            if isinstance(raw_block, dict):
                blocks.append(
                    _ReportBlock(
                        category=_payload_text(raw_block, "category", default["category"]),
                        title=_payload_text(raw_block, "title", default["title"]),
                        summary=_payload_text(raw_block, "summary", default["summary"]),
                        body=_payload_text(raw_block, "body", default["body"]),
                    ),
                )
    while len(blocks) < len(defaults):
        default = defaults[len(blocks)]
        blocks.append(
            _ReportBlock(
                category=default["category"],
                title=default["title"],
                summary=default["summary"],
                body=default["body"],
            ),
        )
    result = tuple(blocks[: len(defaults)])
    return result


def _payload_convergence(
    payload: Mapping[str, Any],
    face_analysis: str,
    saju_text: str,
) -> tuple[_Convergence, ...]:
    raw_items = payload.get("convergence")
    items: list[_Convergence] = []
    if isinstance(raw_items, list):
        for raw_item in raw_items:
            if isinstance(raw_item, dict):
                items.append(
                    _Convergence(
                        face=_payload_text(raw_item, "face", "관상 보조 정보"),
                        saju=_payload_text(raw_item, "saju", "사주 데이터"),
                    ),
                )
    if not items:
        face_line = _first_nonempty_line(face_analysis, "관상 보조 정보")
        saju_line = _first_nonempty_line(saju_text, "사주 데이터")
        items = (
            _Convergence(face=face_line, saju=saju_line),
            _Convergence(face="얼굴 비율의 균형", saju="오행 균형의 보완점"),
            _Convergence(face="현재 표정의 흐름", saju="생활 리듬의 조절"),
        )
    result = tuple(items[:4])
    return result


def _payload_tags(
    payload: Mapping[str, Any],
    defaults: tuple[str, ...],
) -> tuple[str, ...]:
    raw_tags = payload.get("tags")
    tags: list[str] = []
    if isinstance(raw_tags, list):
        tags = [str(item).strip() for item in raw_tags if str(item).strip()]
    if not tags:
        tags = list(defaults)
    result = tuple(tags[:6])
    return result


def _recommendation_cards(
    recommendations: tuple[FaceRecommendation, ...],
) -> tuple[_RecommendationCard, ...]:
    cards: list[_RecommendationCard] = []
    for item in recommendations[:3]:
        tag = " · ".join(item.saju_tags[:2] or item.face_tags[:2])
        cards.append(
            _RecommendationCard(
                name=item.display_name,
                score=min(99, max(1, item.score * 10)),
                reason=item.reason,
                tag=tag or "균형 보완",
            ),
        )
    while len(cards) < 3:
        rank = len(cards) + 1
        cards.append(
            _RecommendationCard(
                name=f"추천상대 {rank}",
                score=max(70, 88 - (rank * 4)),
                reason="추천 DB에 맞는 후보가 부족해 기본 안내 카드로 대체합니다.",
                tag="추천 후보 대기",
            ),
        )
    result = tuple(cards[:3])
    return result


def _pillar_views(
    manse_lookup: ManseLookupResult,
) -> tuple[tuple[str, str, str, str, str], ...]:
    chart = manse_lookup.reading.chart
    pillars = (
        ("시주", chart.hour),
        ("일주 (나)", chart.day),
        ("월주", chart.month),
        ("연주", chart.year),
    )
    rows = []
    for title, pillar in pillars:
        stem_element = STEM_ELEMENTS[pillar.stem_index]
        branch_element = BRANCH_ELEMENTS[pillar.branch_index]
        rows.append(
            (
                title,
                _STEM_HANJA[pillar.stem_index],
                f"{pillar.stem} · {stem_element}",
                _BRANCH_HANJA[pillar.branch_index],
                f"{pillar.branch} · {branch_element}",
            ),
        )
    result = tuple(rows)
    return result


def _default_saju_blocks(reading) -> tuple[Mapping[str, str], ...]:
    defaults = list(_DEFAULT_SAJU_BLOCKS)
    summary = " ".join(reading.summary_lines)
    defaults[0] = {
        **defaults[0],
        "body": f"{summary}\n{reading.interpretation}",
    }
    result = tuple(defaults)
    return result


def _weakest_element(counts: Mapping[str, int]) -> str:
    result = min(ELEMENTS, key=lambda element: (counts[element], ELEMENTS.index(element)))
    return result


def _strongest_element(counts: Mapping[str, int]) -> str:
    result = max(ELEMENTS, key=lambda element: (counts[element], -ELEMENTS.index(element)))
    return result


def _first_nonempty_line(text: str, fallback: str) -> str:
    result = fallback
    for line in text.splitlines():
        cleaned = line.strip(" -#")
        if cleaned:
            result = cleaned
            break
    return result


def _render_report_body(view: _PersonalReportView) -> str:
    result = f"""
<div class="oracle-report">
<div class="wrap">
  <header class="fade">
    <div class="eyebrow">Oracle · 관상 &amp; 사주 종합 리포트</div>
    <div class="ilgan {escape(view.day_master_class)}">{escape(view.day_master_hanja)}<span class="ko">{escape(view.day_master_label)}</span></div>
    <div class="name">{escape(view.name)} 님</div>
    <div class="meta">{escape(view.meta)}</div>
    <p class="essence serif">{escape(view.essence)}</p>
  </header>
  {_render_element_balance(view)}
  {_render_pillars(view)}
  {_render_part("gwansang", "相", "관상 — 얼굴이 말하는 것", view.face_subtitle, view.face_blocks)}
  {_render_part("saju", "命", "사주 — 타고난 기운의 설계도", view.saju_subtitle, view.saju_blocks)}
  {_render_synthesis(view)}
  {_render_tags(view.tags)}
  {_render_recommendations(view)}
  <footer>
    <div class="logo">ORACLE</div>
    <p class="disc">{escape(view.disclaimer)}</p>
  </footer>
</div>
</div>
"""
    return result


def _render_element_balance(view: _PersonalReportView) -> str:
    max_count = max(view.element_counts.values()) if view.element_counts else 1
    bars = []
    for element in ELEMENTS:
        count = view.element_counts[element]
        height = 14 if count == 0 else max(20, int((count / max_count) * 100))
        empty_class = " empty" if count == 0 else ""
        miss_label = '<span class="label-miss">비어 있음</span>' if count == 0 else ""
        bars.append(
            f"""<div class="bar{empty_class}">{miss_label}<div class="col {_ELEMENT_CLASS[element]}" style="height:{height}%"></div><div class="han">{_ELEMENT_HANJA[element]}</div><div class="cnt">{escape(element)} · {count}</div></div>""",
        )
    result = f"""
  <section class="ohaeng fade">
    <h3>오 행 의 균 형</h3>
    <div class="bars">
      {''.join(bars)}
    </div>
    <p class="element-note">{escape(view.element_note)}</p>
  </section>
"""
    return result


def _render_pillars(view: _PersonalReportView) -> str:
    columns = []
    chart = view.pillars
    for index, item in enumerate(chart):
        title, stem_hanja, stem_text, branch_hanja, branch_text = item
        stem_element = stem_text.split(" · ")[-1]
        branch_element = branch_text.split(" · ")[-1]
        stem_class = _ELEMENT_CLASS.get(stem_element, "c-mok")
        branch_class = _ELEMENT_CLASS.get(branch_element, "c-to")
        columns.append(
            f"""
      <div class="pcol">
        <div class="ptitle">{escape(title)}</div>
        <div class="cell gan {stem_class}"><div class="ch">{escape(stem_hanja)}</div><div class="ss">{escape(stem_text)}</div></div>
        <div class="cell {branch_class}"><div class="ch">{escape(branch_hanja)}</div><div class="ss">{escape(branch_text)}</div></div>
      </div>
""",
        )
    result = f"""
  <section class="myeongsik fade">
    <div class="cap serif">사 주 명 식 (四柱命式)</div>
    <div class="grid4">
      {''.join(columns)}
    </div>
  </section>
"""
    return result


def _render_part(
    class_name: str,
    number: str,
    title: str,
    subtitle: str,
    blocks: tuple[_ReportBlock, ...],
) -> str:
    block_html = "\n".join(_render_block(block) for block in blocks)
    result = f"""
  <section class="part {escape(class_name)} fade">
    <div class="part-head">
      <div class="part-num">{escape(number)}</div>
      <div class="part-title">{escape(title)}</div>
      <div class="part-sub">{escape(subtitle)}</div>
    </div>
    {block_html}
  </section>
"""
    return result


def _render_block(block: _ReportBlock) -> str:
    result = f"""
    <div class="block">
      <div class="b-cat">{escape(block.category)}</div>
      <div class="b-title">{escape(block.title)}</div>
      <div class="b-sum">{escape(block.summary)}</div>
      <div class="b-body">{_paragraphs(block.body)}</div>
    </div>
"""
    return result


def _render_synthesis(view: _PersonalReportView) -> str:
    convergence = "\n".join(
        f"""      <div class="cv"><span class="g">{escape(item.face)}</span><span class="eq">＝</span><span class="s">{escape(item.saju)}</span></div>"""
        for item in view.convergence
    )
    result = f"""
  <section class="synth fade">
    <div class="b-title serif">{escape(view.synth_title)}</div>
    <div class="b-body">{_paragraphs(view.synth_body)}</div>
    <div class="converge">
{convergence}
    </div>
    <div class="b-body synth-summary">{_paragraphs(view.synth_summary)}</div>
  </section>
"""
    return result


def _render_tags(tags: tuple[str, ...]) -> str:
    chips = "".join(f'<span class="chip">{escape(tag)}</span>' for tag in tags)
    result = f"""
  <section class="tags fade">
    <h3>나 를 채 워 주 는 키 워 드</h3>
    {chips}
  </section>
"""
    return result


def _render_recommendations(view: _PersonalReportView) -> str:
    cards = "\n".join(
        _render_recommendation_card(index, card)
        for index, card in enumerate(view.recommendation_cards, start=1)
    )
    result = f"""
  <section class="reco fade">
    <div class="reco-head"><div class="b-cat">FACE MATCH · 궁합 좋은 얼굴 추천</div>
      <div class="b-title serif">{escape(view.recommendation_title)}</div></div>
    <p class="reco-lead">{escape(view.recommendation_lead)}</p>
    <div class="cards">
{cards}
    </div>
    <p class="reco-note">※ 내부 얼굴 DB에서 궁합 점수 상위 후보를 추렸어요. 얼굴 이미지는 데모용 placeholder예요.</p>
  </section>
"""
    return result


def _render_recommendation_card(index: int, card: _RecommendationCard) -> str:
    result = f"""
      <div class="card">
        <div class="rank">No.{index}</div>
        <div class="face"><svg viewBox="0 0 24 24"><path d="M12 12a5 5 0 1 0 0-10 5 5 0 0 0 0 10Zm0 2c-4 0-8 2-8 5v1h16v-1c0-3-4-5-8-5Z"/></svg></div>
        <div class="nm">{escape(card.name)}</div>
        <div class="score">{card.score}<span>점</span></div>
        <div class="reason">{escape(card.reason)}</div>
        <span class="mtag">{escape(card.tag)}</span>
      </div>
"""
    return result


def _paragraphs(text: str) -> str:
    parts = [escape(part.strip()) for part in text.splitlines() if part.strip()]
    result = "<br>".join(parts)
    return result


def _font_links() -> str:
    result = """<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Gowun+Batang:wght@400;700&family=Gowun+Dodum&family=Song+Myung&display=swap" rel="stylesheet">"""
    return result


def _report_style() -> str:
    result = """
:root{
  --paper:#F6F1E7; --paper-2:#FBF8F1; --ink:#2A2520; --ink-soft:#6B6256;
  --line:#DAD0BE; --mok:#3A7D5C; --hwa:#C25239; --to:#CC9A3B;
  --geum:#9A958B; --su:#2E4258; --gold:#A8823C;
}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Gowun Dodum",sans-serif;line-height:1.75;-webkit-font-smoothing:antialiased;background-image:radial-gradient(circle at 20% 10%,rgba(58,125,92,.04),transparent 40%),radial-gradient(circle at 85% 30%,rgba(194,82,57,.04),transparent 40%)}
.oracle-report{background:var(--paper);color:var(--ink);font-family:"Gowun Dodum",sans-serif;line-height:1.75;background-image:radial-gradient(circle at 20% 10%,rgba(58,125,92,.04),transparent 40%),radial-gradient(circle at 85% 30%,rgba(194,82,57,.04),transparent 40%);padding-bottom:1px}
.oracle-report *{box-sizing:border-box;margin:0;padding:0}
.wrap{max-width:760px;margin:0 auto;padding:0 22px}
.serif{font-family:"Gowun Batang",serif}
header{padding:64px 0 40px;text-align:center;border-bottom:1px solid var(--line)}
.eyebrow{font-size:12px;letter-spacing:.42em;color:var(--gold);text-transform:uppercase;margin-bottom:26px}
.ilgan{font-family:"Song Myung",serif;font-size:120px;line-height:1;position:relative;display:inline-block}
.ilgan.c-mok{color:var(--mok)}.ilgan.c-hwa{color:var(--hwa)}.ilgan.c-to{color:var(--to)}.ilgan.c-geum{color:var(--geum)}.ilgan.c-su{color:var(--su)}
.ilgan .ko{font-family:"Gowun Batang",serif;font-size:20px;color:var(--ink-soft);position:absolute;bottom:14px;right:-8px;transform:translateX(100%)}
.name{font-family:"Gowun Batang",serif;font-size:30px;font-weight:700;margin-top:18px}
.meta{font-size:14px;color:var(--ink-soft);margin-top:8px;letter-spacing:.04em}
.essence{font-family:"Gowun Batang",serif;font-size:19px;margin-top:24px;color:var(--ink);max-width:30ch;margin-left:auto;margin-right:auto}
.ohaeng{margin:46px 0;padding:30px 26px;background:var(--paper-2);border:1px solid var(--line);border-radius:6px}
.ohaeng h3,.myeongsik .cap,.tags h3{font-family:"Gowun Batang",serif;font-size:13px;letter-spacing:.3em;color:var(--ink-soft);text-align:center;margin-bottom:26px;font-weight:400}
.bars{display:flex;justify-content:space-between;align-items:flex-end;gap:12px;height:140px}
.bar{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:flex-end;height:100%}
.col{width:100%;border-radius:4px 4px 0 0;position:relative;transition:height .9s cubic-bezier(.2,.7,.2,1)}
.bar .han{font-family:"Song Myung",serif;font-size:26px;margin-top:10px;color:var(--ink)}
.bar .cnt{font-size:12px;color:var(--ink-soft);margin-top:2px}
.bar.empty .col{background:transparent;border:1.5px dashed var(--su);height:14px!important}
.bar.empty .label-miss{position:absolute;top:-22px;left:50%;transform:translateX(-50%);font-size:11px;color:var(--su);white-space:nowrap;font-weight:700}
.bar.empty .han{color:var(--su);opacity:.55}
.element-note{text-align:center;font-size:13px;color:var(--ink-soft);margin-top:22px;line-height:1.7}
.myeongsik{margin:46px 0}.myeongsik .cap{margin-bottom:18px}
.grid4{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.pcol{text-align:center}
.pcol .ptitle{font-size:12px;color:var(--ink-soft);margin-bottom:6px}
.cell{border-radius:5px;padding:12px 4px;color:#fff}.cell .ch{font-family:"Song Myung",serif;font-size:34px;line-height:1.1}.cell .ss{font-size:11px;opacity:.92;margin-top:3px}.cell.gan{margin-bottom:6px}
.c-mok{background:var(--mok)}.c-hwa{background:var(--hwa)}.c-to{background:var(--to)}.c-geum{background:var(--geum)}.c-su{background:var(--su)}
.part{margin:56px 0 0}.part-head{display:flex;align-items:baseline;gap:14px;padding-bottom:14px;border-bottom:2px solid var(--ink);margin-bottom:30px}
.part-num{font-family:"Song Myung",serif;font-size:40px;line-height:1;color:var(--gold)}.part-title{font-family:"Gowun Batang",serif;font-size:24px;font-weight:700}.part-sub{font-size:13px;color:var(--ink-soft);margin-left:auto}
.block{margin-bottom:34px}.b-cat{font-size:11px;letter-spacing:.25em;color:var(--gold);text-transform:uppercase;margin-bottom:8px}.b-title{font-family:"Gowun Batang",serif;font-size:21px;font-weight:700;line-height:1.4;margin-bottom:10px}
.b-sum{font-size:14px;color:var(--mok);font-weight:400;margin-bottom:12px;padding-left:14px;border-left:3px solid var(--mok)}.part.saju .b-sum{color:var(--hwa);border-color:var(--hwa)}.b-body{font-size:15.5px;color:var(--ink)}
.synth{margin:60px 0 0;padding:38px 30px;background:linear-gradient(135deg,rgba(58,125,92,.07),rgba(194,82,57,.07));border:1px solid var(--line);border-radius:8px}.synth .b-title{font-size:24px;text-align:center;margin-bottom:18px}.synth-summary{margin-top:22px}
.converge{margin:24px 0 4px;display:flex;flex-direction:column;gap:10px}.cv{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;font-size:13.5px}.cv .g{color:var(--mok);text-align:right}.cv .s{color:var(--hwa)}.cv .eq{color:var(--gold);font-family:"Gowun Batang",serif;font-weight:700}
.tags{margin:46px 0;text-align:center}.tags h3{margin-bottom:18px}.chip{display:inline-block;margin:5px;padding:9px 18px;background:var(--paper-2);border:1px solid var(--su);color:var(--su);border-radius:30px;font-size:14px;font-family:"Gowun Batang",serif}
.reco{margin:60px 0 0}.reco-head{text-align:center;margin-bottom:8px}.reco-head .b-title{font-size:24px}.reco-lead{text-align:center;font-size:14px;color:var(--ink-soft);max-width:42ch;margin:0 auto 30px;line-height:1.75}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:var(--paper-2);border:1px solid var(--line);border-radius:8px;padding:18px 16px 20px;text-align:center;position:relative;transition:transform .25s ease,box-shadow .25s ease}.card:hover{transform:translateY(-4px);box-shadow:0 10px 28px rgba(46,66,88,.12)}
.card .rank{position:absolute;top:12px;left:14px;font-family:"Song Myung",serif;font-size:13px;color:var(--gold)}.face{width:86px;height:86px;border-radius:50%;margin:6px auto 14px;background:linear-gradient(135deg,#dfe6e9,#c7d0d6);display:flex;align-items:center;justify-content:center;border:2px solid var(--su)}.face svg{width:46px;height:46px;fill:var(--su);opacity:.7}
.card .nm{font-family:"Gowun Batang",serif;font-size:16px;font-weight:700}.score{font-family:"Song Myung",serif;font-size:30px;color:var(--su);line-height:1.1;margin:4px 0 2px}.score span{font-size:14px;color:var(--ink-soft)}.reason{font-size:12.5px;color:var(--ink);line-height:1.6;margin-top:8px}.mtag{display:inline-block;margin-top:12px;padding:3px 10px;background:rgba(46,66,88,.08);color:var(--su);border-radius:20px;font-size:11px}.reco-note{text-align:center;font-size:11.5px;color:var(--ink-soft);margin-top:18px}
footer{text-align:center;padding:44px 0 60px;border-top:1px solid var(--line);margin-top:50px}footer .logo{font-family:"Song Myung",serif;font-size:22px;letter-spacing:.2em;color:var(--ink)}footer .disc{font-size:11.5px;color:var(--ink-soft);margin-top:12px;max-width:46ch;margin-left:auto;margin-right:auto;line-height:1.7}
.fade{opacity:0;transform:translateY(16px);animation:rise .8s ease forwards}@keyframes rise{to{opacity:1;transform:none}}@media (prefers-reduced-motion:reduce){.fade{animation:none;opacity:1;transform:none}.col{transition:none}}
@media (max-width:560px){.ilgan{font-size:88px}.name{font-size:24px}.cell .ch{font-size:26px}.part-title{font-size:20px}.cards{grid-template-columns:1fr;gap:12px}.cv{grid-template-columns:1fr;text-align:center;gap:2px}.cv .g{text-align:center}.part-head{display:block}.part-sub{margin-left:0;margin-top:6px}.grid4{gap:6px}.wrap{padding:0 16px}}
"""
    return result.strip()

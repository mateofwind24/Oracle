from __future__ import annotations

import base64
import json
import mimetypes
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any, Mapping

from oracle_report.models import BirthProfile
from oracle_report.saju.calendar import BRANCH_ELEMENTS, STEM_ELEMENTS
from oracle_report.saju.engine import ELEMENTS
from oracle_report.saju.repository import (
    ManseLookupResult,
    birth_time_display_from_profile,
)


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
        "summary": "얼굴의 세로·가로 비율과 좌우 균형을 함께 본 기본 인상입니다.",
        "body": "얼굴 폭과 높이의 균형이 안정적으로 보이면 사람을 대할 때 정돈되고 차분한 첫인상으로 읽습니다. 어느 한 부위가 강하게 튀기보다 전체 윤곽이 고르게 보이면 리포트에서는 균형감과 신뢰감을 중심 키워드로 설명합니다.",
    },
    {
        "category": "타고난 성향과 심리",
        "title": "눈과 눈썹이 만드는 집중감",
        "summary": "눈은 시선의 또렷함, 눈썹은 인상을 정돈하는 선으로 해석합니다.",
        "body": "눈 주변이 또렷하게 보이면 생각을 빠르게 모으고 반응을 분명하게 드러내는 분위기로 풀이합니다. 눈썹이 눈을 안정적으로 감싸면 감정을 급하게 드러내기보다 정돈해서 표현하는 인상으로 설명할 수 있습니다.",
    },
    {
        "category": "재물운과 적성",
        "title": "코와 하관에서 보는 현실감",
        "summary": "코는 중심감, 하관은 마무리와 지속성을 보는 기준입니다.",
        "body": "코가 얼굴 중심에서 안정적으로 보이면 기준을 세우고 현실적으로 판단하려는 인상으로 읽습니다. 하관이 단정하게 이어지면 일을 끝까지 정리하려는 힘과 꾸준함을 더해 주는 요소로 풀이합니다.",
    },
    {
        "category": "연애운과 인간관계",
        "title": "표정 안정성과 소통 방식",
        "summary": "입꼬리와 얼굴 좌우 균형은 대화할 때의 부드러움을 보는 요소입니다.",
        "body": "표정이 안정적으로 보이면 상대가 느끼는 긴장감을 낮추고 편안하게 대화를 시작하는 인상으로 해석합니다. 입 주변과 시선이 자연스럽게 정돈되어 보이면 관계에서는 차분하게 분위기를 맞추는 쪽으로 설명할 수 있습니다.",
    },
    {
        "category": "인생의 흐름 · 삼정",
        "title": "이마·중안부·하관의 비율",
        "summary": "삼정은 얼굴을 위·가운데·아래 세 구역으로 나누어 균형을 보는 전통 기준입니다.",
        "body": "이마는 계획과 생각의 폭, 중안부는 현실감과 실행력, 하관은 마무리와 지속성을 상징하는 식으로 풀이합니다. 세 구역이 고르게 보이면 특정 시기보다 전체 흐름을 균형 있게 가져가는 인상으로 설명합니다.",
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
    {
        "category": "총평 및 인생의 조언",
        "title": "균형을 되찾는 현실적인 조언",
        "summary": "강점은 살리고 부족한 리듬은 일상에서 보완합니다.",
        "body": "사주 해석은 정해진 결론이 아니라 현재 성향을 점검하는 참고 자료입니다. 강하게 드러나는 기운은 장점으로 쓰고 부족한 기운은 생활 습관과 관계 방식에서 천천히 보완하는 쪽이 좋습니다.",
    },
)


@dataclass(frozen=True)
class _ReportBlock:
    category: str
    title: str
    summary: str
    body: str


@dataclass(frozen=True)
class _CompatibilityPersonView:
    name: str
    meta: str
    day_master_hanja: str
    day_master_label: str
    day_master_class: str
    element_note: str


@dataclass(frozen=True)
class _CompatibilityScoreView:
    total: int
    saju: int
    face: int
    mode_bonus: int
    label: str
    summary: str


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
    tags: tuple[str, ...]
    disclaimer: str
    skip_face: bool = False


@dataclass(frozen=True)
class _CompatibilityReportView:
    left: _CompatibilityPersonView
    right: _CompatibilityPersonView
    mode: str
    compatibility_score: _CompatibilityScoreView
    essence: str
    pair_subtitle: str
    pair_blocks: tuple[_ReportBlock, ...]
    saju_subtitle: str
    saju_blocks: tuple[_ReportBlock, ...]
    action_title: str
    action_body: str
    tags: tuple[str, ...]
    disclaimer: str


def render_personal_report_html(
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    generated_text: str,
    full_document: bool = True,
    skip_face: bool = False,
) -> str:
    view = _build_personal_report_view(
        profile,
        manse_lookup,
        face_analysis,
        generated_text,
        skip_face=skip_face,
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
        result = _embed_static_asset_images(result)
    else:
        result = f"""{_font_links()}
<style>
{style}
</style>
{body}
"""
    return result


def render_compatibility_report_html(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    face_analysis: str,
    generated_text: str,
    full_document: bool = True,
) -> str:
    view = _build_compatibility_report_view(
        left_profile,
        right_profile,
        mode,
        left_manse,
        right_manse,
        face_analysis,
        generated_text,
    )
    body = _render_compatibility_report_body(view)
    style = _report_style()
    title = f"Oracle 궁합 리포트 · {view.left.name} × {view.right.name}"
    if full_document:
        result = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{escape(title)}</title>
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
        result = _embed_static_asset_images(result)
    else:
        result = f"""{_font_links()}
<style>
{style}
</style>
{body}
"""
    return result


def _embed_static_asset_images(html: str) -> str:
    def replace(match: re.Match[str]) -> str:
        quote = match.group("quote")
        src = match.group("src")
        data_uri = _static_asset_data_uri(src)
        result = f'src={quote}{data_uri}{quote}'
        return result

    result = re.sub(
        r'src=(?P<quote>["\'])(?P<src>/static/assets/[^"\']+)(?P=quote)',
        replace,
        html,
    )
    return result


def _static_asset_data_uri(src: str) -> str:
    assets_root = (Path(__file__).resolve().parent / "static" / "assets").resolve()
    relative_name = src.removeprefix("/static/assets/")
    asset_path = (assets_root / relative_name).resolve()
    result = src
    if assets_root in asset_path.parents and asset_path.is_file():
        mime_type = mimetypes.guess_type(asset_path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(asset_path.read_bytes()).decode("ascii")
        result = f"data:{mime_type};base64,{encoded}"
    return result


def _build_personal_report_view(
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
    face_analysis: str,
    generated_text: str,
    skip_face: bool = False,
) -> _PersonalReportView:
    payload = _load_generated_payload(generated_text)
    if not payload:
        print(
            "[UI FALLBACK:personal_report] generated_text is not valid JSON; "
            "rendering default UI values."
        )
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
    birth_date_text = profile.birth_datetime.strftime("%Y. %m. %d")
    birth_time_text = _profile_birth_time_display(profile)
    birth_text = f"{birth_date_text} · {birth_time_text}"
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
    disclaimer_fallback = (
        "이 리포트는 얼굴 관찰과 사주/만세력 데이터를 바탕으로 생성된 재미용 "
        "콘텐츠입니다. 운명을 단정하지 않으며 참고로만 즐겨 주세요."
    )
    if skip_face:
        disclaimer_fallback = (
            "이 리포트는 사주/만세력 데이터를 바탕으로 생성된 재미용 콘텐츠입니다. "
            "운명을 단정하지 않으며 참고로만 즐겨 주세요."
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
        face_subtitle=_payload_text(payload, "face_subtitle", "얼굴 비율 · 인상 관찰"),
        face_blocks=_payload_blocks(payload, "face_blocks", _DEFAULT_FACE_BLOCKS),
        saju_subtitle=_payload_text(
            payload,
            "saju_subtitle",
            f"{chart.day.stem}{day_element} · {strongest}강 · {weakest}보완",
        ),
        saju_blocks=_payload_blocks(payload, "saju_blocks", _default_saju_blocks(reading)),
        tags=_payload_tags(payload, (f"{weakest} 보완", "균형", "휴식", "표현")),
        disclaimer=_payload_text(
            payload,
            "disclaimer",
            disclaimer_fallback,
        ),
        skip_face=skip_face,
    )
    return result


def _build_compatibility_report_view(
    left_profile: BirthProfile,
    right_profile: BirthProfile,
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
    face_analysis: str,
    generated_text: str,
) -> _CompatibilityReportView:
    payload = _load_generated_payload(generated_text)
    left_view = _compatibility_person_view(left_profile, left_manse)
    right_view = _compatibility_person_view(right_profile, right_manse)
    result = _CompatibilityReportView(
        left=left_view,
        right=right_view,
        mode=mode,
        compatibility_score=_payload_compatibility_score(payload, mode),
        essence=_payload_text(
            payload,
            "essence",
            f"{left_profile.name} 님과 {right_profile.name} 님은 {mode} 관계에서 서로의 리듬을 맞춰 가는 쪽이 중요한 조합입니다.",
        ),
        pair_subtitle=_payload_text(
            payload,
            "pair_subtitle",
            "얼굴 관찰과 관계 분위기",
        ),
        pair_blocks=_payload_blocks(
            payload,
            "pair_blocks",
            _default_compatibility_pair_blocks(mode, face_analysis),
        ),
        saju_subtitle=_payload_text(
            payload,
            "saju_subtitle",
            "사주 흐름과 상호 보완",
        ),
        saju_blocks=_payload_blocks(
            payload,
            "saju_blocks",
            _default_compatibility_saju_blocks(mode, left_manse, right_manse),
        ),
        action_title=_payload_text(
            payload,
            "action_title",
            "관계를 좋게 만드는 행동 제안",
        ),
        action_body=_payload_text(
            payload,
            "action_body",
            "중요한 대화는 한 번에 결론을 내리기보다 서로의 속도를 확인하면서 이어 가는 편이 좋습니다. 감정이 올라오는 순간에는 바로 판단하기보다 상대가 원하는 반응과 실제 의도를 나누어 확인해 보세요.",
        ),
        tags=_payload_tags(payload, (mode, "상호 보완", "소통 리듬", "관계 균형")),
        disclaimer=_payload_text(
            payload,
            "disclaimer",
            "이 리포트는 사주 데이터와 얼굴 관찰 메모를 바탕으로 생성한 참고용 엔터테인먼트 콘텐츠입니다. 관계를 단정하거나 실제 미래를 예언하지 않습니다.",
        ),
    )
    return result


def _compatibility_person_view(
    profile: BirthProfile,
    manse_lookup: ManseLookupResult,
) -> _CompatibilityPersonView:
    chart = manse_lookup.reading.chart
    day_element = STEM_ELEMENTS[chart.day.stem_index]
    strongest = _strongest_element(manse_lookup.reading.element_counts)
    weakest = _weakest_element(manse_lookup.reading.element_counts)
    birth_date_text = profile.birth_datetime.strftime("%Y. %m. %d")
    birth_time_text = _profile_birth_time_display(profile)
    meta = f"{birth_date_text} · {birth_time_text} · {profile.gender or '성별 미입력'}"
    result = _CompatibilityPersonView(
        name=profile.name,
        meta=meta,
        day_master_hanja=_STEM_HANJA[chart.day.stem_index],
        day_master_label=(
            f"{chart.day.stem}{day_element} · "
            f"{_POLARITY_HANJA.get(chart.day.polarity, chart.day.polarity)}"
            f"{_ELEMENT_HANJA[day_element]}"
        ),
        day_master_class=_ELEMENT_CLASS[day_element],
        element_note=f"{strongest} 기운이 강하고 {weakest} 기운을 보완하면 균형이 좋아집니다.",
    )
    return result


def _profile_birth_time_display(profile: BirthProfile) -> str:
    result = birth_time_display_from_profile(profile)
    if not profile.birth_time_known:
        result = "시간 미상"
    return result


def _default_compatibility_pair_blocks(
    mode: str,
    face_analysis: str,
) -> tuple[Mapping[str, str], ...]:
    face_line = _first_nonempty_line(face_analysis, "두 사람의 얼굴 관찰 메모를 관계 분위기 참고 자료로 사용했습니다.")
    result = (
        {
            "category": "관계의 첫인상",
            "title": f"{mode} 관계에서 먼저 보이는 분위기",
            "summary": "두 사람의 표정 안정감과 시선 분위기를 관계의 첫 리듬으로 읽습니다.",
            "body": f"{face_line} 얼굴 관찰은 관계를 단정하는 기준이 아니라 대화 분위기를 구체화하는 보조 자료입니다.",
        },
        {
            "category": "소통 리듬",
            "title": "말을 주고받을 때 맞춰야 할 속도",
            "summary": "시선과 표정의 안정감은 대화의 긴장도를 낮추는 방향으로 해석합니다.",
            "body": "서로 반응 속도가 다를 수 있으므로 중요한 이야기는 한 번에 몰아가기보다 짧게 확인하며 이어가는 편이 좋습니다. 먼저 분위기를 부드럽게 만든 뒤 핵심을 말하면 관계의 마찰이 줄어듭니다.",
        },
        {
            "category": "관계 강점",
            "title": "서로에게 편안함을 주는 지점",
            "summary": "얼굴 관찰 메모에서 안정적으로 반복되는 인상을 관계 강점으로 연결합니다.",
            "body": "한쪽이 방향을 잡고 다른 한쪽이 분위기를 조율하면 관계가 더 부드럽게 흐릅니다. 서로의 표현 방식이 다르다는 점을 인정하면 장점이 더 잘 살아납니다.",
        },
        {
            "category": "주의할 점",
            "title": "오해가 생기기 쉬운 순간",
            "summary": "표정이나 말투를 바로 결론으로 해석하지 않는 것이 중요합니다.",
            "body": "상대의 반응이 느리거나 조용해 보일 때 관심이 없다고 단정하지 않는 편이 좋습니다. 확인 질문을 짧게 던지고 답을 기다리는 방식이 관계 안정에 도움이 됩니다.",
        },
    )
    return result


def _default_compatibility_saju_blocks(
    mode: str,
    left_manse: ManseLookupResult,
    right_manse: ManseLookupResult,
) -> tuple[Mapping[str, str], ...]:
    left_summary = " ".join(left_manse.reading.summary_lines)
    right_summary = " ".join(right_manse.reading.summary_lines)
    result = (
        {
            "category": "관계 구조",
            "title": "두 사람의 기본 기운",
            "summary": "각자의 일간과 오행 분포를 비교해 관계의 큰 흐름을 봅니다.",
            "body": f"첫 번째 사람은 {left_summary} 두 번째 사람은 {right_summary} 이 차이는 {mode} 관계에서 서로가 어떤 속도로 움직이는지 참고하는 기준이 됩니다.",
        },
        {
            "category": "상호 보완",
            "title": "강한 기운과 부족한 기운의 맞물림",
            "summary": "한쪽의 강점이 다른 쪽의 부족한 리듬을 보완할 수 있는지 살핍니다.",
            "body": "사주 데이터는 좋고 나쁨을 가르는 기준이 아니라 관계에서 어느 부분을 의식적으로 맞추면 좋은지 보여주는 지도에 가깝습니다. 강한 기운은 추진력으로 살리고 약한 기운은 생활 습관과 대화 방식으로 보완하는 편이 좋습니다.",
        },
        {
            "category": "갈등 관리",
            "title": "관계가 흔들릴 때 먼저 볼 지점",
            "summary": "속도 차이와 표현 방식 차이가 갈등의 출발점이 될 수 있습니다.",
            "body": "두 사람의 리듬이 다를 때는 누가 옳은지보다 어떤 방식으로 조율할지에 초점을 맞추는 편이 좋습니다. 특히 감정이 올라온 순간에는 결론보다 상황 정리가 먼저입니다.",
        },
        {
            "category": "현재 관계 흐름",
            "title": f"{mode} 관계에서 지금 점검할 리듬",
            "summary": "현재의 상호작용 패턴을 보고 장점이 살아나는 지점을 찾습니다.",
            "body": "관계의 흐름은 한 번에 정해지는 결론이 아니라 반복되는 반응과 선택이 쌓여 만들어집니다. 서로에게 편안한 속도와 부담스러운 순간을 구분하면 같은 기운도 훨씬 부드럽게 쓰일 수 있습니다.",
        },
        {
            "category": "실천 제안",
            "title": f"{mode} 관계를 오래 좋게 만드는 습관",
            "summary": "반복 가능한 행동 제안으로 관계 운용 방식을 정리합니다.",
            "body": "서로에게 기대하는 반응을 말로 확인하고, 중요한 결정은 하루 정도 간격을 두고 다시 보는 습관이 좋습니다. 작은 약속을 지키는 경험이 쌓이면 관계의 신뢰감도 안정됩니다.",
        },
        {
            "category": "총평 및 조언",
            "title": "서로의 차이를 관계의 언어로 바꾸는 법",
            "summary": "궁합 해석을 단정이 아니라 조율의 참고점으로 정리합니다.",
            "body": "두 사람의 사주는 관계가 좋다 나쁘다를 가르는 판정표가 아니라 서로 다른 리듬을 이해하는 참고 지도에 가깝습니다. 강점은 더 자주 쓰고 부담이 커지는 지점은 작게 조율하면 관계의 안정감이 오래 유지됩니다.",
        },
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
        normalized_value = _normalize_inline_text(value)
        if not _looks_like_generation_error(normalized_value):
            result = normalized_value
    return result


def _looks_like_generation_error(text: str) -> bool:
    normalized_text = _normalize_inline_text(text)
    failed_markers = (
        "분석 오류",
        "분석오류",
        "연산 실패",
        "연산실패",
        "생성하지 못했습니다",
        "생성하지 못했",
    )
    result = any(marker in normalized_text for marker in failed_markers)
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
            if index >= len(defaults):
                break
            default = defaults[min(index, len(defaults) - 1)]
            if isinstance(raw_block, dict):
                blocks.append(
                    _ReportBlock(
                        category=_payload_text(
                            raw_block,
                            "category",
                            default["category"],
                        ),
                        title=_payload_text(raw_block, "title", default["title"]),
                        summary=_payload_text(
                            raw_block,
                            "summary",
                            default["summary"],
                        ),
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
    result = tuple(blocks)
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


def _payload_compatibility_score(
    payload: Mapping[str, Any],
    mode: str,
) -> _CompatibilityScoreView:
    total = _payload_int(payload, "compatibility_score", 75, 65, 96)
    saju = _payload_int(payload, "compatibility_saju_score", total, 60, 96)
    face = _payload_int(payload, "compatibility_face_score", total, 60, 96)
    mode_bonus = _payload_int(payload, "compatibility_mode_bonus", 5, 0, 10)
    result = _CompatibilityScoreView(
        total=total,
        saju=saju,
        face=face,
        mode_bonus=mode_bonus,
        label=_payload_text(
            payload,
            "compatibility_score_label",
            _fallback_compatibility_score_label(total),
        ),
        summary=_payload_text(
            payload,
            "compatibility_score_summary",
            _fallback_compatibility_score_summary(mode),
        ),
    )
    return result


def _payload_int(
    payload: Mapping[str, Any],
    key: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = payload.get(key)
    result = default
    if isinstance(value, bool):
        result = default
    elif isinstance(value, (int, float)):
        result = round(value)
    elif isinstance(value, str):
        try:
            result = round(float(value.strip()))
        except ValueError:
            result = default
    result = max(minimum, min(result, maximum))
    return result


def _fallback_compatibility_score_label(total: int) -> str:
    if total >= 91:
        result = "강한 시너지형"
    elif total >= 83:
        result = "찰떡 보완형"
    elif total >= 74:
        result = "편안한 조율형"
    else:
        result = "천천히 맞춰가는 조합"
    return result


def _fallback_compatibility_score_summary(mode: str) -> str:
    if mode == "연인":
        result = "서로의 속도와 온도를 맞추면 설렘이 더 오래 살아나는 조합이에요."
    elif mode == "직장동료":
        result = "역할과 마감 기준을 잘 나누면 업무 시너지가 살아나는 콤비예요."
    else:
        result = "취향과 대화 리듬을 맞추면 편안하게 오래 가기 좋은 친구 조합이에요."
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
    chart = reading.chart
    counts_text = _element_counts_text(reading.element_counts)
    day_element = STEM_ELEMENTS[chart.day.stem_index]
    day_master = f"{chart.day.stem}{day_element}"
    pillar_text = _pillar_label_text(reading)
    summary = " ".join(reading.summary_lines)
    interpretation = _compact_text(reading.interpretation)
    strongest = _strongest_element(reading.element_counts)
    weakest = _weakest_element(reading.element_counts)
    defaults = (
        {
            **_DEFAULT_SAJU_BLOCKS[0],
            "summary": f"{pillar_text}을 기준으로 전체 기운의 흐름을 봅니다.",
            "body": f"{summary} {interpretation}",
        },
        {
            **_DEFAULT_SAJU_BLOCKS[1],
            "summary": f"일간 {day_master}을 중심으로 내면의 작동 방식을 읽습니다.",
            "body": f"{day_master} 일간은 리포트에서 자신을 상징하는 기준점입니다. 오행 분포는 {counts_text}이며, 가장 강한 기운은 {strongest}, 보완하면 좋은 기운은 {weakest}입니다. 강한 기운은 장점으로 쓰되 부족한 기운이 만드는 빈틈은 생활 리듬에서 의식적으로 보완하는 편이 좋습니다.",
        },
        {
            **_DEFAULT_SAJU_BLOCKS[2],
            "summary": f"{strongest} 기운의 강점과 {weakest} 기운의 보완점을 일과 역할에 연결합니다.",
            "body": f"{strongest} 기운이 강하게 잡히면 익숙한 방식에서는 추진력이나 안정감이 잘 드러납니다. 다만 {weakest} 기운이 약하면 일의 속도, 표현 방식, 회복 루틴 중 한쪽이 비기 쉬우므로 역할을 넓히기보다 균형을 먼저 맞추는 쪽이 좋습니다. 재물운은 결과 예측이 아니라 돈과 일을 다루는 태도에 대한 참고 조언으로만 봅니다.",
        },
        {
            **_DEFAULT_SAJU_BLOCKS[3],
            "summary": "강한 기운과 약한 기운의 차이가 관계 표현 방식에도 드러날 수 있습니다.",
            "body": f"관계에서는 {strongest} 기운이 장점으로 보일 때 자신만의 리듬과 기준이 분명하게 느껴질 수 있습니다. 반대로 {weakest} 기운이 부족하면 상대의 속도나 감정 표현을 세밀하게 맞추는 데 시간이 걸릴 수 있습니다. 중요한 관계일수록 단정적인 판단보다 확인하는 대화와 반복 가능한 약속이 균형을 잡아 줍니다.",
        },
        {
            **_DEFAULT_SAJU_BLOCKS[4],
            "summary": f"올해는 {strongest} 기운을 과하게 밀기보다 {weakest} 기운을 보완하는 쪽이 핵심입니다.",
            "body": f"올해의 조언은 특정 사건을 예언하기보다 현재 명식의 균형을 생활에 적용하는 방식으로 보는 것이 좋습니다. 이미 강한 {strongest} 기운을 더 몰아붙이기보다, 약한 {weakest} 기운을 보완하는 휴식, 정리, 표현, 관계 습관을 하나씩 만드는 편이 안정적입니다. 큰 결정보다는 반복 가능한 작은 루틴이 흐름을 바꿉니다.",
        },
        {
            **_DEFAULT_SAJU_BLOCKS[5],
            "summary": "사주 데이터는 강점과 보완점을 함께 보여주는 참고 지도입니다.",
            "body": f"{summary} 이 리포트의 핵심은 강한 {strongest} 기운을 억누르는 것이 아니라 잘 쓰는 법을 찾고, 약한 {weakest} 기운은 일상에서 조금씩 채우는 데 있습니다. 사주는 결론을 정하는 도구가 아니라 자신을 점검하는 언어로 활용할 때 가장 현실적으로 도움이 됩니다.",
        },
    )
    result = tuple(defaults)
    return result


def _element_counts_text(counts: Mapping[str, int]) -> str:
    result = ", ".join(f"{element} {counts[element]}" for element in ELEMENTS)
    return result


def _pillar_label_text(reading) -> str:
    chart = reading.chart
    result = (
        f"년주 {chart.year.label}, 월주 {chart.month.label}, "
        f"일주 {chart.day.label}, 시주 {chart.hour.label}"
    )
    return result


def _compact_text(text: str) -> str:
    result = " ".join(line.strip() for line in text.splitlines() if line.strip())
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
    if view.skip_face:
        result = _render_saju_only_report_body(view)
        return result
    result = _render_cute_personal_report_body(view)
    return result


def _render_cute_personal_report_body(view: _PersonalReportView) -> str:
    result = f"""
<div class="oracle-report saju-only-report cute-personal-report">
<div class="saju-wrap">
  <div class="saju-page-nav">
    <a class="saju-round" href="/personal" aria-label="입력 화면으로 돌아가기">‹</a>
    <a class="saju-round" href="/" aria-label="처음으로">♡</a>
  </div>

  <header class="saju-hero fade">
    <div class="saju-brand"><span>✧</span>ORACLE<span>✧</span></div>
    <div class="saju-brand-sub">관상 &amp; 사주 리포트</div>
    <div class="saju-brand-line">※</div>
    <span class="saju-cloud saju-cloud-left" aria-hidden="true"></span>
    <span class="saju-spark saju-spark-left" aria-hidden="true">✧</span>
    <span class="saju-spark saju-spark-right" aria-hidden="true">✧</span>
  </header>

  <section class="saju-profile-card fade">
    <div class="saju-day-mark {escape(view.day_master_class)}">
      <span class="saju-day-hanja">{escape(view.day_master_hanja)}</span>
      <span class="saju-day-label">{escape(view.day_master_label)}</span>
    </div>
    <div class="saju-profile-copy">
      <h1>{escape(view.name)} 님</h1>
      <div class="saju-meta">♧ {escape(view.meta)}</div>
      <p>{escape(view.essence)}</p>
    </div>
    <img class="saju-hero-ora" src="/static/assets/oracle-character.png" alt="" aria-hidden="true">
    <span class="saju-floating-heart one" aria-hidden="true">♡</span>
    <span class="saju-floating-heart two" aria-hidden="true">♡</span>
    <span class="saju-floating-heart three" aria-hidden="true">♡</span>
  </section>

  {_render_saju_only_element_cards(view)}
  {_render_cute_personal_keywords(view)}
  {_render_cute_face_blocks(view)}
  {_render_saju_only_blocks(view)}

  <section class="saju-disclaimer fade">
    <div class="saju-disclaimer-icon">💡</div>
    <p>{escape(view.disclaimer)}</p>
    <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
  </section>
</div>
</div>
"""
    return result


def _render_saju_only_report_body(view: _PersonalReportView) -> str:
    result = f"""
<div class="oracle-report saju-only-report">
<div class="saju-wrap">
  <div class="saju-page-nav">
    <a class="saju-round" href="/personal" aria-label="입력 화면으로 돌아가기">‹</a>
    <a class="saju-round" href="/" aria-label="처음으로">♡</a>
  </div>

  <header class="saju-hero fade">
    <div class="saju-brand"><span>✧</span>ORACLE<span>✧</span></div>
    <div class="saju-brand-sub">사주 리포트</div>
    <div class="saju-brand-line">※</div>
    <span class="saju-cloud saju-cloud-left" aria-hidden="true"></span>
    <span class="saju-spark saju-spark-left" aria-hidden="true">✧</span>
    <span class="saju-spark saju-spark-right" aria-hidden="true">✧</span>
  </header>

  <section class="saju-profile-card fade">
    <div class="saju-day-mark {escape(view.day_master_class)}">
      <span class="saju-day-hanja">{escape(view.day_master_hanja)}</span>
      <span class="saju-day-label">{escape(view.day_master_label)}</span>
    </div>
    <div class="saju-profile-copy">
      <h1>{escape(view.name)} 님</h1>
      <div class="saju-meta">♧ {escape(view.meta)}</div>
      <p>{escape(view.essence)}</p>
    </div>
    <img class="saju-hero-ora" src="/static/assets/oracle-character.png" alt="" aria-hidden="true">
    <span class="saju-floating-heart one" aria-hidden="true">♡</span>
    <span class="saju-floating-heart two" aria-hidden="true">♡</span>
    <span class="saju-floating-heart three" aria-hidden="true">♡</span>
  </section>

  {_render_saju_only_element_cards(view)}
  {_render_cute_personal_keywords(view)}
  {_render_saju_only_blocks(view)}

  <section class="saju-disclaimer fade">
    <div class="saju-disclaimer-icon">💡</div>
    <p>{escape(view.disclaimer)}</p>
    <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
  </section>
</div>
</div>
"""
    return result


def _render_compatibility_report_body(view: _CompatibilityReportView) -> str:
    result = f"""
<div class="oracle-report saju-only-report cute-compatibility-report">
<div class="saju-wrap compat-cute-wrap">
  <div class="saju-page-nav">
    <a class="saju-round" href="/compatibility" aria-label="입력 화면으로 돌아가기">‹</a>
    <a class="saju-round" href="/" aria-label="처음으로">♡</a>
  </div>

  <header class="saju-hero compat-cute-hero fade">
    <div class="saju-brand"><span>✧</span>ORACLE<span>✧</span></div>
    <div class="saju-brand-sub">우리 궁합 리포트</div>
    <div class="saju-brand-line">×</div>
    <span class="saju-cloud saju-cloud-left" aria-hidden="true"></span>
    <span class="saju-spark saju-spark-left" aria-hidden="true">✧</span>
    <span class="saju-spark saju-spark-right" aria-hidden="true">✧</span>
  </header>

  <section class="compat-hero-card fade">
    <img class="compat-hero-left" src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
    <div class="compat-hero-main">
      <div class="compat-day-pair">
        {_render_cute_compat_mark(view.left)}
        <span aria-hidden="true">×</span>
        {_render_cute_compat_mark(view.right)}
      </div>
      <h1>{escape(view.left.name)} 님과 {escape(view.right.name)} 님</h1>
      <div class="compat-mode">{escape(view.mode)} 궁합 리포트</div>
      <p>{escape(_short_report_text(view.essence, max_sentences=2, max_chars=190))}</p>
    </div>
    <img class="compat-hero-right" src="/static/assets/oracle-character.png" alt="" aria-hidden="true">
    <span class="compat-heart-float one" aria-hidden="true">♡</span>
    <span class="compat-heart-float two" aria-hidden="true">♡</span>
  </section>

  <section class="compat-profile-card fade">
    {_render_cute_compat_profile(view.left, "첫 번째 사람")}
    {_render_cute_compat_score_heart(view)}
    {_render_cute_compat_profile(view.right, "두 번째 사람")}
  </section>

  {_render_cute_compat_keywords(view)}
  {_render_cute_compat_section("01", "두 사람의 관계 분위기", view.pair_subtitle, view.pair_blocks, "/static/assets/oracle-pair-card.png")}
  {_render_cute_compat_section("02", "사주로 보는 상호 보완", view.saju_subtitle, view.saju_blocks, "/static/assets/oracle-character.png")}
  

  <section class="saju-disclaimer compat-disclaimer fade">
    <div class="saju-disclaimer-icon">💡</div>
    <p>{escape(view.disclaimer)}</p>
    <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
  </section>
</div>
</div>
"""
    return result


def _render_cute_compat_mark(person: _CompatibilityPersonView) -> str:
    result = f"""
      <span class="saju-day-mark compat-day-mark {escape(person.day_master_class)}">
        <span class="saju-day-hanja">{escape(person.day_master_hanja)}</span>
        <span class="saju-day-label">{escape(person.day_master_label)}</span>
      </span>
"""
    return result


def _render_cute_compat_profile(person: _CompatibilityPersonView, label: str) -> str:
    result = f"""
    <article class="compat-person-cute">
      <div class="compat-person-label">♡ {escape(label)} ♡</div>
      <h2>{escape(person.name)} 님</h2>
      <div class="compat-person-meta">{escape(person.meta)}</div>
      {_render_cute_compat_mark(person)}
      <p>{escape(person.element_note)}</p>
      <img src="/static/assets/oracle-solo-card.png" alt="" aria-hidden="true">
    </article>
"""
    return result


def _render_cute_compat_section(
    number: str,
    title: str,
    subtitle: str,
    blocks: tuple[_ReportBlock, ...],
    image: str,
) -> str:
    block_html = "\n".join(
        _render_cute_compat_block(block, index)
        for index, block in enumerate(blocks)
    )
    result = f"""
  <section class="compat-cute-section fade">
    <div class="compat-section-head">
      <h2><span>{escape(number)}</span>{escape(title)} <b aria-hidden="true">♡</b></h2>
      <p>{escape(subtitle)}</p>
    </div>
    <div class="compat-section-body">
      <img class="compat-section-ora" src="{image}" alt="" aria-hidden="true">
      <div class="compat-block-list">
        {block_html}
      </div>
    </div>
  </section>
"""
    return result


def _render_cute_compat_block(block: _ReportBlock, index: int) -> str:
    icons = ("♡", "✧", "☘", "☆")
    icon = icons[index % len(icons)]
    result = f"""
        <article class="compat-cute-block">
          <span class="compat-block-icon" aria-hidden="true">{icon}</span>
          <div>
            <div class="saju-block-cat">{escape(block.category)}</div>
            <h3>{escape(block.title)}</h3>
            <p class="saju-block-summary">{escape(_short_report_text(block.summary, max_sentences=1, max_chars=95))}</p>
            <p>{_paragraphs(_short_report_text(block.body, max_sentences=3, max_chars=260))}</p>
          </div>
        </article>
"""
    return result





def _render_cute_compat_score_heart(view: _CompatibilityReportView) -> str:
    score = view.compatibility_score
    result = f"""
    <div class="compat-score-heart-card">
      <div class="compat-score-title"><span aria-hidden="true">♡</span>궁합 점수<span aria-hidden="true">♡</span></div>
      <div class="compat-score-heart-shape" aria-label="궁합 점수 {score.total}점">
        <div class="compat-score-number">{score.total}</div>
        <div class="compat-score-denominator">/100</div>
      </div>
      <div class="compat-score-one-line">
        <strong>한 줄 평가</strong>
        <p>{escape(_short_report_text(score.summary, max_sentences=1, max_chars=95))}</p>
      </div>
    </div>
"""
    return result


def _render_cute_compat_keywords(view: _CompatibilityReportView) -> str:
    chips = "".join(f'<span class="saju-keyword">{escape(tag)}</span>' for tag in view.tags)
    result = f"""
  <section class="saju-card saju-summary-card cute-keyword-card compat-summary-cute fade">
    <div class="saju-keyword-band">
      <h3>♡ 관계를 채워주는 키워드 ♡</h3>
      <div>{chips}</div>
    </div>
  </section>
"""
    return result


def _render_person_mark(person: _CompatibilityPersonView) -> str:
    result = f"""
      <span class="person-mark {escape(person.day_master_class)}">
        <span class="person-hanja">{escape(person.day_master_hanja)}</span>
        <span class="person-ko">{escape(person.day_master_label)}</span>
      </span>
"""
    return result


def _render_pair_profiles(view: _CompatibilityReportView) -> str:
    result = f"""
  <section class="pair-profiles fade">
    {_render_pair_profile(view.left, "첫 번째 사람")}
    {_render_pair_profile(view.right, "두 번째 사람")}
  </section>
"""
    return result


def _render_pair_profile(person: _CompatibilityPersonView, label: str) -> str:
    result = f"""
    <div class="person-card">
      <div class="b-cat">{escape(label)}</div>
      <div class="person-name serif">{escape(person.name)} 님</div>
      <div class="person-meta">{escape(person.meta)}</div>
      <div class="person-day {escape(person.day_master_class)}">{escape(person.day_master_hanja)}</div>
      <div class="person-label">{escape(person.day_master_label)}</div>
      <p>{escape(person.element_note)}</p>
    </div>
"""
    return result


def _render_action_panel(view: _CompatibilityReportView) -> str:
    result = f"""
  <section class="action-panel fade">
    <div class="b-cat">ACTION GUIDE</div>
    <div class="b-title serif">{escape(view.action_title)}</div>
    <div class="b-body">{_paragraphs(view.action_body)}</div>
  </section>
"""
    return result


def _render_saju_only_element_cards(view: _PersonalReportView) -> str:
    icons = {
        "목": "🌳",
        "화": "🔥",
        "토": "⛰",
        "금": "🪨",
        "수": "💧",
    }
    cards = []
    for element in ELEMENTS:
        count = view.element_counts[element]
        icon_marks = _element_icon_marks(icons[element], count)
        cards.append(
            f"""
      <div class="saju-element-card {escape(_ELEMENT_CLASS[element])}">
        <div class="saju-element-icon">{escape(icon_marks)}</div>
        <div class="saju-element-hanja">{_ELEMENT_HANJA[element]}</div>
        <div class="saju-element-name">{escape(element)}</div>
        <div class="saju-element-count">{count}</div>
      </div>
""",
        )
    result = f"""
  <section class="saju-card saju-elements fade">
    <div class="saju-section-head">
      <h2><span aria-hidden="true">✿</span>오행의 균형</h2>
      <p>사주에 나타난 오행의 분포를 통해 균형을 살펴봐요.</p>
    </div>
    <div class="saju-element-grid">
      {''.join(cards)}
    </div>
    <div class="saju-element-note">
      <img src="/static/assets/oracle-pair-card.png" alt="" aria-hidden="true">
      <p>{escape(view.element_note)}</p>
      <span aria-hidden="true">♡</span>
    </div>
  </section>
"""
    return result


def _element_icon_marks(icon: str, count: int) -> str:
    result = " " if count <= 0 else icon * count
    return result


def _render_saju_only_blocks(view: _PersonalReportView) -> str:
    ora_images = (
        "/static/assets/oracle-character.png",
        "/static/assets/oracle-solo-card.png",
        "/static/assets/oracle-pair-card.png",
    )
    blocks = []
    for index, block in enumerate(view.saju_blocks):
        image = ora_images[index % len(ora_images)]
        blocks.append(
            f"""
      <article class="saju-story-block">
        <img src="{image}" alt="" aria-hidden="true">
        <div>
          <div class="saju-block-cat">{escape(block.category)}</div>
          <h3>{escape(block.title)}</h3>
          <p class="saju-block-summary">{escape(block.summary)}</p>
          <p>{_paragraphs(block.body)}</p>
        </div>
      </article>
""",
        )
    result = f"""
  <section class="saju-card saju-story fade">
    <div class="saju-section-head">
      <h2><span aria-hidden="true">✿</span>사주 - 타고난 기운의 설계도</h2>
      <p>{escape(view.saju_subtitle)}</p>
    </div>
    <div class="saju-story-list">
      {''.join(blocks)}
    </div>
  </section>
"""
    return result


def _render_cute_face_blocks(view: _PersonalReportView) -> str:
    ora_images = (
        "/static/assets/oracle-solo-card.png",
        "/static/assets/oracle-character.png",
        "/static/assets/oracle-pair-card.png",
    )
    blocks = []
    for index, block in enumerate(view.face_blocks):
        image = ora_images[index % len(ora_images)]
        blocks.append(
            f"""
      <article class="saju-story-block cute-face-block">
        <img src="{image}" alt="" aria-hidden="true">
        <div>
          <div class="saju-block-cat">{escape(block.category)}</div>
          <h3>{escape(block.title)}</h3>
          <p class="saju-block-summary">{escape(block.summary)}</p>
          <p>{_paragraphs(block.body)}</p>
        </div>
      </article>
""",
        )
    result = f"""
  <section class="saju-card saju-story cute-face-story fade">
    <div class="saju-section-head">
      <h2><span aria-hidden="true">♡</span>관상 - 얼굴이 말하는 인상</h2>
      <p>{escape(view.face_subtitle)}</p>
    </div>
    <div class="saju-story-list">
      {''.join(blocks)}
    </div>
  </section>
"""
    return result


def _render_cute_personal_keywords(view: _PersonalReportView) -> str:
    chips = "".join(f'<span class="saju-keyword">{escape(tag)}</span>' for tag in view.tags)
    result = f"""
  <section class="saju-card saju-summary-card cute-total-card cute-keyword-card fade">
    <div class="saju-keyword-band">
      <h3>✧ 나를 채워주는 키워드 ✧</h3>
      <div>{chips}</div>
    </div>
  </section>
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


def _render_tags(tags: tuple[str, ...]) -> str:
    chips = "".join(f'<span class="chip">{escape(tag)}</span>' for tag in tags)
    result = f"""
  <section class="tags fade">
    <h3>나 를 채 워 주 는 키 워 드</h3>
    {chips}
  </section>
"""
    return result


def _short_report_text(
    text: str,
    *,
    max_sentences: int,
    max_chars: int,
) -> str:
    normalized_text = _normalize_inline_text(text)
    if len(normalized_text) <= max_chars:
        result = normalized_text
    else:
        sentences = re.findall(r"[^.!?。！？]+[.!?。！？]?", normalized_text)
        selected: list[str] = []
        current_length = 0
        for sentence in sentences:
            clean_sentence = sentence.strip()
            if clean_sentence == "":
                continue
            next_length = current_length + len(clean_sentence)
            if selected and (
                len(selected) >= max_sentences or next_length > max_chars
            ):
                break
            selected.append(clean_sentence)
            current_length = next_length
        if selected:
            result = " ".join(selected)
        else:
            result = normalized_text[:max_chars].rstrip()
        if len(result) > max_chars:
            result = result[:max_chars].rstrip()
        sentence_endings = (".", "!", "?", "。", "！", "？", "요", "다")
        if result and not result.endswith(sentence_endings):
            result = f"{result}..."
    return result


def _paragraphs(text: str) -> str:
    normalized_text = _normalize_inline_text(text)
    result = escape(normalized_text)
    return result


def _normalize_inline_text(text: str) -> str:
    normalized_text = text.replace("\\r\\n", " ")
    normalized_text = normalized_text.replace("\\n", " ")
    normalized_text = normalized_text.replace("\\r", " ")
    normalized_text = normalized_text.replace("\r\n", " ")
    normalized_text = normalized_text.replace("\n", " ")
    normalized_text = normalized_text.replace("\r", " ")
    result = " ".join(normalized_text.split())
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
.saju-only-report{--cute-paper:#fff8ef;--cute-card:rgba(255,255,255,.82);--cute-line:#ffd8ce;--cute-line-soft:#ffe9e3;--cute-pink:#ff6f82;--cute-pink-soft:#fff2f6;--cute-gold:#f5b45f;--cute-ink:#3f211b;--cute-muted:#7e6259;background:linear-gradient(180deg,#fffaf4 0%,#fff4ec 58%,#ffeef1 100%);color:var(--cute-ink);padding:0 0 44px;font-family:"Gowun Dodum",sans-serif}
.saju-wrap{position:relative;max-width:1120px;margin:0 auto;padding:22px 32px 34px}
.saju-page-nav{position:absolute;top:24px;left:0;right:0;z-index:2;display:flex;justify-content:space-between;pointer-events:none}
.saju-round{width:48px;height:48px;display:inline-flex;align-items:center;justify-content:center;border:1px solid var(--cute-line);border-radius:999px;background:rgba(255,255,255,.72);color:#5a332a;font-size:36px;line-height:1;text-decoration:none;box-shadow:0 12px 30px -24px rgba(74,47,38,.45);pointer-events:auto}
.saju-round:last-child{color:#ffb1c0;font-size:32px;transform:rotate(-16deg)}
.saju-hero{position:relative;padding:0 0 28px;text-align:center;border-bottom:0}
.saju-brand{display:inline-flex;align-items:center;gap:22px;font-family:"Song Myung",serif;font-size:42px;line-height:1;color:var(--cute-ink)}
.saju-brand span{color:var(--cute-gold);font-family:"Gowun Dodum",sans-serif;font-size:26px}
.saju-brand-sub{margin-top:14px;color:#8f5d3e;font-family:"Gowun Batang",serif;font-size:16px}
.saju-brand-line{position:relative;display:inline-block;margin-top:12px;color:var(--cute-gold);font-size:14px}
.saju-brand-line::before,.saju-brand-line::after{content:"";position:absolute;top:50%;width:18px;height:1px;background:var(--cute-gold)}
.saju-brand-line::before{right:24px}.saju-brand-line::after{left:24px}
.saju-cloud{position:absolute;left:156px;top:58px;width:74px;height:30px;border:2px solid #ffd7ce;border-top:0;border-radius:0 0 24px 24px;opacity:.92}
.saju-cloud::before,.saju-cloud::after{content:"";position:absolute;bottom:10px;border:2px solid #ffd7ce;border-bottom:0;background:#fff8ef}
.saju-cloud::before{left:13px;width:28px;height:28px;border-radius:999px 999px 0 0}.saju-cloud::after{right:8px;width:38px;height:38px;border-radius:999px 999px 0 0}
.saju-spark{position:absolute;color:var(--cute-gold);font-size:24px}.saju-spark-left{left:46px;top:70px}.saju-spark-right{right:56px;top:102px}
.saju-profile-card,.saju-card,.saju-disclaimer{position:relative;border:1px solid var(--cute-line);border-radius:14px;background:var(--cute-card);box-shadow:0 18px 42px -34px rgba(74,47,38,.42);overflow:hidden}
.saju-profile-card{min-height:310px;display:grid;grid-template-columns:230px 1fr 230px;align-items:center;gap:24px;padding:34px 42px;margin-top:0;background:radial-gradient(circle at 88% 24%,rgba(255,239,242,.88),transparent 30%),rgba(255,255,255,.8)}
.saju-day-mark{width:172px;height:172px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:2px solid #71879c;border-radius:999px;background:rgba(255,255,255,.62);color:#111;justify-self:center}
.saju-day-mark.c-mok,.saju-day-mark.c-hwa,.saju-day-mark.c-to,.saju-day-mark.c-geum,.saju-day-mark.c-su{background:rgba(255,255,255,.62);color:#111}
.saju-day-hanja{font-family:"Song Myung",serif;font-size:90px;line-height:.95}.saju-day-label{margin-top:8px;color:#6f7378;font-family:"Gowun Batang",serif;font-size:15px}
.saju-profile-copy h1{font-family:"Gowun Batang",serif;font-size:38px;line-height:1.2;color:var(--cute-ink)}
.saju-meta{margin-top:18px;color:var(--cute-muted);font-family:"Gowun Batang",serif;font-size:15px}.saju-profile-copy p{margin-top:34px;color:#5f504b;font-family:"Gowun Batang",serif;font-size:17px;line-height:1.75;text-align:center}
.saju-hero-ora{width:220px;height:220px;object-fit:contain;align-self:start;justify-self:center;margin-top:-24px}
.saju-floating-heart{position:absolute;color:#ffb1c0;font-size:34px}.saju-floating-heart.one{right:260px;top:94px}.saju-floating-heart.two{right:104px;top:158px}.saju-floating-heart.three{right:64px;bottom:82px}
.saju-card{margin-top:22px;padding:34px 40px;background:rgba(255,255,255,.8)}
.saju-section-head{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:26px}.saju-section-head h2{font-family:"Gowun Batang",serif;font-size:24px;color:var(--cute-ink)}.saju-section-head h2 span{margin-right:12px;color:#ff8fab}.saju-section-head p{color:#7e6259;font-size:14px}
.saju-element-grid{display:grid;grid-template-columns:repeat(5,1fr);gap:16px}.saju-element-card{min-height:186px;display:flex;flex-direction:column;align-items:center;justify-content:center;border:1px solid var(--cute-line);border-radius:14px;background:rgba(255,255,255,.74);box-shadow:inset 0 0 0 1px rgba(255,255,255,.58)}
.saju-element-card.c-mok{background:linear-gradient(180deg,rgba(58,125,92,.08),rgba(255,255,255,.82))}.saju-element-card.c-hwa{background:linear-gradient(180deg,rgba(255,111,130,.09),rgba(255,255,255,.82))}.saju-element-card.c-to{background:linear-gradient(180deg,rgba(245,180,95,.12),rgba(255,255,255,.82))}.saju-element-card.c-geum{background:linear-gradient(180deg,rgba(154,149,139,.09),rgba(255,255,255,.82))}.saju-element-card.c-su{background:linear-gradient(180deg,rgba(94,166,217,.1),rgba(255,255,255,.82))}
.saju-element-icon{min-height:48px;max-width:112px;display:flex;align-items:center;justify-content:center;flex-wrap:wrap;gap:2px;font-size:34px;line-height:1.05;text-align:center}.saju-element-hanja{margin-top:12px;font-family:"Song Myung",serif;font-size:58px;line-height:1}.saju-element-name{margin-top:10px;color:#6b544d;font-family:"Gowun Batang",serif;font-size:17px}.saju-element-count{margin-top:4px;color:#4b3933;font-family:"Song Myung",serif;font-size:27px}
.saju-element-card.c-mok .saju-element-hanja{color:var(--mok)}.saju-element-card.c-hwa .saju-element-hanja{color:var(--hwa)}.saju-element-card.c-to .saju-element-hanja{color:var(--to)}.saju-element-card.c-geum .saju-element-hanja{color:#555}.saju-element-card.c-su .saju-element-hanja{color:#4e83ad}
.saju-element-note{position:relative;min-height:96px;display:grid;grid-template-columns:104px 1fr 36px;align-items:center;gap:16px;margin-top:30px;padding:16px 28px;border:1px solid var(--cute-line-soft);border-radius:16px;background:linear-gradient(90deg,#fff5f7,#fffafa);text-align:center}.saju-element-note img{width:92px;height:92px;object-fit:contain}.saju-element-note p{color:#5e4a43;font-family:"Gowun Batang",serif;font-size:15px;line-height:1.7}.saju-element-note span{color:#ffb1c0;font-size:28px}
.saju-story-list{display:grid;gap:16px}.saju-story-block{display:grid;grid-template-columns:112px 1fr;gap:22px;align-items:center;min-height:146px;padding:18px 22px;border:1px solid var(--cute-line-soft);border-radius:12px;background:rgba(255,250,250,.76)}.saju-story-block img{width:104px;height:104px;object-fit:contain}.saju-block-cat{color:#d7835b;font-family:"Gowun Batang",serif;font-size:12px;font-weight:700;letter-spacing:.04em}.saju-story-block h3{margin-top:4px;color:var(--cute-ink);font-family:"Gowun Batang",serif;font-size:21px}.saju-story-block p{margin-top:8px;color:#5f504b;font-size:14.5px;line-height:1.72}.saju-story-block .saju-block-summary{color:#d36472;font-family:"Gowun Batang",serif;font-weight:700}
.saju-summary-card{display:grid;grid-template-columns:1fr 178px;gap:20px;align-items:center;padding-bottom:0}.saju-summary-copy{text-align:center}.saju-summary-copy h2{font-family:"Gowun Batang",serif;font-size:24px;color:var(--cute-ink)}.saju-summary-copy h2 span{margin-right:10px;color:#ff8fab}.saju-summary-copy p{margin-top:18px;color:#5f504b;font-size:15px;line-height:1.75;text-align:left}.saju-summary-card>img{width:168px;height:168px;object-fit:contain;align-self:end}.saju-keyword-band{grid-column:1/-1;margin:24px -40px 0;padding:24px 34px;border-top:1px solid var(--cute-line-soft);background:rgba(255,246,248,.76);text-align:center}.saju-keyword-band h3{color:#7e6259;font-family:"Gowun Batang",serif;font-size:15px;font-weight:700}.saju-keyword-band>div{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-top:20px}.saju-keyword{min-height:54px;display:flex;align-items:center;justify-content:center;padding:10px 16px;border:1px solid var(--cute-line);border-radius:14px;background:rgba(255,255,255,.74);color:#5b3b34;font-family:"Gowun Batang",serif;font-size:15px}.cute-keyword-card{display:block;padding:30px 40px}.cute-keyword-card .saju-keyword-band{grid-column:auto;margin:0;padding:4px 0 0;border-top:0;background:transparent}.cute-keyword-card .saju-keyword-band h3{font-size:18px}.cute-keyword-card .saju-keyword-band>div{margin-top:18px}.cute-keyword-card .saju-keyword{min-height:64px;font-size:16px}
.saju-disclaimer{display:grid;grid-template-columns:90px 1fr 190px;align-items:center;gap:22px;margin-top:22px;padding:22px 34px;background:linear-gradient(90deg,#fff5f7,#fffafa)}.saju-disclaimer-icon{color:#f5b45f;font-size:42px;text-align:center}.saju-disclaimer p{color:#6b544d;font-size:15px;line-height:1.75}.saju-disclaimer img{width:180px;height:130px;object-fit:contain;justify-self:center}
.cute-personal-report .saju-brand-sub{color:#9a6647}.cute-personal-report .saju-profile-card{background:radial-gradient(circle at 88% 24%,rgba(255,239,242,.92),transparent 31%),radial-gradient(circle at 18% 58%,rgba(255,249,238,.9),transparent 34%),rgba(255,255,255,.82)}.cute-personal-report .saju-elements{margin-top:22px}.cute-face-story{background:linear-gradient(180deg,rgba(255,255,255,.84),rgba(255,250,251,.78))}.cute-face-story .saju-section-head h2 span{color:#ff8fab}.cute-face-block{background:linear-gradient(90deg,rgba(255,245,247,.86),rgba(255,255,255,.74));border-color:#ffd8df}.cute-face-block .saju-block-cat{color:#d96f83}.cute-face-block .saju-block-summary{color:#b96a7a}.cute-total-card{background:radial-gradient(circle at 92% 18%,rgba(255,238,243,.9),transparent 26%),rgba(255,255,255,.8)}
.cute-compatibility-report{background:linear-gradient(180deg,#fffaf4 0%,#fff5ef 52%,#ffeef2 100%)}.compat-cute-wrap{max-width:1120px}.compat-cute-hero .saju-brand-sub{color:#8f5d3e}.compat-cute-hero .saju-brand-line{color:#ffb1c0}.compat-hero-card{position:relative;min-height:340px;display:grid;grid-template-columns:220px 1fr 220px;align-items:center;gap:16px;padding:34px 38px;border:1px solid var(--cute-line);border-radius:14px;background:radial-gradient(circle at 15% 24%,rgba(255,244,247,.9),transparent 28%),radial-gradient(circle at 85% 28%,rgba(255,249,239,.92),transparent 30%),rgba(255,255,255,.82);box-shadow:0 18px 42px -34px rgba(74,47,38,.42);overflow:hidden}.compat-hero-left,.compat-hero-right{width:190px;height:210px;object-fit:contain;justify-self:center}.compat-hero-main{text-align:center}.compat-day-pair{display:flex;align-items:center;justify-content:center;gap:28px}.compat-day-pair>span{color:#ff9fad;font-size:46px;font-family:"Gowun Batang",serif}.compat-day-mark{width:116px;height:116px;border-width:1.5px}.compat-day-mark .saju-day-hanja{font-size:62px}.compat-day-mark .saju-day-label{font-size:12px}.compat-hero-main h1{margin-top:22px;color:var(--cute-ink);font-family:"Gowun Batang",serif;font-size:34px;line-height:1.25}.compat-mode{margin-top:10px;color:#8f5d3e;font-family:"Gowun Batang",serif;font-size:15px}.compat-hero-main p{max-width:720px;margin:22px auto 0;color:#5f504b;font-size:16px;line-height:1.8}.compat-heart-float{position:absolute;color:#ffb1c0;font-size:28px}.compat-heart-float.one{left:132px;top:46px}.compat-heart-float.two{right:146px;bottom:78px}.compat-profile-card{position:relative;display:grid;grid-template-columns:1fr 110px 1fr;align-items:center;gap:22px;margin-top:22px;padding:24px;border:1px solid var(--cute-line);border-radius:14px;background:rgba(255,255,255,.8);box-shadow:0 18px 42px -34px rgba(74,47,38,.42)}.compat-person-cute{position:relative;min-height:318px;padding:28px 28px 22px;border:1px solid var(--cute-line);border-radius:14px;background:linear-gradient(180deg,rgba(255,250,251,.86),rgba(255,255,255,.74));text-align:center;overflow:hidden}.compat-person-cute:nth-of-type(2){background:linear-gradient(180deg,rgba(248,255,250,.86),rgba(255,255,255,.74));border-color:#dcecdf}.compat-person-label{color:#ff8fab;font-family:"Gowun Batang",serif;font-size:14px;font-weight:700}.compat-person-cute h2{margin-top:16px;color:var(--cute-ink);font-family:"Gowun Batang",serif;font-size:26px}.compat-person-meta{margin-top:10px;color:#7e6259;font-size:14px}.compat-person-cute .compat-day-mark{margin:22px auto 0}.compat-person-cute p{max-width:28ch;margin:18px auto 0;color:#5f504b;font-size:14px;line-height:1.7}.compat-person-cute>img{position:absolute;right:18px;bottom:8px;width:86px;height:86px;object-fit:contain}.compat-profile-heart{width:62px;height:62px;display:flex;align-items:center;justify-content:center;justify-self:center;border:2px solid #ffb7c4;border-radius:999px;background:#fff4f7;color:#ff7890;font-size:42px;box-shadow:0 12px 28px -22px rgba(255,111,130,.8)}.compat-score-card{display:grid;grid-template-columns:260px 1fr;gap:24px;align-items:center;margin-top:22px;padding:24px 30px;border:1px solid var(--cute-line);border-radius:14px;background:linear-gradient(135deg,rgba(255,246,248,.92),rgba(255,255,255,.82));box-shadow:0 18px 42px -34px rgba(74,47,38,.42)}.compat-score-main{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:178px;border:1px solid var(--cute-line-soft);border-radius:18px;background:#fffafd;text-align:center}.compat-score-main span{color:#d36472;font-family:"Gowun Batang",serif;font-size:14px;font-weight:700}.compat-score-main strong{margin-top:6px;color:#ff7890;font-family:"Song Myung",serif;font-size:76px;line-height:1}.compat-score-main strong em{margin-left:4px;color:#7e6259;font-family:"Gowun Batang",serif;font-size:20px;font-style:normal}.compat-score-main b{margin-top:8px;color:#5b3b34;font-family:"Gowun Batang",serif;font-size:18px}.compat-score-card p{color:#5f504b;font-family:"Gowun Batang",serif;font-size:18px;line-height:1.72;text-align:center}.compat-score-breakdown{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin-top:20px}.compat-score-breakdown span{display:flex;align-items:center;justify-content:center;min-height:48px;border:1px solid var(--cute-line-soft);border-radius:999px;background:rgba(255,255,255,.74);color:#7e6259;font-family:"Gowun Batang",serif;font-size:14px}.compat-cute-section{position:relative;margin-top:22px;padding:30px 34px;border:1px solid var(--cute-line);border-radius:14px;background:rgba(255,255,255,.8);box-shadow:0 18px 42px -34px rgba(74,47,38,.42);overflow:hidden}.compat-section-head{display:flex;align-items:center;justify-content:space-between;gap:20px;margin-bottom:20px}.compat-section-head h2{display:flex;align-items:center;gap:14px;color:var(--cute-ink);font-family:"Gowun Batang",serif;font-size:27px}.compat-section-head h2 span{color:#ff7890;font-family:"Song Myung",serif;font-size:34px}.compat-section-head h2 b{color:#ffb1c0;font-family:"Gowun Dodum",sans-serif;font-size:18px}.compat-section-head p{padding:8px 16px;border-radius:999px;background:#fff2f6;color:#d36472;font-family:"Gowun Batang",serif;font-size:13px}.compat-section-body{display:grid;grid-template-columns:130px 1fr;gap:24px;align-items:start}.compat-section-ora{width:118px;height:118px;object-fit:contain}.compat-block-list{display:grid;gap:14px}.compat-cute-block{display:grid;grid-template-columns:54px 1fr;gap:16px;padding:16px 18px;border:1px solid var(--cute-line-soft);border-radius:12px;background:rgba(255,250,250,.75)}.compat-block-icon{width:48px;height:48px;display:flex;align-items:center;justify-content:center;border-radius:999px;background:#fff5f7;color:#ff8fab;font-size:26px}.compat-cute-block h3{margin-top:4px;color:var(--cute-ink);font-family:"Gowun Batang",serif;font-size:20px}.compat-cute-block p{margin-top:8px;color:#5f504b;font-size:14.5px;line-height:1.72}.compat-cute-block .saju-block-summary{color:#d36472;font-family:"Gowun Batang",serif;font-weight:700}.compat-action-cute{background:radial-gradient(circle at 86% 70%,rgba(255,238,243,.9),transparent 28%),rgba(255,255,255,.8)}.compat-action-body{display:grid;grid-template-columns:150px 1fr;gap:24px;align-items:center}.compat-action-body img{width:140px;height:130px;object-fit:contain}.compat-action-body p{color:#5f504b;font-size:15px;line-height:1.8}.compat-summary-cute{background:linear-gradient(180deg,rgba(255,246,248,.86),rgba(255,255,255,.8))}.compat-summary-cute .saju-summary-copy h2 span:last-child{margin-left:10px;margin-right:0}.compat-disclaimer{background:linear-gradient(90deg,#fff7f2,#fff5f8)}
.compat-cute-block p:not(.saju-block-summary){display:-webkit-box;-webkit-line-clamp:5;-webkit-box-orient:vertical;overflow:hidden}
.compat-profile-card{grid-template-columns:minmax(0,1fr) minmax(220px,260px) minmax(0,1fr);gap:18px;align-items:center;padding:20px}.compat-person-cute{min-height:268px;padding:22px 22px 18px}.compat-person-cute h2{margin-top:12px;font-size:24px}.compat-person-meta{margin-top:8px}.compat-person-cute .compat-day-mark{margin-top:16px}.compat-person-cute p{max-width:25ch;margin-top:14px;font-size:13.5px}.compat-person-cute>img{width:72px;height:72px}.compat-score-heart-card{align-self:center;display:flex;flex-direction:column;align-items:center;justify-content:center;width:100%;max-width:252px;min-height:218px;padding:2px 2px 0;border:0;border-radius:0;background:transparent;text-align:center;box-shadow:none}.compat-score-title{display:flex;align-items:center;justify-content:center;gap:10px;color:#ff7890;font-family:"Gowun Batang",serif;font-size:14px;font-weight:700;letter-spacing:.02em}.compat-score-title span{color:#ff9faf;font-size:14px}.compat-score-heart-shape{position:relative;width:158px;height:116px;display:flex;flex-direction:column;align-items:center;justify-content:center;margin-top:0;color:#ff7291}.compat-score-heart-shape::before{content:"♡";position:absolute;left:50%;top:50%;transform:translate(-50%,-50%) scaleX(1.08);color:#ff8fab;font-family:"Gowun Batang",serif;font-size:166px;line-height:.78;text-shadow:0 2px 0 rgba(255,143,171,.14)}.compat-score-number,.compat-score-denominator{position:relative;z-index:1}.compat-score-number{margin-top:4px;color:#e84f75;font-family:"Song Myung",serif;font-size:48px;line-height:1}.compat-score-denominator{margin-top:0;color:#e84f75;font-family:"Gowun Batang",serif;font-size:13px}.compat-score-one-line{width:100%;margin-top:6px;padding:10px 12px;border:1px solid rgba(255,196,206,.72);border-radius:14px;background:rgba(255,255,255,.48)}.compat-score-one-line strong{display:block;color:#f06682;font-family:"Gowun Batang",serif;font-size:14px}.compat-score-one-line p{margin-top:5px;color:#5f504b;font-size:12.8px;line-height:1.5}.compat-score-breakdown{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;width:100%;margin-top:10px}.compat-score-breakdown span{display:flex;align-items:center;justify-content:center;min-height:34px;border:1px solid var(--cute-line-soft);border-radius:999px;background:rgba(255,255,255,.7);color:#7e6259;font-family:"Gowun Batang",serif;font-size:12px}
header{padding:64px 0 40px;text-align:center;border-bottom:1px solid var(--line)}
.eyebrow{font-size:12px;letter-spacing:.42em;color:var(--gold);text-transform:uppercase;margin-bottom:26px}
.ilgan-wrap{margin:18px 0;display:flex;justify-content:center}
.pair-mark{display:flex;align-items:center;justify-content:center;gap:18px;margin:4px 0 18px}.pair-x{font-family:"Gowun Batang",serif;font-size:34px;color:var(--gold)}
.person-mark{display:flex;flex-direction:column;align-items:center;justify-content:center;width:116px;height:116px;border-radius:50%;color:var(--ink);background:var(--paper-2);border:2px solid currentColor}.person-mark .person-hanja{font-family:"Song Myung",serif;font-size:54px;line-height:1;color:#111}.person-mark .person-ko{font-family:"Gowun Batang",serif;font-size:12px;margin-top:6px;color:var(--ink-soft)}
.name{font-family:"Gowun Batang",serif;font-size:30px;font-weight:700;margin-top:18px}
.meta{font-size:14px;color:var(--ink-soft);margin-top:8px;letter-spacing:.04em}
.essence{font-family:"Gowun Batang",serif;font-size:19px;line-height:1.78;margin-top:24px;color:var(--ink)}
.pair-profiles{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;margin:46px 0}.person-card{background:var(--paper-2);border:1px solid var(--line);border-radius:8px;padding:24px 22px;text-align:center}.person-name{font-size:22px;font-weight:700;margin:4px 0}.person-meta{font-size:13px;color:var(--ink-soft);margin-bottom:16px}.person-day{font-family:"Song Myung",serif;font-size:48px;line-height:1;color:#111;background:var(--paper-2);border:2px solid currentColor;width:76px;height:76px;border-radius:50%;display:flex;align-items:center;justify-content:center;margin:0 auto 10px}.person-label{font-size:13px;color:var(--ink-soft);margin-bottom:12px}.person-card p{font-size:13.5px;color:var(--ink);line-height:1.65}
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
.cell{border-radius:5px;padding:12px 4px;color:#fff}.cell .ch{font-family:"Song Myung",serif;font-size:34px;line-height:1.1;color:#111}.cell .ss{font-size:11px;opacity:.92;margin-top:3px}.cell.gan{margin-bottom:6px}
.c-mok{background:var(--mok)}.c-hwa{background:var(--hwa)}.c-to{background:var(--to)}.c-geum{background:var(--geum)}.c-su{background:var(--su)}
.person-mark.c-mok,.person-day.c-mok{color:#111;background:rgba(58,125,92,.12);border-color:var(--mok)}.person-mark.c-hwa,.person-day.c-hwa{color:#111;background:rgba(194,82,57,.12);border-color:var(--hwa)}.person-mark.c-to,.person-day.c-to{color:#111;background:rgba(204,154,59,.16);border-color:var(--to)}.person-mark.c-geum,.person-day.c-geum{color:#111;background:rgba(154,149,139,.16);border-color:var(--geum)}.person-mark.c-su,.person-day.c-su{color:#111;background:rgba(46,66,88,.12);border-color:var(--su)}
.part{margin:56px 0 0}.part-head{display:flex;align-items:baseline;gap:14px;padding-bottom:14px;border-bottom:2px solid var(--ink);margin-bottom:30px}
.part-num{font-family:"Song Myung",serif;font-size:40px;line-height:1;color:var(--gold)}.part-title{font-family:"Gowun Batang",serif;font-size:24px;font-weight:700}.part-sub{font-size:13px;color:var(--ink-soft);margin-left:auto}
.block{margin-bottom:34px}.b-cat{font-size:11px;letter-spacing:.25em;color:var(--gold);text-transform:uppercase;margin-bottom:8px}.b-title{font-family:"Gowun Batang",serif;font-size:21px;font-weight:700;line-height:1.4;margin-bottom:10px}
.b-sum{font-size:15.5px;line-height:1.78;color:var(--mok);font-weight:400;margin-bottom:12px;padding-left:14px;border-left:3px solid var(--mok)}.part.saju .b-sum{color:var(--hwa);border-color:var(--hwa)}.b-body{font-size:15.5px;line-height:1.78;color:var(--ink)}
.synth{margin:60px 0 0;padding:38px 30px;background:linear-gradient(135deg,rgba(58,125,92,.07),rgba(194,82,57,.07));border:1px solid var(--line);border-radius:8px}.synth .b-title{font-size:24px;text-align:center;margin-bottom:18px}.synth-summary{margin-top:22px}
.converge{margin:24px 0 4px;display:flex;flex-direction:column;gap:10px}.cv{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;gap:10px;font-size:13.5px}.cv .g{color:var(--mok);text-align:right}.cv .s{color:var(--hwa)}.cv .eq{color:var(--gold);font-family:"Gowun Batang",serif;font-weight:700}
.action-panel{margin:44px 0 0;padding:30px 28px;background:var(--paper-2);border:1px solid var(--line);border-radius:8px}.action-panel .b-title{font-size:23px;margin-bottom:14px}
.tags{margin:46px 0;text-align:center}.tags h3{margin-bottom:18px}.chip{display:inline-block;margin:5px;padding:9px 18px;background:var(--paper-2);border:1px solid var(--su);color:var(--su);border-radius:30px;font-size:14px;font-family:"Gowun Batang",serif}
.reco{margin:60px 0 0}.reco-head{text-align:center;margin-bottom:8px}.reco-head .b-title{font-size:24px}.reco-lead{text-align:center;font-size:14px;color:var(--ink-soft);max-width:42ch;margin:0 auto 30px;line-height:1.75}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.card{background:var(--paper-2);border:1px solid var(--line);border-radius:8px;padding:18px 16px 20px;text-align:center;position:relative;transition:transform .25s ease,box-shadow .25s ease}.card:hover{transform:translateY(-4px);box-shadow:0 10px 28px rgba(46,66,88,.12)}
.card .rank{position:absolute;top:12px;left:14px;font-family:"Song Myung",serif;font-size:13px;color:var(--gold)}.face{width:86px;height:86px;border-radius:50%;margin:6px auto 14px;background:linear-gradient(135deg,#dfe6e9,#c7d0d6);display:flex;align-items:center;justify-content:center;border:2px solid var(--su)}.face svg{width:46px;height:46px;fill:var(--su);opacity:.7}
.card .nm{font-family:"Gowun Batang",serif;font-size:16px;font-weight:700}.score{font-family:"Song Myung",serif;font-size:30px;color:var(--su);line-height:1.1;margin:4px 0 2px}.score span{font-size:14px;color:var(--ink-soft)}.reason{font-size:12.5px;color:var(--ink);line-height:1.6;margin-top:8px}.mtag{display:inline-block;margin-top:12px;padding:3px 10px;background:rgba(46,66,88,.08);color:var(--su);border-radius:20px;font-size:11px}.reco-note{text-align:center;font-size:11.5px;color:var(--ink-soft);margin-top:18px}
footer{text-align:center;padding:44px 0 60px;border-top:1px solid var(--line);margin-top:50px}footer .logo{font-family:"Song Myung",serif;font-size:22px;letter-spacing:.2em;color:var(--ink)}footer .disc{font-size:11.5px;color:var(--ink-soft);margin-top:12px;max-width:46ch;margin-left:auto;margin-right:auto;line-height:1.7}
.fade{opacity:0;transform:translateY(16px);animation:rise .8s ease forwards}@keyframes rise{to{opacity:1;transform:none}}@media (prefers-reduced-motion:reduce){.fade{animation:none;opacity:1;transform:none}.col{transition:none}}
@media (max-width:760px){.compat-profile-card{grid-template-columns:1fr;padding:18px}.compat-score-heart-card{max-width:260px;min-height:220px}.compat-score-heart-shape{width:158px;height:116px}.compat-score-heart-shape::before{font-size:166px}.compat-score-number{font-size:48px}.compat-score-breakdown{grid-template-columns:1fr}}
@media (max-width:560px){.name{font-size:24px}.cell .ch{font-size:26px}.part-title{font-size:20px}.cards,.pair-profiles{grid-template-columns:1fr;gap:12px}.pair-mark{gap:10px}.person-mark{width:92px;height:92px}.person-mark .person-hanja{font-size:42px}.pair-x{font-size:26px}.cv{grid-template-columns:1fr;text-align:center;gap:2px}.cv .g{text-align:center}.part-head{display:block}.part-sub{margin-left:0;margin-top:6px}.grid4{gap:6px}.wrap{padding:0 16px}}
"""
    return result.strip()

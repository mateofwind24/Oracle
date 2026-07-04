from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from oracle_report.models import BirthProfile
from oracle_report.saju.calendar import (
    time_branch_index_from_datetime as _chart_time_branch_index_from_datetime,
)
from oracle_report.saju.engine import ELEMENTS, SajuReading, build_saju_reading


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
MANSE_TIME_BRANCH_HANJA = (
    "子時",
    "丑時",
    "寅時",
    "卯時",
    "辰時",
    "巳時",
    "午時",
    "未時",
    "申時",
    "酉時",
    "戌時",
    "亥時",
)
MANSE_TIME_BRANCH_RANGES = (
    "23:30-01:30",
    "01:30-03:30",
    "03:30-05:30",
    "05:30-07:30",
    "07:30-09:30",
    "09:30-11:30",
    "11:30-13:30",
    "13:30-15:30",
    "15:30-17:30",
    "17:30-19:30",
    "19:30-21:30",
    "21:30-23:30",
)
_REPRESENTATIVE_TIMES = (
    "00:30",
    "02:30",
    "04:30",
    "06:30",
    "08:30",
    "10:30",
    "12:30",
    "14:30",
    "16:30",
    "18:30",
    "20:30",
    "22:30",
)
UNKNOWN_BIRTH_TIME_REPRESENTATIVE = "12:30"
_UNKNOWN_BIRTH_TIME_DISPLAY = "시간 미상 (오시(午時) 보조 기준)"


@dataclass(frozen=True)
class ManseLookupResult:
    reading: SajuReading
    formatted_text: str
    gender: str
    time_branch_label: str
    daeun_direction: str


class ManseRepository:
    def __init__(self, db_path: Path | None = None) -> None:
        self._db_path = db_path

    def lookup(self, profile: BirthProfile) -> ManseLookupResult:
        result = build_manse_lookup(profile)
        return result


def build_manse_lookup(profile: BirthProfile) -> ManseLookupResult:
    gender = normalize_gender(profile.gender)
    reading = build_saju_reading(profile.birth_datetime)
    time_branch_index = time_branch_index_from_datetime(profile.birth_datetime)
    time_branch_label = MANSE_TIME_BRANCH_LABELS[time_branch_index]
    daeun_direction = _daeun_direction(reading.chart.year.stem_index, gender)
    formatted_text = _format_manse_lookup(
        profile=profile,
        reading=reading,
        gender=gender,
        daeun_direction=daeun_direction,
    )
    result = ManseLookupResult(
        reading=reading,
        formatted_text=formatted_text,
        gender=gender,
        time_branch_label=time_branch_label,
        daeun_direction=daeun_direction,
    )
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
    result = _chart_time_branch_index_from_datetime(birth_datetime)
    return result


def time_branch_label_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = MANSE_TIME_BRANCH_LABELS[index]
    return result


def time_branch_display_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = f"{MANSE_TIME_BRANCH_LABELS[index]}({MANSE_TIME_BRANCH_HANJA[index]})"
    return result


def time_branch_range_display_from_index(index: int) -> str:
    _validate_time_branch_index(index)
    result = (
        f"{time_branch_display_from_index(index)} "
        f"{MANSE_TIME_BRANCH_RANGES[index]}"
    )
    return result


def time_branch_display_from_datetime(birth_datetime: datetime) -> str:
    index = time_branch_index_from_datetime(birth_datetime)
    result = time_branch_display_from_index(index)
    return result


def birth_time_display_from_profile(profile: BirthProfile) -> str:
    result = time_branch_display_from_datetime(profile.birth_datetime)
    if not profile.birth_time_known:
        result = _UNKNOWN_BIRTH_TIME_DISPLAY
    return result


def birth_datetime_display_from_profile(profile: BirthProfile) -> str:
    birth_date = profile.birth_datetime.date().isoformat()
    birth_time = birth_time_display_from_profile(profile)
    result = f"{birth_date} {birth_time}"
    return result


def representative_time_from_time_branch(value: str) -> str | None:
    cleaned = _normalize_time_branch_token(value)
    result = None
    for index, label in enumerate(MANSE_TIME_BRANCH_LABELS):
        hanja = MANSE_TIME_BRANCH_HANJA[index]
        names = {
            _normalize_time_branch_token(label),
            _normalize_time_branch_token(hanja),
            _normalize_time_branch_token(f"{label}({hanja})"),
            _normalize_time_branch_token(f"{label} {hanja}"),
        }
        if cleaned in names:
            result = _REPRESENTATIVE_TIMES[index]
    return result


# ==========================================
# 십신 계산을 위한 헬퍼 함수 추가
# ==========================================

def _get_ten_god_stem(dm_index: int, target_index: int) -> str:
    """천간의 십신을 계산합니다."""
    from oracle_report.saju.calendar import STEM_ELEMENTS, STEM_POLARITIES
    ELEMENTS_ORDER = ("목", "화", "토", "금", "수")
    
    dm_el = STEM_ELEMENTS[dm_index]
    dm_pol = STEM_POLARITIES[dm_index]
    tgt_el = STEM_ELEMENTS[target_index]
    tgt_pol = STEM_POLARITIES[target_index]
    
    dm_el_idx = ELEMENTS_ORDER.index(dm_el)
    tgt_el_idx = ELEMENTS_ORDER.index(tgt_el)
    
    # 오행 간의 거리 계산 (상생상극 매핑)
    diff = (tgt_el_idx - dm_el_idx) % 5
    same_pol = (dm_pol == tgt_pol)
    
    if diff == 0:
        return "비견" if same_pol else "겁재"
    elif diff == 1:
        return "식신" if same_pol else "상관"
    elif diff == 2:
        return "편재" if same_pol else "정재"
    elif diff == 3:
        return "편관" if same_pol else "정관"
    else:
        return "편인" if same_pol else "정인"


def _get_ten_god_branch(dm_index: int, target_index: int) -> str:
    """지지의 십신을 계산합니다. (명리 표준 용(用) 음양 기준 적용)"""
    from oracle_report.saju.calendar import STEM_ELEMENTS, STEM_POLARITIES, BRANCH_ELEMENTS
    ELEMENTS_ORDER = ("목", "화", "토", "금", "수")
    # 자(0)부터 해(11)까지의 십신 계산용 표준 음양 배열
    BRANCH_POLARITIES = ("음", "음", "양", "음", "양", "양", "음", "음", "양", "음", "양", "양")
    
    dm_el = STEM_ELEMENTS[dm_index]
    dm_pol = STEM_POLARITIES[dm_index]
    tgt_el = BRANCH_ELEMENTS[target_index]
    tgt_pol = BRANCH_POLARITIES[target_index]
    
    dm_el_idx = ELEMENTS_ORDER.index(dm_el)
    tgt_el_idx = ELEMENTS_ORDER.index(tgt_el)
    
    diff = (tgt_el_idx - dm_el_idx) % 5
    same_pol = (dm_pol == tgt_pol)
    
    if diff == 0:
        return "비견" if same_pol else "겁재"
    elif diff == 1:
        return "식신" if same_pol else "상관"
    elif diff == 2:
        return "편재" if same_pol else "정재"
    elif diff == 3:
        return "편관" if same_pol else "정관"
    else:
        return "편인" if same_pol else "정인"


# ==============================================================================
# 26종 신살/귀인 실시간 연산 가속 엔진
# ==============================================================================

def _calculate_all_shinsals(chart) -> dict[str, list[str]]:
    """사주 4주를 분석하여 26가지 신살 및 귀인을 정밀 도출합니다."""
    from oracle_report.saju.calendar import HEAVENLY_STEMS, EARTHLY_BRANCHES
    
    dm_idx = chart.day.stem_index
    dm_stem = chart.day.stem
    dm_branch = chart.day.branch
    mb_idx = chart.month.branch_index
    
    pillars = {
        "year": chart.year,
        "month": chart.month,
        "day": chart.day,
        "hour": chart.hour
    }
    
    result = {k: [] for k in pillars.keys()}
    
    # 삼기귀인 체크를 위한 전체 천간 모음
    all_stems = [p.stem for p in pillars.values()]
    has_samgi = False
    if ("갑" in all_stems and "무" in all_stems and "경" in all_stems) or \
       ("을" in all_stems and "병" in all_stems and "정" in all_stems) or \
       ("신" in all_stems and "임" in all_stems and "계" in all_stems):
        has_samgi = True

    # 일지 기준 공망 구하기
    day_ganji_idx = (chart.day.stem_index - chart.day.branch_index) % 12
    gongmang_branches = []
    if day_ganji_idx == 0: gongmang_branches = ["술", "해"]
    elif day_ganji_idx == 10: gongmang_branches = ["신", "유"]
    elif day_ganji_idx == 8: gongmang_branches = ["오", "미"]
    elif day_ganji_idx == 6: gongmang_branches = ["진", "사"]
    elif day_ganji_idx == 4: gongmang_branches = ["인", "묘"]
    elif day_ganji_idx == 2: gongmang_branches = ["자", "축"]

    for k, p in pillars.items():
        s = p.stem
        b = p.branch
        label = p.label
        sal_list = []
        
        # 1.1 천을귀인
        if (dm_stem in ("갑", "경", "무") and b in ("축", "미")) or \
           (dm_stem in ("을", "기") and b in ("자", "신")) or \
           (dm_stem in ("병", "정") and b in ("해", "유")) or \
           (dm_stem == "신" and b in ("인", "오")) or \
           (dm_stem in ("임", "계") and b in ("사", "묘")):
            sal_list.append("천을귀인")
            
        # 1.2 문창귀인
        if (dm_stem == "갑" and b == "사") or (dm_stem == "을" and b == "오") or \
           (dm_stem == "병" and b == "신") or (dm_stem == "정" and b == "유") or \
           (dm_stem == "무" and b == "신") or (dm_stem == "기" and b == "유") or \
           (dm_stem == "경" and b == "해") or (dm_stem == "신" and b == "자") or \
           (dm_stem == "임" and b == "인") or (dm_stem == "계" and b == "묘"):
            sal_list.append("문창귀인")
            
        # 1.3 양인살
        if (dm_stem == "갑" and b == "卯") or (dm_stem == "丙" and b == "午") or \
           (dm_stem == "戊" and b == "午") or (dm_stem == "庚" and b == "酉") or \
           (dm_stem == "壬" and b == "子"):
            sal_list.append("양인살")
            
        # 1.4 도화살 / 1.5 역마살 / 1.6 화개살 / 2.11 겁살 / 2.12 수옥살 / 2.13 망신살 / 2.14 반안살 (신살 삼합 기준 지지 연산)
        ref_branch = chart.day.branch if k != "day" else chart.year.branch
        if ref_branch in ("해", "묘", "미"):
            if b == "자": sal_list.append("도화살")
            elif b == "사": sal_list.append("역마살")
            elif b == "미": sal_list.append("화개살")
            elif b == "신": sal_list.append("겁살")
            elif b == "유": sal_list.append("수옥살")
            elif b == "인": sal_list.append("망신살")
            elif b == "축": sal_list.append("반안살")
        elif ref_branch in ("인", "오", "술"):
            if b == "묘": sal_list.append("도화살")
            elif b == "신": sal_list.append("역마살")
            elif b == "술": sal_list.append("화개살")
            elif b == "해": sal_list.append("겁살")
            elif b == "자": sal_list.append("수옥살")
            elif b == "사": sal_list.append("망신살")
            elif b == "미": sal_list.append("반안살")
        elif ref_branch in ("사", "유", "축"):
            if b == "오": sal_list.append("도화살")
            elif b == "해": sal_list.append("역마살")
            elif b == "축": sal_list.append("화개살")
            elif b == "인": sal_list.append("겁살")
            elif b == "묘": sal_list.append("수옥살")
            elif b == "신": sal_list.append("망신살")
            elif b == "술": sal_list.append("반안살")
        elif ref_branch in ("신", "자", "진"):
            if b == "유": sal_list.append("도화살")
            elif b == "인": sal_list.append("역마살")
            elif b == "진": sal_list.append("화개살")
            elif b == "사": sal_list.append("겁살")
            elif b == "오": sal_list.append("수옥살")
            elif b == "해": sal_list.append("망신살")
            elif b == "미": sal_list.append("반안살")

        # 1.7 공망살
        if b in gongmang_branches:
            sal_list.append("공망살")

        # 1.8 원진살
        if (dm_branch == "자" and b == "미") or (dm_branch == "축" and b == "오") or \
           (dm_branch == "인" and b == "유") or (dm_branch == "묘" and b == "신") or \
           (dm_branch == "진" and b == "해") or (dm_branch == "사" and b == "술") or \
           (dm_branch == "미" and b == "자") or (dm_branch == "오" and b == "축") or \
           (dm_branch == "유" and b == "인") or (dm_branch == "신" and b == "묘") or \
           (dm_branch == "해" and b == "진") or (dm_branch == "술" and b == "사"):
            sal_list.append("원진살")

        # 1.9 귀문관살
        if (dm_branch == "자" and b == "미") or (dm_branch == "축" and b == "인") or \
           (dm_branch == "인" and b == "축") or (dm_branch == "묘" and b == "신") or \
           (dm_branch == "진" and b == "해") or (dm_branch == "사" and b == "술") or \
           (dm_branch == "미" and b == "자") or (dm_branch == "신" and b == "묘") or \
           (dm_branch == "해" and b == "진") or (dm_branch == "술" and b == "사"):
            if "원진살" not in sal_list: sal_list.append("귀문관살")

        # 1.10 백호살
        if label in ("갑진", "을미", "병진", "정축", "무진", "임술", "계축"):
            sal_list.append("백호살")

        # 1.11 괴강살
        if label in ("경진", "경술", "임진", "임술", "무진", "무술"):
            sal_list.append("괴강살")

        # 2.1 복성귀인
        if (dm_stem == "갑" and b == "인") or (dm_stem == "을" and b == "축") or \
           (dm_stem == "병" and b == "자") or (dm_stem == "정" and b == "유") or \
           (dm_stem == "무" and b == "오") or (dm_stem == "기" and b == "미") or \
           (dm_stem == "경" and b == "신") or (dm_stem == "신" and b == "마") or \
           (dm_stem == "임" and b == "진") or (dm_stem == "계" and b == "사"):
            sal_list.append("복성귀인")

        # 2.2 금여
        if (dm_stem == "갑" and b == "진") or (dm_stem == "을" and b == "사") or \
           (dm_stem == "병" and b == "미") or (dm_stem == "정" and b == "신") or \
           (dm_stem == "무" and b == "미") or (dm_stem == "기" and b == "신") or \
           (dm_stem == "경" and b == "술") or (dm_stem == "신" and b == "해") or \
           (dm_stem == "임" and b == "축") or (dm_stem == "계" and b == "인"):
            sal_list.append("금여")

        # 2.3 건록
        if (dm_stem == "갑" and b == "인") or (dm_stem == "을" and b == "묘") or \
           (dm_stem == "병" and b == "사") or (dm_stem == "정" and b == "오") or \
           (dm_stem == "무" and b == "사") or (dm_stem == "기" and b == "오") or \
           (dm_stem == "경" and b == "신") or (dm_stem == "신" and b == "유") or \
           (dm_stem == "임" and b == "해") or (dm_stem == "계" and b == "자"):
            sal_list.append("건록")

        # 2.4 암록
        if (dm_stem == "갑" and b == "해") or (dm_stem == "을" and b == "술") or \
           (dm_stem == "병" and b == "신") or (dm_stem == "정" and b == "미") or \
           (dm_stem == "무" and b == "신") or (dm_stem == "기" and b == "미") or \
           (dm_stem == "경" and b == "사") or (dm_stem == "신" and b == "진") or \
           (dm_stem == "임" and b == "인") or (dm_stem == "계" and b == "축"):
            sal_list.append("암록")

        # 2.5 삼기귀인 (일주나 특정 기둥에 대표 부여)
        if has_samgi and k == "day":
            sal_list.append("삼기귀인")

        # 2.6 육수
        if label in ("갑자", "갑술", "병자", "병술", "무자", "무술"):
            sal_list.append("육수")

        # 2.7 천의성
        check_map = {"자":"해", "축":"자", "인":"축", "묘":"인", "진":"묘", "사":"진", "오":"사", "미":"오", "신":"미", "유":"신", "술":"유", "해":"술"}
        if EARTHLY_BRANCHES[mb_idx] in check_map and b == check_map[EARTHLY_BRANCHES[mb_idx]]:
            sal_list.append("천의성")

        # 2.8 현침살
        if s in ("갑", "신") or b in ("묘", "오", "미"):
            sal_list.append("현침살")

        # 2.9 홍염살
        if (dm_stem == "갑" and b == "오") or (dm_stem == "을" and b == "오") or \
           (dm_stem == "병" and b == "인") or (dm_stem == "정" and b == "미") or \
           (dm_stem == "무" and b == "진") or (dm_stem == "기" and b == "진") or \
           (dm_stem == "경" and b == "술") or (dm_stem == "신" and b == "유") or \
           (dm_stem == "임" and b == "신") or (dm_stem == "계" and b == "해"):
            sal_list.append("홍염살")

        # 2.10 급각살
        if (EARTHLY_BRANCHES[mb_idx] in ("인", "묘", "진") and b in ("축", "미")) or \
           (EARTHLY_BRANCHES[mb_idx] in ("사", "오", "미") and b in ("묘", "미")) or \
           (EARTHLY_BRANCHES[mb_idx] in ("신", "유", "술") and b in ("인", "술")) or \
           (EARTHLY_BRANCHES[mb_idx] in ("자", "축", "해") and b in ("축", "진")):
            sal_list.append("급각살")

        # 2.15 음양살
        if label in ("병자", "무자", "병오", "무오", "임자", "임오", "무신", "신묘"):
            sal_list.append("음양살")

        result[k] = sal_list
        
    return result


# ==============================================================================
# 기존 _format_manse_lookup 함수 교체
# ==============================================================================
def _format_manse_lookup(
    profile: BirthProfile,
    reading: SajuReading,
    gender: str,
    daeun_direction: str,
) -> str:
    chart = reading.chart
    dm_idx = chart.day.stem_index
    
    # 십신 데이터 구하기
    year_stem_tg = _get_ten_god_stem(dm_idx, chart.year.stem_index)
    year_branch_tg = _get_ten_god_branch(dm_idx, chart.year.branch_index)
    month_stem_tg = _get_ten_god_stem(dm_idx, chart.month.stem_index)
    month_branch_tg = _get_ten_god_branch(dm_idx, chart.month.branch_index)
    day_branch_tg = _get_ten_god_branch(dm_idx, chart.day.branch_index)
    hour_stem_tg = _get_ten_god_stem(dm_idx, chart.hour.stem_index)
    hour_branch_tg = _get_ten_god_branch(dm_idx, chart.hour.branch_index)
    
    # 26종 신살 통합 매트릭스 연산 수행
    shinsal_matrix = _calculate_all_shinsals(chart)
    
    # 💡 [수정] 문법 에러 방지를 위해 일주의 특수기운 텍스트를 미리 안전하게 조립합니다.
    day_sal_list = shinsal_matrix['day']
    day_sal_str = f" / 특수기운: {', '.join(day_sal_list)}" if day_sal_list else ""
    
    def _make_pillar_text(label, stem_tg, branch_tg, sal_list):
        sal_str = f" / 특수기운: {', '.join(sal_list)}" if sal_list else ""
        return f"{label} (천간: {stem_tg} / 지지: {branch_tg}{sal_str})"

    counts = ", ".join(f"{element} {reading.element_counts[element]}" for element in ELEMENTS)
    birth_date = profile.birth_datetime.date().isoformat()
    time_branch_label = birth_time_display_from_profile(profile)
    
    result = "\n".join(
        (
            "[만세력/사주명식]",
            f"- 기준일: {birth_date} {time_branch_label} ({gender})",
            f"- 년주: {_make_pillar_text(chart.year.label, year_stem_tg, year_branch_tg, shinsal_matrix['year'])}",
            f"- 월주: {_make_pillar_text(chart.month.label, month_stem_tg, month_branch_tg, shinsal_matrix['month'])}",
            f"- 일주: {chart.day.label} (천간: 본인_일간 / 지지: {day_branch_tg}{day_sal_str})",
            f"- 시주: {_make_pillar_text(chart.hour.label, hour_stem_tg, hour_branch_tg, shinsal_matrix['hour'])}",
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

def _daeun_direction(year_stem_index: int, gender: str) -> str:
    year_polarity = "양" if year_stem_index % 2 == 0 else "음"
    result = "역행"
    if (gender == "남성" and year_polarity == "양") or (
        gender == "여성" and year_polarity == "음"
    ):
        result = "순행"
    return result


def _validate_time_branch_index(index: int) -> None:
    if not 0 <= index < len(MANSE_TIME_BRANCH_LABELS):
        raise ValueError("시간 지지 인덱스가 범위를 벗어났습니다.")


def _normalize_time_branch_token(value: str) -> str:
    result = (
        value.strip()
        .lower()
        .replace(" ", "")
        .replace("(", "")
        .replace(")", "")
    )
    return result

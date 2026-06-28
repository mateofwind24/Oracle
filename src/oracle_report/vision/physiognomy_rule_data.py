from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceReference:
    id: str
    title: str
    url: str
    note: str


@dataclass(frozen=True)
class PhysiognomyRuleRange:
    min_value: float | None
    max_value: float | None
    tag: str
    observation: str
    interpretation: str


@dataclass(frozen=True)
class PhysiognomyRule:
    id: str
    metric: str
    title: str
    basis: str
    source_ids: tuple[str, ...]
    ranges: tuple[PhysiognomyRuleRange, ...]


RULE_SOURCE_REFERENCES = (
    SourceReference(
        id="encykorea_face",
        title="한국민족문화대백과사전: 얼굴",
        url="https://encykorea.aks.ac.kr/Article/E0036194",
        note="오관, 육부, 삼재, 삼정, 오악, 십이궁 등 전통 관상 분류 체계.",
    ),
    SourceReference(
        id="gimpo_three_zones",
        title="김포신문: 얼굴 관상은 3등분 균형이 맞아야",
        url="https://www.igimpo.com/news/articleView.html?idxno=62270",
        note="상정, 중정, 하정의 3등분 구조와 균형 기준.",
    ),
    SourceReference(
        id="skku_12_palaces",
        title="성대신문: 관상을 시작하려면 십이궁부터 공부하라",
        url="https://www.skkuw.com/news/articleView.html?idxno=10698",
        note="미간, 눈썹, 눈꺼풀, 코, 하관 등 십이궁의 주요 위치.",
    ),
    SourceReference(
        id="newswire_12_palaces",
        title="뉴스와이어: 12궁을 기준으로 본 좋은 관상",
        url="https://www.newswire.co.kr/newsRead.php?no=727912",
        note="명궁, 재백궁, 형제궁, 전택궁, 노복궁 등 주요 부위 요약.",
    ),
    SourceReference(
        id="woman_donga_age_zones",
        title="여성동아: 관상을 주관하는 부위는 연령대별로 다릅니다",
        url="https://woman.donga.com/people/article/all/12/6163768/1",
        note="눈썹, 미간, 코, 광대, 인중, 입술, 턱을 나누어 보는 현대 해설.",
    ),
    SourceReference(
        id="mebyface_mian_xiang",
        title="MeByFace: Mian Xiang Chinese Face Reading",
        url="https://www.mebyface.com/learn/mian-xiang-chinese-face-reading",
        note="삼분 구역, 오관, 십이궁, 인중 등 중국 면상 체계 요약.",
    ),
    SourceReference(
        id="medipharm_ratio",
        title="메디팜헬스뉴스: 얼굴 황금비율과 관상",
        url="https://www.medipharmhealth.co.kr/mobile/article.html?no=78071",
        note="미용 관점의 상중하 얼굴 비율과 얼굴형 인상 분류.",
    ),
)

PHYSIOGNOMY_SAFETY_NOTE = (
    "랜드마크 비율을 바탕으로 만든 엔터테인먼트 참고 자료이며 실제 성격, 건강, "
    "신원, 능력, 미래를 판단하지 않습니다."
)

UNSUPPORTED_PHYSIOGNOMY_FEATURES = (
    "귀 모양",
    "피부색과 기색",
    "점과 흉터",
    "주름",
    "머리카락",
    "음성",
    "자세와 걸음",
)

PHYSIOGNOMY_RULES = (
    PhysiognomyRule(
        id="third_balance",
        metric="third_balance_error",
        title="삼정 균형",
        basis="상정, 중정, 하정을 비슷한 비중으로 보는 전통 관상 기준",
        source_ids=("encykorea_face", "gimpo_three_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.055,
                tag="삼정 균형형",
                observation="상정, 중정, 하정의 편차가 작아 전체 세로 균형이 안정적입니다.",
                interpretation=(
                    "전통 관상에서는 얼굴 흐름이 고르게 잡힌 인상으로 보므로 "
                    "리포트에는 균형감과 안정감을 보조 키워드로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.055,
                max_value=0.105,
                tag="삼정 약간 편차형",
                observation="세 구역 중 한 구역이 조금 더 강조되어 보입니다.",
                interpretation=(
                    "전통 관상에서는 특정 시기나 역할감이 부각된 인상으로 읽으므로 "
                    "리포트에는 강조된 구역을 개성 요소로만 다룹니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.105,
                max_value=None,
                tag="삼정 편차 강조형",
                observation="상중하 세로 구역의 차이가 또렷하게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 균형보다 특정 부위의 힘을 먼저 보므로 "
                    "리포트에는 강한 인상과 보완 균형을 함께 언급합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="upper_zone",
        metric="upper_zone_ratio",
        title="상정 비율",
        basis="이마에서 눈썹까지의 상정 구역",
        source_ids=("gimpo_three_zones", "woman_donga_age_zones", "medipharm_ratio"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.29,
                tag="상정 짧은 편",
                observation="이마-눈썹 구역이 전체 높이에 비해 짧은 편입니다.",
                interpretation=(
                    "전통 관상 해석에서는 상정보다 현재 행동감이나 실무감을 "
                    "더 앞세워 읽는 보조 소재로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.29,
                max_value=0.39,
                tag="상정 균형형",
                observation="이마-눈썹 구역이 삼정 기준에서 무난한 폭으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 초반 인상과 계획성이 과하거나 부족하지 않은 "
                    "균형 소재로 풀 수 있습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.39,
                max_value=None,
                tag="상정 긴 편",
                observation="이마-눈썹 구역이 상대적으로 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 이마 영역의 존재감이 강한 인상으로 보므로 "
                    "리포트에는 생각의 폭이나 정돈된 첫인상 같은 보조 표현을 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="middle_zone",
        metric="middle_zone_ratio",
        title="중정 비율",
        basis="눈썹에서 코까지의 중정 구역",
        source_ids=("gimpo_three_zones", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.29,
                tag="중정 짧은 편",
                observation="눈썹-코 구역이 전체 높이에 비해 짧은 편입니다.",
                interpretation=(
                    "전통 관상에서는 코와 관골의 중심감이 부드럽게 보이는 쪽으로 "
                    "해석하므로 리포트에는 부담 없는 인상을 보조로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.29,
                max_value=0.39,
                tag="중정 균형형",
                observation="눈썹-코 구역이 삼정 기준에서 균형적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 얼굴 중심부가 안정된 인상으로 보므로 "
                    "리포트에는 현실감과 중심감을 보조 키워드로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.39,
                max_value=None,
                tag="중정 긴 편",
                observation="눈썹-코 구역이 상대적으로 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 중정의 존재감이 강한 인상으로 보므로 "
                    "리포트에는 활동성과 추진감의 인상 소재로만 반영합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="lower_zone",
        metric="lower_zone_ratio",
        title="하정 비율",
        basis="코 아래에서 턱까지의 하정 구역",
        source_ids=("gimpo_three_zones", "skku_12_palaces", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.29,
                tag="하정 짧은 편",
                observation="코 아래-턱 구역이 전체 높이에 비해 짧은 편입니다.",
                interpretation=(
                    "전통 관상에서는 하관의 무게감이 약한 인상으로 읽을 수 있어 "
                    "리포트에는 가벼운 인상과 부드러운 마무리감을 함께 적습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.29,
                max_value=0.40,
                tag="하정 균형형",
                observation="코 아래-턱 구역이 삼정 기준에서 균형적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 하관이 안정된 인상으로 보므로 리포트에는 "
                    "마무리감과 신뢰감을 보조 키워드로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.40,
                max_value=None,
                tag="하정 긴 편",
                observation="코 아래-턱 구역이 상대적으로 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 하관의 존재감이 큰 인상으로 읽으므로 "
                    "리포트에는 끈기와 안정감의 보조 소재로만 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="face_aspect",
        metric="face_aspect_ratio",
        title="얼굴형 세로/가로",
        basis="얼굴 세로 길이를 광대 폭으로 나눈 윤곽 비율",
        source_ids=("medipharm_ratio", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=1.18,
                tag="가로 안정형 윤곽",
                observation="얼굴 가로 폭이 상대적으로 안정감 있게 보입니다.",
                interpretation=(
                    "얼굴형 분류에서는 편안하고 단단한 첫인상으로 읽기 쉬우므로 "
                    "리포트에 안정적인 윤곽 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=1.18,
                max_value=1.45,
                tag="균형형 윤곽",
                observation="세로와 가로 윤곽이 과하게 치우치지 않은 편입니다.",
                interpretation=(
                    "얼굴형 분류에서는 균형 잡힌 인상으로 보므로 리포트에는 "
                    "조화로운 분위기를 보조 해석으로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=1.45,
                max_value=None,
                tag="세로 강조형 윤곽",
                observation="얼굴 세로 길이가 상대적으로 길게 관찰됩니다.",
                interpretation=(
                    "얼굴형 분류에서는 차분하고 길게 흐르는 인상으로 읽을 수 있어 "
                    "리포트에는 정돈된 분위기를 보조로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="eye_spacing",
        metric="eye_spacing_ratio",
        title="미간/눈 사이 간격",
        basis="양쪽 안쪽 눈꼬리 사이 간격을 얼굴 폭으로 나눈 비율",
        source_ids=("skku_12_palaces", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.24,
                tag="미간 좁은 편",
                observation="눈 사이 간격이 얼굴 폭에 비해 좁은 편입니다.",
                interpretation=(
                    "십이궁의 명궁 해석에서는 미간을 중요한 관찰 지점으로 보므로 "
                    "리포트에는 집중된 인상이라는 보조 표현을 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.24,
                max_value=0.34,
                tag="미간 균형형",
                observation="눈 사이 간격이 균형적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 미간이 답답하지 않은 인상을 좋게 보므로 "
                    "리포트에는 열린 인상과 안정감을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.34,
                max_value=None,
                tag="미간 넓은 편",
                observation="눈 사이 간격이 얼굴 폭에 비해 넓은 편입니다.",
                interpretation=(
                    "전통 관상에서는 미간의 여유를 넓은 인상으로 읽으므로 "
                    "리포트에는 여유와 부드러움을 보조 소재로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="eye_width",
        metric="eye_width_ratio",
        title="눈 가로 크기",
        basis="눈 가로폭을 얼굴 폭으로 나눈 비율",
        source_ids=("encykorea_face", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.17,
                tag="눈 가로폭 작은 편",
                observation="눈의 가로폭이 얼굴 폭에 비해 작게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 시선이 응축된 인상으로 읽는 편이므로 "
                    "리포트에는 신중함과 내면 집중감을 보조 표현으로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.17,
                max_value=0.24,
                tag="눈 가로폭 균형형",
                observation="눈의 가로폭이 얼굴 폭 대비 무난하게 균형을 이룹니다.",
                interpretation=(
                    "전통 관상에서는 눈의 폭이 과하지 않으면 또렷하면서도 편안한 인상으로 보므로 "
                    "리포트에는 안정적인 시선 흐름을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.24,
                max_value=None,
                tag="눈 가로폭 넓은 편",
                observation="눈의 가로폭이 얼굴 폭에 비해 넓게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 시야가 열려 보이는 인상으로 풀 수 있어 "
                    "리포트에는 개방감과 표현 리듬을 보조 표현으로 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="eye_aspect",
        metric="eye_aspect_ratio",
        title="눈 세로 개방감",
        basis="눈 세로 높이를 눈 가로폭으로 나눈 비율",
        source_ids=("skku_12_palaces", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.18,
                tag="눈 세로폭 얕은 편",
                observation="눈의 세로 개방감이 크지 않고 가로 흐름이 더 강조됩니다.",
                interpretation=(
                    "전통 관상에서는 길고 차분한 시선으로 읽는 경우가 많아 "
                    "리포트에는 냉정함보다 절제된 집중감을 보조 해석으로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.18,
                max_value=0.30,
                tag="눈 세로폭 균형형",
                observation="눈의 세로 개방감이 과하지도 답답하지도 않게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 시선과 표정의 안정성이 무난한 인상으로 보므로 "
                    "리포트에는 자연스러운 소통감을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.30,
                max_value=None,
                tag="눈 세로폭 열린 편",
                observation="눈의 세로 개방감이 크게 보여 또렷한 표정성이 느껴집니다.",
                interpretation=(
                    "전통 관상에서는 반응성이 잘 드러나는 눈으로 읽히기 쉬워 "
                    "리포트에는 표현력과 생동감을 보조 소재로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="brow_eye_span",
        metric="brow_eye_span_ratio",
        title="눈썹 길이",
        basis="눈썹이 눈을 충분히 감싸는지를 보는 전통 기준",
        source_ids=("skku_12_palaces", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.96,
                tag="눈썹 짧은 편",
                observation="눈썹 길이가 눈 폭에 비해 짧게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 눈썹의 흐름을 대인 인상으로 해석하므로 "
                    "리포트에는 또렷하고 간결한 인상을 보조로 적습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.96,
                max_value=1.20,
                tag="눈썹 균형형",
                observation="눈썹이 눈 폭을 무난하게 감싸는 편입니다.",
                interpretation=(
                    "전통 관상에서는 눈썹이 눈을 감싸는 모양을 안정적으로 보므로 "
                    "리포트에는 정돈된 관계감과 부드러운 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=1.20,
                max_value=None,
                tag="눈썹 긴 편",
                observation="눈썹 길이가 눈 폭보다 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 긴 눈썹을 눈 주변의 여유로운 흐름으로 보므로 "
                    "리포트에는 포용적인 인상이라는 보조 표현을 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="brow_eye_gap",
        metric="brow_eye_gap_ratio",
        title="눈썹-눈꺼풀 간격",
        basis="눈썹과 윗눈꺼풀 사이 전택궁 영역",
        source_ids=("skku_12_palaces", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.065,
                tag="눈썹 가까운 편",
                observation="눈썹과 눈 사이 간격이 가까운 편입니다.",
                interpretation=(
                    "전통 관상에서는 눈 주변 간격을 압축된 인상으로 읽으므로 "
                    "리포트에는 또렷함과 밀도감을 보조로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.065,
                max_value=0.12,
                tag="눈썹 간격 균형형",
                observation="눈썹과 눈 사이 간격이 안정적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 눈썹과 눈 사이가 답답하지 않은 구성을 좋게 보므로 "
                    "리포트에는 차분하고 정돈된 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.12,
                max_value=None,
                tag="눈썹 여유형",
                observation="눈썹과 눈 사이 간격이 넓은 편입니다.",
                interpretation=(
                    "전통 관상에서는 눈 위 공간의 여유를 부드러운 인상으로 읽으므로 "
                    "리포트에는 여백감과 편안함을 보조로 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="eye_tail_tilt",
        metric="eye_tail_tilt",
        title="눈꼬리 기울기",
        basis="바깥 눈꼬리와 안쪽 눈꼬리의 높이 차이",
        source_ids=("encykorea_face", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=-0.015,
                tag="눈꼬리 하향형",
                observation="바깥 눈꼬리가 안쪽보다 낮게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 부드럽고 차분한 눈매로 읽는 경우가 있어 "
                    "리포트에는 잔잔한 인상과 신중한 표현 리듬을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=-0.015,
                max_value=0.02,
                tag="눈꼬리 균형형",
                observation="눈꼬리 기울기가 크지 않아 안정적인 눈매로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 눈매의 균형을 정돈된 인상으로 해석하므로 "
                    "리포트에는 편안하고 안정적인 시선감을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.02,
                max_value=None,
                tag="눈꼬리 상향형",
                observation="바깥 눈꼬리가 안쪽보다 높게 보여 눈매 상승감이 느껴집니다.",
                interpretation=(
                    "전통 관상에서는 눈매의 상승감을 또렷한 추진 인상으로 읽기도 하므로 "
                    "리포트에는 활기와 선명함을 보조 표현으로 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="nose_width",
        metric="nose_width_ratio",
        title="코 폭",
        basis="콧볼 폭을 얼굴 폭으로 나눈 중정 중심 비율",
        source_ids=("skku_12_palaces", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.15,
                tag="코 폭 좁은 편",
                observation="콧볼 폭이 얼굴 폭에 비해 좁은 편입니다.",
                interpretation=(
                    "전통 관상에서는 코를 얼굴 중심부의 핵심으로 보므로 "
                    "리포트에는 섬세하고 정돈된 중심 인상으로만 반영합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.15,
                max_value=0.23,
                tag="코 폭 균형형",
                observation="콧볼 폭이 얼굴 폭 대비 균형적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 코와 주변부 조화를 중요하게 보므로 "
                    "리포트에는 중심감과 안정감을 보조 키워드로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.23,
                max_value=None,
                tag="코 폭 넓은 편",
                observation="콧볼 폭이 얼굴 폭에 비해 넓은 편입니다.",
                interpretation=(
                    "전통 관상에서는 코의 존재감이 강한 인상으로 보므로 "
                    "리포트에는 선명한 중심 인상을 보조 소재로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="nose_length",
        metric="nose_length_ratio",
        title="코 길이",
        basis="미간 아래 코 시작점부터 코끝 아래까지의 세로 비율",
        source_ids=("skku_12_palaces", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.20,
                tag="코 길이 짧은 편",
                observation="코의 세로 길이가 얼굴 높이에 비해 짧게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 중심부가 압축된 인상으로 읽는 경우가 있어 "
                    "리포트에는 간결한 중심감과 부담 없는 분위기를 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.20,
                max_value=0.31,
                tag="코 길이 균형형",
                observation="코의 세로 길이가 얼굴 높이 대비 무난한 균형으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 얼굴 중심부의 안정성을 좋게 보므로 "
                    "리포트에는 차분한 중심감과 정돈감을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.31,
                max_value=None,
                tag="코 길이 긴 편",
                observation="코의 세로 길이가 상대적으로 길게 보여 중심 축이 강조됩니다.",
                interpretation=(
                    "전통 관상에서는 코의 세로 존재감을 또렷한 중심 인상으로 읽을 수 있어 "
                    "리포트에는 진중함과 선명한 축을 보조 표현으로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="nose_length_width",
        metric="nose_length_width_ratio",
        title="코 길이 대비 폭",
        basis="코의 세로 길이를 코 폭으로 나눈 비율",
        source_ids=("mebyface_mian_xiang", "medipharm_ratio"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=1.15,
                tag="코 짧고 넓은 편",
                observation="코의 세로감보다 폭감이 먼저 보이는 편입니다.",
                interpretation=(
                    "전통 관상에서는 중심 인상이 단단하게 느껴질 수 있어 "
                    "리포트에는 묵직한 존재감과 현실감을 보조 소재로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=1.15,
                max_value=1.75,
                tag="코 비율 균형형",
                observation="코의 세로감과 폭감이 과하게 치우치지 않은 편입니다.",
                interpretation=(
                    "전통 관상에서는 중심부 조화가 안정적이라고 읽을 수 있어 "
                    "리포트에는 균형 잡힌 중심 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=1.75,
                max_value=None,
                tag="코 길고 가는 편",
                observation="코의 세로 흐름이 폭보다 더 강조되어 보입니다.",
                interpretation=(
                    "전통 관상에서는 선이 길고 정돈된 중심으로 보기도 하므로 "
                    "리포트에는 섬세함과 정리된 인상을 보조 표현으로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="mouth_width",
        metric="mouth_width_ratio",
        title="입 폭",
        basis="입꼬리 사이 폭을 얼굴 폭으로 나눈 오관 비율",
        source_ids=("encykorea_face", "woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.30,
                tag="입 폭 좁은 편",
                observation="입 폭이 얼굴 폭에 비해 좁은 편입니다.",
                interpretation=(
                    "전통 관상에서는 입을 표현과 마무리 인상으로 보므로 "
                    "리포트에는 절제된 표현감이라는 보조 문장을 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.30,
                max_value=0.43,
                tag="입 폭 균형형",
                observation="입 폭이 얼굴 폭 대비 균형적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 입의 균형을 편안한 표현감으로 읽으므로 "
                    "리포트에는 자연스러운 소통 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.43,
                max_value=None,
                tag="입 폭 넓은 편",
                observation="입 폭이 얼굴 폭에 비해 넓은 편입니다.",
                interpretation=(
                    "전통 관상에서는 입의 존재감이 큰 인상으로 보므로 "
                    "리포트에는 표현력이 또렷한 분위기를 보조로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="mouth_height",
        metric="mouth_height_ratio",
        title="입 높이",
        basis="윗입술과 아랫입술 사이 세로 높이를 얼굴 높이로 나눈 비율",
        source_ids=("encykorea_face", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.028,
                tag="입 높이 얕은 편",
                observation="입의 세로 높이가 크지 않아 입매가 얇고 차분하게 보입니다.",
                interpretation=(
                    "전통 관상에서는 입매의 높이를 표현 밀도로 읽는 경우가 있어 "
                    "리포트에는 절제된 표현감과 담백한 소통을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.028,
                max_value=0.050,
                tag="입 높이 균형형",
                observation="입의 세로 높이가 과하지 않게 안정적으로 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 입매 균형을 자연스러운 표현감으로 보므로 "
                    "리포트에는 편안한 전달감과 무난한 소통 리듬을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.050,
                max_value=None,
                tag="입 높이 도드라진 편",
                observation="입의 세로 높이가 도드라져 표정 전달력이 크게 보입니다.",
                interpretation=(
                    "전통 관상에서는 입매 존재감을 표현력으로 연결하기도 하므로 "
                    "리포트에는 말의 온도감과 감정 표현성을 보조로 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="philtrum_chin",
        metric="philtrum_chin_ratio",
        title="인중-턱 흐름",
        basis="코 아래에서 윗입술까지와 하정 전체의 상대 비율",
        source_ids=("woman_donga_age_zones", "mebyface_mian_xiang"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.20,
                tag="인중 짧은 편",
                observation="인중 구간이 하정 전체에 비해 짧게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 인중과 입 주변을 하정 흐름으로 보므로 "
                    "리포트에는 산뜻한 마무리 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.20,
                max_value=0.34,
                tag="인중 균형형",
                observation="인중과 턱으로 이어지는 하정 흐름이 균형적으로 보입니다.",
                interpretation=(
                    "전통 관상에서는 인중, 입술, 턱의 연결을 마무리 인상으로 보므로 "
                    "리포트에는 안정적인 하관 흐름을 보조 해석으로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.34,
                max_value=None,
                tag="인중 긴 편",
                observation="인중 구간이 하정 전체에 비해 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 인중의 길이를 하관의 존재감으로 보므로 "
                    "리포트에는 차분하고 길게 이어지는 인상을 보조로 사용합니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="chin_length",
        metric="chin_length_ratio",
        title="턱 길이",
        basis="아랫입술 아래부터 턱끝까지의 세로 비율",
        source_ids=("newswire_12_palaces", "woman_donga_age_zones", "medipharm_ratio"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.16,
                tag="턱 길이 짧은 편",
                observation="턱끝까지의 마무리 길이가 짧아 하관 마감이 가볍게 보입니다.",
                interpretation=(
                    "전통 관상에서는 턱의 길이를 마무리감으로 보므로 "
                    "리포트에는 가벼운 마무리와 부드러운 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.16,
                max_value=0.26,
                tag="턱 길이 균형형",
                observation="턱 길이가 전체 얼굴 높이 대비 무난하게 균형을 이룹니다.",
                interpretation=(
                    "전통 관상에서는 턱의 균형을 안정된 마무리로 읽으므로 "
                    "리포트에는 단정함과 신뢰감을 보조 키워드로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.26,
                max_value=None,
                tag="턱 길이 긴 편",
                observation="턱끝까지 이어지는 길이가 상대적으로 길게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 하관의 지속감을 끈기 있는 인상으로 읽는 경우가 있어 "
                    "리포트에는 차분한 지속력과 마무리감을 보조 표현으로 넣습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="jaw_width",
        metric="jaw_width_ratio",
        title="하관 폭",
        basis="턱선 폭을 얼굴 폭으로 나눈 하관 안정감",
        source_ids=("skku_12_palaces", "newswire_12_palaces", "medipharm_ratio"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.58,
                tag="하관 폭 좁은 편",
                observation="턱선 폭이 광대 폭에 비해 좁은 편입니다.",
                interpretation=(
                    "전통 관상에서는 하관을 마무리와 안정감의 영역으로 보므로 "
                    "리포트에는 부드러운 윤곽과 가벼운 하관 인상을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.58,
                max_value=0.76,
                tag="하관 폭 균형형",
                observation="턱선 폭이 얼굴 폭과 무난하게 조화를 이룹니다.",
                interpretation=(
                    "전통 관상에서는 하관의 폭과 무게감을 안정감으로 읽으므로 "
                    "리포트에는 단정한 마무리 인상을 보조로 사용합니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.76,
                max_value=None,
                tag="하관 폭 넓은 편",
                observation="턱선 폭이 얼굴 폭에 비해 넓게 관찰됩니다.",
                interpretation=(
                    "전통 관상에서는 하관의 존재감이 큰 인상으로 보므로 "
                    "리포트에는 묵직한 안정감과 선명한 윤곽을 보조로 적습니다."
                ),
            ),
        ),
    ),
    PhysiognomyRule(
        id="mouth_balance",
        metric="mouth_balance_delta",
        title="입꼬리 좌우 균형",
        basis="양쪽 입꼬리 높이 차이로 보는 표정 안정성",
        source_ids=("encykorea_face", "woman_donga_age_zones"),
        ranges=(
            PhysiognomyRuleRange(
                min_value=None,
                max_value=0.025,
                tag="입꼬리 균형형",
                observation="양쪽 입꼬리 높이 차이가 작아 표정이 안정적으로 보입니다.",
                interpretation=(
                    "전통 관상에서는 얼굴 표정도 함께 보므로 리포트에는 "
                    "편안하고 안정적인 표정 흐름을 보조로 넣습니다."
                ),
            ),
            PhysiognomyRuleRange(
                min_value=0.025,
                max_value=None,
                tag="입꼬리 비대칭형",
                observation="양쪽 입꼬리 높이 차이가 있어 표정 비대칭이 관찰됩니다.",
                interpretation=(
                    "촬영 순간의 표정 영향일 수 있으므로 리포트에는 단정하지 않고 "
                    "현재 표정이 한쪽으로 기울어 보인다는 보조 관찰만 남깁니다."
                ),
            ),
        ),
    ),
)

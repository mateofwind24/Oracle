from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Any

from oracle_report.vision.physiognomy_rule_repository import PhysiognomyRuleMatch


_PERSONAL_CATEGORIES = {
    "basic": "타고난 인상과 기본 상",
    "strength": "강점으로 읽히는 복과 기세",
    "relationship": "관계와 대인운",
    "direction": "앞으로 살릴 운의 방향",
    "advice": "조심할 점과 생활 조언",
}

_PAIR_CATEGORIES = {
    "first_impression": "첫인상과 분위기",
    "communication": "소통 리듬",
    "strength": "관계 강점",
    "caution": "주의할 점",
}

_PART_METRICS = {
    "balance": (
        "third_balance_error",
        "upper_zone_ratio",
        "middle_zone_ratio",
        "lower_zone_ratio",
        "face_aspect_ratio",
    ),
    "eyes": (
        "eye_width_ratio",
        "eye_aspect_ratio",
        "eye_spacing_ratio",
        "eye_tail_tilt",
        "brow_eye_span_ratio",
        "brow_eye_gap_ratio",
    ),
    "nose": (
        "nose_length_ratio",
        "nose_width_ratio",
        "nose_length_width_ratio",
    ),
    "mouth": (
        "mouth_width_ratio",
        "mouth_height_ratio",
        "mouth_balance_delta",
    ),
    "jaw": (
        "lower_zone_ratio",
        "philtrum_chin_ratio",
        "chin_length_ratio",
        "jaw_width_ratio",
    ),
}

_PART_LABELS = {
    "balance": "전체 균형",
    "eyes": "눈과 눈썹",
    "nose": "코와 중심감",
    "mouth": "입매와 표정",
    "jaw": "하관과 마무리",
}

_PERSONAL_BLOCK_PARTS = {
    "basic": ("balance", "nose"),
    "strength": ("eyes", "nose"),
    "relationship": ("eyes", "mouth"),
    "direction": ("balance", "jaw"),
    "advice": ("mouth", "jaw"),
}

_PAIR_BLOCK_PARTS = {
    "first_impression": ("balance", "nose"),
    "communication": ("eyes", "mouth"),
    "strength": ("nose", "jaw"),
    "caution": ("mouth", "jaw"),
}


@dataclass(frozen=True)
class FacePartProfile:
    key: str
    label: str
    tone: str
    matches: tuple[PhysiognomyRuleMatch, ...]
    evidence: str
    impression: str
    strength: str
    caution: str


_PERSONAL_TEMPLATES: dict[str, tuple[dict[str, str], ...]] = {
    "basic": (
        {
            "title": "{main_impression} 첫인상",
            "summary": "{primary_summary} 전체적으로 {main_impression} 흐름이 먼저 보여요.",
            "body": "{primary_evidence} {secondary_evidence} 이런 흐름은 처음 마주했을 때 과하게 튀기보다 {main_impression} 분위기로 전달될 수 있어요. 중요한 일을 시작할 때는 이 안정감을 바탕으로 순서를 세우고 차분하게 중심을 잡으면 좋아요.",
        },
        {
            "title": "차분하게 정돈된 기본 상",
            "summary": "{primary_summary} 얼굴의 기본 비율에서 정돈된 인상이 읽혀요.",
            "body": "{primary_evidence} 여기에 {secondary_label} 쪽의 {secondary_impression} 느낌이 더해져 전체 인상이 한쪽으로 치우치지 않아요. 복잡한 상황에서도 먼저 판을 살피고 필요한 부분부터 정리하는 방식이 잘 맞을 수 있어요.",
        },
        {
            "title": "중심을 잃지 않는 인상",
            "summary": "{primary_summary} 안정적인 구조가 첫인상의 바탕을 만들어줘요.",
            "body": "{primary_evidence} {secondary_evidence} 이 조합은 급하게 밀어붙이기보다 상황을 확인하고 균형을 맞추려는 인상으로 이어져요. 계획을 너무 크게 벌리기보다 작은 기준을 세워 꾸준히 이어가면 장점이 잘 살아나요.",
        },
        {
            "title": "{primary_label}이 살아나는 얼굴 흐름",
            "summary": "{primary_summary} 얼굴 전체에서 안정적인 기준점이 보여요.",
            "body": "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 분위기가 함께 읽혀요. 이런 인상은 주변에 신뢰감을 주기 쉬우니, 중요한 선택에서는 속도보다 일관성을 우선하면 좋아요.",
        },
        {
            "title": "부드럽게 균형 잡힌 기본기",
            "summary": "{primary_summary} 전체 인상은 과장보다 균형 쪽에 가까워요.",
            "body": "{primary_evidence} {secondary_evidence} 이 흐름은 스스로를 드러낼 때도 지나치게 강하게 보이기보다 안정적으로 설득하는 힘으로 이어질 수 있어요. 생활에서는 정리된 루틴을 만들수록 얼굴에서 보이는 장점이 더 자연스럽게 살아나요.",
        },
    ),
    "strength": (
        {
            "title": "{main_impression} 힘이 있는 장점",
            "summary": "{primary_summary} 강점은 또렷함과 안정감이 함께 보이는 쪽이에요.",
            "body": "{primary_evidence} {secondary_evidence} 이런 조합은 중요한 순간에 시선을 모으고 중심을 잡는 장점으로 이어질 수 있어요. 판단이 필요한 자리에서는 급히 결론을 내기보다 관찰한 내용을 차분히 정리해 말하면 더 좋은 인상을 줄 수 있어요.",
        },
        {
            "title": "차분히 힘을 모으는 기세",
            "summary": "{secondary_summary} 안정적인 중심감이 강점으로 읽혀요.",
            "body": "{secondary_evidence} 여기에 {primary_label}의 {primary_impression} 흐름이 더해져, 무리하게 앞서기보다 필요한 순간에 존재감이 살아나는 편이에요. 일을 풀어갈 때는 자신 있는 기준을 짧고 분명하게 표현하면 좋아요.",
        },
        {
            "title": "정돈된 관찰력이 주는 힘",
            "summary": "{primary_summary} 섬세하게 보고 반응하는 장점이 보여요.",
            "body": "{primary_evidence} {secondary_evidence} 이 흐름은 사람이나 상황의 변화를 그냥 지나치지 않고 포착하는 힘으로 연결될 수 있어요. 다만 생각이 많아질 때는 핵심을 하나만 골라 행동으로 옮기는 연습이 도움이 돼요.",
        },
        {
            "title": "{secondary_label}에서 살아나는 추진감",
            "summary": "{secondary_summary} 중심을 잡고 앞으로 나아가는 분위기가 있어요.",
            "body": "{secondary_evidence} {primary_evidence} 이런 얼굴 흐름은 한 번 방향을 정하면 쉽게 흐트러지지 않는 장점으로 읽혀요. 주변의 반응을 보되, 스스로 정한 기준을 꾸준히 유지할 때 기세가 더 좋아져요.",
        },
        {
            "title": "부담 없이 드러나는 존재감",
            "summary": "{primary_summary} 강한 표현보다 자연스러운 설득력이 돋보여요.",
            "body": "{primary_evidence} {secondary_evidence} 이 조합은 큰소리로 밀어붙이기보다 차근차근 신뢰를 쌓는 방식에 잘 맞아요. 작은 성과를 꾸준히 보여주면 얼굴에서 보이는 안정감이 실제 강점으로 이어질 수 있어요.",
        },
    ),
    "relationship": (
        {
            "title": "{main_impression} 대화 인상",
            "summary": "{primary_summary} 관계에서는 반응을 살피는 분위기가 먼저 보여요.",
            "body": "{primary_evidence} {secondary_evidence} 이런 흐름은 대화에서 상대의 반응을 보고 속도를 맞추는 인상으로 이어질 수 있어요. 중요한 이야기를 할 때는 표정과 말의 속도를 조금만 낮추면 부드러운 장점이 더 잘 전달돼요.",
        },
        {
            "title": "편안하게 맞춰 가는 관계 흐름",
            "summary": "{secondary_summary} 표현 방식에서 자연스러운 소통감이 보여요.",
            "body": "{secondary_evidence} 여기에 {primary_label}의 {primary_impression} 느낌이 더해져, 관계 안에서 너무 앞서기보다 분위기를 읽는 쪽에 강점이 있어요. 다만 배려가 길어지면 말하고 싶은 핵심이 흐려질 수 있으니 필요한 말은 짧게 정리해두면 좋아요.",
        },
        {
            "title": "상대를 살피는 부드러운 시선",
            "summary": "{primary_summary} 눈 주변 흐름에서 관계 감각이 읽혀요.",
            "body": "{primary_evidence} {secondary_evidence} 이런 인상은 처음부터 강하게 다가가기보다 상대의 분위기를 확인하며 가까워지는 방식에 잘 맞아요. 친밀한 관계일수록 감정을 오래 누르지 말고 편안한 표현으로 풀어내면 좋아요.",
        },
        {
            "title": "말보다 분위기를 먼저 읽는 편",
            "summary": "{primary_summary} 대인 관계에서는 관찰력이 장점이 될 수 있어요.",
            "body": "{primary_evidence} {secondary_evidence} 이 조합은 상대가 무엇을 불편해하는지 빠르게 알아차리는 인상으로 이어져요. 대신 모든 반응을 혼자 해석하려 하지 말고, 중요한 부분은 질문으로 확인하면 오해를 줄일 수 있어요.",
        },
        {
            "title": "자연스럽게 온도를 맞추는 소통",
            "summary": "{secondary_summary} 표현이 과하지 않고 편안하게 이어지는 흐름이에요.",
            "body": "{secondary_evidence} {primary_evidence} 이런 얼굴 흐름은 관계에서 급격한 변화보다 안정적인 친밀감을 쌓는 데 유리해요. 새로운 사람을 만날 때도 처음부터 많은 것을 보여주기보다 차분히 리듬을 맞추면 좋아요.",
        },
    ),
    "direction": (
        {
            "title": "{main_impression} 방향을 살리기",
            "summary": "{primary_summary} 앞으로는 안정된 기준을 꾸준히 살리는 흐름이 좋아요.",
            "body": "{primary_evidence} {secondary_evidence} 이 인상은 새로운 기회를 잡을 때도 갑작스러운 변화보다 준비된 선택에서 더 잘 살아나요. 자신의 속도를 정하고 반복 가능한 루틴을 만들면 운의 방향을 안정적으로 키울 수 있어요.",
        },
        {
            "title": "균형을 기준으로 확장하는 흐름",
            "summary": "{secondary_summary} 마무리감과 지속력이 방향의 열쇠가 될 수 있어요.",
            "body": "{secondary_evidence} 여기에 {primary_label}의 {primary_impression} 흐름이 더해져, 시작보다 유지와 정리에 강점이 생길 수 있어요. 앞으로는 큰 변화 하나보다 매일 지키는 기준 하나를 만드는 쪽이 더 잘 맞아요.",
        },
        {
            "title": "차분히 쌓아 올리는 운의 방향",
            "summary": "{primary_summary} 급하게 넓히기보다 안정적으로 키우는 흐름이에요.",
            "body": "{primary_evidence} {secondary_evidence} 이런 얼굴 흐름은 한 번에 성과를 만들기보다 시간을 두고 신뢰를 쌓을 때 더 빛나요. 일이나 관계 모두에서 마무리 기준을 분명히 하면 좋은 흐름을 오래 가져갈 수 있어요.",
        },
        {
            "title": "중심을 지키며 넓혀 가는 방향",
            "summary": "{primary_summary} 얼굴 전체의 기준점이 방향성을 잡아줘요.",
            "body": "{primary_evidence} {secondary_evidence} 이 조합은 주변 상황이 바뀌어도 스스로의 중심을 다시 잡는 힘으로 이어질 수 있어요. 앞으로는 선택지를 많이 벌리기보다 자신에게 맞는 기준을 좁혀가는 전략이 좋아요.",
        },
        {
            "title": "꾸준함으로 살아나는 흐름",
            "summary": "{secondary_summary} 시간이 갈수록 안정감이 장점으로 쌓일 수 있어요.",
            "body": "{secondary_evidence} {primary_evidence} 이런 인상은 짧은 승부보다 긴 호흡의 관계나 일에서 더 자연스럽게 빛나요. 지금부터는 잘하는 방식을 반복 가능한 습관으로 만드는 것이 가장 현실적인 방향이에요.",
        },
    ),
    "advice": (
        {
            "title": "편안한 표현을 놓치지 않기",
            "summary": "{primary_summary} 생활에서는 표현의 균형을 의식하면 좋아요.",
            "body": "{primary_evidence} {secondary_evidence} 이 흐름은 감정을 너무 급하게 드러내거나 반대로 오래 눌러두면 장점이 흐려질 수 있어요. 중요한 대화에서는 먼저 한 문장으로 마음을 정리하고, 끝까지 차분히 마무리하는 습관이 도움이 돼요.",
        },
        {
            "title": "마무리 리듬을 안정시키기",
            "summary": "{secondary_summary} 끝맺음을 차분히 하는 습관이 잘 맞아요.",
            "body": "{secondary_evidence} {primary_evidence} 이런 인상은 시작보다 마무리에서 신뢰가 쌓일 때 더 좋아져요. 약속이나 일정을 작게라도 끝까지 지키는 방식을 생활화하면 얼굴에서 보이는 안정감이 실제 평판으로 이어질 수 있어요.",
        },
        {
            "title": "속도보다 온도를 맞추기",
            "summary": "{primary_summary} 표현의 속도를 조절하면 장점이 더 부드러워져요.",
            "body": "{primary_evidence} {secondary_evidence} 이 조합은 말이나 반응이 빠를 때 의도보다 강하게 전달될 수 있어요. 중요한 관계에서는 바로 답하기보다 한 박자 쉬고, 상대가 이해한 내용을 확인하는 습관을 들이면 좋아요.",
        },
        {
            "title": "차분한 루틴으로 균형 잡기",
            "summary": "{secondary_summary} 생활 리듬을 일정하게 잡을수록 안정감이 살아나요.",
            "body": "{secondary_evidence} {primary_evidence} 이런 얼굴 흐름은 몸과 마음이 바빠질 때 표정의 여유가 줄어들 수 있어요. 하루 중 정리하는 시간을 짧게라도 고정하면 대화와 일의 마무리가 한결 편안해질 수 있어요.",
        },
        {
            "title": "부드럽게 말하고 분명히 끝내기",
            "summary": "{primary_summary} 소통에서는 부드러움과 기준을 함께 챙기면 좋아요.",
            "body": "{primary_evidence} {secondary_evidence} 이 인상은 다정하게 시작하되 결론이 흐려지면 피로해질 수 있어요. 부탁이나 거절을 할 때는 이유를 길게 늘이기보다 핵심을 분명히 말하고 따뜻하게 마무리하면 좋아요.",
        },
    ),
}

_PAIR_TEMPLATES: dict[str, tuple[dict[str, str], ...]] = {
    "first_impression": (
        {
            "title": "서로 다른 분위기가 맞물리는 첫인상",
            "summary": "두 사람의 얼굴 흐름은 {left_main} 쪽과 {right_main} 쪽이 함께 보여요.",
            "body": "{left_name}님은 {left_evidence} {right_name}님은 {right_evidence} 그래서 첫인상에서는 한쪽으로만 기울기보다 서로 다른 속도가 만나는 느낌이 생길 수 있어요. 처음부터 결론을 맞추려 하기보다 각자의 리듬을 확인하면 관계가 더 편안해져요.",
        },
        {
            "title": "차분함과 반응성이 만나는 분위기",
            "summary": "두 사람은 같은 장점보다 서로 다른 표현 방식이 먼저 보여요.",
            "body": "{left_name}님에게서는 {left_main} 흐름이, {right_name}님에게서는 {right_main} 흐름이 읽혀요. 이 차이는 어색함보다 서로의 빈 곳을 채워주는 분위기로 이어질 수 있어요. 첫 대화에서는 누가 맞는지보다 어떤 속도가 편한지 맞춰보면 좋아요.",
        },
        {
            "title": "서로의 중심을 살피는 조합",
            "summary": "얼굴의 중심감과 균형에서 관계의 기본 톤이 보여요.",
            "body": "{left_evidence} {right_evidence} 두 사람 모두 자기 방식의 중심이 있기 때문에 관계가 급하게 흔들리기보다는 천천히 맞춰지는 편이에요. 다만 초반에는 표현 방식이 다를 수 있으니 작은 반응을 자주 확인하면 좋아요.",
        },
        {
            "title": "첫인상에서 보이는 온도 차이",
            "summary": "{left_name}님과 {right_name}님은 서로 다른 온도의 인상이 섞여요.",
            "body": "{left_name}님은 {left_main} 분위기가 있고, {right_name}님은 {right_main} 분위기가 있어요. 이 온도 차이는 관계에 입체감을 줄 수 있지만 때로는 속도 차이로 느껴질 수 있어요. 처음에는 서로의 편한 대화 간격을 찾는 것이 중요해요.",
        },
        {
            "title": "부드럽게 균형을 찾는 첫 흐름",
            "summary": "두 사람의 얼굴 관찰에서는 조율할 여지가 있는 안정감이 보여요.",
            "body": "{left_evidence} {right_evidence} 이 조합은 처음부터 강하게 부딪히기보다 조금씩 서로의 기준을 확인하며 맞춰가는 관계에 가까워요. 급한 판단보다 편안한 반복 경험을 쌓으면 첫인상의 장점이 더 잘 살아나요.",
        },
    ),
    "communication": (
        {
            "title": "대화 속도를 맞추면 편한 조합",
            "summary": "소통에서는 {left_name}님의 {left_main} 흐름과 {right_name}님의 {right_main} 흐름이 만나요.",
            "body": "{left_evidence} {right_evidence} 이 차이는 대화에서 한 사람은 먼저 살피고, 다른 사람은 표현을 통해 풀어내는 식으로 나타날 수 있어요. 중요한 이야기는 바로 결론 내기보다 서로가 들은 내용을 한 번씩 확인하면 좋아요.",
        },
        {
            "title": "표현과 관찰이 섞이는 리듬",
            "summary": "두 사람은 말의 양보다 반응의 타이밍을 맞추는 게 중요해요.",
            "body": "{left_name}님은 {left_main} 인상이 있고, {right_name}님은 {right_main} 인상이 있어요. 대화에서는 이 차이가 매력으로 보일 수도 있지만 피곤할 때는 오해로 바뀔 수 있어요. 감정이 올라올수록 짧게 묻고 천천히 답하는 방식이 잘 맞아요.",
        },
        {
            "title": "서로의 반응을 읽는 소통",
            "summary": "눈과 입매 흐름에서 관계의 대화 리듬이 보여요.",
            "body": "{left_evidence} {right_evidence} 이런 조합은 말보다 표정이나 분위기로 먼저 반응을 주고받기 쉬워요. 좋은 흐름을 유지하려면 침묵을 부정적으로 해석하지 말고, 필요한 순간에는 직접 확인하는 대화가 좋아요.",
        },
        {
            "title": "속도 차이를 조율하는 대화",
            "summary": "소통에서는 빠른 반응과 차분한 확인이 함께 필요해요.",
            "body": "{left_name}님에게는 {left_main} 흐름이, {right_name}님에게는 {right_main} 흐름이 보여요. 한쪽이 빨리 표현하고 다른 한쪽이 천천히 정리하면 타이밍 차이가 생길 수 있어요. 대화의 결론보다 과정의 속도를 맞추는 것이 관계를 편하게 만들어줘요.",
        },
        {
            "title": "부드럽게 주고받는 말의 흐름",
            "summary": "두 사람의 소통은 편안한 분위기를 만들 여지가 있어요.",
            "body": "{left_evidence} {right_evidence} 이 흐름은 서로가 방어적으로 굳지만 않으면 안정적인 대화로 이어질 수 있어요. 특히 중요한 이야기는 한 번에 몰아 하기보다 짧게 나누어 말하면 더 자연스럽게 풀려요.",
        },
    ),
    "strength": (
        {
            "title": "서로의 부족한 리듬을 채우는 힘",
            "summary": "관계 강점은 서로 다른 중심감이 보완되는 데 있어요.",
            "body": "{left_evidence} {right_evidence} 두 사람은 같은 방식으로 움직이기보다 각자의 강점을 나눠 맡을 때 안정감이 커질 수 있어요. 한 사람은 방향을 잡고 다른 한 사람은 분위기를 살피는 식으로 역할을 나누면 좋아요.",
        },
        {
            "title": "차이를 장점으로 바꾸는 조합",
            "summary": "{left_main} 흐름과 {right_main} 흐름이 만나 관계에 균형을 줄 수 있어요.",
            "body": "{left_name}님은 {left_evidence} {right_name}님은 {right_evidence} 서로의 표현 방식이 다르기 때문에 처음에는 낯설 수 있지만, 익숙해지면 한쪽이 놓친 부분을 다른 쪽이 챙기는 장점이 생겨요.",
        },
        {
            "title": "안정감이 쌓이는 관계 장점",
            "summary": "두 사람은 시간을 두고 맞춰갈수록 강점이 잘 드러나요.",
            "body": "{left_evidence} {right_evidence} 이 조합은 빠른 설렘보다 신뢰가 쌓일 때 더 좋은 흐름을 만들 수 있어요. 서로의 장점을 칭찬으로 확인해주면 관계의 안정감이 훨씬 선명해져요.",
        },
        {
            "title": "현실적인 보완이 가능한 관계",
            "summary": "얼굴 흐름에서는 서로 도와줄 수 있는 역할 차이가 보여요.",
            "body": "{left_name}님의 {left_main} 분위기와 {right_name}님의 {right_main} 분위기는 같은 결을 반복하기보다 다른 결을 더해요. 그래서 역할을 분명히 나누면 관계의 피로가 줄고, 서로가 더 편하게 강점을 낼 수 있어요.",
        },
        {
            "title": "함께 있을 때 살아나는 균형",
            "summary": "두 사람은 서로 다른 인상을 통해 관계의 폭을 넓힐 수 있어요.",
            "body": "{left_evidence} {right_evidence} 이 흐름은 혼자일 때보다 함께 있을 때 더 다양한 선택지를 만들 수 있는 조합이에요. 의견이 다를 때도 차이를 문제로 보기보다 역할의 차이로 해석하면 좋아요.",
        },
    ),
    "caution": (
        {
            "title": "속도 차이를 오해하지 않기",
            "summary": "주의할 점은 서로의 표현 속도를 다르게 받아들일 수 있다는 점이에요.",
            "body": "{left_evidence} {right_evidence} 한쪽은 바로 표현하고 다른 한쪽은 정리할 시간이 필요할 수 있어요. 감정이 올라올 때는 결론을 재촉하기보다 잠시 시간을 두고 다시 이야기하는 약속을 만드는 게 좋아요.",
        },
        {
            "title": "말의 결을 부드럽게 맞추기",
            "summary": "관계가 편하려면 표현의 강도와 마무리를 함께 조절해야 해요.",
            "body": "{left_name}님은 {left_main} 흐름이 있고, {right_name}님은 {right_main} 흐름이 있어요. 이 차이가 피곤할 때는 말투나 표정을 다르게 해석하는 원인이 될 수 있어요. 중요한 대화에서는 먼저 의도를 말하고, 마지막에는 서로 이해한 내용을 확인하면 좋아요.",
        },
        {
            "title": "혼자 해석하지 않기",
            "summary": "표정과 분위기만으로 상대의 마음을 단정하지 않는 게 중요해요.",
            "body": "{left_evidence} {right_evidence} 두 사람 모두 분위기를 읽는 힘이 있지만, 그만큼 혼자 결론을 내리기 쉬운 순간도 생길 수 있어요. 불편한 지점은 돌려 말하기보다 짧고 부드럽게 직접 확인하는 편이 좋아요.",
        },
        {
            "title": "관계의 마무리를 미루지 않기",
            "summary": "주의할 점은 작은 감정을 오래 쌓아두지 않는 거예요.",
            "body": "{left_name}님의 {left_main} 흐름과 {right_name}님의 {right_main} 흐름은 평소에는 잘 맞아도 피로할 때 엇갈릴 수 있어요. 그날 생긴 불편함은 너무 오래 묵히지 말고, 짧게 정리해서 풀어내는 습관이 필요해요.",
        },
        {
            "title": "편안함 속에서도 기준 지키기",
            "summary": "서로 편해질수록 작은 약속과 말의 마무리가 중요해요.",
            "body": "{left_evidence} {right_evidence} 관계가 익숙해질수록 표현을 생략하거나 상대가 알아줄 거라고 생각하기 쉬워요. 사소한 약속일수록 분명히 말하고 지키는 방식이 두 사람의 안정감을 지켜줘요.",
        },
    ),
}


def build_personal_face_payload(
    matches: tuple[PhysiognomyRuleMatch, ...],
    seed_text: str = "",
) -> dict[str, Any]:
    profiles = _build_part_profiles(matches)
    blocks = [
        _build_personal_block(category_key, profiles, seed_text)
        for category_key in _PERSONAL_CATEGORIES
    ]
    result = {
        "face_subtitle": _build_personal_subtitle(profiles, seed_text),
        "face_blocks": blocks,
        "face_summary": _build_personal_summary(profiles, seed_text),
    }
    return result


def build_pair_face_payload(
    left_matches: tuple[PhysiognomyRuleMatch, ...],
    right_matches: tuple[PhysiognomyRuleMatch, ...],
    left_name: str,
    right_name: str,
    seed_text: str = "",
) -> dict[str, Any]:
    left_profiles = _build_part_profiles(left_matches)
    right_profiles = _build_part_profiles(right_matches)
    blocks = [
        _build_pair_block(
            category_key,
            left_profiles,
            right_profiles,
            left_name,
            right_name,
            seed_text,
        )
        for category_key in _PAIR_CATEGORIES
    ]
    result = {
        "pair_subtitle": _build_pair_subtitle(left_profiles, right_profiles, seed_text),
        "pair_blocks": blocks,
        "face_summary": (
            f"{left_name}님과 {right_name}님은 서로의 표현 속도와 중심감을 "
            "확인하며 맞춰갈 때 관계 분위기가 더 안정적으로 살아나요."
        ),
    }
    return result


def _build_part_profiles(
    matches: tuple[PhysiognomyRuleMatch, ...],
) -> dict[str, FacePartProfile]:
    result = {
        part_key: _build_part_profile(part_key, matches)
        for part_key in _PART_METRICS
    }
    return result


def _build_part_profile(
    part_key: str,
    matches: tuple[PhysiognomyRuleMatch, ...],
) -> FacePartProfile:
    part_matches = tuple(
        match for match in matches if match.metric in _PART_METRICS[part_key]
    )
    tone = _part_tone(part_key, part_matches)
    evidence = _join_sentences(
        tuple(match.observation for match in part_matches[:2]),
        f"{_PART_LABELS[part_key]}에서 무난한 균형이 관찰돼요.",
    )
    impression = _part_impression(part_key, tone)
    strength = _part_strength(part_key, tone)
    caution = _part_caution(part_key, tone)
    result = FacePartProfile(
        key=part_key,
        label=_PART_LABELS[part_key],
        tone=tone,
        matches=part_matches,
        evidence=evidence,
        impression=impression,
        strength=strength,
        caution=caution,
    )
    return result


def _part_tone(part_key: str, matches: tuple[PhysiognomyRuleMatch, ...]) -> str:
    tags = " ".join(match.tag for match in matches)
    result = "balanced"
    if part_key == "balance":
        if "편차 강조" in tags or "세로 강조" in tags:
            result = "defined"
        elif "가로 안정" in tags or "짧은 편" in tags:
            result = "soft"
    elif part_key == "eyes":
        if "상향" in tags or "넓은 편" in tags or "열린 편" in tags:
            result = "clear"
        elif "여유" in tags or "넓은" in tags:
            result = "relaxed"
        elif "좁은" in tags or "얕은" in tags:
            result = "focused"
    elif part_key == "nose":
        if "긴 편" in tags or "길고" in tags:
            result = "defined"
        elif "짧" in tags or "좁은" in tags:
            result = "soft"
    elif part_key == "mouth":
        if "넓은" in tags or "도드라진" in tags:
            result = "expressive"
        elif "좁은" in tags or "얕은" in tags:
            result = "reserved"
        elif "비대칭" in tags:
            result = "sensitive"
    elif part_key == "jaw":
        if "넓은" in tags or "긴 편" in tags:
            result = "firm"
        elif "좁은" in tags or "짧은" in tags:
            result = "light"
    return result


def _part_impression(part_key: str, tone: str) -> str:
    table = {
        "balance": {
            "balanced": "균형 잡힌 안정감",
            "defined": "또렷한 존재감",
            "soft": "부드럽고 부담 없는 안정감",
        },
        "eyes": {
            "balanced": "차분하게 정돈된 시선",
            "clear": "또렷하고 반응성 있는 시선",
            "relaxed": "여유 있게 살피는 시선",
            "focused": "집중해서 바라보는 시선",
        },
        "nose": {
            "balanced": "중심이 안정된 인상",
            "defined": "방향성이 또렷한 중심감",
            "soft": "부드럽게 잡힌 중심감",
        },
        "mouth": {
            "balanced": "자연스럽고 편안한 표현",
            "expressive": "표현이 살아 있는 분위기",
            "reserved": "말을 아끼고 정리하는 분위기",
            "sensitive": "섬세하게 반응하는 표현",
        },
        "jaw": {
            "balanced": "마무리가 안정된 하관",
            "firm": "버티는 힘이 느껴지는 하관",
            "light": "가볍고 유연한 마무리감",
        },
    }
    result = table[part_key].get(tone, table[part_key]["balanced"])
    return result


def _part_strength(part_key: str, tone: str) -> str:
    impression = _part_impression(part_key, tone)
    result = f"{impression}을 바탕으로 상황을 차분히 정리하는 힘"
    if part_key == "eyes":
        result = f"{impression}으로 상대 반응을 읽는 힘"
    elif part_key == "mouth":
        result = f"{impression}을 통해 관계의 온도를 맞추는 힘"
    elif part_key == "jaw":
        result = f"{impression}으로 일을 끝까지 이어가는 힘"
    return result


def _part_caution(part_key: str, tone: str) -> str:
    result = "속도를 급하게 올리기보다 기준을 차분히 확인하는 것"
    if part_key == "eyes":
        result = "상대 반응을 혼자 단정하지 않고 직접 확인하는 것"
    elif part_key == "mouth":
        result = "표현을 오래 참거나 한 번에 몰아내지 않는 것"
    elif part_key == "jaw":
        result = "마무리를 미루지 않고 작은 약속을 끝까지 지키는 것"
    elif tone in ("defined", "firm", "clear"):
        result = "또렷한 인상이 강하게 전달되지 않도록 말의 온도를 맞추는 것"
    return result


_PERSONAL_EXPANSION_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "basic": {
        "titles": (
            "{primary_label}에서 시작되는 첫인상",
            "차분히 정리되는 얼굴의 기본 흐름",
            "{main_impression}이 만드는 안정된 분위기",
            "과하지 않게 중심이 잡힌 인상",
            "{secondary_label}과 함께 보이는 기본기",
        ),
        "summaries": (
            "{primary_summary} 첫인상은 단정한 흐름으로 정리돼요.",
            "{secondary_summary} 전체 분위기는 안정 쪽으로 읽혀요.",
            "{part_pair_label}이 함께 얼굴의 기준점을 만들어줘요.",
            "{main_impression}이 먼저 보이고, 세부 인상도 한쪽으로 크게 치우치지 않아요.",
            "얼굴의 기본 구조에서는 급한 인상보다 차분한 정돈감이 더 잘 보여요.",
        ),
        "bodies": (
            "{primary_evidence} {secondary_evidence} 이 근거들을 함께 보면 처음 마주했을 때 튀는 인상보다 차분히 중심을 잡는 분위기로 이어질 수 있어요. 중요한 일을 시작할 때는 먼저 기준을 세우고 순서대로 움직이면 장점이 더 자연스럽게 살아나요.",
            "{primary_evidence} 여기에 {secondary_label}의 {secondary_impression} 흐름이 더해져 얼굴의 기본 상이 안정적으로 정리돼요. 생활 장면에서는 갑자기 방향을 바꾸기보다 정한 기준을 지키는 태도가 신뢰감을 만들 수 있어요.",
            "{primary_evidence} {secondary_evidence} 풀어서 보면 얼굴의 큰 흐름이 한쪽으로 과하게 밀리기보다 균형을 맞추려는 쪽에 가까워요. 새로운 환경에서는 말을 많이 앞세우기보다 차분히 관찰한 뒤 움직이는 방식이 잘 맞아요.",
            "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 느낌이 보여요. 이런 조합은 첫인상에서 부담을 주기보다 안정된 기준을 가진 사람처럼 보일 수 있어요. 중요한 선택 앞에서는 속도보다 일관성을 먼저 잡는 것이 좋아요.",
            "{primary_evidence} {secondary_evidence} 이 인상은 결정론적인 의미라기보다 얼굴에서 보이는 균형감과 중심감을 참고하는 정도로 보면 좋아요. 일상에서는 복잡한 상황을 한 번에 해결하려 하기보다 작은 단위로 나눠 정리할 때 장점이 잘 드러나요.",
        ),
    },
    "strength": {
        "titles": (
            "{primary_label}에서 읽히는 장점",
            "{secondary_label}이 받쳐주는 기세",
            "차분히 드러나는 존재감",
            "{main_impression}을 살리는 힘",
            "관찰력과 중심감이 만나는 강점",
        ),
        "summaries": (
            "{primary_summary} 강점은 무리한 표현보다 자연스러운 설득력에 가까워요.",
            "{secondary_summary} 필요한 순간에 중심을 잡는 힘이 보일 수 있어요.",
            "{part_pair_label}이 함께 장점의 방향을 만들어줘요.",
            "{main_impression}은 사람이나 상황을 차분히 읽는 힘으로 이어질 수 있어요.",
            "얼굴에서 보이는 강점은 급한 추진보다 꾸준히 신뢰를 쌓는 쪽에 가까워요.",
        ),
        "bodies": (
            "{primary_evidence} {secondary_evidence} 이 조합은 중요한 순간에 분위기를 살피면서도 필요한 기준을 잡는 장점으로 이어질 수 있어요. 판단이 필요한 자리에서는 관찰한 내용을 짧고 분명하게 정리해 말하면 좋은 인상을 줄 수 있어요.",
            "{primary_evidence} 여기에 {secondary_label}의 {secondary_impression} 흐름이 더해져, 앞에 나설 때도 과한 압박보다 안정적인 존재감으로 전달될 수 있어요. 일을 풀어갈 때는 자신 있는 기준을 세우고 작게 실행하는 방식이 잘 맞아요.",
            "{primary_evidence} {secondary_evidence} 현실적으로 보면 이 강점은 사람의 반응과 상황의 흐름을 놓치지 않는 데서 살아날 수 있어요. 생각이 많아질 때는 핵심을 하나로 좁히고 바로 행동할 수 있는 작은 단계를 만들면 좋아요.",
            "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 분위기가 함께 보여요. 이런 인상은 큰소리로 밀어붙이기보다 차근차근 신뢰를 쌓을 때 더 자연스럽게 힘을 얻어요. 약속한 것을 꾸준히 보여주는 방식이 장점을 크게 만들어줘요.",
            "{primary_evidence} {secondary_evidence} 이 흐름은 단순히 좋고 나쁨이 아니라 어떤 방식으로 힘을 쓰면 편한지를 보여주는 참고점이에요. 강점을 살리려면 상황을 먼저 읽고, 그다음 자신의 의견을 너무 길지 않게 표현하는 습관이 도움이 돼요.",
        ),
    },
    "relationship": {
        "titles": (
            "{primary_label}에서 보이는 관계 감각",
            "편안하게 온도를 맞추는 대화",
            "{secondary_label}이 만드는 소통 분위기",
            "상대를 살피며 반응하는 인상",
            "부드럽게 이어지는 대인 흐름",
        ),
        "summaries": (
            "{primary_summary} 관계에서는 상대의 반응을 살피는 흐름이 보여요.",
            "{secondary_summary} 표현 방식은 편안한 소통감으로 이어질 수 있어요.",
            "{part_pair_label}이 대화의 온도를 만들어줘요.",
            "{main_impression}은 관계에서 성급함보다 조율 쪽으로 나타날 수 있어요.",
            "대인 관계에서는 표정과 시선의 리듬을 부드럽게 맞추는 것이 좋아요.",
        ),
        "bodies": (
            "{primary_evidence} {secondary_evidence} 이런 근거를 함께 보면 대화에서 상대의 반응을 보고 속도를 맞추려는 인상으로 이어질 수 있어요. 중요한 이야기를 할 때는 표정과 말의 속도를 조금 낮추면 장점이 더 편안하게 전달돼요.",
            "{primary_evidence} 여기에 {secondary_label}의 {secondary_impression} 흐름이 더해져 관계 안에서 너무 앞서기보다 분위기를 읽는 쪽에 강점이 생길 수 있어요. 다만 배려가 길어지면 핵심이 흐려질 수 있으니 필요한 말은 짧게 정리해두면 좋아요.",
            "{primary_evidence} {secondary_evidence} 현실적인 관계 장면으로 옮기면 처음부터 강하게 다가가기보다 상대가 편안해지는 지점을 확인하며 가까워지는 방식에 잘 맞아요. 친밀한 관계일수록 감정을 오래 누르지 말고 부드럽게 풀어내면 좋아요.",
            "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 분위기가 보여요. 이 조합은 말보다 표정이나 분위기를 먼저 읽는 장점으로 이어질 수 있지만, 혼자 단정하면 오해가 생길 수 있어요. 중요한 부분은 질문으로 확인하는 습관이 도움이 돼요.",
            "{primary_evidence} {secondary_evidence} 이 흐름은 관계에서 급격한 변화보다 안정적인 친밀감을 쌓는 데 유리한 참고점이에요. 새로운 사람을 만날 때도 처음부터 많은 것을 보여주기보다 차분히 리듬을 맞추면 좋아요.",
        ),
    },
    "direction": {
        "titles": (
            "{primary_label} 기준으로 보는 방향",
            "꾸준히 쌓아가는 운의 흐름",
            "{secondary_label}에서 보는 활용법",
            "중심을 지키며 넓히는 방향",
            "생활 속에서 살아나는 안정감",
        ),
        "summaries": (
            "{primary_summary} 앞으로는 안정된 기준을 꾸준히 살리는 흐름이 좋아요.",
            "{secondary_summary} 마무리감과 지속력이 방향의 열쇠가 될 수 있어요.",
            "{part_pair_label}은 긴 호흡으로 볼수록 장점이 커져요.",
            "{main_impression}은 선택지를 넓히기보다 기준을 좁힐 때 더 잘 살아나요.",
            "운의 방향은 갑작스러운 변화보다 반복 가능한 루틴에서 안정적으로 커질 수 있어요.",
        ),
        "bodies": (
            "{primary_evidence} {secondary_evidence} 이 근거를 함께 보면 새로운 기회를 잡을 때도 갑작스러운 변화보다 준비된 선택에서 장점이 더 잘 살아날 수 있어요. 자신의 속도를 정하고 반복 가능한 루틴을 만들면 앞으로의 흐름을 안정적으로 키울 수 있어요.",
            "{primary_evidence} 여기에 {secondary_label}의 {secondary_impression} 흐름이 더해져 시작보다 유지와 정리에 강점이 생길 수 있어요. 앞으로는 큰 변화 하나보다 매일 지키는 기준 하나를 만드는 쪽이 현실적으로 잘 맞아요.",
            "{primary_evidence} {secondary_evidence} 이 인상은 한 번에 성과를 만들기보다 시간을 두고 신뢰를 쌓을 때 더 자연스럽게 빛나요. 일이나 관계 모두에서 마무리 기준을 분명히 하면 좋은 흐름을 오래 가져갈 수 있어요.",
            "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 분위기가 관찰돼요. 주변 상황이 바뀌어도 스스로의 중심을 다시 잡는 힘으로 이어질 수 있으니, 선택지를 너무 많이 벌리기보다 자신에게 맞는 기준을 좁혀가는 전략이 좋아요.",
            "{primary_evidence} {secondary_evidence} 이것은 미래를 단정하는 뜻이 아니라 얼굴에서 보이는 활용 방향을 생활 조언으로 바꾼 참고점이에요. 지금부터는 잘하는 방식을 반복 가능한 습관으로 만드는 것이 가장 현실적인 방향이에요.",
        ),
    },
    "advice": {
        "titles": (
            "표현의 온도를 편안하게 맞추기",
            "{secondary_label}에서 찾는 생활 조언",
            "작은 루틴으로 균형 잡기",
            "부드럽게 말하고 분명히 끝내기",
            "{primary_label}에서 보는 조심할 점",
        ),
        "summaries": (
            "{primary_summary} 생활에서는 표현의 균형을 의식하면 좋아요.",
            "{secondary_summary} 끝맺음을 차분히 하는 습관이 잘 맞아요.",
            "{part_pair_label}은 속도보다 마무리를 챙길 때 안정돼요.",
            "{main_impression}이 강점으로 보이려면 말의 온도와 기준을 함께 잡는 것이 좋아요.",
            "조심할 점은 장점을 누르는 것이 아니라 편안하게 쓰는 방식을 찾는 데 있어요.",
        ),
        "bodies": (
            "{primary_evidence} {secondary_evidence} 이 근거를 보면 감정을 너무 급하게 드러내거나 반대로 오래 눌러두면 장점이 흐려질 수 있어요. 중요한 대화에서는 먼저 한 문장으로 마음을 정리하고, 끝까지 차분히 마무리하는 습관이 도움이 돼요.",
            "{primary_evidence} 여기에 {secondary_label}의 {secondary_impression} 흐름이 더해져 시작보다 마무리에서 신뢰가 쌓일 때 더 좋아져요. 약속이나 일정을 작게라도 끝까지 지키는 방식을 생활화하면 얼굴에서 보이는 안정감이 실제 관계에서도 살아날 수 있어요.",
            "{primary_evidence} {secondary_evidence} 이 조합은 말이나 반응이 빠를 때 의도보다 강하게 전달될 수 있다는 점만 조심하면 좋아요. 중요한 관계에서는 바로 답하기보다 한 박자 쉬고, 상대가 이해한 내용을 확인하는 습관을 들이면 좋아요.",
            "{primary_evidence} 또한 {secondary_label}에서는 {secondary_impression} 분위기가 보여요. 몸과 마음이 바빠질 때 표정의 여유가 줄어들 수 있으니, 하루 중 정리하는 시간을 짧게라도 고정하면 대화와 일의 마무리가 편안해져요.",
            "{primary_evidence} {secondary_evidence} 이 인상은 다정하게 시작하되 결론이 흐려지면 피로해질 수 있다는 참고점도 함께 줘요. 부탁이나 거절을 할 때는 이유를 길게 늘이기보다 핵심을 분명히 말하고 따뜻하게 마무리하면 좋아요.",
        ),
    },
}

_PAIR_EXPANSION_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "first_impression": {
        "titles": (
            "서로의 분위기를 살피는 첫 흐름",
            "차분함과 반응성이 만나는 첫인상",
            "서로 다른 중심이 맞물리는 조합",
            "첫인상에서 보이는 온도 차이",
            "부드럽게 균형을 찾는 관계 분위기",
        ),
        "summaries": (
            "두 사람의 얼굴 흐름은 {left_main} 쪽과 {right_main} 쪽이 함께 보여요.",
            "{left_name}님과 {right_name}님은 같은 결보다 서로 다른 표현 방식이 먼저 보여요.",
            "첫인상에서는 한쪽으로 기울기보다 서로의 속도를 확인하는 흐름이 좋아요.",
            "얼굴의 중심감과 균형에서 관계의 기본 톤이 부드럽게 읽혀요.",
            "두 사람은 처음부터 단정하기보다 천천히 맞춰갈 여지가 있는 조합이에요.",
        ),
        "bodies": (
            "{left_name}님은 {left_evidence} {right_name}님은 {right_evidence} 이 근거를 함께 보면 첫인상에서 서로 다른 속도가 만나는 느낌이 생길 수 있어요. 처음부터 결론을 맞추려 하기보다 각자의 리듬을 확인하면 관계가 더 편안해져요.",
            "{left_name}님에게서는 {left_main} 흐름이, {right_name}님에게서는 {right_main} 흐름이 읽혀요. 이 차이는 어색함보다 서로의 빈 곳을 채워주는 분위기로 이어질 수 있어요. 첫 대화에서는 누가 맞는지보다 어떤 속도가 편한지 맞춰보면 좋아요.",
            "{left_evidence} {right_evidence} 두 사람 모두 자기 방식의 중심이 있기 때문에 관계가 급하게 흔들리기보다는 천천히 맞춰지는 편이에요. 초반에는 표현 방식이 다를 수 있으니 작은 반응을 자주 확인하면 좋아요.",
            "{left_name}님은 {left_main} 분위기가 있고, {right_name}님은 {right_main} 분위기가 있어요. 이 온도 차이는 관계에 입체감을 줄 수 있지만 때로는 속도 차이로 느껴질 수 있어요. 처음에는 서로의 편한 대화 간격을 찾는 것이 중요해요.",
            "{left_evidence} {right_evidence} 이 조합은 처음부터 강하게 부딪히기보다 조금씩 서로의 기준을 확인하며 맞춰가는 관계에 가까워요. 급한 판단보다 편안한 반복 경험을 쌓으면 첫인상의 장점이 더 잘 살아나요.",
        ),
    },
    "communication": {
        "titles": (
            "대화 속도를 맞추면 편한 조합",
            "표현과 관찰이 섞이는 리듬",
            "서로의 반응을 읽는 소통",
            "속도 차이를 조율하는 대화",
            "부드럽게 주고받는 말의 흐름",
        ),
        "summaries": (
            "소통에서는 {left_name}님의 {left_main} 흐름과 {right_name}님의 {right_main} 흐름이 만나요.",
            "두 사람은 말의 양보다 반응의 타이밍을 맞추는 게 중요해요.",
            "눈과 입매 흐름에서 관계의 대화 리듬이 보여요.",
            "소통에서는 빠른 반응과 차분한 확인이 함께 필요해요.",
            "두 사람의 소통은 편안한 분위기를 만들 여지가 있어요.",
        ),
        "bodies": (
            "{left_evidence} {right_evidence} 이 차이는 대화에서 한 사람은 먼저 살피고, 다른 사람은 표현을 통해 풀어내는 식으로 나타날 수 있어요. 중요한 이야기는 바로 결론 내기보다 서로가 들은 내용을 한 번씩 확인하면 좋아요.",
            "{left_name}님은 {left_main} 인상이 있고, {right_name}님은 {right_main} 인상이 있어요. 대화에서는 이 차이가 매력으로 보일 수도 있지만 피곤할 때는 오해로 바뀔 수 있어요. 감정이 올라올수록 짧게 묻고 천천히 답하는 방식이 잘 맞아요.",
            "{left_evidence} {right_evidence} 이런 조합은 말보다 표정이나 분위기로 먼저 반응을 주고받기 쉬워요. 좋은 흐름을 유지하려면 침묵을 부정적으로 해석하지 말고, 필요한 순간에는 직접 확인하는 대화가 좋아요.",
            "{left_name}님에게는 {left_main} 흐름이, {right_name}님에게는 {right_main} 흐름이 보여요. 한쪽이 빨리 표현하고 다른 한쪽이 천천히 정리하면 타이밍 차이가 생길 수 있어요. 대화의 결론보다 과정의 속도를 맞추는 것이 관계를 편하게 만들어줘요.",
            "{left_evidence} {right_evidence} 이 흐름은 서로가 방어적으로 굳지만 않으면 안정적인 대화로 이어질 수 있어요. 특히 중요한 이야기는 한 번에 몰아 하기보다 짧게 나누어 말하면 더 자연스럽게 풀려요.",
        ),
    },
    "strength": {
        "titles": (
            "서로의 부족한 리듬을 채우는 힘",
            "차이를 장점으로 바꾸는 조합",
            "안정감이 쌓이는 관계 장점",
            "현실적인 보완이 가능한 관계",
            "함께 있을 때 살아나는 균형",
        ),
        "summaries": (
            "관계 강점은 서로 다른 중심감이 보완되는 데 있어요.",
            "{left_main} 흐름과 {right_main} 흐름이 만나 관계에 균형을 줄 수 있어요.",
            "두 사람은 시간을 두고 맞춰갈수록 강점이 잘 드러나요.",
            "얼굴 흐름에서는 서로 도와줄 수 있는 역할 차이가 보여요.",
            "두 사람은 서로 다른 인상을 통해 관계의 폭을 넓힐 수 있어요.",
        ),
        "bodies": (
            "{left_evidence} {right_evidence} 두 사람은 같은 방식으로 움직이기보다 각자의 강점을 나눠 맡을 때 안정감이 커질 수 있어요. 한 사람은 방향을 잡고 다른 한 사람은 분위기를 살피는 식으로 역할을 나누면 좋아요.",
            "{left_name}님은 {left_evidence} {right_name}님은 {right_evidence} 서로의 표현 방식이 다르기 때문에 처음에는 낯설 수 있지만, 익숙해지면 한쪽이 놓친 부분을 다른 쪽이 챙기는 장점이 생겨요.",
            "{left_evidence} {right_evidence} 이 조합은 빠른 설렘보다 신뢰가 쌓일 때 더 좋은 흐름을 만들 수 있어요. 서로의 장점을 칭찬으로 확인해주면 관계의 안정감이 훨씬 선명해져요.",
            "{left_name}님의 {left_main} 분위기와 {right_name}님의 {right_main} 분위기는 같은 결을 반복하기보다 다른 결을 더해요. 그래서 역할을 분명히 나누면 관계의 피로가 줄고, 서로가 더 편하게 강점을 낼 수 있어요.",
            "{left_evidence} {right_evidence} 이 흐름은 혼자일 때보다 함께 있을 때 더 다양한 선택지를 만들 수 있는 조합이에요. 의견이 다를 때도 차이를 문제로 보기보다 역할의 차이로 해석하면 좋아요.",
        ),
    },
    "caution": {
        "titles": (
            "속도 차이를 오해하지 않기",
            "말의 결을 부드럽게 맞추기",
            "혼자 해석하지 않기",
            "관계의 마무리를 미루지 않기",
            "편안함 속에서도 기준 지키기",
        ),
        "summaries": (
            "주의할 점은 서로의 표현 속도를 다르게 받아들일 수 있다는 점이에요.",
            "관계가 편하려면 표현의 강도와 마무리를 함께 조절해야 해요.",
            "표정과 분위기만으로 상대의 마음을 단정하지 않는 게 중요해요.",
            "작은 감정을 오래 쌓아두지 않는 것이 관계를 편하게 만들어줘요.",
            "서로 편해질수록 작은 약속과 말의 마무리가 중요해요.",
        ),
        "bodies": (
            "{left_evidence} {right_evidence} 한쪽은 바로 표현하고 다른 한쪽은 정리할 시간이 필요할 수 있어요. 감정이 올라올 때는 결론을 재촉하기보다 잠시 시간을 두고 다시 이야기하는 약속을 만드는 게 좋아요.",
            "{left_name}님은 {left_main} 흐름이 있고, {right_name}님은 {right_main} 흐름이 있어요. 이 차이가 피곤할 때는 말투나 표정을 다르게 해석하는 원인이 될 수 있어요. 중요한 대화에서는 먼저 의도를 말하고, 마지막에는 서로 이해한 내용을 확인하면 좋아요.",
            "{left_evidence} {right_evidence} 두 사람 모두 분위기를 읽는 힘이 있지만, 그만큼 혼자 결론을 내리기 쉬운 순간도 생길 수 있어요. 불편한 지점은 돌려 말하기보다 짧고 부드럽게 직접 확인하는 편이 좋아요.",
            "{left_name}님의 {left_main} 흐름과 {right_name}님의 {right_main} 흐름은 평소에는 잘 맞아도 피로할 때 엇갈릴 수 있어요. 그날 생긴 불편함은 너무 오래 묵히지 말고, 짧게 정리해서 풀어내는 습관이 필요해요.",
            "{left_evidence} {right_evidence} 관계가 익숙해질수록 표현을 생략하거나 상대가 알아줄 거라고 생각하기 쉬워요. 사소한 약속일수록 분명히 말하고 지키는 방식이 두 사람의 안정감을 지켜줘요.",
        ),
    },
}


def _expanded_personal_templates(category_key: str) -> tuple[dict[str, str], ...]:
    result = _PERSONAL_TEMPLATES[category_key] + _compose_template_variations(
        _PERSONAL_EXPANSION_RULES[category_key],
    )
    return result


def _expanded_pair_templates(category_key: str) -> tuple[dict[str, str], ...]:
    result = _PAIR_TEMPLATES[category_key] + _compose_template_variations(
        _PAIR_EXPANSION_RULES[category_key],
    )
    return result


def _compose_template_variations(
    rules: dict[str, tuple[str, ...]],
) -> tuple[dict[str, str], ...]:
    titles = rules["titles"]
    summaries = rules["summaries"]
    bodies = rules["bodies"]
    result = tuple(
        {
            "title": title,
            "summary": summaries[(title_index + body_index) % len(summaries)],
            "body": body,
        }
        for title_index, title in enumerate(titles)
        for body_index, body in enumerate(bodies)
    )
    return result


def _build_personal_block(
    category_key: str,
    profiles: dict[str, FacePartProfile],
    seed_text: str,
) -> dict[str, str]:
    primary, secondary = (
        profiles[part_key] for part_key in _PERSONAL_BLOCK_PARTS[category_key]
    )
    template = _choose_template(
        _expanded_personal_templates(category_key),
        seed_text,
        category_key,
    )
    values = _personal_template_values(primary, secondary, seed_text, category_key)
    result = {
        "category": _PERSONAL_CATEGORIES[category_key],
        "title": template["title"].format(**values),
        "summary": template["summary"].format(**values),
        "body": template["body"].format(**values),
    }
    return result


def _build_pair_block(
    category_key: str,
    left_profiles: dict[str, FacePartProfile],
    right_profiles: dict[str, FacePartProfile],
    left_name: str,
    right_name: str,
    seed_text: str,
) -> dict[str, str]:
    primary_part, secondary_part = _PAIR_BLOCK_PARTS[category_key]
    left_profile = left_profiles[primary_part]
    right_profile = right_profiles[secondary_part]
    template = _choose_template(
        _expanded_pair_templates(category_key),
        seed_text,
        f"pair:{category_key}",
    )
    values = {
        "left_name": left_name,
        "right_name": right_name,
        "left_main": left_profile.impression,
        "right_main": right_profile.impression,
        "left_evidence": _evidence_phrase(
            left_profile,
            seed_text,
            f"pair:{category_key}:left",
        ),
        "right_evidence": _evidence_phrase(
            right_profile,
            seed_text,
            f"pair:{category_key}:right",
        ),
    }
    result = {
        "category": _PAIR_CATEGORIES[category_key],
        "title": template["title"].format(**values),
        "summary": template["summary"].format(**values),
        "body": template["body"].format(**values),
    }
    return result


def _personal_template_values(
    primary: FacePartProfile,
    secondary: FacePartProfile,
    seed_text: str,
    category_key: str,
) -> dict[str, str]:
    result = {
        "main_impression": primary.impression,
        "primary_label": primary.label,
        "primary_impression": primary.impression,
        "primary_summary": f"{primary.label}에서는 {primary.impression}이 읽혀요.",
        "primary_evidence": _evidence_phrase(
            primary,
            seed_text,
            f"{category_key}:primary",
        ),
        "secondary_label": secondary.label,
        "secondary_impression": secondary.impression,
        "secondary_summary": f"{secondary.label}에서는 {secondary.impression}이 보여요.",
        "secondary_evidence": _evidence_phrase(
            secondary,
            seed_text,
            f"{category_key}:secondary",
        ),
        "part_pair_label": _part_pair_label(primary, secondary),
        "strength": primary.strength,
        "caution": secondary.caution,
    }
    return result


def _part_pair_label(primary: FacePartProfile, secondary: FacePartProfile) -> str:
    result = f"{_compact_part_label(primary.label)} 흐름과 {_compact_part_label(secondary.label)} 흐름"
    return result


def _compact_part_label(label: str) -> str:
    result = label.replace("과 ", "/").replace("와 ", "/")
    return result


def _evidence_phrase(profile: FacePartProfile, seed_text: str, salt: str) -> str:
    observations = _profile_observations(profile)
    first = observations[0]
    second = observations[1] if len(observations) > 1 else "같은 흐름도 함께 보여요."
    templates = (
        "{label}을 보면 {first} 이 흐름이 {impression}으로 자연스럽게 이어져요.",
        "{first} 그래서 {label} 쪽은 {impression}으로 정리해서 볼 수 있어요.",
        "{label}에서는 {first} {second} 이런 근거가 {impression}을 만들어줘요.",
        "먼저 {label} 쪽에서 {first} 이 부분이 {impression}을 보여주는 단서가 돼요.",
        "{first} 여기에 {second} 이 점이 더해져 {label}의 {impression}이 부드럽게 살아나요.",
        "{label}의 흐름을 풀어보면 {first} 그래서 전체적으로 {impression}에 가까워요.",
        "{first} 이 관찰은 단정적인 의미보다 {label}에서 {impression}이 보인다는 참고점으로 보면 좋아요.",
        "{label}에서는 {first} 이 점이 첫인상에서 {impression}을 느끼게 해요.",
    )
    template = _choose_template_text(templates, seed_text, f"evidence:{salt}")
    result = template.format(
        label=profile.label,
        first=first,
        second=second,
        impression=profile.impression,
    )
    result = " ".join(result.split())
    return result


def _profile_observations(profile: FacePartProfile) -> tuple[str, ...]:
    observations = tuple(
        _soften_observation(match.observation)
        for match in profile.matches
        if match.observation.strip()
    )
    result = observations[:2]
    if not result:
        result = (f"{profile.label}에서 무난한 균형이 관찰돼요.",)
    return result


def _soften_observation(text: str) -> str:
    result = text.strip()
    replacements = (
        ("관찰됩니다.", "관찰돼요."),
        ("관찰됩니다", "관찰돼요"),
        ("안정적입니다.", "안정적이에요."),
        ("안정적입니다", "안정적이에요"),
        ("균형적입니다.", "균형적이에요."),
        ("균형적입니다", "균형적이에요"),
        ("편입니다.", "편이에요."),
        ("편입니다", "편이에요"),
        ("보입니다.", "보여요."),
        ("보입니다", "보여요"),
        ("나타납니다.", "나타나요."),
        ("나타납니다", "나타나요"),
        ("있습니다.", "있어요."),
        ("있습니다", "있어요"),
        ("입니다.", "이에요."),
        ("입니다", "이에요"),
    )
    for old_text, new_text in replacements:
        result = result.replace(old_text, new_text)
    return result


def _build_personal_subtitle(
    profiles: dict[str, FacePartProfile],
    seed_text: str,
) -> str:
    balance = profiles["balance"]
    nose = profiles["nose"]
    options = (
        f"{balance.impression}과 {nose.impression}이 함께 보이는 얼굴 흐름이에요.",
        f"전체적으로 {balance.impression}을 바탕으로 {nose.impression}이 더해져요.",
        f"{balance.label}과 {nose.label}에서 안정적인 인상 포인트가 읽혀요.",
        f"과한 표현보다 {balance.impression}이 먼저 느껴지는 인상이에요.",
        f"{nose.impression}을 중심으로 차분한 얼굴 분위기가 이어져요.",
    )
    result = _choose_template_text(options, seed_text, "subtitle")
    return result


def _build_personal_summary(
    profiles: dict[str, FacePartProfile],
    seed_text: str,
) -> str:
    eyes = profiles["eyes"]
    mouth = profiles["mouth"]
    jaw = profiles["jaw"]
    options = (
        f"{eyes.impression}과 {mouth.impression}이 어우러져, 관계에서는 편안한 온도로 자신을 표현하는 흐름이 강점이에요.",
        f"{jaw.impression}이 바탕을 잡아주고 {eyes.impression}이 더해져, 차분히 살피며 꾸준히 이어가는 장점이 보여요.",
        f"얼굴 전체에서는 {profiles['balance'].impression}이 먼저 읽히고, 생활에서는 {jaw.strength}이 잘 살아날 수 있어요.",
        f"{mouth.impression}과 {eyes.impression}이 함께 보여서, 대화에서는 속도보다 온도를 맞출 때 장점이 더 잘 드러나요.",
        f"{profiles['nose'].impression}을 기준으로 {jaw.impression}이 이어져, 중요한 순간에 흔들림을 줄이고 마무리하는 힘이 보여요.",
    )
    result = _choose_template_text(options, seed_text, "summary")
    return result


def _build_pair_subtitle(
    left_profiles: dict[str, FacePartProfile],
    right_profiles: dict[str, FacePartProfile],
    seed_text: str,
) -> str:
    options = (
        f"{left_profiles['balance'].impression}과 {right_profiles['balance'].impression}이 만나는 관계 분위기",
        f"{left_profiles['eyes'].impression}과 {right_profiles['mouth'].impression}이 섞이는 소통 흐름",
        f"서로 다른 중심감이 조율되는 얼굴 관찰",
        f"표현 속도와 안정감을 맞춰가는 관계 인상",
        f"차분한 관찰과 자연스러운 표현이 만나는 흐름",
    )
    result = _choose_template_text(options, seed_text, "pair-subtitle")
    return result


def _join_sentences(values: tuple[str, ...], default: str) -> str:
    cleaned = tuple(value.strip() for value in values if value.strip())
    result = default
    if cleaned:
        result = " ".join(cleaned)
    return result


def _choose_template(
    templates: tuple[dict[str, str], ...],
    seed_text: str,
    salt: str,
) -> dict[str, str]:
    index = _seeded_index(len(templates), seed_text, salt)
    result = templates[index]
    return result


def _choose_template_text(
    templates: tuple[str, ...],
    seed_text: str,
    salt: str,
) -> str:
    index = _seeded_index(len(templates), seed_text, salt)
    result = templates[index]
    return result


def _seeded_index(count: int, seed_text: str, salt: str) -> int:
    digest = hashlib.sha256(f"{seed_text}:{salt}".encode("utf-8")).hexdigest()
    rng = random.Random(int(digest[:16], 16))
    result = rng.randrange(count)
    return result

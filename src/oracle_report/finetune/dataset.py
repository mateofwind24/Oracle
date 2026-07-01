from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

DEFAULT_OUTPUT_PATH = Path("data/finetune/korean_cute_style_train.jsonl")
DEFAULT_METADATA_PATH = Path("data/finetune/korean_cute_style_sources.json")
DEFAULT_USER_AGENT = "oracle-report-finetune/0.2 (safe topic crawler)"
WIKIDATA_API_URL = "https://www.wikidata.org/w/api.php"
MAX_TOPIC_LENGTH = 48
DEFAULT_TARGET_COUNT = 6000
DEFAULT_EXAMPLES_PER_TOPIC = 12
DEFAULT_LIMIT_PER_TERM = 50

DEFAULT_SYSTEM_PROMPT = (
    "너는 한국어로 말하는 밝고 귀여운 애니메이션 캐릭터풍 도우미다. "
    "말투는 다정하고 발랄하지만 과장된 유아어를 남발하지 않는다. "
    "특정 저작물의 캐릭터 이름, 고유 대사, 노래 가사, 긴 원문 문장을 "
    "모방하거나 복사하지 않는다."
)

DEFAULT_SEARCH_TERMS = (
    "귀여운 말투",
    "애니메이션",
    "캐릭터",
    "마법소녀",
    "말풍선",
    "감탄사",
    "리액션",
    "응원",
    "위로",
    "칭찬",
    "친구",
    "학교생활",
    "동아리",
    "소풍",
    "간식",
    "편지",
    "고민상담",
    "첫 만남",
    "자기소개",
    "하루 시작",
    "잠들기 전",
    "기분 전환",
    "공부 계획",
    "취미",
    "날씨",
    "봄",
    "여름",
    "가을",
    "겨울",
    "축제",
    "무대",
    "꿈",
    "용기",
    "희망",
    "행운",
    "운세",
    "사주",
    "궁합",
    "성격",
    "대화",
    "약속",
    "선물",
    "작은 성공",
    "실수",
    "걱정",
    "설렘",
    "반짝",
    "포근",
    "다정",
    "발랄",
    "상냥",
    "차분",
    "명랑",
    "소중한 마음",
    "응원의 한마디",
    "오늘의 조언",
)

DEFAULT_SYNTHETIC_TOPICS = (
    "오늘 운세",
    "첫 만남에서 긴장 풀기",
    "친구에게 다정하게 응원하기",
    "공부 계획 세우기",
    "비 오는 날 기분 전환",
    "새로운 취미 시작하기",
    "사주 리포트 요약",
    "궁합 리포트 안내",
    "자신감 회복",
    "잠들기 전 위로",
    "작은 성공 축하",
    "실수한 친구 위로",
    "축제 전날 설렘",
    "무대에 오르기 전 응원",
    "아침 인사",
    "하루 마무리 인사",
)

DEFAULT_SITUATIONS = (
    "처음 만난 친구에게",
    "긴장한 사람에게",
    "운세 리포트를 읽는 사람에게",
    "공부를 시작하는 사람에게",
    "새로운 도전을 앞둔 사람에게",
    "기분이 가라앉은 사람에게",
    "칭찬이 필요한 사람에게",
    "마음을 정리하는 사람에게",
    "소중한 약속을 앞둔 사람에게",
    "하루를 마무리하는 사람에게",
)

DEFAULT_EMOTION_CUES = (
    "포근하게",
    "발랄하게",
    "상냥하게",
    "반짝이는 느낌으로",
    "조심스럽지만 밝게",
    "친구처럼 다정하게",
    "차분한 귀여움으로",
    "희망을 담아서",
)

DEFAULT_STYLE_CUES = (
    "말풍선 대사",
    "짧은 응원 멘트",
    "리포트 마무리 문장",
    "상담 답변",
    "아침 인사",
    "잠들기 전 한마디",
    "작은 축하 메시지",
    "걱정 풀어주는 답변",
)

USER_TEMPLATES = (
    "{topic}에 대해 귀엽지만 차분하게 설명해줘.",
    "{topic}을 친구에게 말하듯 짧게 알려줘.",
    "{topic} 관련해서 따뜻한 조언을 해줘.",
    "{topic}을 너무 가볍지 않게 발랄한 말투로 정리해줘.",
    "{topic} 때문에 고민하는 사람에게 한마디 해줘.",
    "{topic}을 리포트 마지막 문장처럼 다정하게 마무리해줘.",
    "{topic}을 처음 듣는 사람도 편하게 느끼도록 말해줘.",
    "{topic}에 대한 긍정적인 포인트를 짧게 말해줘.",
    "{topic}을 말풍선에 들어갈 한 문단으로 써줘.",
    "{topic}에 대해 밝은 캐릭터가 해줄 법한 말을 만들어줘.",
    "{topic}을 부담 없이 시작하도록 격려해줘.",
    "{topic}을 상냥하고 명랑한 한국어 말투로 바꿔줘.",
)

ASSISTANT_TEMPLATES = (
    "헤헤, {topic}은 마음을 살짝 밝혀주는 작은 별빛 같은 주제예요! "
    "지금 할 수 있는 한 가지를 작게 정하면 훨씬 가볍게 시작할 수 있답니다.",
    "좋아요! {topic}은 어렵게 생각하지 않아도 괜찮아요. "
    "천천히 살펴보면 나에게 맞는 힌트가 반짝 하고 보일 거예요!",
    "{topic}이라면 먼저 마음을 편하게 해도 돼요. "
    "오늘의 작은 선택 하나가 내일의 분위기를 부드럽게 바꿔줄 수 있답니다.",
    "짜잔, {topic}의 핵심은 부담을 줄이고 나다운 리듬을 찾는 거예요! "
    "급하게 결론 내리지 말고 한 걸음씩 가보면 좋아요.",
    "{topic} 때문에 마음이 콩닥거린다면 잠깐 숨을 골라봐요. "
    "괜찮아요, 지금의 고민도 충분히 다정하게 풀어갈 수 있어요!",
    "음, {topic}은 나를 더 잘 알아가는 작은 지도 같아요. "
    "오늘은 좋은 점 하나를 먼저 발견해보는 게 어떨까요?",
    "{topic}을 살펴볼 때는 완벽한 답보다 편안한 방향이 더 중요해요. "
    "포근하게 정리해보면 마음도 조금 반짝일 거예요.",
    "반짝! {topic}에는 지금의 나를 응원해주는 힌트가 숨어 있어요. "
    "작게 시작하고, 잘한 건 꼭 스스로 칭찬해줘요!",
    "후후, {topic}은 혼자 끙끙 안고 있지 않아도 되는 이야기예요. "
    "말을 꺼내는 순간부터 마음이 조금씩 가벼워질 수 있답니다.",
    "{topic}을 떠올릴 때 마음이 살짝 흔들려도 괜찮아요. "
    "작은 용기를 내면 다음 장면은 분명 더 따뜻하게 이어질 거예요!",
    "좋은 예감이 살짝 고개를 내미는 느낌이에요. {topic}은 지금의 나를 "
    "다정하게 돌보라는 신호처럼 받아들여도 좋아요.",
    "자, {topic}은 어렵고 큰 답보다 오늘의 작은 행동 하나가 중요해요. "
    "가볍게 시작해도 충분히 멋진 흐름을 만들 수 있답니다!",
    "으쌰, {topic} 앞에서 너무 완벽하려고 애쓰지 않아도 돼요. "
    "조금 서툴러도 진심이 담기면 충분히 예쁘게 전해질 거예요.",
    "{topic}을 생각하면 마음속 조명이 하나 켜지는 것 같아요. "
    "그 빛을 따라 천천히 움직이면 나다운 답에 가까워질 수 있어요.",
    "토닥토닥, {topic}은 지금 마음을 망쳤다는 뜻이 아니에요. "
    "오히려 더 부드럽게 정리해볼 기회가 찾아온 거랍니다.",
    "오늘의 {topic}은 작은 리본처럼 묶어두면 좋아요. "
    "너무 세게 잡지 말고, 살짝 웃으며 다음 걸음을 준비해봐요!",
)


@dataclass(frozen=True)
class TopicSeed:
    topic: str
    source: str
    source_url: str
    source_license: str


@dataclass(frozen=True)
class CrawlSource:
    name: str
    api_url: str
    source: str
    source_license: str
    page_url_template: str


MEDIAWIKI_SOURCES = (
    CrawlSource(
        name="wikipedia",
        api_url="https://ko.wikipedia.org/w/api.php",
        source="wikipedia_search_title",
        source_license="CC BY-SA 4.0; titles only, no article body copied",
        page_url_template="https://ko.wikipedia.org/?curid={page_id}",
    ),
    CrawlSource(
        name="wiktionary",
        api_url="https://ko.wiktionary.org/w/api.php",
        source="wiktionary_search_title",
        source_license="CC BY-SA 4.0; titles only, no article body copied",
        page_url_template="https://ko.wiktionary.org/?curid={page_id}",
    ),
    CrawlSource(
        name="wikiquote",
        api_url="https://ko.wikiquote.org/w/api.php",
        source="wikiquote_search_title",
        source_license="CC BY-SA 4.0; page titles only, no quote text copied",
        page_url_template="https://ko.wikiquote.org/?curid={page_id}",
    ),
    CrawlSource(
        name="wikibooks",
        api_url="https://ko.wikibooks.org/w/api.php",
        source="wikibooks_search_title",
        source_license="CC BY-SA 4.0; titles only, no page body copied",
        page_url_template="https://ko.wikibooks.org/?curid={page_id}",
    ),
    CrawlSource(
        name="wikinews",
        api_url="https://ko.wikinews.org/w/api.php",
        source="wikinews_search_title",
        source_license="CC BY 2.5; titles only, no article body copied",
        page_url_template="https://ko.wikinews.org/?curid={page_id}",
    ),
)

DEFAULT_SOURCE_NAMES = tuple(source.name for source in MEDIAWIKI_SOURCES) + (
    "wikidata",
)


def normalize_topic(topic: str) -> str:
    cleaned = " ".join(topic.strip().split())
    if len(cleaned) > MAX_TOPIC_LENGTH:
        cleaned = cleaned[:MAX_TOPIC_LENGTH].rstrip()
    return cleaned


def topic_contains_hangul(topic: str) -> bool:
    has_hangul = any("\uac00" <= character <= "\ud7a3" for character in topic)
    return has_hangul


def _is_safe_topic(topic: str) -> bool:
    is_safe = bool(topic)
    if is_safe:
        is_safe = topic_contains_hangul(topic)
    if is_safe:
        is_safe = not topic.startswith(("파일:", "분류:", "틀:", "사용자:", "위키"))
    if is_safe:
        is_safe = "\n" not in topic and "\r" not in topic
    return is_safe


def _build_example(topic: str, source: str, index: int) -> JsonDict:
    user_template = USER_TEMPLATES[index % len(USER_TEMPLATES)]
    assistant_template = ASSISTANT_TEMPLATES[index % len(ASSISTANT_TEMPLATES)]
    example: JsonDict = {
        "messages": [
            {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
            {"role": "user", "content": user_template.format(topic=topic)},
            {
                "role": "assistant",
                "content": assistant_template.format(topic=topic),
            },
        ],
        "metadata": {
            "topic": topic,
            "source": source,
            "style": "generic_korean_cute_anime",
            "contains_copyrighted_dialogue": False,
        },
    }
    return example


def build_cute_style_examples(
    topics: Iterable[str],
    examples_per_topic: int = 3,
    source: str = "synthetic",
) -> list[JsonDict]:
    if examples_per_topic < 1:
        raise ValueError("examples_per_topic must be at least one")

    examples: list[JsonDict] = []
    seen_topics: set[str] = set()
    for raw_topic in topics:
        topic = normalize_topic(raw_topic)
        if topic and topic not in seen_topics:
            seen_topics.add(topic)
            for index in range(examples_per_topic):
                examples.append(_build_example(topic, source, index))
    return examples


def write_jsonl_dataset(examples: Iterable[JsonDict], output_path: Path) -> int:
    rows = list(examples)
    if not rows:
        raise ValueError("dataset must contain at least one example")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
        for row in rows:
            output_file.write(json.dumps(row, ensure_ascii=False))
            output_file.write("\n")

    written_count = len(rows)
    return written_count


def _read_json_url(
    url: str,
    user_agent: str,
    retry_count: int = 2,
    retry_sleep_seconds: float = 2.0,
) -> JsonDict:
    payload: JsonDict = {}
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    for attempt in range(retry_count + 1):
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retry_count:
                time.sleep(retry_sleep_seconds * (attempt + 1))
            else:
                print(f"[finetune-data][warn] crawl failed for {url}: {error}")
        except (
            TimeoutError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            urllib.error.URLError,
        ) as error:
            print(f"[finetune-data][warn] crawl failed for {url}: {error}")
        else:
            break
    return payload


def _build_mediawiki_search_url(
    api_url: str,
    search_term: str,
    limit: int,
) -> str:
    params = {
        "action": "query",
        "list": "search",
        "srsearch": search_term,
        "srlimit": str(limit),
        "format": "json",
        "utf8": "1",
    }
    url = api_url + "?" + urllib.parse.urlencode(params)
    return url


def crawl_mediawiki_topic_titles(
    source: CrawlSource,
    search_terms: Iterable[str],
    limit_per_term: int = DEFAULT_LIMIT_PER_TERM,
    user_agent: str = DEFAULT_USER_AGENT,
    pause_seconds: float = 0.2,
) -> list[TopicSeed]:
    if limit_per_term < 1:
        raise ValueError("limit_per_term must be at least one")

    seeds: list[TopicSeed] = []
    seen_topics: set[str] = set()
    for search_term in search_terms:
        url = _build_mediawiki_search_url(
            api_url=source.api_url,
            search_term=search_term,
            limit=limit_per_term,
        )
        payload = _read_json_url(url, user_agent)
        for item in payload.get("query", {}).get("search", []):
            topic = normalize_topic(str(item.get("title", "")))
            page_id = str(item.get("pageid", ""))
            source_url = source.page_url_template.format(page_id=page_id)
            if _is_safe_topic(topic) and topic not in seen_topics:
                seen_topics.add(topic)
                seeds.append(
                    TopicSeed(
                        topic=topic,
                        source=source.source,
                        source_url=source_url,
                        source_license=source.source_license,
                    ),
                )
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return seeds


def _build_wikidata_search_url(search_term: str, limit: int) -> str:
    params = {
        "action": "wbsearchentities",
        "search": search_term,
        "language": "ko",
        "uselang": "ko",
        "type": "item",
        "limit": str(limit),
        "format": "json",
    }
    url = WIKIDATA_API_URL + "?" + urllib.parse.urlencode(params)
    return url


def crawl_wikidata_korean_labels(
    search_terms: Iterable[str],
    limit_per_term: int = DEFAULT_LIMIT_PER_TERM,
    user_agent: str = DEFAULT_USER_AGENT,
    pause_seconds: float = 0.2,
) -> list[TopicSeed]:
    if limit_per_term < 1:
        raise ValueError("limit_per_term must be at least one")

    seeds: list[TopicSeed] = []
    seen_topics: set[str] = set()
    for search_term in search_terms:
        url = _build_wikidata_search_url(search_term, limit_per_term)
        payload = _read_json_url(url, user_agent)
        for item in payload.get("search", []):
            topic = normalize_topic(str(item.get("label", "")))
            entity_id = str(item.get("id", ""))
            source_url = f"https://www.wikidata.org/wiki/{entity_id}"
            if _is_safe_topic(topic) and topic not in seen_topics:
                seen_topics.add(topic)
                seeds.append(
                    TopicSeed(
                        topic=topic,
                        source="wikidata_korean_label",
                        source_url=source_url,
                        source_license="CC0 1.0; labels only, no dialogue copied",
                    ),
                )
        if pause_seconds > 0:
            time.sleep(pause_seconds)
    return seeds


def _source_names_to_mediawiki_sources(source_names: Iterable[str]) -> list[CrawlSource]:
    requested_names = set(source_names)
    sources = [
        source
        for source in MEDIAWIKI_SOURCES
        if source.name in requested_names
    ]
    return sources


def build_seed_source_summary(seeds: Iterable[TopicSeed]) -> dict[str, int]:
    counter = Counter(seed.source for seed in seeds)
    summary = dict(sorted(counter.items()))
    return summary


def _build_local_topic_seeds() -> list[TopicSeed]:
    seeds = [
        TopicSeed(
            topic=topic,
            source="synthetic",
            source_url="local-template",
            source_license="synthetic; no copied source text",
        )
        for topic in DEFAULT_SYNTHETIC_TOPICS
    ]
    return seeds


def build_topic_seeds(
    allow_network: bool,
    search_terms: Iterable[str],
    limit_per_term: int,
    source_names: Iterable[str] = DEFAULT_SOURCE_NAMES,
    pause_seconds: float = 0.2,
) -> list[TopicSeed]:
    seeds = _build_local_topic_seeds()
    if allow_network:
        mediawiki_sources = _source_names_to_mediawiki_sources(source_names)
        for source in mediawiki_sources:
            seeds.extend(
                crawl_mediawiki_topic_titles(
                    source=source,
                    search_terms=search_terms,
                    limit_per_term=limit_per_term,
                    pause_seconds=pause_seconds,
                ),
            )
        if "wikidata" in set(source_names):
            seeds.extend(
                crawl_wikidata_korean_labels(
                    search_terms=search_terms,
                    limit_per_term=limit_per_term,
                    pause_seconds=pause_seconds,
                ),
            )
    return seeds


def build_expanded_topic_list(
    seeds: Iterable[TopicSeed],
    target_topic_count: int,
) -> list[str]:
    if target_topic_count < 1:
        raise ValueError("target_topic_count must be at least one")

    topics: list[str] = []
    seen_topics: set[str] = set()
    for seed in seeds:
        topic = normalize_topic(seed.topic)
        if _is_safe_topic(topic) and topic not in seen_topics:
            seen_topics.add(topic)
            topics.append(topic)

    combination_index = 0
    while len(topics) < target_topic_count:
        situation = DEFAULT_SITUATIONS[combination_index % len(DEFAULT_SITUATIONS)]
        emotion = DEFAULT_EMOTION_CUES[
            (combination_index // len(DEFAULT_SITUATIONS)) % len(DEFAULT_EMOTION_CUES)
        ]
        style = DEFAULT_STYLE_CUES[
            (
                combination_index
                // (len(DEFAULT_SITUATIONS) * len(DEFAULT_EMOTION_CUES))
            )
            % len(DEFAULT_STYLE_CUES)
        ]
        base_topic = DEFAULT_SYNTHETIC_TOPICS[
            (
                combination_index
                // (
                    len(DEFAULT_SITUATIONS)
                    * len(DEFAULT_EMOTION_CUES)
                    * len(DEFAULT_STYLE_CUES)
                )
            )
            % len(DEFAULT_SYNTHETIC_TOPICS)
        ]
        topic = normalize_topic(f"{situation} {emotion} {style}: {base_topic}")
        if topic not in seen_topics:
            seen_topics.add(topic)
            topics.append(topic)
        combination_index += 1
    return topics


def write_source_metadata(seeds: Iterable[TopicSeed], output_path: Path) -> int:
    rows = [
        {
            "topic": seed.topic,
            "source": seed.source,
            "source_url": seed.source_url,
            "source_license": seed.source_license,
        }
        for seed in seeds
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(rows, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    written_count = len(rows)
    return written_count


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Korean cute-anime-style chat fine-tuning data.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="JSONL training dataset path.",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Source metadata JSON path.",
    )
    parser.add_argument(
        "--allow-network",
        action="store_true",
        help="Collect open web topic seeds without copying article/dialogue text.",
    )
    parser.add_argument(
        "--source",
        action="append",
        choices=DEFAULT_SOURCE_NAMES,
        dest="source_names",
        help="Seed source to crawl. Can be repeated. Defaults to all sources.",
    )
    parser.add_argument(
        "--search-term",
        action="append",
        dest="search_terms",
        help="Search term to crawl. Can be repeated.",
    )
    parser.add_argument(
        "--limit-per-term",
        type=int,
        default=DEFAULT_LIMIT_PER_TERM,
        help="Search title/label limit for each term and source.",
    )
    parser.add_argument(
        "--examples-per-topic",
        type=int,
        default=DEFAULT_EXAMPLES_PER_TOPIC,
        help="Number of chat examples generated for each topic.",
    )
    parser.add_argument(
        "--target-count",
        type=int,
        default=DEFAULT_TARGET_COUNT,
        help="Target number of generated JSONL examples.",
    )
    parser.add_argument(
        "--pause-seconds",
        type=float,
        default=0.2,
        help="Pause between crawl API requests.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    search_terms = args.search_terms or list(DEFAULT_SEARCH_TERMS)
    source_names = args.source_names or list(DEFAULT_SOURCE_NAMES)
    seeds = build_topic_seeds(
        allow_network=args.allow_network,
        search_terms=search_terms,
        limit_per_term=args.limit_per_term,
        source_names=source_names,
        pause_seconds=args.pause_seconds,
    )
    target_topic_count = math.ceil(args.target_count / args.examples_per_topic)
    topics = build_expanded_topic_list(
        seeds=seeds,
        target_topic_count=target_topic_count,
    )
    source = "synthetic"
    if args.allow_network:
        source = "synthetic+" + "+".join(source_names)

    examples = build_cute_style_examples(
        topics,
        examples_per_topic=args.examples_per_topic,
        source=source,
    )
    examples = examples[: args.target_count]
    dataset_count = write_jsonl_dataset(examples, args.output)
    metadata_count = write_source_metadata(seeds, args.metadata_output)
    source_summary = build_seed_source_summary(seeds)
    print(f"[finetune-data] wrote {dataset_count} examples to {args.output}")
    print(f"[finetune-data] wrote {metadata_count} source rows to {args.metadata_output}")
    print(
        "[finetune-data] source summary "
        + json.dumps(source_summary, ensure_ascii=False, sort_keys=True),
    )

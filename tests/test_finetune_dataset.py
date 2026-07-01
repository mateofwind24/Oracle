from __future__ import annotations

import json
from pathlib import Path

import pytest

from oracle_report.finetune.dataset import (
    TopicSeed,
    build_cute_style_examples,
    build_expanded_topic_list,
    build_seed_source_summary,
    topic_contains_hangul,
    write_jsonl_dataset,
)
from oracle_report.finetune.train_qlora import DEFAULT_MODEL_ID, _parse_args
from oracle_report.finetune.validate import validate_adapter_dir, validate_dataset_file


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_build_cute_style_examples_uses_safe_chat_format() -> None:
    examples = build_cute_style_examples(["날씨"], examples_per_topic=1)

    example = examples[0]
    messages = example["messages"]
    system_message = messages[0]["content"]
    user_message = messages[1]["content"]
    assistant_message = messages[2]["content"]

    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert "한국어" in system_message
    assert "특정 저작물" in system_message
    assert "날씨" in user_message
    assert "날씨" in assistant_message
    assert any(token in assistant_message for token in ("요!", "답니다", "헤헤"))
    assert example["metadata"]["source"] == "synthetic"


def test_write_jsonl_dataset_round_trips_examples(tmp_path) -> None:
    output_path = tmp_path / "train.jsonl"
    examples = build_cute_style_examples(["운세"], examples_per_topic=2)

    written_count = write_jsonl_dataset(examples, output_path)
    rows = [
        json.loads(line)
        for line in output_path.read_text(encoding="utf-8").splitlines()
    ]

    assert written_count == 2
    assert len(rows) == 2
    assert rows[0]["messages"][1]["role"] == "user"
    assert rows[0]["metadata"]["topic"] == "운세"


def test_write_jsonl_dataset_rejects_empty_examples(tmp_path) -> None:
    output_path = tmp_path / "empty.jsonl"

    with pytest.raises(ValueError, match="at least one"):
        write_jsonl_dataset([], output_path)


def test_topic_contains_hangul_rejects_latin_only_titles() -> None:
    assert topic_contains_hangul("오늘 운세") is True
    assert topic_contains_hangul("Gmail") is False


def test_build_expanded_topic_list_reaches_requested_size() -> None:
    seeds = [
        TopicSeed(
            topic="응원",
            source="synthetic",
            source_url="local-template",
            source_license="synthetic",
        ),
    ]

    topics = build_expanded_topic_list(seeds, target_topic_count=40)

    assert len(topics) == 40
    assert "응원" in topics
    assert len(set(topics)) == 40


def test_build_seed_source_summary_counts_multiple_sources() -> None:
    seeds = [
        TopicSeed("응원", "synthetic", "local-template", "synthetic"),
        TopicSeed("마법소녀", "wikipedia_search_title", "https://example", "cc"),
        TopicSeed("반짝", "wiktionary_search_title", "https://example", "cc"),
    ]

    summary = build_seed_source_summary(seeds)

    assert summary == {
        "synthetic": 1,
        "wikipedia_search_title": 1,
        "wiktionary_search_title": 1,
    }


def test_validate_dataset_file_reports_style_metrics(tmp_path) -> None:
    output_path = tmp_path / "train.jsonl"
    examples = build_cute_style_examples(["응원"], examples_per_topic=3)
    write_jsonl_dataset(examples, output_path)

    report = validate_dataset_file(output_path, min_examples=3)

    assert report["ok"] is True
    assert report["example_count"] == 3
    assert report["assistant_hangul_count"] == 3
    assert report["style_marker_count"] >= 1


def test_validate_adapter_dir_requires_adapter_files(tmp_path) -> None:
    report = validate_adapter_dir(tmp_path)

    assert report["ok"] is False
    assert "adapter_config.json" in report["missing_files"]


def test_train_qlora_defaults_to_gemma4_e2b_text_style() -> None:
    args = _parse_args([])

    assert args.model_id == DEFAULT_MODEL_ID
    assert args.model_id == "unsloth/gemma-4-E2B-it"
    assert args.chat_template == "gemma-4"
    assert args.include_multimodal is False
    assert args.gpu_memory_utilization == 0.9
    assert args.min_vram_gb == 8.0
    assert args.allow_low_vram is False
    assert args.num_train_epochs == 1.0
    assert args.max_steps == -1
    assert args.dataset.as_posix() == "data/finetune/korean_cute_style_train.jsonl"


def test_shell_entrypoints_install_and_validate() -> None:
    train_script = (ROOT_DIR / "train.sh").read_text(encoding="utf-8")
    validation_script = (ROOT_DIR / "validation.sh").read_text(encoding="utf-8")
    common_script = (ROOT_DIR / "scripts" / "finetune" / "common.sh").read_text(
        encoding="utf-8",
    )

    assert "ensure_finetune_requirements" in train_script
    assert "build_korean_cute_dataset.py" in train_script
    assert "train_qlora.py" in train_script
    assert "validate_finetune.py" in validation_script
    assert "fine-tune dependencies already installed; skipping" in common_script

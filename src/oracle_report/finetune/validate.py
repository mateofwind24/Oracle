from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any


DEFAULT_DATASET_PATH = Path("data/finetune/korean_cute_style_train.jsonl")
DEFAULT_METADATA_PATH = Path("data/finetune/korean_cute_style_sources.json")
DEFAULT_ADAPTER_DIR = Path("runs/finetune/korean-cute-lora")
DEFAULT_MIN_EXAMPLES = 1000
STYLE_MARKERS = (
    "헤헤",
    "좋아요",
    "짜잔",
    "반짝",
    "토닥",
    "답니다",
    "거예요",
    "괜찮아요",
)


def _contains_hangul(text: str) -> bool:
    has_hangul = any("\uac00" <= character <= "\ud7a3" for character in text)
    return has_hangul


def _read_jsonl(path: Path) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    invalid_count = 0
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(row)
                else:
                    invalid_count += 1
            except json.JSONDecodeError:
                invalid_count += 1
    else:
        invalid_count += 1
    result = (rows, invalid_count)
    return result


def _has_expected_roles(row: dict[str, Any]) -> bool:
    messages = row.get("messages", [])
    roles = [
        message.get("role")
        for message in messages
        if isinstance(message, dict)
    ]
    has_expected_roles = roles == ["system", "user", "assistant"]
    return has_expected_roles


def _assistant_text(row: dict[str, Any]) -> str:
    text = ""
    messages = row.get("messages", [])
    if isinstance(messages, list) and len(messages) >= 3:
        assistant = messages[2]
        if isinstance(assistant, dict):
            text = str(assistant.get("content", ""))
    return text


def validate_dataset_file(
    dataset_path: Path,
    min_examples: int = DEFAULT_MIN_EXAMPLES,
) -> dict[str, Any]:
    rows, invalid_json_count = _read_jsonl(dataset_path)
    role_error_count = 0
    assistant_hangul_count = 0
    style_marker_count = 0
    unsafe_metadata_count = 0

    for row in rows:
        if not _has_expected_roles(row):
            role_error_count += 1
        assistant_text = _assistant_text(row)
        if _contains_hangul(assistant_text):
            assistant_hangul_count += 1
        if any(marker in assistant_text for marker in STYLE_MARKERS):
            style_marker_count += 1
        metadata = row.get("metadata", {})
        if metadata.get("contains_copyrighted_dialogue") is not False:
            unsafe_metadata_count += 1

    ok = (
        len(rows) >= min_examples
        and invalid_json_count == 0
        and role_error_count == 0
        and unsafe_metadata_count == 0
        and assistant_hangul_count == len(rows)
        and style_marker_count > 0
    )
    report: dict[str, Any] = {
        "ok": ok,
        "path": str(dataset_path),
        "example_count": len(rows),
        "min_examples": min_examples,
        "invalid_json_count": invalid_json_count,
        "role_error_count": role_error_count,
        "assistant_hangul_count": assistant_hangul_count,
        "style_marker_count": style_marker_count,
        "unsafe_metadata_count": unsafe_metadata_count,
    }
    return report


def validate_metadata_file(metadata_path: Path) -> dict[str, Any]:
    source_count = 0
    unique_source_count = 0
    invalid_count = 0
    if metadata_path.exists():
        try:
            rows = json.loads(metadata_path.read_text(encoding="utf-8"))
            if isinstance(rows, list):
                source_count = len(rows)
                unique_source_count = len(
                    {
                        row.get("source")
                        for row in rows
                        if isinstance(row, dict) and row.get("source")
                    },
                )
            else:
                invalid_count = 1
        except json.JSONDecodeError:
            invalid_count = 1
    else:
        invalid_count = 1

    ok = invalid_count == 0 and source_count > 0 and unique_source_count > 0
    report: dict[str, Any] = {
        "ok": ok,
        "path": str(metadata_path),
        "source_count": source_count,
        "unique_source_count": unique_source_count,
        "invalid_count": invalid_count,
    }
    return report


def validate_adapter_dir(adapter_dir: Path) -> dict[str, Any]:
    adapter_config = adapter_dir / "adapter_config.json"
    adapter_safetensors = adapter_dir / "adapter_model.safetensors"
    adapter_bin = adapter_dir / "adapter_model.bin"
    missing_files: list[str] = []
    if not adapter_config.exists():
        missing_files.append("adapter_config.json")
    if not adapter_safetensors.exists() and not adapter_bin.exists():
        missing_files.append("adapter_model.safetensors|adapter_model.bin")

    ok = adapter_dir.exists() and not missing_files
    report: dict[str, Any] = {
        "ok": ok,
        "path": str(adapter_dir),
        "missing_files": missing_files,
    }
    return report


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate fine-tuning dataset and trained adapter artifacts.",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA_PATH)
    parser.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    parser.add_argument("--min-examples", type=int, default=DEFAULT_MIN_EXAMPLES)
    parser.add_argument(
        "--allow-missing-adapter",
        action="store_true",
        help="Only validate dataset/source files and do not fail on missing adapter.",
    )
    args = parser.parse_args(argv)
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    dataset_report = validate_dataset_file(args.dataset, args.min_examples)
    metadata_report = validate_metadata_file(args.metadata)
    adapter_report = validate_adapter_dir(args.adapter_dir)
    ok = dataset_report["ok"] and metadata_report["ok"]
    if not args.allow_missing_adapter:
        ok = ok and adapter_report["ok"]

    report = {
        "ok": ok,
        "dataset": dataset_report,
        "metadata": metadata_report,
        "adapter": adapter_report,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not ok:
        sys.exit(1)

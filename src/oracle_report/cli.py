from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from oracle_report.config import (
    load_capture_config,
    load_report_llm_config,
)
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import BirthProfile
from oracle_report.prompt_templates import (
    list_prompt_template_info,
)
from oracle_report.saju.engine import SajuReading
from oracle_report.saju.repository import (
    UNKNOWN_BIRTH_TIME_REPRESENTATIVE,
    representative_time_from_time_branch,
)
from oracle_report.vision.runtime import run_capture


_DEFAULT_FACE_ANALYSIS_TEXT = "관상 분석 결과를 여기에 넣습니다."
_DEFAULT_FACE_DB_PATH = "data/face_recommendations.sqlite"
_UNKNOWN_BIRTH_TIME_VALUES = frozenset(("", "모름", "미상", "unknown", "none"))
_TOKEN_TABLE_HEADERS = (
    "name",
    "id_slot",
    "prefix_tokens",
    "body_template_tokens",
    "full_template_tokens",
)


def _apply_temperature_override(args: argparse.Namespace) -> None:
    if getattr(args, "temperature", None) is not None:
        os.environ["ORACLE_LLM_TEMPERATURE"] = str(args.temperature)
        os.environ["ORACLE_REPORT_LLM_TEMPERATURE"] = str(args.temperature)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _apply_temperature_override(args)
    result = 0
    try:
        if args.command == "capture":
            result = _run_capture_command(args)
        elif args.command == "serve":
            result = _run_serve_command(args)
        elif args.command == "token":
            result = _run_token_command(args)
        else:
            parser.print_help()
            result = 2
    except KeyboardInterrupt:
        print("cancelled")
        result = 130
    except Exception as exc:
        print(f"error: {exc}")
        result = 1
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oracle-report")
    subparsers = parser.add_subparsers(dest="command")

    capture = subparsers.add_parser("capture", help="capture a face image")
    _add_capture_args(capture)

    serve = subparsers.add_parser("serve", help="run the lightweight Flask UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--debug", action="store_true")
    serve.add_argument("--temperature", type=float, help="LLM generation temperature")
    serve.add_argument("--speculative", action="store_true", help="Enable speculative work stealing in distributed inference")
    serve.add_argument(
        "--no-local-fallback",
        action="store_true",
        dest="no_local_fallback",
        help=(
            "Disable local worker fallback. "
            "This node will only orchestrate and aggregate results from slave nodes, "
            "without performing any LLM inference itself."
        ),
    )

    token = subparsers.add_parser(
        "token",
        help="print prompt prefix token sizes",
    )
    token.add_argument(
        "--offline",
        action="store_true",
        help="estimate token counts without calling llama.cpp /tokenize",
    )

    result = parser
    return result


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-preview", action="store_true")





def _run_capture_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    artifact = run_capture(config, output_dir=args.output_dir)
    print(artifact.image_path)
    result = 0
    return result


def _run_serve_command(args: argparse.Namespace) -> int:
    from oracle_report.web import create_app

    if args.speculative:
        os.environ["ORACLE_DISTRIBUTED_SPECULATIVE"] = "1"

    if args.no_local_fallback:
        os.environ["ORACLE_DISTRIBUTED_LOCAL_FALLBACK"] = "0"

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    result = 0
    return result





def _run_token_command(args: argparse.Namespace) -> int:
    config = load_report_llm_config()
    counter = _PromptTokenCounter(config.base_url, offline=args.offline)
    rows = []
    for info in list_prompt_template_info():
        prefix_tokens = counter.count(info.prefix)
        body_tokens = counter.count(info.body_template)
        full_text = _join_prompt_parts(info.prefix, info.body_template)
        full_tokens = counter.count(full_text)
        slot_text = "" if info.slot_id is None else str(info.slot_id)
        rows.append(
            (
                info.name,
                slot_text,
                str(prefix_tokens),
                str(body_tokens),
                str(full_tokens),
            ),
        )
    print(f"source={counter.source}")
    for line in _format_table(_TOKEN_TABLE_HEADERS, rows):
        print(line)
    result = 0
    return result


def _format_table(
    headers: tuple[str, ...],
    rows: list[tuple[str, ...]],
) -> list[str]:
    all_rows = [headers, *rows]
    widths = _table_widths(all_rows)
    result = [_format_table_row(headers, widths)]
    for row in rows:
        result.append(_format_table_row(row, widths))
    return result


def _table_widths(rows: list[tuple[str, ...]]) -> tuple[int, ...]:
    widths = [0] * len(rows[0])
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    result = tuple(widths)
    return result


def _format_table_row(row: tuple[str, ...], widths: tuple[int, ...]) -> str:
    cells = []
    last_index = len(row) - 1
    for index, cell in enumerate(row):
        if index == last_index:
            cells.append(cell)
        else:
            cells.append(cell.ljust(widths[index]))
    result = "  ".join(cells).rstrip()
    return result





class _PromptTokenCounter:
    def __init__(self, base_url: str, offline: bool) -> None:
        self._base_url = base_url
        self._offline = offline
        self._server_failed = False

    @property
    def source(self) -> str:
        result = "llama.cpp /tokenize"
        if self._offline or self._server_failed:
            result = "estimated"
        return result

    def count(self, text: str) -> int:
        result = _estimate_token_count(text)
        if not self._offline and not self._server_failed:
            try:
                result = _server_token_count(self._base_url, text)
            except Exception as exc:
                self._server_failed = True
                print(f"[token][warn] using estimated counts: {exc}")
                result = _estimate_token_count(text)
        return result


def _server_token_count(base_url: str, text: str) -> int:
    import requests

    token_url = _tokenize_url(base_url)
    response = requests.post(
        token_url,
        json={"content": text},
        timeout=15.0,
    )
    if response.status_code < 200 or response.status_code >= 300:
        raise RuntimeError(f"HTTP {response.status_code} from {token_url}")
    root = response.json()
    result = 0
    tokens = root.get("tokens")
    if isinstance(tokens, list):
        result = len(tokens)
    else:
        count = root.get("count", root.get("n_tokens", 0))
        if isinstance(count, int):
            result = count
    return result


def _tokenize_url(base_url: str) -> str:
    cleaned_url = base_url.rstrip("/")
    if cleaned_url.endswith("/v1"):
        cleaned_url = cleaned_url[:-3]
    result = f"{cleaned_url}/tokenize"
    return result


def _estimate_token_count(text: str) -> int:
    result = 0
    if text.strip() != "":
        result = max(1, len(text.encode("utf-8")) // 4)
    return result


def _join_prompt_parts(prefix: str, body: str) -> str:
    result = body.strip()
    if prefix.strip() != "":
        result = f"{prefix.strip()}\n\n{body.strip()}".strip()
    return result


def _override_capture_config(args: argparse.Namespace):
    config = load_capture_config()
    camera_index = config.camera_index if args.camera_index is None else args.camera_index
    output_dir = config.output_dir if args.output_dir is None else args.output_dir
    show_preview = config.show_preview and not args.no_preview
    result = replace(
        config,
        camera_index=camera_index,
        output_dir=output_dir,
        show_preview=show_preview,
    )
    return result





if __name__ == '__main__':
    import sys
    sys.exit(main())

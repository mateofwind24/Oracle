from __future__ import annotations

import argparse
import os
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from oracle_report.config import (
    load_capture_config,
    load_face_llm_config,
    load_report_llm_config,
)
from oracle_report.llm import LlamaCppChatClient
from oracle_report.models import BirthProfile
from oracle_report.prompt_templates import (
    list_prompt_template_info,
    render_debug_prompt_template,
)
from oracle_report.recommender import format_recommendations, recommend_faces
from oracle_report.report import (
    build_couple_saju_reading_prompt,
    build_saju_reading_prompt,
)
from oracle_report.saju.engine import SajuReading
from oracle_report.saju.repository import (
    ManseLookupResult,
    ManseRepository,
    UNKNOWN_BIRTH_TIME_REPRESENTATIVE,
    birth_datetime_display_from_profile,
    birth_time_display_from_profile,
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


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = 0
    try:
        if args.command == "capture":
            result = _run_capture_command(args)
        elif args.command == "serve":
            result = _run_serve_command(args)
        elif args.command == "prompt":
            result = _run_prompt_command(args)
        elif args.command == "prompt-run":
            result = _run_prompt_result_command(args)
        elif args.command == "llm":
            result = _run_prompt_result_command(args)
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
    serve.add_argument("--distributed-warmup", action="store_true")

    prompt = subparsers.add_parser("prompt", help="print workflow prompt inputs")
    _add_prompt_args(prompt)

    prompt_run = subparsers.add_parser("prompt-run", help="run one workflow prompt")
    _add_prompt_args(prompt_run)

    llm = subparsers.add_parser(
        "llm",
        help="run one workflow prompt and print only LLM output",
    )
    _add_prompt_args(llm)

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


def _add_prompt_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "target",
        choices=(
            "personal-face-analysis",
            "compatibility-face-analysis",
            "face-analysis-copule",
            "saju-reading",
            "saju-reading-couple",
            "personal-final",
            "compatibility-final",
        ),
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--birth-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--birth-time", default="", help="HH:MM or 시진")
    parser.add_argument("--gender", required=True)
    parser.add_argument("--right-name", default="")
    parser.add_argument("--right-birth-date", default="")
    parser.add_argument("--right-birth-time", default="")
    parser.add_argument("--right-gender", default="")
    parser.add_argument("--mode", default="연인")
    parser.add_argument("--person-label", default="첫 번째 사람")
    parser.add_argument("--target-gender", default="")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--manse-db", type=Path)
    parser.add_argument("--face-db", type=Path)
    parser.add_argument("--face-analysis", default="")
    parser.add_argument("--face-analysis-file", type=Path)
    parser.add_argument("--recommendation-text", default="")
    parser.add_argument("--recommendation-file", type=Path)


def _run_capture_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    artifact = run_capture(config, output_dir=args.output_dir)
    print(artifact.image_path)
    result = 0
    return result


def _run_serve_command(args: argparse.Namespace) -> int:
    from oracle_report.web import create_app

    if args.distributed_warmup:
        os.environ["ORACLE_DISTRIBUTED_WARMUP"] = "1"

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    result = 0
    return result


def _run_prompt_command(args: argparse.Namespace) -> int:
    prompt_text = _build_prompt_text(args)
    print(prompt_text)
    result = 0
    return result


def _run_prompt_result_command(args: argparse.Namespace) -> int:
    prompt_text = _build_llm_prompt_text(args)
    output_text = prompt_text
    if args.target in (
        "personal-face-analysis",
        "compatibility-face-analysis",
        "face-analysis-copule",
    ):
        # Landmark rules face analysis does not require LLM generation.
        output_text = prompt_text
    elif args.target in (
        "saju-reading",
        "saju-reading-couple",
        "personal-final",
        "compatibility-final",
    ):
        output_text = LlamaCppChatClient(load_report_llm_config()).generate(
            prompt_text,
            image_path=None,
        )
    print(output_text)
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


def _build_llm_prompt_text(args: argparse.Namespace) -> str:
    result = _build_prompt_text(args)
    if args.target == "saju-reading":
        profile = _build_prompt_birth_profile(args)
        manse_lookup = _lookup_manse(args, profile)
        result = build_saju_reading_prompt(profile, manse_lookup.formatted_text)
    elif args.target == "saju-reading-couple":
        result = _build_couple_saju_reading_prompt_text(args)
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


def _run_landmark_analysis_on_image(image_path: Path | None) -> str:
    if image_path is None or not image_path.exists():
        return "## 관상정보\n- 오류: 이미지 경로가 지정되지 않았거나 파일이 존재하지 않습니다."
    
    import cv2
    frame = cv2.imread(str(image_path))
    if frame is None:
        return "## 관상정보\n- 오류: 이미지를 로드할 수 없습니다."
        
    from oracle_report.config import load_capture_config
    from oracle_report.vision.camera import build_capture_processors
    
    config = load_capture_config()
    detector, analyzer = build_capture_processors(config)
    
    faces = detector.detect(frame)
    if not faces:
        return "## 관상정보\n- 오류: 얼굴을 감지하지 못했습니다."
        
    face = faces[0]
    quality = analyzer.analyze(frame, face)
    if not quality.ready:
        return "## 관상정보\n- 오류: 랜드마크 분석을 위한 이미지 품질이 충분하지 않습니다."
        
    return quality.face_analysis


def _build_prompt_text(args: argparse.Namespace) -> str:
    profile = _build_prompt_birth_profile(args)
    result = ""
    if args.target in ("personal-face-analysis", "compatibility-face-analysis", "face-analysis-copule"):
        result = _run_landmark_analysis_on_image(args.image)
    elif args.target == "saju-reading":
        result = _lookup_manse(args, profile).formatted_text
    elif args.target == "saju-reading-couple":
        result = _build_pair_manse_text(args, profile)
    elif args.target == "personal-final":
        result = _build_personal_final_debug_prompt_text(args, profile)
    elif args.target == "compatibility-final":
        result = _build_compatibility_final_debug_prompt_text(args, profile)
    return result


def _build_prompt_birth_profile(args: argparse.Namespace) -> BirthProfile:
    parse_time, birth_time_known = _normalize_birth_time(args.birth_time)
    birth_datetime = _parse_birth_datetime(args.birth_date, parse_time)
    result = BirthProfile(
        name=args.name.strip(),
        birth_datetime=birth_datetime,
        gender=args.gender.strip(),
        birth_time_known=birth_time_known,
    )
    return result


def _lookup_manse(
    args: argparse.Namespace,
    profile: BirthProfile,
) -> ManseLookupResult:
    del args
    result = ManseRepository().lookup(profile)
    return result


def _build_recommendation_text(
    args: argparse.Namespace,
    reading: SajuReading,
) -> str:
    result = _read_text_option(
        args.recommendation_text,
        args.recommendation_file,
        "",
    )
    if result == "":
        db_path = _configured_path(
            args.face_db,
            "ORACLE_FACE_DB_PATH",
            _DEFAULT_FACE_DB_PATH,
        )
        recommendations = recommend_faces(
            db_path,
            args.target_gender,
            reading,
        )
        result = format_recommendations(recommendations)
    return result


def _build_personal_final_debug_prompt_text(
    args: argparse.Namespace,
    profile: BirthProfile,
) -> str:
    manse_lookup = _lookup_manse(args, profile)
    face_analysis = _read_text_option(
        args.face_analysis,
        args.face_analysis_file,
        _DEFAULT_FACE_ANALYSIS_TEXT,
    )
    recommendation_text = _build_recommendation_text(args, manse_lookup.reading)
    result = render_debug_prompt_template(
        "personal_final",
        {
            "name": profile.name,
            "gender": _gender_text(profile),
            "birth_datetime": birth_datetime_display_from_profile(profile),
            "birth_time_text": birth_time_display_from_profile(profile),
            "timezone": profile.timezone,
            "saju_text": manse_lookup.formatted_text,
            "face_analysis": face_analysis,
            "recommendation_text": recommendation_text,
        },
    )
    return result




def _build_couple_saju_reading_prompt_text(args: argparse.Namespace) -> str:
    left_profile = _build_prompt_birth_profile(args)
    right_profile = _build_right_prompt_birth_profile(args)
    left_manse = _lookup_manse(args, left_profile)
    right_manse = _lookup_manse(args, right_profile)
    result = build_couple_saju_reading_prompt(
        left_profile,
        right_profile,
        args.mode,
        left_manse.formatted_text,
        right_manse.formatted_text,
    )
    return result


def _build_pair_manse_text(
    args: argparse.Namespace,
    left_profile: BirthProfile,
) -> str:
    right_profile = _build_right_prompt_birth_profile(args)
    left_manse = _lookup_manse(args, left_profile)
    right_manse = _lookup_manse(args, right_profile)
    result = "\n\n".join((left_manse.formatted_text, right_manse.formatted_text))
    return result


def _gender_text(profile: BirthProfile) -> str:
    result = profile.gender
    if result == "":
        result = "미입력"
    return result


def _build_compatibility_final_debug_prompt_text(
    args: argparse.Namespace,
    left_profile: BirthProfile,
) -> str:
    right_profile = _build_right_prompt_birth_profile(args)
    left_manse = _lookup_manse(args, left_profile)
    right_manse = _lookup_manse(args, right_profile)
    face_analysis = _read_text_option(
        args.face_analysis,
        args.face_analysis_file,
        _DEFAULT_FACE_ANALYSIS_TEXT,
    )
    result = render_debug_prompt_template(
        "compatibility_final",
        {
            "left_name": left_profile.name,
            "left_gender": _gender_text(left_profile),
            "left_birth_datetime": birth_datetime_display_from_profile(left_profile),
            "left_birth_time_text": birth_time_display_from_profile(left_profile),
            "right_name": right_profile.name,
            "right_gender": _gender_text(right_profile),
            "right_birth_datetime": birth_datetime_display_from_profile(right_profile),
            "right_birth_time_text": birth_time_display_from_profile(right_profile),
            "mode": args.mode,
            "left_saju_text": left_manse.formatted_text,
            "right_saju_text": right_manse.formatted_text,
            "face_analysis": face_analysis,
        },
    )
    return result


def _build_right_prompt_birth_profile(args: argparse.Namespace) -> BirthProfile:
    if args.right_name.strip() == "":
        raise ValueError("--right-name is required for compatibility-final.")
    if args.right_birth_date.strip() == "":
        raise ValueError("--right-birth-date is required for compatibility-final.")
    if args.right_gender.strip() == "":
        raise ValueError("--right-gender is required for compatibility-final.")
    parse_time, birth_time_known = _normalize_birth_time(args.right_birth_time)
    birth_datetime = _parse_birth_datetime(args.right_birth_date, parse_time)
    result = BirthProfile(
        name=args.right_name.strip(),
        birth_datetime=birth_datetime,
        gender=args.right_gender.strip(),
        birth_time_known=birth_time_known,
    )
    return result


def _configured_path(
    argument_path: Path | None,
    env_name: str,
    default_path: str,
) -> Path:
    result = argument_path
    if result is None:
        result = Path(os.getenv(env_name, default_path))
    return result


def _read_text_option(
    inline_text: str,
    file_path: Path | None,
    default_text: str,
) -> str:
    result = default_text
    if inline_text.strip() != "":
        result = inline_text
    if file_path is not None:
        result = file_path.read_text(encoding="utf-8")
    return result


def _normalize_birth_time(birth_time: str) -> tuple[str, bool]:
    cleaned_time = birth_time.strip()
    birth_time_known = cleaned_time.lower() not in _UNKNOWN_BIRTH_TIME_VALUES
    parse_time = cleaned_time
    if not birth_time_known:
        parse_time = UNKNOWN_BIRTH_TIME_REPRESENTATIVE
    else:
        time_branch_time = representative_time_from_time_branch(cleaned_time)
        if time_branch_time is not None:
            parse_time = time_branch_time
    result = (parse_time, birth_time_known)
    return result


def _parse_birth_datetime(birth_date: str, birth_time: str) -> datetime:
    result = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    return result


if __name__ == '__main__':
    import sys
    sys.exit(main())

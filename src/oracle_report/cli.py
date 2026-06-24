from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from oracle_report.config import load_capture_config, load_llm_config
from oracle_report.models import BirthProfile
from oracle_report.physiognomy import FaceReadingInput
from oracle_report.report import ReportRequest, generate_report
from oracle_report.vision.runtime import run_capture


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    result = 0
    try:
        if args.command == "capture":
            result = _run_capture_command(args)
        elif args.command == "report":
            result = _run_report_command(args)
        elif args.command == "run":
            result = _run_full_command(args)
        elif args.command == "serve":
            result = _run_serve_command(args)
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

    report = subparsers.add_parser("report", help="generate a report from inputs")
    _add_report_args(report)

    run = subparsers.add_parser("run", help="capture then generate a report")
    _add_capture_args(run)
    _add_report_args(run)

    serve = subparsers.add_parser("serve", help="run the lightweight Flask UI")
    serve.add_argument("--host", default="0.0.0.0")
    serve.add_argument("--port", type=int, default=8501)
    serve.add_argument("--debug", action="store_true")

    result = parser
    return result


def _add_capture_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--camera-index", type=int)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--face-analysis-mode", type=int, choices=(1, 2))


def _add_report_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True)
    parser.add_argument("--birth-date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--birth-time", required=True, help="HH:MM")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--output", type=Path)


def _run_capture_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    artifact = run_capture(config, output_dir=args.output_dir)
    print(artifact.image_path)
    result = 0
    return result


def _run_report_command(args: argparse.Namespace) -> int:
    request = _build_report_request(args, image_path=args.image)
    artifact = generate_report(request, load_llm_config())
    _print_report_result(artifact.markdown, artifact.output_path)
    result = 0
    return result


def _run_full_command(args: argparse.Namespace) -> int:
    config = _override_capture_config(args)
    capture_artifact = run_capture(config, output_dir=args.output_dir)
    request = _build_report_request(args, image_path=capture_artifact.image_path)
    face_input = FaceReadingInput(
        image_path=capture_artifact.image_path,
        quality=capture_artifact.quality,
    )
    request = ReportRequest(
        birth_profile=request.birth_profile,
        face_input=face_input,
        output_path=request.output_path,
    )
    artifact = generate_report(request, load_llm_config())
    _print_report_result(artifact.markdown, artifact.output_path)
    result = 0
    return result


def _run_serve_command(args: argparse.Namespace) -> int:
    from oracle_report.web import create_app

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)
    result = 0
    return result


def _override_capture_config(args: argparse.Namespace):
    config = load_capture_config()
    camera_index = config.camera_index if args.camera_index is None else args.camera_index
    output_dir = config.output_dir if args.output_dir is None else args.output_dir
    show_preview = config.show_preview and not args.no_preview
    face_analysis_mode = (
        config.face_analysis_mode
        if args.face_analysis_mode is None
        else args.face_analysis_mode
    )
    result = replace(
        config,
        camera_index=camera_index,
        output_dir=output_dir,
        show_preview=show_preview,
        face_analysis_mode=face_analysis_mode,
    )
    return result


def _build_report_request(
    args: argparse.Namespace,
    image_path: Path | None,
) -> ReportRequest:
    birth_datetime = _parse_birth_datetime(args.birth_date, args.birth_time)
    birth_profile = BirthProfile(name=args.name, birth_datetime=birth_datetime)
    face_input = FaceReadingInput(image_path=image_path, quality=None)
    result = ReportRequest(
        birth_profile=birth_profile,
        face_input=face_input,
        output_path=args.output,
    )
    return result


def _parse_birth_datetime(birth_date: str, birth_time: str) -> datetime:
    result = datetime.strptime(f"{birth_date} {birth_time}", "%Y-%m-%d %H:%M")
    return result


def _print_report_result(markdown: str, output_path: Path | None) -> None:
    if output_path is None:
        print(markdown)
    else:
        print(output_path)

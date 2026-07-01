from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np

from oracle_report.config import CaptureConfig
from oracle_report.models import CaptureArtifact, CaptureDecision, FaceBox, FaceQuality
from oracle_report.vision.capture import FaceCaptureHarness, save_capture_artifact
from oracle_report.vision.camera import (
    build_capture_processors,
    draw_overlay,
    open_camera,
)
from oracle_report.vision.landmarks import (
    LandmarkMetrics,
    _evaluate_physio_rules,
    _format_prompt_observation_context,
    _format_prompt_metric_snapshot,
    _format_prompt_rule_hints,
    _physiognomy_rule_repository,
    build_rule_based_face_payload_json,
    build_rule_based_face_analysis,
    rule_matches_to_json,
)


FrameCallback = Callable[..., None]


def run_capture(
    config: CaptureConfig,
    output_dir: Path | None = None,
    frame_callback: FrameCallback | None = None,
    show_capture_guide: bool = True,
) -> CaptureArtifact:
    destination = output_dir or config.output_dir
    if config.mock_capture_enabled:
        return _build_mock_capture_artifact(config, destination)
    cv2, capture = open_camera(config)
    detector, analyzer = build_capture_processors(config)
    harness = FaceCaptureHarness(
        detector=detector,
        quality_analyzer=analyzer,
        min_face_seconds=config.min_face_seconds,
        face_min_size_px=config.face_min_size_px,
    )
    artifact: CaptureArtifact | None = None
    latest_decision = CaptureDecision(
        state="searching",
        elapsed_seconds=0.0,
        face=None,
        quality=None,
        should_capture=False,
        message="정면 얼굴을 카메라 중앙에 맞춰 주세요.",
    )

    try:
        while artifact is None:
            ok, raw_frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read camera frame")
            latest_decision = harness.observe(raw_frame)
            faces = [] if latest_decision.face is None else [latest_decision.face]
            preview_frame = raw_frame.copy()
            draw_overlay(
                cv2,
                preview_frame,
                latest_decision.message,
                faces,
                latest_decision.state == "warning",
                latest_decision.landmark_points,
                show_capture_guide,
            )
            if frame_callback is not None:
                _publish_preview_frame(frame_callback, cv2, preview_frame, latest_decision)
            if config.show_preview:
                cv2.imshow("oracle-report", preview_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt("capture cancelled")
            if latest_decision.should_capture:
                artifact = save_capture_artifact(raw_frame, latest_decision, destination)
    finally:
        capture.release()
        if config.show_preview:
            cv2.destroyAllWindows()

    result = artifact
    return result


def _publish_preview_frame(
    frame_callback: FrameCallback,
    cv2: Any,
    preview_frame: Any,
    decision: CaptureDecision,
) -> None:
    try:
        frame_callback(cv2, preview_frame, decision)
    except TypeError:
        frame_callback(cv2, preview_frame)


def _build_mock_capture_artifact(
    config: CaptureConfig,
    output_dir: Path,
) -> CaptureArtifact:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "capture.jpg"
    _write_mock_capture_image(image_path, config.frame_width, config.frame_height)

    face = FaceBox(
        x=max(0, config.frame_width // 4),
        y=max(0, config.frame_height // 6),
        width=max(96, config.frame_width // 2),
        height=max(96, (config.frame_height * 2) // 3),
        confidence=1.0,
    )
    metrics = _mock_landmark_metrics(config)
    matches = _evaluate_physio_rules(metrics, _physiognomy_rule_repository())
    quality = FaceQuality(
        ready=True,
        warnings=(),
        eye_count=metrics.eye_count,
        eyebrow_score=metrics.eyebrow_score,
        frontality_score=metrics.frontality_score,
        occlusion_score=metrics.occlusion_score,
        landmark_points=((220, 170), (420, 170), (320, 250), (260, 340), (380, 340)),
        landmark_metrics_text=_format_prompt_metric_snapshot(metrics),
        landmark_context_text=_format_prompt_observation_context(matches),
        landmark_rules_text=_format_prompt_rule_hints(matches),
        face_analysis=build_rule_based_face_analysis(metrics, matches),
        face_payload_json=build_rule_based_face_payload_json(metrics, matches),
        landmark_matches_json=rule_matches_to_json(matches),
    )
    return CaptureArtifact(
        image_path=image_path,
        face=face,
        captured_at=datetime.now(),
        quality=quality,
        landmark_points=quality.landmark_points,
        face_analysis=quality.face_analysis,
    )


def _write_mock_capture_image(image_path: Path, width: int, height: int) -> None:
    custom_mock_path = Path("mock_face.jpg")
    if custom_mock_path.exists():
        import shutil
        try:
            shutil.copy(custom_mock_path, image_path)
            return
        except Exception as e:
            print(f"[Warning] Failed to copy custom mock_face.jpg: {e}")

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError("Pillow is required to generate mock capture artifacts.") from exc

    safe_width = max(320, width)
    safe_height = max(240, height)
    image = Image.new("RGB", (safe_width, safe_height), color=(24, 28, 34))
    draw = ImageDraw.Draw(image)
    draw.rectangle(
        (
            safe_width // 4,
            safe_height // 6,
            (safe_width * 3) // 4,
            (safe_height * 5) // 6,
        ),
        outline=(0, 210, 255),
        width=4,
    )
    draw.text((24, 24), "mock landmark capture", fill=(255, 255, 255))
    image.save(image_path, format="JPEG")


def _mock_landmark_metrics(config: CaptureConfig) -> LandmarkMetrics:
    values = asdict(
        LandmarkMetrics(
            frontality_score=0.94,
            occlusion_score=0.96,
            eye_count=2,
            eyebrow_score=0.08,
            face_aspect_ratio=1.32,
            eye_width_ratio=0.182,
            eye_height_ratio=0.051,
            eye_aspect_ratio=0.279,
            eye_spacing_ratio=0.286,
            eye_tail_tilt=0.012,
            nose_length_ratio=0.241,
            mouth_width_ratio=0.362,
            mouth_height_ratio=0.043,
            lower_face_ratio=0.334,
            nose_length_width_ratio=1.418,
            mouth_corner_delta=0.004,
            upper_zone_ratio=0.332,
            middle_zone_ratio=0.338,
            lower_zone_ratio=0.330,
            third_balance_error=0.008,
            brow_eye_span_ratio=1.080,
            brow_eye_gap_ratio=0.081,
            nose_width_ratio=0.190,
            philtrum_chin_ratio=0.262,
            chin_length_ratio=0.214,
            jaw_width_ratio=0.662,
            mouth_balance_delta=0.006,
        )
    )
    if config.mock_landmark_metrics_json.strip() != "":
        try:
            overrides = json.loads(config.mock_landmark_metrics_json)
        except json.JSONDecodeError as exc:
            raise RuntimeError("ORACLE_MOCK_LANDMARK_METRICS_JSON must be valid JSON.") from exc
        if not isinstance(overrides, dict):
            raise RuntimeError("ORACLE_MOCK_LANDMARK_METRICS_JSON must be a JSON object.")
        for key, raw_value in overrides.items():
            if key not in values:
                raise RuntimeError(f"unsupported mock landmark metric: {key}")
            if key == "eye_count":
                values[key] = int(raw_value)
            else:
                values[key] = float(raw_value)
    return LandmarkMetrics(**values)

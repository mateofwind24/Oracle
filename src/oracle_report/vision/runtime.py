from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from oracle_report.config import CaptureConfig
from oracle_report.models import CaptureArtifact, CaptureDecision
from oracle_report.vision.capture import FaceCaptureHarness, save_capture_artifact
from oracle_report.vision.camera import (
    build_capture_processors,
    draw_overlay,
    open_camera,
)


FrameCallback = Callable[[Any, Any], None]


def run_capture(
    config: CaptureConfig,
    output_dir: Path | None = None,
    frame_callback: FrameCallback | None = None,
) -> CaptureArtifact:
    destination = output_dir or config.output_dir
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
            )
            if frame_callback is not None:
                frame_callback(cv2, preview_frame)
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

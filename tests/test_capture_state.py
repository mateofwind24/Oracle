from __future__ import annotations

import numpy as np

from oracle_report.models import FaceBox, FaceQuality
from oracle_report.vision.capture import FaceCaptureHarness


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        result = self.now
        return result


class FakeDetector:
    def __init__(self, face: FaceBox | list[FaceBox] | None) -> None:
        self.face = face

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        if self.face is None:
            result = []
        elif isinstance(self.face, list):
            result = self.face
        else:
            result = [self.face]
        return result


class FakeAnalyzer:
    def __init__(self, quality: FaceQuality) -> None:
        self.quality = quality

    def analyze(self, frame: np.ndarray, face: FaceBox) -> FaceQuality:
        result = self.quality
        return result


def test_capture_requires_two_seconds_of_stable_face() -> None:
    clock = FakeClock()
    detector = FakeDetector(FaceBox(10, 10, 120, 120))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    first = harness.observe(frame)
    clock.now = 1.5
    second = harness.observe(frame)
    clock.now = 2.1
    third = harness.observe(frame)

    assert first.should_capture is False
    assert second.should_capture is False
    assert third.should_capture is True
    assert third.state == "captured"


def test_quality_warning_blocks_capture() -> None:
    clock = FakeClock()
    detector = FakeDetector(FaceBox(10, 10, 120, 120))
    analyzer = FakeAnalyzer(
        FaceQuality(
            ready=False,
            warnings=("눈을 감았거나 눈 영역이 충분히 보이지 않습니다.",),
            eye_count=0,
            eyebrow_score=0.05,
        ),
    )
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    harness.observe(frame)
    clock.now = 2.2
    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "눈" in decision.message


def test_multiple_faces_are_warning() -> None:
    clock = FakeClock()
    detector = FakeDetector(
        [
            FaceBox(10, 10, 120, 120),
            FaceBox(160, 10, 120, 120),
        ],
    )
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "한 명" in decision.message


def test_tracking_resets_when_face_disappears() -> None:
    clock = FakeClock()
    detector = FakeDetector(FaceBox(10, 10, 120, 120))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((240, 320, 3), dtype=np.uint8)

    harness.observe(frame)
    clock.now = 1.8
    detector.face = None
    missing = harness.observe(frame)
    clock.now = 2.3
    detector.face = FaceBox(10, 10, 120, 120)
    after_reset = harness.observe(frame)

    assert missing.state == "searching"
    assert after_reset.elapsed_seconds == 0.0
    assert after_reset.should_capture is False

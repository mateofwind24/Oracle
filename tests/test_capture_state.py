from __future__ import annotations

import numpy as np

from oracle_report.models import FaceBox, FaceQuality
from oracle_report.vision.capture import FaceCaptureHarness
from oracle_report.vision.framing import build_capture_guide


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


def test_capture_guide_places_larger_head_box_near_center() -> None:
    guide = build_capture_guide(640, 480)

    assert guide.head_box == FaceBox(229, 96, 182, 228)


def test_capture_requires_two_seconds_of_stable_face() -> None:
    clock = FakeClock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeDetector(_guide_face(frame))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)

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
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeDetector(_guide_face(frame))
    analyzer = FakeAnalyzer(
        FaceQuality(
            ready=False,
            warnings=("눈을 감았거나 눈 영역이 충분히 보이지 않습니다.",),
            eye_count=0,
            eyebrow_score=0.05,
        ),
    )
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)

    harness.observe(frame)
    clock.now = 2.2
    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "눈" in decision.message


def test_multiple_faces_are_warning() -> None:
    clock = FakeClock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    guide_face = _guide_face(frame)
    detector = FakeDetector(
        [
            guide_face,
            FaceBox(410, 80, 150, 188),
        ],
    )
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)

    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "한 명" in decision.message


def test_tracking_resets_when_face_disappears() -> None:
    clock = FakeClock()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    detector = FakeDetector(_guide_face(frame))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)

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


def test_face_must_be_in_center_guide_before_capture() -> None:
    clock = FakeClock()
    detector = FakeDetector(FaceBox(10, 160, 150, 188))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    harness.observe(frame)
    clock.now = 2.2
    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "중앙" in decision.message


def test_face_must_match_guide_depth_before_capture() -> None:
    clock = FakeClock()
    detector = FakeDetector(FaceBox(270, 117, 100, 125))
    analyzer = FakeAnalyzer(FaceQuality(ready=True, eye_count=2, eyebrow_score=0.05))
    harness = FaceCaptureHarness(detector, analyzer, clock=clock)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    harness.observe(frame)
    clock.now = 2.2
    decision = harness.observe(frame)

    assert decision.should_capture is False
    assert decision.state == "warning"
    assert "가까이" in decision.message


def _guide_face(frame: np.ndarray) -> FaceBox:
    guide = build_capture_guide(frame.shape[1], frame.shape[0])
    result = guide.head_box
    return result

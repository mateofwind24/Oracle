from __future__ import annotations

import pytest

from oracle_report.config import load_capture_config


def test_default_capture_config_uses_lightweight_face_detection(monkeypatch) -> None:
    monkeypatch.delenv("ORACLE_CAMERA_FPS", raising=False)
    monkeypatch.delenv("ORACLE_FACE_DETECTION_SCALE", raising=False)
    monkeypatch.delenv("ORACLE_FACE_DETECTION_INTERVAL", raising=False)
    monkeypatch.delenv("ORACLE_FACE_ANALYSIS_MODE", raising=False)

    config = load_capture_config()

    assert config.camera_fps == 15
    assert config.face_detection_scale == 0.5
    assert config.face_detection_interval == 2
    assert config.face_analysis_mode == 2


def test_rejects_invalid_face_detection_scale(monkeypatch) -> None:
    monkeypatch.setenv("ORACLE_FACE_DETECTION_SCALE", "1.5")

    with pytest.raises(ValueError, match="ORACLE_FACE_DETECTION_SCALE"):
        load_capture_config()



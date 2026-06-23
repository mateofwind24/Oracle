from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:

    def _load_dotenv() -> None:
        pass


_LLM_BASE_URL_ENV_NAMES = (
    "ORACLE_LLM_BASE_URL",
    "ORACLE_FACE_LLM_BASE_URL",
    "ORACLE_REPORT_LLM_BASE_URL",
)


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int
    frame_width: int
    frame_height: int
    camera_fps: int
    min_face_seconds: float
    face_min_size_px: int
    face_detection_scale: float
    face_detection_interval: int
    output_dir: Path
    show_preview: bool
    eye_min_count: int
    eyebrow_min_edge_density: float


@dataclass(frozen=True)
class LlmConfig:
    model: str
    base_url: str
    timeout_seconds: float
    max_output_tokens: int
    temperature: float
    send_image: bool


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    debug: bool


def load_capture_config() -> CaptureConfig:
    _load_dotenv()
    result = CaptureConfig(
        camera_index=_read_int("ORACLE_CAMERA_INDEX", 0),
        frame_width=_read_int("ORACLE_FRAME_WIDTH", 640),
        frame_height=_read_int("ORACLE_FRAME_HEIGHT", 480),
        camera_fps=_read_positive_int("ORACLE_CAMERA_FPS", 15),
        min_face_seconds=_read_float("ORACLE_MIN_FACE_SECONDS", 2.0),
        face_min_size_px=_read_int("ORACLE_FACE_MIN_SIZE_PX", 96),
        face_detection_scale=_read_detection_scale(),
        face_detection_interval=_read_positive_int(
            "ORACLE_FACE_DETECTION_INTERVAL",
            2,
        ),
        output_dir=Path(os.getenv("ORACLE_OUTPUT_DIR", "runs")),
        show_preview=_read_bool("ORACLE_SHOW_PREVIEW", True),
        eye_min_count=_read_int("ORACLE_EYE_MIN_COUNT", 2),
        eyebrow_min_edge_density=_read_float(
            "ORACLE_EYEBROW_MIN_EDGE_DENSITY",
            0.018,
        ),
    )
    return result


def load_llm_config() -> LlmConfig:
    result = load_report_llm_config()
    return result


def load_face_llm_config() -> LlmConfig:
    result = _load_llm_config("ORACLE_FACE_LLM", send_image_default=True)
    return result


def load_report_llm_config() -> LlmConfig:
    result = _load_llm_config("ORACLE_REPORT_LLM", send_image_default=False)
    return result


def load_app_config() -> AppConfig:
    _load_dotenv()
    result = AppConfig(
        host=os.getenv("ORACLE_APP_HOST", "0.0.0.0"),
        port=_read_positive_int("ORACLE_APP_PORT", 8501),
        debug=_read_bool("ORACLE_APP_DEBUG", False),
    )
    return result


def _load_llm_config(prefix: str, send_image_default: bool) -> LlmConfig:
    _load_dotenv()
    _validate_configured_local_llm_urls()
    base_url = os.getenv(
        f"{prefix}_BASE_URL",
        os.getenv("ORACLE_LLM_BASE_URL", "http://127.0.0.1:8080/v1"),
    )
    _validate_local_llm_url(base_url)
    result = LlmConfig(
        model=os.getenv(
            f"{prefix}_MODEL",
            os.getenv("ORACLE_LLM_MODEL", "local-model"),
        ),
        base_url=base_url,
        timeout_seconds=_read_float(
            f"{prefix}_TIMEOUT_SECONDS",
            _read_float("ORACLE_LLM_TIMEOUT_SECONDS", 60.0),
        ),
        max_output_tokens=_read_int(
            f"{prefix}_MAX_OUTPUT_TOKENS",
            _read_int("ORACLE_MAX_OUTPUT_TOKENS", 1800),
        ),
        temperature=_read_float(
            f"{prefix}_TEMPERATURE",
            _read_float("ORACLE_LLM_TEMPERATURE", 0.7),
        ),
        send_image=_read_bool(
            f"{prefix}_SEND_IMAGE",
            _read_bool("ORACLE_LLM_SEND_IMAGE", send_image_default),
        ),
    )
    return result


def _validate_configured_local_llm_urls() -> None:
    for name in _LLM_BASE_URL_ENV_NAMES:
        configured_url = os.getenv(name)
        if configured_url is not None and configured_url.strip() != "":
            _validate_local_llm_url(configured_url)


def _validate_local_llm_url(base_url: str) -> None:
    parsed_url = urlparse(base_url)
    host = parsed_url.hostname
    allowed_hosts = {"127.0.0.1", "localhost", "::1"}
    if parsed_url.scheme not in {"http", "https"} or host not in allowed_hosts:
        raise ValueError(
            "ORACLE_LLM_BASE_URL must point to a local llama.cpp server "
            "(localhost, 127.0.0.1, or ::1).",
        )


def _read_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    result = default
    if raw_value is not None and raw_value.strip() != "":
        result = int(raw_value)
    return result


def _read_positive_int(name: str, default: int) -> int:
    result = _read_int(name, default)
    if result <= 0:
        raise ValueError(f"{name} must be greater than 0.")
    return result


def _read_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    result = default
    if raw_value is not None and raw_value.strip() != "":
        result = float(raw_value)
    return result


def _read_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    result = default
    if raw_value is not None and raw_value.strip() != "":
        result = raw_value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return result


def _read_detection_scale() -> float:
    result = _read_float("ORACLE_FACE_DETECTION_SCALE", 0.5)
    if result <= 0.0 or result > 1.0:
        raise ValueError("ORACLE_FACE_DETECTION_SCALE must be > 0.0 and <= 1.0.")
    return result

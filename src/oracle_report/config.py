from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:

    def _load_dotenv() -> None:
        pass


_LLM_BASE_URL_ENV_NAMES = (
    "ORACLE_LLM_BASE_URL",
    "ORACLE_REPORT_LLM_BASE_URL",
)
_CAMERA_BACKENDS = frozenset(("auto", "default", "v4l2", "dshow", "msmf"))

_MOCK_LANDMARK_METRICS_JSON = (
    '{"third_balance_error":0.008,"face_aspect_ratio":1.32,'
    '"eye_width_ratio":0.182,"eye_spacing_ratio":0.286,"eye_tail_tilt":0.012,'
    '"brow_eye_gap_ratio":0.081,"nose_length_ratio":0.241,"nose_width_ratio":0.190,'
    '"mouth_width_ratio":0.362,"mouth_balance_delta":0.006,'
    '"chin_length_ratio":0.214,"jaw_width_ratio":0.662}'
)
_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON = (
    '{"third_balance_error":0.018,"face_aspect_ratio":1.38,'
    '"eye_width_ratio":0.19,"eye_spacing_ratio":0.28,"eye_tail_tilt":0.018,'
    '"brow_eye_gap_ratio":0.078,"nose_length_ratio":0.25,"nose_width_ratio":0.19,'
    '"mouth_width_ratio":0.34,"mouth_balance_delta":0.004,'
    '"chin_length_ratio":0.21,"jaw_width_ratio":0.64}'
)
_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON = (
    '{"third_balance_error":0.035,"face_aspect_ratio":1.24,'
    '"eye_width_ratio":0.17,"eye_spacing_ratio":0.31,"eye_tail_tilt":-0.012,'
    '"brow_eye_gap_ratio":0.092,"nose_length_ratio":0.22,"nose_width_ratio":0.17,'
    '"mouth_width_ratio":0.43,"mouth_balance_delta":0.012,'
    '"chin_length_ratio":0.18,"jaw_width_ratio":0.72}'
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
    camera_auto_detect: bool = True
    camera_backend: str = "auto"
    mock_capture_enabled: bool = False
    mock_landmark_metrics_json: str = ""
    mock_pair_left_landmark_metrics_json: str = ""
    mock_pair_right_landmark_metrics_json: str = ""


@dataclass(frozen=True)
class LlmConfig:
    model: str
    base_url: str
    timeout_seconds: float
    max_output_tokens: int
    temperature: float
    prompt_cache: bool = False
    reasoning: bool = False


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    debug: bool
    distributed_role: str | None = None
    distributed_split: bool = False
    distributed_warmup: bool = False
    distributed_speculative: bool = False
    distributed_local_fallback: bool = True
    master_addr: str | None = None
    slave_addrs: tuple[str, ...] = field(default_factory=tuple)


def load_capture_config() -> CaptureConfig:
    _load_dotenv()
    mock_capture_enabled = _read_bool("ORACLE_MOCK_CAPTURE_ENABLED", False)
    mock_landmark_metrics_json = os.getenv("ORACLE_MOCK_LANDMARK_METRICS_JSON", "")
    pair_left_metrics_json = os.getenv("ORACLE_MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON", "")
    pair_right_metrics_json = os.getenv("ORACLE_MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON", "")
    if mock_capture_enabled:
        if mock_landmark_metrics_json.strip() == "":
            mock_landmark_metrics_json = _MOCK_LANDMARK_METRICS_JSON
        if pair_left_metrics_json.strip() == "":
            pair_left_metrics_json = _MOCK_PAIR_LEFT_LANDMARK_METRICS_JSON
        if pair_right_metrics_json.strip() == "":
            pair_right_metrics_json = _MOCK_PAIR_RIGHT_LANDMARK_METRICS_JSON
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
        camera_auto_detect=_read_bool("ORACLE_CAMERA_AUTO_DETECT", True),
        camera_backend=_read_camera_backend(),
        mock_capture_enabled=mock_capture_enabled,
        mock_landmark_metrics_json=mock_landmark_metrics_json,
        mock_pair_left_landmark_metrics_json=pair_left_metrics_json,
        mock_pair_right_landmark_metrics_json=pair_right_metrics_json,
    )
    return result


def load_llm_config() -> LlmConfig:
    result = load_report_llm_config()
    return result


def load_report_llm_config() -> LlmConfig:
    result = _load_llm_config("ORACLE_REPORT_LLM")
    return result


def load_app_config() -> AppConfig:
    _load_dotenv()
    role = os.getenv("ORACLE_DISTRIBUTED_ROLE")
    if role == "":
        role = None

    slave_addrs_str = os.getenv("ORACLE_SLAVE_ADDRS", "").strip()
    if slave_addrs_str:
        slave_addrs = tuple(
            addr.strip() for addr in slave_addrs_str.split(",") if addr.strip()
        )
    else:
        slave_addrs = ()

    result = AppConfig(
        host=os.getenv("ORACLE_APP_HOST", "0.0.0.0"),
        port=_read_positive_int("ORACLE_APP_PORT", 8501),
        debug=_read_bool("ORACLE_APP_DEBUG", False),
        distributed_role=role,
        distributed_split=_read_bool("ORACLE_DISTRIBUTED_SPLIT", False),
        distributed_warmup=_read_bool("ORACLE_DISTRIBUTED_WARMUP", False),
        distributed_speculative=_read_bool("ORACLE_DISTRIBUTED_SPECULATIVE", False),
        distributed_local_fallback=_read_bool("ORACLE_DISTRIBUTED_LOCAL_FALLBACK", True),
        master_addr=os.getenv("ORACLE_MASTER_ADDR"),
        slave_addrs=slave_addrs,
    )
    return result


def _load_llm_config(prefix: str) -> LlmConfig:
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
        reasoning=_read_bool(
            f"{prefix}_REASONING",
            _read_bool("ORACLE_REASONING", False),
        ),
        max_output_tokens=_read_int(
            f"{prefix}_MAX_OUTPUT_TOKENS",
            _read_int("ORACLE_MAX_OUTPUT_TOKENS", 4096),
        ),
        temperature=_read_float(
            f"{prefix}_TEMPERATURE",
            _read_float("ORACLE_LLM_TEMPERATURE", 0.7),
        ),
        prompt_cache=_read_bool(
            f"{prefix}_PROMPT_CACHE",
            _read_bool("ORACLE_LLM_PROMPT_CACHE", False),
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


def _read_camera_backend() -> str:
    raw_value = os.getenv("ORACLE_CAMERA_BACKEND", "auto")
    result = raw_value.strip().lower()
    if result == "":
        result = "auto"
    if result not in _CAMERA_BACKENDS:
        raise ValueError(
            "ORACLE_CAMERA_BACKEND must be one of: auto, default, v4l2, dshow, msmf.",
        )
    return result


def _read_detection_scale() -> float:
    result = _read_float("ORACLE_FACE_DETECTION_SCALE", 0.5)
    if result <= 0.0 or result > 1.0:
        raise ValueError("ORACLE_FACE_DETECTION_SCALE must be > 0.0 and <= 1.0.")
    return result

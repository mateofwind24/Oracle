#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Defaults are repo-relative
RUN_ORACLE_APP_HOST="${RUN_ORACLE_APP_HOST:-0.0.0.0}"
RUN_ORACLE_APP_PORT="${RUN_ORACLE_APP_PORT:-8501}"
RUN_ORACLE_APP_DEBUG="${RUN_ORACLE_APP_DEBUG:-0}"
RUN_ORACLE_DISTRIBUTED_WARMUP="${RUN_ORACLE_DISTRIBUTED_WARMUP:-0}"
RUN_ORACLE_REASONING="${RUN_ORACLE_REASONING:-0}"

RUN_ORACLE_LLM_BASE_URL="${RUN_ORACLE_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
RUN_ORACLE_LLM_MODEL="${RUN_ORACLE_LLM_MODEL:-local-model}"
RUN_ORACLE_FACE_LLM_MODEL="${RUN_ORACLE_FACE_LLM_MODEL:-$RUN_ORACLE_LLM_MODEL}"
RUN_ORACLE_FACE_LLM_SEND_IMAGE="${RUN_ORACLE_FACE_LLM_SEND_IMAGE:-0}"
RUN_ORACLE_REPORT_LLM_MODEL="${RUN_ORACLE_REPORT_LLM_MODEL:-$RUN_ORACLE_LLM_MODEL}"
RUN_ORACLE_REPORT_LLM_SEND_IMAGE="${RUN_ORACLE_REPORT_LLM_SEND_IMAGE:-0}"
RUN_ORACLE_LLM_TIMEOUT_SECONDS="${RUN_ORACLE_LLM_TIMEOUT_SECONDS:-${ORACLE_LLM_TIMEOUT_SECONDS:-300}}"
RUN_ORACLE_REPORT_LLM_TIMEOUT_SECONDS="${RUN_ORACLE_REPORT_LLM_TIMEOUT_SECONDS:-${ORACLE_REPORT_LLM_TIMEOUT_SECONDS:-3600}}"

RUN_ORACLE_START_LLAMA_SERVER="${RUN_ORACLE_START_LLAMA_SERVER:-1}"
RUN_ORACLE_LLAMA_MODEL_PATH=""
RUN_ORACLE_LLAMA_SERVER_BIN="${RUN_ORACLE_LLAMA_SERVER_BIN:-llama-server}"
RUN_LLAMA_CONTEXT_SIZE="${RUN_LLAMA_CONTEXT_SIZE:-8192}"
RUN_LLAMA_PARALLEL="${RUN_LLAMA_PARALLEL:-}"
KVFIX_LLAMA_CONTEXT_SIZE=20480

RUN_ORACLE_CAMERA_INDEX="${RUN_ORACLE_CAMERA_INDEX:-0}"
RUN_ORACLE_FRAME_WIDTH="${RUN_ORACLE_FRAME_WIDTH:-640}"
RUN_ORACLE_FRAME_HEIGHT="${RUN_ORACLE_FRAME_HEIGHT:-480}"
RUN_ORACLE_CAMERA_FPS="${RUN_ORACLE_CAMERA_FPS:-15}"
RUN_ORACLE_MIN_FACE_SECONDS="${RUN_ORACLE_MIN_FACE_SECONDS:-2.0}"
RUN_ORACLE_FACE_MIN_SIZE_PX="${RUN_ORACLE_FACE_MIN_SIZE_PX:-96}"
RUN_ORACLE_FACE_DETECTION_SCALE="${RUN_ORACLE_FACE_DETECTION_SCALE:-0.5}"
RUN_ORACLE_FACE_DETECTION_INTERVAL="${RUN_ORACLE_FACE_DETECTION_INTERVAL:-2}"
RUN_ORACLE_SHOW_PREVIEW="${RUN_ORACLE_SHOW_PREVIEW:-0}"
RUN_ORACLE_FACE_ANALYSIS_MODE="${RUN_ORACLE_FACE_ANALYSIS_MODE:-1}"

RUN_ORACLE_OUTPUT_DIR="${RUN_ORACLE_OUTPUT_DIR:-$ROOT_DIR/runs}"
RUN_ORACLE_FACE_DB_PATH="${RUN_ORACLE_FACE_DB_PATH:-$ROOT_DIR/data/face_recommendations.sqlite}"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
ORACLE_LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
LLAMA_LOG_DIR="$ROOT_DIR/runs/logs"
LLAMA_PID_FILE="$ROOT_DIR/runs/llama-server.pid"
GEMMA3_1B_Q4_MODEL="$ROOT_DIR/models/gemma-3-1b-it-Q4_0.gguf"
GEMMA4_E2B_Q2_MODEL="$ROOT_DIR/models/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
GEMMA3_1B_Q4_MODEL_URL="https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf"
GEMMA3_1B_Q4_MODEL_SHA256="27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e"
GEMMA4_E2B_Q2_MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
GEMMA4_E2B_Q2_MODEL_SHA256="dd279a54c0c0dc9724ed11d7f73ad7fb4489a45f58fefe9447da2429a727de0c"
PACKAGED_MODEL_URL="$GEMMA4_E2B_Q2_MODEL_URL"
PACKAGED_MODEL_SHA256="$GEMMA4_E2B_Q2_MODEL_SHA256"

# Execution configs set by parse_args
LLAMA_THREADS=""
LLAMA_NGL=""
LLAMA_BATCH_SIZE=""
LLAMA_EXTRA_ARGS=""
PYTHON_ENV="auto"
POSITIONAL_ARGS=()
RUN_LLAMA_CONTEXT_SIZE_EXPLICIT=0
LLAMA_SERVER_STARTED=0
LLAMA_SERVER_PID=""

log() {
  printf '[run] %s\n' "$*"
}

fail() {
  printf '[run][error] %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_linux() {
  [[ "$(uname -s)" == "Linux" ]]
}

run_with_elevation() {
  if [[ "$(id -u)" -eq 0 ]]; then
    "$@"
    return
  fi
  if ! command_exists sudo; then
    return 1
  fi
  if sudo -n true >/dev/null 2>&1; then
    sudo "$@"
    return
  fi
  if [[ -t 0 ]]; then
    sudo "$@"
    return
  fi
  return 1
}

process_running() {
  local pid="$1"
  local state
  if ! kill -0 "$pid" >/dev/null 2>&1; then
    return 1
  fi
  state="$(ps -p "$pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
  if [[ "$state" == Z* ]]; then
    return 1
  fi
  return 0
}

cleanup_llama_server() {
  local pid
  local recorded_pid
  local attempt

  if [[ "${LLAMA_SERVER_STARTED:-0}" != "1" ]]; then
    return
  fi

  pid="${LLAMA_SERVER_PID:-}"
  if [[ -z "$pid" ]]; then
    return
  fi

  if process_running "$pid"; then
    log "stopping llama.cpp server pid=$pid"
    kill "$pid" >/dev/null 2>&1 || true
    for attempt in $(seq 1 30); do
      if ! process_running "$pid"; then
        break
      fi
      sleep 0.2
    done
    if process_running "$pid"; then
      log "forcing llama.cpp server stop pid=$pid"
      kill -KILL "$pid" >/dev/null 2>&1 || true
    fi
    wait "$pid" >/dev/null 2>&1 || true
  fi

  if [[ -f "$LLAMA_PID_FILE" ]]; then
    recorded_pid="$(cat "$LLAMA_PID_FILE" 2>/dev/null || true)"
    if [[ "$recorded_pid" == "$pid" ]]; then
      rm -f "$LLAMA_PID_FILE"
    fi
  fi

  LLAMA_SERVER_STARTED=0
  LLAMA_SERVER_PID=""
}

on_run_exit() {
  local status
  status="$?"
  set +e
  cleanup_llama_server
  exit "$status"
}

trap on_run_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

print_help() {
  cat <<EOF
Usage: $0 [options] [command] [command_args...]

Commands:
  debug <cmd> [args...]    Run in debug mode (saves outputs to runs/debug/)
  kvfix <cmd> [args...]    Run with fixed prompt cache slots enabled (ctx default: 20480)
  release <cmd> [args...]  Run in release mode (temp output dir, deleted after run)
  capture                  Run capture only
  prompt <args...>         Debug prompt generation
  prompt-run <args...>     Run prompt generation with LLM call
  token                    Print prompts.json prefix token sizes
  (empty)                  Start Flask web server (default)

Wrapper Options:
  -h, --help               Show this help message
  -m, --model-path PATH    Path to the GGUF model file
  -p, --port PORT          Port for the Flask app (default: 8501)
  --host HOST              Host for the Flask app (default: 0.0.0.0)
  --debug                  Enable debug mode for Flask app and logging
  -t, --threads THREADS    Number of threads for llama.cpp server
  -ngl, --ngl LAYERS       Number of GPU layers to offload to GPU (llama.cpp)
  -c, --ctx-size SIZE      Context size for llama.cpp (default: 8192)
  --parallel N             Number of llama.cpp slots
  -b, --batch-size SIZE    Batch size for llama.cpp
  --distributed-role ROLE  Distributed role: master, slave, or hybrid
  --distributed-split      Split prompts for parallel execution
  --distributed-warmup     Warmup LLM KV cache on start
  --reasoning              Enable reasoning mode (think tags) for LLM
  --mock-capture           Enable mock camera capture mode using mock_face.jpg
  --master-addr ADDR       Master address (e.g., http://192.168.0.5:8501)
  --slave-addrs ADDRS      Comma-separated list of slave addresses
  --python-env ENV         Force Python env type (active-conda, active-venv, conda, uv, venv, auto)
  --llama-dir DIR          Path to llama.cpp repository
  --extra-llama-args ARGS  Additional raw command-line arguments for llama-server
EOF
}

parse_args() {
  POSITIONAL_ARGS=()
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        print_help
        exit 0
        ;;
      -m|--model-path)
        RUN_ORACLE_LLAMA_MODEL_PATH="$2"
        shift 2
        ;;
      -p|--port)
        RUN_ORACLE_APP_PORT="$2"
        shift 2
        ;;
      --host)
        RUN_ORACLE_APP_HOST="$2"
        shift 2
        ;;
      -t|--threads)
        LLAMA_THREADS="$2"
        shift 2
        ;;
      -ngl|--ngl|--n-gpu-layers)
        LLAMA_NGL="$2"
        shift 2
        ;;
      -c|--ctx-size|--context-size)
        RUN_LLAMA_CONTEXT_SIZE="$2"
        RUN_LLAMA_CONTEXT_SIZE_EXPLICIT=1
        shift 2
        ;;
      --parallel)
        RUN_LLAMA_PARALLEL="$2"
        shift 2
        ;;
      -b|--batch-size)
        LLAMA_BATCH_SIZE="$2"
        shift 2
        ;;
      --distributed-role)
        RUN_ORACLE_DISTRIBUTED_ROLE="$2"
        shift 2
        ;;
      --distributed-split)
        RUN_ORACLE_DISTRIBUTED_SPLIT=1
        shift 1
        ;;
      --distributed-warmup)
        RUN_ORACLE_DISTRIBUTED_WARMUP=1
        shift 1
        ;;
      --reasoning)
        RUN_ORACLE_REASONING=1
        shift 1
        ;;
      --mock-capture)
        RUN_ORACLE_MOCK_CAPTURE_ENABLED=1
        shift 1
        ;;
      --debug)
        RUN_ORACLE_APP_DEBUG=1
        shift 1
        ;;
      --master-addr)
        RUN_ORACLE_MASTER_ADDR="$2"
        shift 2
        ;;
      --slave-addrs)
        RUN_ORACLE_SLAVE_ADDRS="$2"
        shift 2
        ;;
      --python-env)
        PYTHON_ENV="$2"
        shift 2
        ;;
      --llama-dir)
        ORACLE_LLAMA_CPP_DIR="$2"
        shift 2
        ;;
      --extra-llama-args)
        LLAMA_EXTRA_ARGS="$2"
        shift 2
        ;;
      *)
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
  done
}

apply_kvfix_mode() {
  if [[ "${POSITIONAL_ARGS[0]:-}" == "kvfix" ]]; then
    export ORACLE_LLM_PROMPT_CACHE=1
    if [[ "$RUN_LLAMA_CONTEXT_SIZE_EXPLICIT" != "1" ]]; then
      RUN_LLAMA_CONTEXT_SIZE="$KVFIX_LLAMA_CONTEXT_SIZE"
    fi
    if [[ -z "${RUN_LLAMA_PARALLEL:-}" ]]; then
      RUN_LLAMA_PARALLEL=5
    fi
    POSITIONAL_ARGS=("${POSITIONAL_ARGS[@]:1}")
  fi
}

detect_cuda() {
  if command_exists nvcc; then
    return 0
  fi
  local p
  for p in /usr/local/cuda/bin /usr/local/cuda-*/bin; do
    if [[ -x "$p/nvcc" ]]; then
      export PATH="$p:$PATH"
      if [[ -d "${p%/bin}/lib64" ]]; then
        export LD_LIBRARY_PATH="${p%/bin}/lib64:${LD_LIBRARY_PATH:-}"
      fi
      return 0
    fi
  done
  if command_exists nvidia-smi; then
    return 0
  fi
  return 1
}

setup_python_env() {
  local env_type="${PYTHON_ENV:-auto}"

  # 1. Active Conda env
  if [[ "$env_type" == "active-conda" || ( "$env_type" == "auto" && -n "${CONDA_PREFIX:-}" ) ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      log "Using active Conda environment: ${CONDA_DEFAULT_ENV:-oracle}"
      return 0
    elif [[ "$env_type" == "active-conda" ]]; then
      fail "active-conda specified but no Conda environment is active."
    fi
  fi

  # 2. Active Virtualenv (venv, uv, etc.)
  if [[ "$env_type" == "active-venv" || ( "$env_type" == "auto" && -n "${VIRTUAL_ENV:-}" ) ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
      log "Using active virtual environment: $VIRTUAL_ENV"
      return 0
    elif [[ "$env_type" == "active-venv" ]]; then
      fail "active-venv specified but no virtual environment is active."
    fi
  fi

  # 2. Conda 'oracle' env
  if [[ "$env_type" == "conda" || "$env_type" == "auto" ]] && command_exists conda; then
    if conda env list | grep -q -E "^oracle[[:space:]]"; then
      local conda_path
      conda_path="$(conda info --base)"
      if [[ -f "$conda_path/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_path/etc/profile.d/conda.sh"
        conda activate oracle
        if [[ "${CONDA_DEFAULT_ENV:-}" == "oracle" ]]; then
          return 0
        fi
      fi
    fi
    if [[ "$env_type" == "conda" ]]; then
      fail "Conda was specified, but 'oracle' environment could not be found/activated."
    fi
  fi

  # 3. UV virtualenv
  if [[ "$env_type" == "uv" || "$env_type" == "auto" ]] && command_exists uv; then
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_DIR/bin/activate"
      return 0
    fi
  fi

  # 4. Fallback to standard venv
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
    return 0
  fi

  fail "Python environment not found. Run ./build.sh first, then run ./run.sh."
}

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.env"
    set +a
  else
    fail ".env not found. Run ./build.sh first or restore .env from .env.example."
  fi
}

apply_run_config() {
  export ORACLE_APP_HOST="$RUN_ORACLE_APP_HOST"
  export ORACLE_APP_PORT="$RUN_ORACLE_APP_PORT"
  export ORACLE_APP_DEBUG="$RUN_ORACLE_APP_DEBUG"

  export ORACLE_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_LLM_MODEL="$RUN_ORACLE_LLM_MODEL"
  export ORACLE_LLM_TIMEOUT_SECONDS="$RUN_ORACLE_LLM_TIMEOUT_SECONDS"

  local default_max_tokens=1800
  if [[ "${RUN_ORACLE_REASONING:-${ORACLE_REASONING:-0}}" == "1" ]]; then
    default_max_tokens=4096
  fi

  export ORACLE_FACE_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_FACE_LLM_MODEL="$RUN_ORACLE_FACE_LLM_MODEL"
  export ORACLE_FACE_LLM_SEND_IMAGE="$RUN_ORACLE_FACE_LLM_SEND_IMAGE"
  export ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS="${ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS:-2048}"

  export ORACLE_REPORT_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_REPORT_LLM_MODEL="$RUN_ORACLE_REPORT_LLM_MODEL"
  export ORACLE_REPORT_LLM_TIMEOUT_SECONDS="$RUN_ORACLE_REPORT_LLM_TIMEOUT_SECONDS"
  export ORACLE_REPORT_LLM_SEND_IMAGE="$RUN_ORACLE_REPORT_LLM_SEND_IMAGE"
  export ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS="${ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS:-$default_max_tokens}"
  export ORACLE_LLM_PROMPT_CACHE="${ORACLE_LLM_PROMPT_CACHE:-0}"

  export ORACLE_START_LLAMA_SERVER="$RUN_ORACLE_START_LLAMA_SERVER"
  export ORACLE_LLAMA_MODEL_PATH="${ORACLE_LLAMA_MODEL_PATH:-$RUN_ORACLE_LLAMA_MODEL_PATH}"
  export ORACLE_LLAMA_SERVER_BIN="$RUN_ORACLE_LLAMA_SERVER_BIN"
  export LLAMA_CONTEXT_SIZE="$RUN_LLAMA_CONTEXT_SIZE"
  if [[ -n "${RUN_LLAMA_PARALLEL:-}" ]]; then
    export LLAMA_PARALLEL="$RUN_LLAMA_PARALLEL"
  else
    unset LLAMA_PARALLEL
  fi

  export ORACLE_CAMERA_INDEX="$RUN_ORACLE_CAMERA_INDEX"
  export ORACLE_FRAME_WIDTH="$RUN_ORACLE_FRAME_WIDTH"
  export ORACLE_FRAME_HEIGHT="$RUN_ORACLE_FRAME_HEIGHT"
  export ORACLE_CAMERA_FPS="$RUN_ORACLE_CAMERA_FPS"
  export ORACLE_MIN_FACE_SECONDS="$RUN_ORACLE_MIN_FACE_SECONDS"
  export ORACLE_FACE_MIN_SIZE_PX="$RUN_ORACLE_FACE_MIN_SIZE_PX"
  export ORACLE_FACE_DETECTION_SCALE="$RUN_ORACLE_FACE_DETECTION_SCALE"
  export ORACLE_FACE_DETECTION_INTERVAL="$RUN_ORACLE_FACE_DETECTION_INTERVAL"
  export ORACLE_SHOW_PREVIEW="$RUN_ORACLE_SHOW_PREVIEW"
  export ORACLE_MOCK_CAPTURE_ENABLED="${RUN_ORACLE_MOCK_CAPTURE_ENABLED:-${ORACLE_MOCK_CAPTURE_ENABLED:-0}}"

  export ORACLE_OUTPUT_DIR="$RUN_ORACLE_OUTPUT_DIR"
  export ORACLE_FACE_DB_PATH="$RUN_ORACLE_FACE_DB_PATH"

  export ORACLE_DISTRIBUTED_ROLE="${RUN_ORACLE_DISTRIBUTED_ROLE:-${ORACLE_DISTRIBUTED_ROLE:-}}"
  export ORACLE_DISTRIBUTED_SPLIT="${RUN_ORACLE_DISTRIBUTED_SPLIT:-${ORACLE_DISTRIBUTED_SPLIT:-0}}"
  export ORACLE_DISTRIBUTED_WARMUP="${RUN_ORACLE_DISTRIBUTED_WARMUP:-${RUN_ORACLE_DISTRIBUTED_WARMUP:-0}}"
  export ORACLE_REASONING="${RUN_ORACLE_REASONING:-${ORACLE_REASONING:-0}}"
  export ORACLE_MASTER_ADDR="${RUN_ORACLE_MASTER_ADDR:-${ORACLE_MASTER_ADDR:-}}"
  export ORACLE_SLAVE_ADDRS="${RUN_ORACLE_SLAVE_ADDRS:-${ORACLE_SLAVE_ADDRS:-}}"

  export ORACLE_LLAMA_CPP_DIR="$ORACLE_LLAMA_CPP_DIR"
}

camera_device_paths() {
  local device_path
  for device_path in /dev/video*; do
    if [[ -e "$device_path" ]]; then
      printf '%s\n' "$device_path"
    fi
  done
}

camera_devices_need_access_fix() {
  local device_path
  for device_path in "$@"; do
    if [[ ! -r "$device_path" || ! -w "$device_path" ]]; then
      return 0
    fi
  done
  return 1
}

grant_camera_device_access() {
  local devices=("$@")
  local current_user
  current_user="$(id -un)"
  if command_exists setfacl; then
    run_with_elevation setfacl -m "u:${current_user}:rw" "${devices[@]}"
    return
  fi
  run_with_elevation chmod a+rw "${devices[@]}"
}

ensure_camera_device_access() {
  if [[ "${RUN_ORACLE_AUTO_CAMERA_PERMISSIONS:-1}" != "1" ]]; then
    return
  fi
  if ! is_linux; then
    return
  fi

  local devices=()
  local blocked_devices=()
  local device_path
  while IFS= read -r device_path; do
    if [[ -n "$device_path" ]]; then
      devices+=("$device_path")
      if [[ ! -r "$device_path" || ! -w "$device_path" ]]; then
        blocked_devices+=("$device_path")
      fi
    fi
  done < <(camera_device_paths)

  if [[ "${#devices[@]}" -eq 0 ]]; then
    return
  fi
  if ! camera_devices_need_access_fix "${devices[@]}"; then
    return
  fi

  log "camera devices detected without user access; attempting permission repair: ${blocked_devices[*]}"
  if grant_camera_device_access "${blocked_devices[@]}"; then
    if camera_devices_need_access_fix "${blocked_devices[@]}"; then
      log "Warning: camera permission repair ran, but access is still unavailable. Re-login or add $(id -un) to the video group."
    else
      log "camera device permissions repaired for current user"
    fi
    return
  fi

  log "Warning: could not automatically grant camera permissions. Try: sudo usermod -aG video $(id -un)"
}

llm_host_port() {
  python - "$ORACLE_LLM_BASE_URL" <<'PY'
import sys
from urllib.parse import urlparse

parsed = urlparse(sys.argv[1])
host = parsed.hostname or "127.0.0.1"
port = parsed.port or (443 if parsed.scheme == "https" else 80)
print(host, port)
PY
}

server_ready() {
  python - "$ORACLE_LLM_BASE_URL" <<'PY'
import sys
from urllib.error import URLError
from urllib.request import urlopen

url = sys.argv[1].rstrip("/") + "/models"
try:
    with urlopen(url, timeout=2) as response:
        ok = 200 <= response.status < 300
except URLError:
    ok = False
except TimeoutError:
    ok = False
raise SystemExit(0 if ok else 1)
PY
}

find_llama_server() {
  if [[ -n "${ORACLE_LLAMA_SERVER_BIN:-}" ]] &&
    command -v "$ORACLE_LLAMA_SERVER_BIN" >/dev/null 2>&1; then
    command -v "$ORACLE_LLAMA_SERVER_BIN"
  elif command_exists llama-server; then
    command -v llama-server
  elif [[ -d "$ORACLE_LLAMA_CPP_DIR" ]]; then
    local found_bin
    found_bin="$(find "$ORACLE_LLAMA_CPP_DIR" -type f -name 'llama-server' -executable | head -n 1)"
    if [[ -n "$found_bin" ]]; then
      printf '%s\n' "$found_bin"
    else
      return 1
    fi
  else
    return 1
  fi
}

default_model_url_for_path() {
  local model_path
  local model_name
  local result
  model_path="$1"
  model_name="${model_path##*/}"
  result="$PACKAGED_MODEL_URL"
  if [[ "$model_name" == "gemma-3-1b-it-Q4_0.gguf" ]]; then
    result="$GEMMA3_1B_Q4_MODEL_URL"
  elif [[ "$model_name" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    result="$GEMMA4_E2B_Q2_MODEL_URL"
  fi
  printf '%s\n' "$result"
}

default_model_hash_for_path() {
  local model_path
  local model_name
  local result
  model_path="$1"
  model_name="${model_path##*/}"
  result="$PACKAGED_MODEL_SHA256"
  if [[ "$model_name" == "gemma-3-1b-it-Q4_0.gguf" ]]; then
    result="$GEMMA3_1B_Q4_MODEL_SHA256"
  elif [[ "$model_name" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    result="$GEMMA4_E2B_Q2_MODEL_SHA256"
  fi
  printf '%s\n' "$result"
}

configured_model_url_for_path() {
  local model_path
  local configured_url
  local result
  model_path="$1"
  configured_url="${ORACLE_LLAMA_MODEL_URL:-}"
  result="$(default_model_url_for_path "$model_path")"
  if [[ -n "$configured_url" && "$configured_url" != "$PACKAGED_MODEL_URL" ]]; then
    result="$configured_url"
  fi
  printf '%s\n' "$result"
}

configured_model_hash_for_path() {
  local model_path
  local configured_hash
  local result
  model_path="$1"
  configured_hash="${ORACLE_LLAMA_MODEL_SHA256:-}"
  result="$(default_model_hash_for_path "$model_path")"
  
  local model_name="${model_path##*/}"
  if [[ "$model_name" == "gemma-3-1b-it-Q4_0.gguf" ]]; then
    if [[ -n "$configured_hash" ]]; then
      result="$configured_hash"
    fi
  else
    if [[ -n "${RUN_ORACLE_LLAMA_MODEL_PATH:-}" ]]; then
      result=""
    else
      local env_model_path="${ORACLE_LLAMA_MODEL_PATH:-}"
      local env_model_name="${env_model_path##*/}"
      if [[ -n "$env_model_name" && "$model_name" == "$env_model_name" && -n "$configured_hash" ]]; then
        result="$configured_hash"
      else
        result=""
      fi
    fi
  fi
  printf '%s\n' "$result"
}

verify_model_hash() {
  local model_path
  local expected_hash
  local actual_hash
  model_path="$1"
  expected_hash="$(configured_model_hash_for_path "$model_path")"
  if [[ -z "$expected_hash" ]]; then
    return
  fi
  if ! command_exists sha256sum; then
    return
  fi
  actual_hash="$(sha256sum "$model_path" | awk '{print $1}')"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    fail "model checksum mismatch for $model_path; expected $expected_hash, got $actual_hash"
  fi
}

find_repo_model_file() {
  local model_file
  model_file=""
  if [[ -d "$ROOT_DIR/models" ]]; then
    model_file="$(find "$ROOT_DIR/models" -type f -name '*.gguf' |
      sort |
      head -n 1)"
  fi
  printf '%s\n' "$model_file"
}

known_repo_model_hash_for_path() {
  local model_path
  local model_name
  local result
  model_path="$1"
  model_name="${model_path##*/}"
  result=""
  if [[ "$model_name" == "gemma-3-1b-it-Q4_0.gguf" ]]; then
    result="$GEMMA3_1B_Q4_MODEL_SHA256"
  fi
  printf '%s\n' "$result"
}

verify_repo_model_file_if_known() {
  local model_path
  local model_hash
  model_path="$1"
  model_hash="$(known_repo_model_hash_for_path "$model_path")"
  if [[ -z "$model_hash" ]]; then
    return
  fi
  if ! command_exists sha256sum; then
    return
  fi
  local actual_hash
  actual_hash="$(sha256sum "$model_path" | awk '{print $1}')"
  if [[ "$actual_hash" != "$model_hash" ]]; then
    fail "model checksum mismatch for $model_path; expected $model_hash, got $actual_hash"
  fi
}

download_model_file() {
  local model_path
  local model_tmp_path
  local model_url
  model_path="$1"
  model_tmp_path="${model_path}.tmp"
  model_url="$(configured_model_url_for_path "$model_path")"
  command_exists curl || fail "curl is required to download the packaged GGUF model"
  mkdir -p "$(dirname "$model_path")"
  log "downloading packaged GGUF model from $model_url"
  curl --fail --location --continue-at - --retry 5 --retry-delay 2 \
    --retry-all-errors --output "$model_tmp_path" "$model_url"
  verify_model_hash "$model_tmp_path"
  mv "$model_tmp_path" "$model_path"
}

ensure_model_file() {
  local model_path
  local existing_model_path
  model_path="$ORACLE_LLAMA_MODEL_PATH"
  if [[ -z "$model_path" ]]; then
    fail "Model path not set. Specify model with --model-path option"
  fi
  if [[ "${model_path##*/}" == "model.gguf" ]]; then
    log "models/model.gguf is a legacy default; using Gemma 4 E2B Q2"
    model_path="$GEMMA4_E2B_Q2_MODEL"
    export ORACLE_LLAMA_MODEL_PATH="$model_path"
  fi
  if [[ -f "$model_path" ]]; then
    verify_model_hash "$model_path"
    return
  fi

  if [[ "${model_path##*/}" == "gemma-3-1b-it-Q4_0.gguf" || "${model_path##*/}" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    download_model_file "$model_path"
    verify_model_hash "$model_path"
    return
  fi

  if [[ "$model_path" != "$GEMMA4_E2B_Q2_MODEL" ]]; then
    download_model_file "$model_path"
    verify_model_hash "$model_path"
    return
  fi

  existing_model_path="$(find_repo_model_file)"
  if [[ -n "$existing_model_path" ]]; then
    verify_repo_model_file_if_known "$existing_model_path"
    export ORACLE_LLAMA_MODEL_PATH="$existing_model_path"
    log "using existing repo model at $existing_model_path; skipping model download"
    return
  fi

  download_model_file "$model_path"
  verify_model_hash "$model_path"
}

start_llama_server() {
  ORACLE_LLM_BASE_URL="${ORACLE_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
  if server_ready; then
    log "llama.cpp server already reachable"
    return
  fi

  if [[ "${ORACLE_START_LLAMA_SERVER:-1}" != "1" ]]; then
    fail "local llama.cpp server is not reachable at $ORACLE_LLM_BASE_URL"
  fi

  local model_path
  ensure_model_file
  model_path="$ORACLE_LLAMA_MODEL_PATH"

  local server_bin
  server_bin="$(find_llama_server)" ||
    fail "llama-server not found; run ./build.sh first"

  read -r host port < <(llm_host_port)
  mkdir -p "$LLAMA_LOG_DIR" "$ROOT_DIR/runs"
  log "starting llama.cpp server on $host:$port"

  local server_args=(
    -m "$model_path"
    --host "$host"
    --port "$port"
    -c "${LLAMA_CONTEXT_SIZE:-4096}"
    --cache-type-k q4_0
    --cache-type-v q4_0
  )

  if [[ "${ORACLE_REASONING:-0}" == "1" ]]; then
    server_args+=(--reasoning on --reasoning-format deepseek)
  else
    server_args+=(--reasoning off --reasoning-format none)
  fi

  if [[ -n "${LLAMA_PARALLEL:-}" ]]; then
    server_args+=(--parallel "$LLAMA_PARALLEL")
  fi

  # Check GPU/CUDA and automatically set GPU layers if NGL is not explicitly set
  local use_cuda=0
  if detect_cuda; then
    use_cuda=1
  fi

  if [[ -n "$LLAMA_NGL" ]]; then
    server_args+=(-ngl "$LLAMA_NGL")
  elif [[ "$use_cuda" -eq 1 ]]; then
    log "CUDA detected, automatically setting --n-gpu-layers 99 to offload all layers to GPU"
    server_args+=(-ngl 99)
  fi

  if [[ -n "$LLAMA_THREADS" ]]; then
    server_args+=(-t "$LLAMA_THREADS")
  fi

  if [[ -n "$LLAMA_BATCH_SIZE" ]]; then
    server_args+=(-b "$LLAMA_BATCH_SIZE")
  fi

  if [[ -n "$LLAMA_EXTRA_ARGS" ]]; then
    # split arguments safely
    # shellcheck disable=SC2206
    server_args+=($LLAMA_EXTRA_ARGS)
  fi

  log "llama-server arguments: ${server_args[*]}"

  "$server_bin" "${server_args[@]}" >"$LLAMA_LOG_DIR/llama-server.log" 2>&1 &
  LLAMA_SERVER_PID="$!"
  LLAMA_SERVER_STARTED=1
  printf '%s\n' "$LLAMA_SERVER_PID" >"$LLAMA_PID_FILE"

  local attempt
  for attempt in $(seq 1 180); do
    if server_ready; then
      log "llama.cpp server ready"
      return
    fi
    sleep 1
  done

  fail "llama.cpp server did not become ready; see $LLAMA_LOG_DIR/llama-server.log"
}

run_oracle() {
  if [[ "$#" -gt 0 ]]; then
    python -m oracle_report.cli "$@"
    return
  fi

  local debug_args=()
  if [[ "${ORACLE_APP_DEBUG:-0}" == "1" ]]; then
    debug_args=(--debug)
  fi
  log "starting Oracle Flask UI at http://${ORACLE_APP_HOST}:${ORACLE_APP_PORT}"
  python -m oracle_report.cli serve \
    --host "$ORACLE_APP_HOST" \
    --port "$ORACLE_APP_PORT" \
    "${debug_args[@]}"
}

timestamp() {
  date +"%Y%m%d_%H%M%S"
}

require_wrapped_command() {
  local mode
  mode="$1"
  shift
  if [[ "$#" -eq 0 ]]; then
    fail "usage: ./run.sh $mode <capture|prompt|prompt-run|serve> [args...]"
  fi
}

reject_release_output_args() {
  local arg
  for arg in "$@"; do
    case "$arg" in
      --output | --output=* | --output-dir | --output-dir=*)
        fail "release mode does not allow $arg because it must not save outputs"
        ;;
    esac
  done
}

run_debug_mode() {
  require_wrapped_command debug "$@"
  local debug_dir
  local status
  debug_dir="$ROOT_DIR/runs/debug/$(timestamp)"
  mkdir -p "$debug_dir/artifacts"
  export ORACLE_OUTPUT_DIR="$debug_dir/artifacts"
  printf './run.sh debug' >"$debug_dir/command.txt"
  printf ' %q' "$@" >>"$debug_dir/command.txt"
  printf '\n' >>"$debug_dir/command.txt"
  log "debug output dir: $debug_dir"
  if needs_llm_server "$@"; then
    start_llama_server
  fi
  set +e
  run_oracle "$@" 2>&1 | tee "$debug_dir/output.log"
  status="${PIPESTATUS[0]}"
  set -e
  return "$status"
}

run_release_mode() {
  require_wrapped_command release "$@"
  reject_release_output_args "$@"
  local release_dir
  local status
  release_dir="$(mktemp -d)"
  export ORACLE_OUTPUT_DIR="$release_dir"
  if needs_llm_server "$@"; then
    start_llama_server
  fi
  set +e
  run_oracle "$@"
  status="$?"
  set -e
  rm -rf "$release_dir"
  return "$status"
}

needs_llm_server() {
  local result
  result=0
  if [[ "$#" -gt 0 ]]; then
    case "$1" in
      capture | prompt | --help | -h)
        result=1
        ;;
      prompt-run)
        case "${2:-}" in
          saju-reading | --help | -h | "")
            result=1
            ;;
          *)
            result=0
            ;;
        esac
        ;;
      token)
        case "${2:-}" in
          --offline)
            result=1
            ;;
          *)
            result=0
            ;;
        esac
        ;;
      *)
        result=0
        ;;
      esac
  fi
  return "$result"
}

needs_camera_device() {
  if [[ "$#" -eq 0 ]]; then
    return 0
  fi
  case "$1" in
    capture | serve)
      return 0
      ;;
    debug | release)
      case "${2:-}" in
        capture | serve)
          return 0
          ;;
      esac
      ;;
  esac
  return 1
}

main() {
  # Parse options
  parse_args "$@"

  local requested_cmd="${POSITIONAL_ARGS[0]:-}"
  local requested_subcmd="${POSITIONAL_ARGS[1]:-}"
  if [[ "$requested_cmd" == "build" ]] ||
    [[ "$requested_cmd" == "kvfix" && "$requested_subcmd" == "build" ]]; then
    fail "run.sh is execution-only. Use ./build.sh for build/install tasks."
  fi

  # Activate python environment
  setup_python_env
  load_env
  apply_kvfix_mode

  local cmd="${POSITIONAL_ARGS[0]:-}"

  # Apply arguments to runtime variables
  if [[ -n "$RUN_ORACLE_LLAMA_MODEL_PATH" ]]; then
    export ORACLE_LLAMA_MODEL_PATH="$RUN_ORACLE_LLAMA_MODEL_PATH"
  else
    # Fallback to env or default
    export ORACLE_LLAMA_MODEL_PATH="${ORACLE_LLAMA_MODEL_PATH:-$GEMMA4_E2B_Q2_MODEL}"
  fi

  apply_run_config

  if needs_camera_device "${POSITIONAL_ARGS[@]}"; then
    ensure_camera_device_access
  fi

  # Re-evaluate command from positional arguments
  case "$cmd" in
    debug)
      local debug_args=("${POSITIONAL_ARGS[@]:1}")
      if [[ "${#debug_args[@]}" -eq 0 ]]; then
        debug_args=("serve")
      fi
      run_debug_mode "${debug_args[@]}"
      ;;
    release)
      local release_args=("${POSITIONAL_ARGS[@]:1}")
      if [[ "${#release_args[@]}" -eq 0 ]]; then
        release_args=("serve")
      fi
      run_release_mode "${release_args[@]}"
      ;;
    *)
      if needs_llm_server "${POSITIONAL_ARGS[@]}"; then
        start_llama_server
      fi
      run_oracle "${POSITIONAL_ARGS[@]}"
      ;;
  esac
}

main "$@"

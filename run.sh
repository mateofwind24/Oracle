#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# Edit this block on Raspberry Pi before running ./run.sh.
# Defaults are repo-relative, so /home/willtek/work/oracle works after clone.
RUN_ORACLE_APP_HOST="${RUN_ORACLE_APP_HOST:-0.0.0.0}"
RUN_ORACLE_APP_PORT="${RUN_ORACLE_APP_PORT:-8501}"
RUN_ORACLE_APP_DEBUG="${RUN_ORACLE_APP_DEBUG:-0}"

RUN_ORACLE_LLM_BASE_URL="${RUN_ORACLE_LLM_BASE_URL:-http://127.0.0.1:8080/v1}"
RUN_ORACLE_LLM_MODEL="${RUN_ORACLE_LLM_MODEL:-local-model}"
RUN_ORACLE_FACE_LLM_MODEL="${RUN_ORACLE_FACE_LLM_MODEL:-$RUN_ORACLE_LLM_MODEL}"
RUN_ORACLE_FACE_LLM_SEND_IMAGE="${RUN_ORACLE_FACE_LLM_SEND_IMAGE:-1}"
RUN_ORACLE_REPORT_LLM_MODEL="${RUN_ORACLE_REPORT_LLM_MODEL:-$RUN_ORACLE_LLM_MODEL}"
RUN_ORACLE_REPORT_LLM_SEND_IMAGE="${RUN_ORACLE_REPORT_LLM_SEND_IMAGE:-0}"
RUN_ORACLE_LLM_TIMEOUT_SECONDS="${RUN_ORACLE_LLM_TIMEOUT_SECONDS:-120}"
RUN_ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS="${RUN_ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS:-700}"
RUN_ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS="${RUN_ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS:-1800}"

RUN_ORACLE_START_LLAMA_SERVER="${RUN_ORACLE_START_LLAMA_SERVER:-1}"
RUN_ORACLE_LLAMA_MODEL_PATH="${RUN_ORACLE_LLAMA_MODEL_PATH:-$ROOT_DIR/models/model.gguf}"
RUN_ORACLE_LLAMA_SERVER_BIN="${RUN_ORACLE_LLAMA_SERVER_BIN:-llama-server}"
RUN_LLAMA_CONTEXT_SIZE="${RUN_LLAMA_CONTEXT_SIZE:-4096}"

RUN_ORACLE_CAMERA_INDEX="${RUN_ORACLE_CAMERA_INDEX:-0}"
RUN_ORACLE_FRAME_WIDTH="${RUN_ORACLE_FRAME_WIDTH:-640}"
RUN_ORACLE_FRAME_HEIGHT="${RUN_ORACLE_FRAME_HEIGHT:-480}"
RUN_ORACLE_CAMERA_FPS="${RUN_ORACLE_CAMERA_FPS:-15}"
RUN_ORACLE_MIN_FACE_SECONDS="${RUN_ORACLE_MIN_FACE_SECONDS:-2.0}"
RUN_ORACLE_FACE_MIN_SIZE_PX="${RUN_ORACLE_FACE_MIN_SIZE_PX:-96}"
RUN_ORACLE_FACE_DETECTION_SCALE="${RUN_ORACLE_FACE_DETECTION_SCALE:-0.5}"
RUN_ORACLE_FACE_DETECTION_INTERVAL="${RUN_ORACLE_FACE_DETECTION_INTERVAL:-2}"
RUN_ORACLE_SHOW_PREVIEW="${RUN_ORACLE_SHOW_PREVIEW:-0}"

RUN_ORACLE_OUTPUT_DIR="${RUN_ORACLE_OUTPUT_DIR:-$ROOT_DIR/runs}"
RUN_ORACLE_MANSE_DB_PATH="${RUN_ORACLE_MANSE_DB_PATH:-$ROOT_DIR/data/manse.sqlite}"
RUN_ORACLE_FACE_DB_PATH="${RUN_ORACLE_FACE_DB_PATH:-$ROOT_DIR/data/face_recommendations.sqlite}"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$DEPS_DIR/llama.cpp}"
LLAMA_LOG_DIR="$ROOT_DIR/runs/logs"
LLAMA_PID_FILE="$ROOT_DIR/runs/llama-server.pid"

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

activate_venv() {
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
  elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
  else
    log "venv not found; running build.sh first"
    "$ROOT_DIR/build.sh"
    if [[ -f "$VENV_DIR/bin/activate" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_DIR/bin/activate"
    elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
      # shellcheck source=/dev/null
      source "$VENV_DIR/Scripts/activate"
    else
      fail "venv activation failed after build"
    fi
  fi
}

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.env"
    set +a
  elif [[ -f "$ROOT_DIR/.env.example" ]]; then
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    set -a
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.env"
    set +a
    log "created .env from .env.example"
  fi
}

apply_run_config() {
  export ORACLE_APP_HOST="$RUN_ORACLE_APP_HOST"
  export ORACLE_APP_PORT="$RUN_ORACLE_APP_PORT"
  export ORACLE_APP_DEBUG="$RUN_ORACLE_APP_DEBUG"

  export ORACLE_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_LLM_MODEL="$RUN_ORACLE_LLM_MODEL"
  export ORACLE_LLM_TIMEOUT_SECONDS="$RUN_ORACLE_LLM_TIMEOUT_SECONDS"

  export ORACLE_FACE_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_FACE_LLM_MODEL="$RUN_ORACLE_FACE_LLM_MODEL"
  export ORACLE_FACE_LLM_SEND_IMAGE="$RUN_ORACLE_FACE_LLM_SEND_IMAGE"
  export ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS="$RUN_ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS"

  export ORACLE_REPORT_LLM_BASE_URL="$RUN_ORACLE_LLM_BASE_URL"
  export ORACLE_REPORT_LLM_MODEL="$RUN_ORACLE_REPORT_LLM_MODEL"
  export ORACLE_REPORT_LLM_SEND_IMAGE="$RUN_ORACLE_REPORT_LLM_SEND_IMAGE"
  export ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS="$RUN_ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS"

  export ORACLE_START_LLAMA_SERVER="$RUN_ORACLE_START_LLAMA_SERVER"
  export ORACLE_LLAMA_MODEL_PATH="$RUN_ORACLE_LLAMA_MODEL_PATH"
  export ORACLE_LLAMA_SERVER_BIN="$RUN_ORACLE_LLAMA_SERVER_BIN"
  export LLAMA_CONTEXT_SIZE="$RUN_LLAMA_CONTEXT_SIZE"

  export ORACLE_CAMERA_INDEX="$RUN_ORACLE_CAMERA_INDEX"
  export ORACLE_FRAME_WIDTH="$RUN_ORACLE_FRAME_WIDTH"
  export ORACLE_FRAME_HEIGHT="$RUN_ORACLE_FRAME_HEIGHT"
  export ORACLE_CAMERA_FPS="$RUN_ORACLE_CAMERA_FPS"
  export ORACLE_MIN_FACE_SECONDS="$RUN_ORACLE_MIN_FACE_SECONDS"
  export ORACLE_FACE_MIN_SIZE_PX="$RUN_ORACLE_FACE_MIN_SIZE_PX"
  export ORACLE_FACE_DETECTION_SCALE="$RUN_ORACLE_FACE_DETECTION_SCALE"
  export ORACLE_FACE_DETECTION_INTERVAL="$RUN_ORACLE_FACE_DETECTION_INTERVAL"
  export ORACLE_SHOW_PREVIEW="$RUN_ORACLE_SHOW_PREVIEW"

  export ORACLE_OUTPUT_DIR="$RUN_ORACLE_OUTPUT_DIR"
  export ORACLE_MANSE_DB_PATH="$RUN_ORACLE_MANSE_DB_PATH"
  export ORACLE_FACE_DB_PATH="$RUN_ORACLE_FACE_DB_PATH"
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
  elif [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then
    printf '%s\n' "$LLAMA_CPP_DIR/build/bin/llama-server"
  else
    return 1
  fi
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
  model_path="${ORACLE_LLAMA_MODEL_PATH:-${LLAMA_MODEL:-}}"
  if [[ -z "$model_path" || ! -f "$model_path" ]]; then
    fail "set ORACLE_LLAMA_MODEL_PATH in .env to an existing .gguf model"
  fi

  local server_bin
  server_bin="$(find_llama_server)" ||
    fail "llama-server not found; run ./build.sh first"

  read -r host port < <(llm_host_port)
  mkdir -p "$LLAMA_LOG_DIR" "$ROOT_DIR/runs"
  log "starting llama.cpp server on $host:$port"
  "$server_bin" \
    -m "$model_path" \
    --host "$host" \
    --port "$port" \
    -c "${LLAMA_CONTEXT_SIZE:-4096}" \
    >"$LLAMA_LOG_DIR/llama-server.log" 2>&1 &
  printf '%s\n' "$!" >"$LLAMA_PID_FILE"

  local attempt
  for attempt in $(seq 1 60); do
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
    oracle-report "$@"
    return
  fi

  local debug_args=()
  if [[ "${ORACLE_APP_DEBUG:-0}" == "1" ]]; then
    debug_args=(--debug)
  fi
  log "starting Oracle Flask UI at http://${ORACLE_APP_HOST}:${ORACLE_APP_PORT}"
  oracle-report serve \
    --host "$ORACLE_APP_HOST" \
    --port "$ORACLE_APP_PORT" \
    "${debug_args[@]}"
}

needs_llm_server() {
  local result
  result=0
  if [[ "$#" -gt 0 ]]; then
    case "$1" in
      capture | --help | -h)
        result=1
        ;;
      *)
        result=0
        ;;
    esac
  fi
  return "$result"
}

main() {
  activate_venv
  load_env
  apply_run_config
  if needs_llm_server "$@"; then
    start_llama_server
  fi
  run_oracle "$@"
}

main "$@"

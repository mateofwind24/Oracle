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
RUN_ORACLE_FACE_LLM_SEND_IMAGE="${RUN_ORACLE_FACE_LLM_SEND_IMAGE:-0}"
RUN_ORACLE_REPORT_LLM_MODEL="${RUN_ORACLE_REPORT_LLM_MODEL:-$RUN_ORACLE_LLM_MODEL}"
RUN_ORACLE_REPORT_LLM_SEND_IMAGE="${RUN_ORACLE_REPORT_LLM_SEND_IMAGE:-0}"
RUN_ORACLE_LLM_TIMEOUT_SECONDS="${RUN_ORACLE_LLM_TIMEOUT_SECONDS:-120}"
RUN_ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS="${RUN_ORACLE_FACE_LLM_MAX_OUTPUT_TOKENS:-700}"
RUN_ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS="${RUN_ORACLE_REPORT_LLM_MAX_OUTPUT_TOKENS:-1800}"

RUN_ORACLE_START_LLAMA_SERVER="${RUN_ORACLE_START_LLAMA_SERVER:-1}"
RUN_ORACLE_LLAMA_MODEL_PATH="${RUN_ORACLE_LLAMA_MODEL_PATH:-$ROOT_DIR/models/gemma-4-E2B-it-UD-Q2_K_XL.gguf}"
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
RUN_ORACLE_FACE_ANALYSIS_MODE="${RUN_ORACLE_FACE_ANALYSIS_MODE:-1}"

RUN_ORACLE_OUTPUT_DIR="${RUN_ORACLE_OUTPUT_DIR:-$ROOT_DIR/runs}"
RUN_ORACLE_MANSE_DB_PATH="${RUN_ORACLE_MANSE_DB_PATH:-$ROOT_DIR/data/manse.sqlite}"
RUN_ORACLE_FACE_DB_PATH="${RUN_ORACLE_FACE_DB_PATH:-$ROOT_DIR/data/face_recommendations.sqlite}"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$DEPS_DIR/llama.cpp}"
LLAMA_LOG_DIR="$ROOT_DIR/runs/logs"
LLAMA_PID_FILE="$ROOT_DIR/runs/llama-server.pid"
GEMMA3_1B_Q4_MODEL_URL="https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf"
GEMMA3_1B_Q4_MODEL_SHA256="27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e"
GEMMA4_E2B_Q2_MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
GEMMA4_E2B_Q2_MODEL_SHA256="dd279a54c0c0dc9724ed11d7f73ad7fb4489a45f58fefe9447da2429a727de0c"
PACKAGED_MODEL_URL="$GEMMA4_E2B_Q2_MODEL_URL"
PACKAGED_MODEL_SHA256="$GEMMA4_E2B_Q2_MODEL_SHA256"

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
  export ORACLE_FACE_ANALYSIS_MODE="$RUN_ORACLE_FACE_ANALYSIS_MODE"

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
  if [[ -n "$configured_hash" && "$configured_hash" != "$PACKAGED_MODEL_SHA256" ]]; then
    result="$configured_hash"
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
    model_file="$(find "$ROOT_DIR/models" -maxdepth 1 -type f -name '*.gguf' |
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
  model_path="${ORACLE_LLAMA_MODEL_PATH:-${LLAMA_MODEL:-}}"
  if [[ -z "$model_path" ]]; then
    fail "set ORACLE_LLAMA_MODEL_PATH in .env to an existing .gguf model"
  fi
  if [[ "${model_path##*/}" == "model.gguf" ]]; then
    log "models/model.gguf is a legacy default; using Gemma 4 E2B Q2"
    model_path="$ROOT_DIR/models/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
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

  if [[ "$model_path" != "$ROOT_DIR/models/gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
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
  model_path="${ORACLE_LLAMA_MODEL_PATH:-${LLAMA_MODEL:-}}"

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
      *)
        result=0
        ;;
    esac
  fi
  return "$result"
}

main() {
  if [[ "${1:-}" == "build" ]]; then
    "$ROOT_DIR/build.sh"
    return
  fi

  activate_venv
  load_env
  apply_run_config

  case "${1:-}" in
    debug)
      shift
      run_debug_mode "$@"
      ;;
    release)
      shift
      run_release_mode "$@"
      ;;
    *)
      if needs_llm_server "$@"; then
        start_llama_server
      fi
      run_oracle "$@"
      ;;
  esac
}

main "$@"

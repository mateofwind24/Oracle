#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$DEPS_DIR/llama.cpp}"
PACKAGED_MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-IQ2_M.gguf"
PACKAGED_MODEL_SHA256="60f84cb5b9512175f219506da4a5d98d30b112855c474a3a6f06f6596dc7fd9b"
GEMMA3_1B_Q4_MODEL_PATH="$ROOT_DIR/models/gemma-3-1b-it-Q4_0.gguf"
GEMMA3_1B_Q4_MODEL_URL="https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf"
GEMMA3_1B_Q4_MODEL_SHA256="27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e"

log() {
  printf '[build] %s\n' "$*"
}

fail() {
  printf '[build][error] %s\n' "$*" >&2
  exit 1
}

command_exists() {
  command -v "$1" >/dev/null 2>&1
}

is_linux() {
  [[ "$(uname -s)" == "Linux" ]]
}

sudo_cmd() {
  if [[ "$(id -u)" -eq 0 ]]; then
    printf ''
  elif command_exists sudo; then
    printf 'sudo'
  else
    printf '__missing_sudo__'
  fi
}

install_apt_packages() {
  if ! is_linux || ! command_exists apt-get || ! command_exists dpkg-query; then
    return
  fi

  local packages=("$@")
  local missing=()
  local package
  for package in "${packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null |
      grep -q 'install ok installed'; then
      missing+=("$package")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    log "apt packages already installed"
    return
  fi

  local sudo_bin
  sudo_bin="$(sudo_cmd)"
  if [[ "$sudo_bin" == "__missing_sudo__" ]]; then
    fail "missing apt packages (${missing[*]}), but sudo is not available"
  fi

  log "installing apt packages: ${missing[*]}"
  $sudo_bin apt-get update
  $sudo_bin apt-get install -y "${missing[@]}"
}

python_cmd() {
  if command_exists python3; then
    printf 'python3'
  elif command_exists python; then
    printf 'python'
  else
    fail "python3 is required"
  fi
}

activate_venv() {
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
  elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
  else
    fail "venv activation file not found under $VENV_DIR"
  fi
}

python_deps_ready() {
  python - <<'PY'
import importlib.util

modules = (
    "cv2",
    "dotenv",
    "flask",
    "mediapipe",
    "numpy",
    "oracle_report",
    "pytest",
    "requests",
)
missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
}

ensure_python_env() {
  local py
  local venv_created
  py="$(python_cmd)"
  venv_created=0

  if [[ ! -d "$VENV_DIR" ]]; then
    log "creating Python venv at $VENV_DIR"
    if is_linux; then
      "$py" -m venv --system-site-packages "$VENV_DIR"
    else
      "$py" -m venv "$VENV_DIR"
    fi
    venv_created=1
  else
    log "Python venv already exists"
  fi

  activate_venv
  if [[ "$venv_created" -eq 0 ]] && python_deps_ready &&
    command_exists oracle-report &&
    command_exists oracle-build-manse-db; then
    log "Python dependencies already installed"
    return
  fi

  python -m pip install --upgrade pip setuptools wheel

  if python -c 'import cv2' >/dev/null 2>&1; then
    log "OpenCV already importable; installing Python app/quality/test deps"
    python -m pip install -e ".[quality,test]"
  else
    log "OpenCV not importable; installing Python camera/quality/test deps"
    python -m pip install -e ".[camera,quality,test]"
  fi

  python -c 'import cv2' >/dev/null 2>&1 ||
    fail "OpenCV import failed after installation"
}

ensure_llama_cpp() {
  if command_exists llama-server; then
    log "llama-server already installed on PATH"
    return
  fi

  if [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then
    log "llama.cpp already built at $LLAMA_CPP_DIR"
    return
  fi

  command_exists git || fail "git is required to clone llama.cpp"
  command_exists cmake || fail "cmake is required to build llama.cpp"

  mkdir -p "$DEPS_DIR"
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    log "cloning llama.cpp into $LLAMA_CPP_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  else
    log "llama.cpp source already exists"
  fi

  log "configuring llama.cpp"
  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=ON \
    -DGGML_OPENMP=ON

  log "building llama-server"
  cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server \
    -j "${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"
}

ensure_env_file() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    log ".env already exists"
  else
    cp "$ROOT_DIR/.env.example" "$ROOT_DIR/.env"
    log "created .env from .env.example"
  fi
}

load_env() {
  if [[ -f "$ROOT_DIR/.env" ]]; then
    set -a
    # shellcheck source=/dev/null
    source "$ROOT_DIR/.env"
    set +a
  fi
}

ensure_runtime_dirs() {
  mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/models" "$ROOT_DIR/runs"
  log "runtime directories ready"
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

verify_file_hash() {
  local file_path
  local expected_hash
  local actual_hash
  file_path="$1"
  expected_hash="$2"
  if [[ -z "$expected_hash" ]]; then
    return
  fi
  if ! command_exists sha256sum; then
    return
  fi
  actual_hash="$(sha256sum "$file_path" | awk '{print $1}')"
  if [[ "$actual_hash" != "$expected_hash" ]]; then
    fail "checksum mismatch for $file_path; expected $expected_hash, got $actual_hash"
  fi
}

download_model_file() {
  local model_path
  local model_tmp_path
  local model_url
  local model_hash
  model_path="$1"
  model_url="$2"
  model_hash="$3"
  model_tmp_path="${model_path}.tmp"
  command_exists curl || fail "curl is required to download the packaged GGUF model"
  mkdir -p "$(dirname "$model_path")"
  log "downloading packaged GGUF model from $model_url"
  curl --fail --location --continue-at - --retry 5 --retry-delay 2 \
    --retry-all-errors --output "$model_tmp_path" "$model_url"
  verify_file_hash "$model_tmp_path" "$model_hash"
  mv "$model_tmp_path" "$model_path"
  log "model file ready at $model_path"
}

ensure_model_file_at() {
  local model_path
  local model_url
  local model_hash
  model_path="$1"
  model_url="$2"
  model_hash="$3"
  if [[ -f "$model_path" ]]; then
    verify_file_hash "$model_path" "$model_hash"
    log "model file ready at $model_path"
    return
  fi

  download_model_file "$model_path" "$model_url" "$model_hash"
  verify_file_hash "$model_path" "$model_hash"
}

ensure_model_file() {
  local model_path
  local model_url
  local model_hash
  model_path="${ORACLE_LLAMA_MODEL_PATH:-$ROOT_DIR/models/model.gguf}"
  model_url="$(configured_model_url_for_path "$model_path")"
  model_hash="$(configured_model_hash_for_path "$model_path")"
  ensure_model_file_at "$model_path" "$model_url" "$model_hash"
}

ensure_gemma3_1b_q4_model_file() {
  local model_path
  local model_url
  local model_hash
  if [[ "${ORACLE_DOWNLOAD_GEMMA3_1B_Q4_MODEL:-1}" != "1" ]]; then
    log "skipping Gemma 3 1B Q4 model download"
    return
  fi

  model_path="${ORACLE_GEMMA3_1B_Q4_MODEL_PATH:-$GEMMA3_1B_Q4_MODEL_PATH}"
  model_url="${ORACLE_GEMMA3_1B_Q4_MODEL_URL:-$GEMMA3_1B_Q4_MODEL_URL}"
  model_hash="${ORACLE_GEMMA3_1B_Q4_MODEL_SHA256:-$GEMMA3_1B_Q4_MODEL_SHA256}"
  ensure_model_file_at "$model_path" "$model_url" "$model_hash"
}

ensure_manse_db() {
  if [[ "${ORACLE_BUILD_MANSE_DB:-1}" != "1" ]]; then
    log "skipping manse DB build because ORACLE_BUILD_MANSE_DB is not 1"
    return
  fi

  local db_path
  local start_year
  local end_year
  db_path="${ORACLE_MANSE_DB_PATH:-$ROOT_DIR/data/manse.sqlite}"
  start_year="${ORACLE_MANSE_START_YEAR:-1900}"
  end_year="${ORACLE_MANSE_END_YEAR:-2100}"

  log "ensuring manse DB at $db_path for $start_year-$end_year"
  oracle-build-manse-db \
    --db "$db_path" \
    --start-year "$start_year" \
    --end-year "$end_year"
}

run_verification() {
  if [[ "${ORACLE_SKIP_TESTS:-0}" == "1" ]]; then
    log "skipping tests because ORACLE_SKIP_TESTS=1"
  else
    log "running tests"
    python -m pytest
  fi
  log "checking CLI import"
  oracle-report --help >/dev/null
}

main() {
  install_apt_packages \
    python3 \
    python3-venv \
    python3-pip \
    python3-opencv \
    opencv-data \
    libatlas-base-dev \
    git \
    cmake \
    build-essential \
    curl \
    ca-certificates

  ensure_python_env
  ensure_llama_cpp
  ensure_env_file
  load_env
  ensure_runtime_dirs
  ensure_model_file
  ensure_gemma3_1b_q4_model_file
  ensure_manse_db
  run_verification
  log "build complete"
}

main "$@"

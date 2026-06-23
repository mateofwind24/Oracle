#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$DEPS_DIR/llama.cpp}"

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
    log "OpenCV already importable; installing Python app/test deps"
    python -m pip install -e ".[test]"
  else
    log "OpenCV not importable; installing Python camera/test deps"
    python -m pip install -e ".[camera,test]"
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

ensure_runtime_dirs() {
  mkdir -p "$ROOT_DIR/data" "$ROOT_DIR/models" "$ROOT_DIR/runs"
  log "runtime directories ready"
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
    libatlas-base-dev \
    git \
    cmake \
    build-essential \
    curl \
    ca-certificates

  ensure_python_env
  ensure_llama_cpp
  ensure_env_file
  ensure_runtime_dirs
  ensure_manse_db
  run_verification
  log "build complete"
}

main "$@"

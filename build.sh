#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${ORACLE_VENV_DIR:-$ROOT_DIR/.venv}"
DEPS_DIR="${ORACLE_DEPS_DIR:-$ROOT_DIR/.deps}"
DEFAULT_LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
LLAMA_CPP_DIR=""

GEMMA3_1B_Q4_MODEL_PATH="$ROOT_DIR/models/gemma-3-1b-it-Q4_0.gguf"
GEMMA4_E2B_Q2_MODEL_PATH="$ROOT_DIR/models/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
GEMMA3_1B_Q4_MODEL_URL="https://huggingface.co/unsloth/gemma-3-1b-it-GGUF/resolve/main/gemma-3-1b-it-Q4_0.gguf"
GEMMA3_1B_Q4_MODEL_SHA256="27ee88e03be02e9ba73def9a819d570d8ad73716e50769e87f374ae394b0276e"
GEMMA4_E2B_Q2_MODEL_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-UD-Q2_K_XL.gguf"
GEMMA4_E2B_Q2_MODEL_SHA256="dd279a54c0c0dc9724ed11d7f73ad7fb4489a45f58fefe9447da2429a727de0c"
PACKAGED_MODEL_URL="$GEMMA4_E2B_Q2_MODEL_URL"
PACKAGED_MODEL_SHA256="$GEMMA4_E2B_Q2_MODEL_SHA256"

# Options populated by parser
BUILD_PROFILE="default"
BUILD_JOBS=""
FORCE_CUDA="0"
FORCE_CUDA_EXPLICIT="0"
PYTHON_ENV="auto"
PYTHON_ENV_EXPLICIT="0"
MODEL_PATH=""
python_env_mode="venv"
SKIP_SYSTEMD_SERVICES="0"

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
  elif command_exists sudo && sudo -n true 2>/dev/null; then
    printf 'sudo'
  else
    printf '__missing_sudo__'
  fi
}

usage() {
  cat <<EOF
Usage: $0 [profile] [options]

Profiles:
  hwonly                  Build for this laptop: CPU-only llama.cpp, local .venv,
                          and no systemd service generation

Options:
  -h, --help               Show this help message
  -j, --jobs JOBS          Number of build jobs (default: auto-detected CPU cores)
  --cuda                   Force build llama.cpp with CUDA support
  --cpu                    Force build llama.cpp in CPU-only mode (default)
  --auto-gpu               Auto-detect CUDA capability
  --python-env ENV         Force python env type: active-conda, active-venv, conda, uv, venv, or auto (default: auto)
  --model-path PATH        Set GGUF model path (default: models/gemma-4-E2B-it-UD-Q2_K_XL.gguf)
  --llama-dir DIR          Directory for llama.cpp source/build (default: ./llama.cpp)
EOF
  exit 0
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|--help)
        usage
        ;;
      -j|--jobs)
        BUILD_JOBS="$2"
        shift 2
        ;;
      --cuda)
        FORCE_CUDA="1"
        FORCE_CUDA_EXPLICIT="1"
        shift
        ;;
      --cpu)
        FORCE_CUDA="0"
        FORCE_CUDA_EXPLICIT="1"
        shift
        ;;
      --auto-gpu)
        FORCE_CUDA="auto"
        FORCE_CUDA_EXPLICIT="1"
        shift
        ;;
      --python-env)
        PYTHON_ENV="$2"
        PYTHON_ENV_EXPLICIT="1"
        shift 2
        ;;
      --model-path)
        MODEL_PATH="$2"
        shift 2
        ;;
      --llama-dir)
        LLAMA_CPP_DIR="$2"
        shift 2
        ;;
      hwonly)
        BUILD_PROFILE="hwonly"
        shift
        ;;
      *)
        log "Warning: Unknown option $1"
        shift
        ;;
    esac
  done
}

apply_build_profile() {
  if [[ "$BUILD_PROFILE" == "hwonly" ]]; then
    log "Using hwonly laptop build profile"
    if [[ "$FORCE_CUDA_EXPLICIT" != "1" ]]; then
      FORCE_CUDA="0"
    fi
    if [[ "$PYTHON_ENV_EXPLICIT" != "1" ]]; then
      if command_exists uv; then
        PYTHON_ENV="uv"
      else
        PYTHON_ENV="venv"
      fi
    fi
    SKIP_SYSTEMD_SERVICES="1"
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

install_apt_packages() {
  if ! is_linux || ! command_exists apt-get || ! command_exists dpkg-query; then
    return
  fi

  local packages=("$@")
  local filtered_packages=()
  local pkg
  for pkg in "${packages[@]}"; do
    if [[ "$python_env_mode" == "conda" || "$python_env_mode" == "active-conda" || "$python_env_mode" == "uv" || "$python_env_mode" == "active-venv" ]]; then
      # Skip python/opencv system packages in conda/uv
      if [[ "$pkg" =~ ^python3.* || "$pkg" == "python3-opencv" || "$pkg" == "opencv-data" || "$pkg" == "libatlas-base-dev" ]]; then
        continue
      fi
    fi
    filtered_packages+=("$pkg")
  done

  if [[ "${#filtered_packages[@]}" -eq 0 ]]; then
    return
  fi

  local missing=()
  for pkg in "${filtered_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null |
      grep -q 'install ok installed'; then
      missing+=("$pkg")
    fi
  done

  if [[ "${#missing[@]}" -eq 0 ]]; then
    log "apt packages already installed"
    return
  fi

  local sudo_bin
  sudo_bin="$(sudo_cmd)"
  if [[ "$sudo_bin" == "__missing_sudo__" ]]; then
    log "Warning: missing apt packages (${missing[*]}), but sudo is not available. Continuing anyway..."
    return
  fi

  log "installing apt packages: ${missing[*]}"
  if ! $sudo_bin apt-get update || ! $sudo_bin apt-get install -y "${missing[@]}"; then
    log "Warning: failed to install some apt packages. Continuing anyway..."
  fi
}

install_python_bootstrap_packages() {
  local env_type="${PYTHON_ENV:-auto}"
  if ! is_linux || ! command_exists apt-get || ! command_exists dpkg-query; then
    return
  fi

  if [[ "$env_type" == "active-conda" || "$env_type" == "active-venv" ]]; then
    return
  fi

  if [[ "$env_type" == "auto" && ( -n "${CONDA_PREFIX:-}" || -n "${VIRTUAL_ENV:-}" ) ]]; then
    return
  fi

  install_apt_packages \
    python3 \
    python3-venv \
    python3-pip
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

activate_python_env() {
  if [[ -f "$VENV_DIR/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/bin/activate"
  elif [[ -f "$VENV_DIR/Scripts/activate" ]]; then
    # shellcheck source=/dev/null
    source "$VENV_DIR/Scripts/activate"
  else
    fail "venv activation failed: $VENV_DIR"
  fi
}

setup_python_env() {
  local env_type="${PYTHON_ENV:-auto}"
  log "Setting up Python environment (mode: $env_type)..."

  # 1. Active Conda env
  if [[ "$env_type" == "active-conda" || ( "$env_type" == "auto" && -n "${CONDA_PREFIX:-}" ) ]]; then
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
      log "Using already active Conda environment: ${CONDA_DEFAULT_ENV:-oracle} (Prefix: $CONDA_PREFIX)"
      if python -c "import pip" >/dev/null 2>&1; then
        python_env_mode="active-conda"
        return 0
      else
        log "Warning: Conda prefix is set but python does not have pip. Falling back."
      fi
    elif [[ "$env_type" == "active-conda" ]]; then
      fail "active-conda specified but no Conda environment is active."
    fi
  fi

  # 2. Active Virtualenv (venv, uv, etc.)
  if [[ "$env_type" == "active-venv" || ( "$env_type" == "auto" && -n "${VIRTUAL_ENV:-}" ) ]]; then
    if [[ -n "${VIRTUAL_ENV:-}" ]]; then
      log "Using already active virtual environment (Prefix: $VIRTUAL_ENV)"
      if python -c "import pip" >/dev/null 2>&1; then
        python_env_mode="active-venv"
        return 0
      else
        log "Warning: VIRTUAL_ENV is set but python does not have pip. Falling back."
      fi
    elif [[ "$env_type" == "active-venv" ]]; then
      fail "active-venv specified but no virtual environment is active."
    fi
  fi

  # 2. Conda 'oracle' env
  if [[ "$env_type" == "conda" || "$env_type" == "auto" ]] && command_exists conda; then
    if conda env list | grep -q -E "^oracle[[:space:]]"; then
      log "Conda environment 'oracle' exists. Activating..."
      local conda_path
      conda_path="$(conda info --base)"
      if [[ -f "$conda_path/etc/profile.d/conda.sh" ]]; then
        # shellcheck source=/dev/null
        source "$conda_path/etc/profile.d/conda.sh"
        conda activate oracle
        if [[ "${CONDA_DEFAULT_ENV:-}" == "oracle" ]]; then
          python_env_mode="conda"
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
    log "uv found. Creating/using virtualenv via uv..."
    if [[ ! -d "$VENV_DIR" ]]; then
      uv venv "$VENV_DIR"
    fi
    activate_python_env
    python_env_mode="uv"
    return 0
  fi

  # 4. Fallback to standard venv
  log "Using standard python venv at $VENV_DIR"
  local py
  py="$(python_cmd)"
  if [[ -d "$VENV_DIR" && ! -f "$VENV_DIR/bin/activate" && ! -f "$VENV_DIR/Scripts/activate" ]]; then
    log "Removing invalid Python environment at $VENV_DIR"
    rm -rf "$VENV_DIR"
  fi
  if [[ ! -d "$VENV_DIR" ]]; then
    if is_linux; then
      "$py" -m venv --system-site-packages "$VENV_DIR"
    else
      "$py" -m venv "$VENV_DIR"
    fi
  fi
  activate_python_env
  python_env_mode="venv"
  return 0
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
    "PIL",
    "pytest",
    "requests",
)
missing = [name for name in modules if importlib.util.find_spec(name) is None]
raise SystemExit(0 if not missing else 1)
PY
}

install_deps() {
  if [[ "$python_env_mode" == "uv" ]]; then
    uv pip install --upgrade pip setuptools wheel
    if python -c 'import cv2' >/dev/null 2>&1; then
      log "OpenCV already importable; installing quality/test deps with uv"
      uv pip install -e ".[quality,test]"
    else
      log "OpenCV not importable; installing camera/quality/test deps with uv"
      uv pip install -e ".[camera,quality,test]"
    fi
  else
    python -m pip install --upgrade pip setuptools wheel
    if python -c 'import cv2' >/dev/null 2>&1; then
      log "OpenCV already importable; installing quality/test deps with pip"
      python -m pip install -e ".[quality,test]"
    else
      log "OpenCV not importable; installing camera/quality/test deps with pip"
      python -m pip install -e ".[camera,quality,test]"
    fi
  fi
}

ensure_python_env() {
  if python_deps_ready &&
    command_exists oracle-report; then
    log "Python dependencies already installed"
    return
  fi

  log "Installing Python dependencies..."
  install_deps

  python -c 'import cv2' >/dev/null 2>&1 ||
    fail "OpenCV import failed after installation"
}

llama_server_built() {
  [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ||
     -x "$LLAMA_CPP_DIR/build/bin/llama-server.exe" ||
     -x "$LLAMA_CPP_DIR/build/bin/Release/llama-server.exe" ]]
}

llama_cuda_cache_enabled() {
  local cache_path="$LLAMA_CPP_DIR/build/CMakeCache.txt"
  [[ -f "$cache_path" ]] && grep -Eq '^GGML_CUDA(:[A-Z]+)?=ON$' "$cache_path"
}

ensure_llama_cpp() {
  if command_exists llama-server; then
    log "llama-server already installed on PATH"
    return
  fi

  if llama_server_built; then
    if [[ "${FORCE_CUDA:-0}" == "0" ]] && llama_cuda_cache_enabled; then
      log "llama.cpp was built with CUDA; reconfiguring CPU-only build"
    else
      log "llama.cpp already built at $LLAMA_CPP_DIR"
      return
    fi
  fi

  command_exists git || fail "git is required to clone llama.cpp"
  command_exists cmake || fail "cmake is required to build llama.cpp"

  mkdir -p "$(dirname "$LLAMA_CPP_DIR")"
  if [[ ! -d "$LLAMA_CPP_DIR/.git" ]]; then
    log "cloning llama.cpp into $LLAMA_CPP_DIR"
    git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR"
  else
    log "llama.cpp source already exists"
  fi

  log "configuring llama.cpp"
  local cmake_flags=(
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=ON
    -DGGML_OPENMP=ON
  )

  local use_cuda=0
  if [[ "${FORCE_CUDA:-auto}" == "1" ]]; then
    use_cuda=1
  elif [[ "${FORCE_CUDA:-auto}" == "0" ]]; then
    use_cuda=0
  else
    if detect_cuda; then
      use_cuda=1
    fi
  fi

  if [[ "$use_cuda" -eq 1 ]]; then
    log "CUDA detected or requested. Building llama.cpp with CUDA support (-DGGML_CUDA=ON)..."
    cmake_flags+=(-DGGML_CUDA=ON)
  else
    log "Building llama.cpp in CPU-only mode..."
    cmake_flags+=(-DGGML_CUDA=OFF)
  fi

  cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" "${cmake_flags[@]}"

  log "building llama-server"
  local jobs="${BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '2')}"
  log "Running cmake --build with jobs: $jobs"
  cmake --build "$LLAMA_CPP_DIR/build" --config Release --target llama-server -j "$jobs"
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
  if [[ "$model_name" == "gemma-3-1b-it-Q4_0.gguf" || "$model_name" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    if [[ -n "$configured_hash" ]]; then
      result="$configured_hash"
    fi
  else
    if [[ -n "${MODEL_PATH:-}" ]]; then
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
  elif [[ "$model_name" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    result="$GEMMA4_E2B_Q2_MODEL_SHA256"
  fi
  printf '%s\n' "$result"
}

verify_repo_model_file_if_known() {
  local model_path
  local model_hash
  model_path="$1"
  model_hash="$(known_repo_model_hash_for_path "$model_path")"
  verify_file_hash "$model_path" "$model_hash"
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

ensure_model_file() {
  local model_path
  local model_url
  local model_hash
  local existing_model_path
  model_path="${ORACLE_LLAMA_MODEL_PATH:-$GEMMA4_E2B_Q2_MODEL_PATH}"
  if [[ "${model_path##*/}" == "model.gguf" ]]; then
    log "models/model.gguf is a legacy default; using Gemma 4 E2B Q2 at $GEMMA4_E2B_Q2_MODEL_PATH"
    model_path="$GEMMA4_E2B_Q2_MODEL_PATH"
    export ORACLE_LLAMA_MODEL_PATH="$model_path"
  fi
  if [[ -f "$model_path" ]]; then
    model_hash="$(configured_model_hash_for_path "$model_path")"
    verify_file_hash "$model_path" "$model_hash"
    log "model file ready at $model_path"
    return
  fi

  if [[ "${model_path##*/}" == "gemma-3-1b-it-Q4_0.gguf" || "${model_path##*/}" == "gemma-4-E2B-it-UD-Q2_K_XL.gguf" ]]; then
    model_url="$(configured_model_url_for_path "$model_path")"
    model_hash="$(configured_model_hash_for_path "$model_path")"
    download_model_file "$model_path" "$model_url" "$model_hash"
    verify_file_hash "$model_path" "$model_hash"
    return
  fi

  existing_model_path="$(find_repo_model_file)"
  if [[ -n "$existing_model_path" ]]; then
    verify_repo_model_file_if_known "$existing_model_path"
    export ORACLE_LLAMA_MODEL_PATH="$existing_model_path"
    log "using existing repo model at $existing_model_path; skipping model download"
    return
  fi

  model_url="$(configured_model_url_for_path "$model_path")"
  model_hash="$(configured_model_hash_for_path "$model_path")"
  download_model_file "$model_path" "$model_url" "$model_hash"
  verify_file_hash "$model_path" "$model_hash"
}

generate_systemd_services() {
  if [[ "${SKIP_SYSTEMD_SERVICES:-0}" == "1" ]]; then
    log "skipping systemd service generation"
    return
  fi

  local template_dir="$ROOT_DIR/systemd"
  local current_user
  current_user="$(id -un)"

  log "generating systemd service files dynamically for user=$current_user, dir=$ROOT_DIR"

  if [[ -f "$template_dir/llama-server.service.template" ]]; then
    sed -e "s|{{ORACLE_DIR}}|$ROOT_DIR|g" \
        -e "s|{{ORACLE_USER}}|$current_user|g" \
        "$template_dir/llama-server.service.template" > "$template_dir/llama-server.service"
  fi

  if [[ -f "$template_dir/oracle-report.service.template" ]]; then
    sed -e "s|{{ORACLE_DIR}}|$ROOT_DIR|g" \
        -e "s|{{ORACLE_USER}}|$current_user|g" \
        "$template_dir/oracle-report.service.template" > "$template_dir/oracle-report.service"
  fi
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
  parse_args "$@"
  apply_build_profile

  if [[ -z "$LLAMA_CPP_DIR" ]]; then
    LLAMA_CPP_DIR="$DEFAULT_LLAMA_CPP_DIR"
  fi

  if [[ -n "$MODEL_PATH" ]]; then
    export ORACLE_LLAMA_MODEL_PATH="$MODEL_PATH"
  fi

  install_python_bootstrap_packages

  # Call setup_python_env first to determine the package environment mode
  setup_python_env

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
    ca-certificates \
    fonts-noto-cjk

  ensure_python_env
  ensure_llama_cpp
  ensure_env_file
  load_env
  ensure_runtime_dirs
  ensure_model_file
  generate_systemd_services
  run_verification
  log "build complete"
}

main "$@"

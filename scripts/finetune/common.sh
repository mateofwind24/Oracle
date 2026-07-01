#!/usr/bin/env bash
set -euo pipefail

FINETUNE_VENV_DIR="${ORACLE_FINETUNE_VENV_DIR:-$ROOT_DIR/runs/finetune-venv}"
PYTHON_BIN="${ORACLE_FINETUNE_PYTHON:-python3}"
TORCH_INDEX_URL="${ORACLE_FINETUNE_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu126}"

venv_python() {
  local result
  result="$FINETUNE_VENV_DIR/bin/python"
  if [[ -x "$FINETUNE_VENV_DIR/Scripts/python.exe" ]]; then
    result="$FINETUNE_VENV_DIR/Scripts/python.exe"
  fi
  printf '%s\n' "$result"
}

ensure_venv() {
  local python_path
  python_path="$(venv_python)"
  if [[ -x "$python_path" ]]; then
    printf '[finetune] using existing venv: %s\n' "$FINETUNE_VENV_DIR"
  else
    printf '[finetune] creating venv: %s\n' "$FINETUNE_VENV_DIR"
    "$PYTHON_BIN" -m venv "$FINETUNE_VENV_DIR"
  fi
}

ensure_base_tools() {
  local python_path
  python_path="$(venv_python)"
  "$python_path" -m pip install --upgrade pip "setuptools<82" wheel
}

torch_ready() {
  local python_path
  local result
  python_path="$(venv_python)"
  result="0"
  if "$python_path" - <<'PY'
import importlib.util

spec = importlib.util.find_spec("torch")
ok = False
if spec is not None:
    import torch
    ok = torch.cuda.is_available()
raise SystemExit(0 if ok else 1)
PY
  then
    result="1"
  fi
  printf '%s\n' "$result"
}

ensure_torch() {
  local ready
  ready="$(torch_ready)"
  if [[ "$ready" == "1" ]]; then
    printf '[finetune] CUDA torch already installed; skipping torch install\n'
  else
    printf '[finetune] installing CUDA torch from %s\n' "$TORCH_INDEX_URL"
    "$(venv_python)" -m pip install \
      --index-url "$TORCH_INDEX_URL" \
      torch torchvision torchaudio
  fi
}

finetune_requirements_ready() {
  local python_path
  local result
  python_path="$(venv_python)"
  result="0"
  if "$python_path" - <<'PY'
import importlib.util

required = (
    "bitsandbytes",
    "datasets",
    "peft",
    "transformers",
    "trl",
    "unsloth",
)
ok = all(importlib.util.find_spec(name) is not None for name in required)
if ok:
    import transformers

    ok = transformers.__version__ == "5.5.0"
raise SystemExit(0 if ok else 1)
PY
  then
    result="1"
  fi
  printf '%s\n' "$result"
}

ensure_finetune_requirements() {
  local ready
  ready="$(finetune_requirements_ready)"
  if [[ "$ready" == "1" ]]; then
    printf '[finetune] fine-tune dependencies already installed; skipping\n'
  else
    printf '[finetune] installing fine-tune dependencies\n'
    "$(venv_python)" -m pip install -r "$ROOT_DIR/requirements-finetune.txt"
  fi
}

dataset_row_count() {
  local dataset_path
  dataset_path="$1"
  if [[ -f "$dataset_path" ]]; then
    "$(venv_python)" - "$dataset_path" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
print(len(path.read_text(encoding="utf-8").splitlines()))
PY
  else
    printf '0\n'
  fi
}

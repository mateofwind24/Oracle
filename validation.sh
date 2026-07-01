#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/finetune/common.sh"

DATASET_PATH="${ORACLE_FINETUNE_DATASET:-$ROOT_DIR/data/finetune/korean_cute_style_train.jsonl}"
METADATA_PATH="${ORACLE_FINETUNE_METADATA:-$ROOT_DIR/data/finetune/korean_cute_style_sources.json}"
OUTPUT_DIR="${ORACLE_FINETUNE_OUTPUT_DIR:-$ROOT_DIR/runs/finetune/korean-cute-lora}"
MIN_EXAMPLES="${ORACLE_FINETUNE_MIN_VALIDATION_EXAMPLES:-1000}"

ensure_venv
ensure_base_tools
ensure_torch
ensure_finetune_requirements

printf '[finetune] validating dataset and adapter artifacts\n'
"$(venv_python)" "$ROOT_DIR/scripts/finetune/validate_finetune.py" \
  --dataset "$DATASET_PATH" \
  --metadata "$METADATA_PATH" \
  --adapter-dir "$OUTPUT_DIR" \
  --min-examples "$MIN_EXAMPLES" \
  "$@"

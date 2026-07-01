#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$ROOT_DIR/scripts/finetune/common.sh"

DATASET_PATH="${ORACLE_FINETUNE_DATASET:-$ROOT_DIR/data/finetune/korean_cute_style_train.jsonl}"
METADATA_PATH="${ORACLE_FINETUNE_METADATA:-$ROOT_DIR/data/finetune/korean_cute_style_sources.json}"
OUTPUT_DIR="${ORACLE_FINETUNE_OUTPUT_DIR:-$ROOT_DIR/runs/finetune/korean-cute-lora}"
TARGET_COUNT="${ORACLE_FINETUNE_TARGET_COUNT:-6000}"
EXAMPLES_PER_TOPIC="${ORACLE_FINETUNE_EXAMPLES_PER_TOPIC:-12}"
LIMIT_PER_TERM="${ORACLE_FINETUNE_LIMIT_PER_TERM:-50}"
PAUSE_SECONDS="${ORACLE_FINETUNE_CRAWL_PAUSE_SECONDS:-0.2}"
MAX_SEQ_LENGTH="${ORACLE_FINETUNE_MAX_SEQ_LENGTH:-1024}"
BATCH_SIZE="${ORACLE_FINETUNE_BATCH_SIZE:-1}"
GRAD_ACCUMULATION="${ORACLE_FINETUNE_GRAD_ACCUMULATION:-8}"
NUM_TRAIN_EPOCHS="${ORACLE_FINETUNE_NUM_TRAIN_EPOCHS:-1}"
LORA_RANK="${ORACLE_FINETUNE_LORA_RANK:-16}"
LORA_ALPHA="${ORACLE_FINETUNE_LORA_ALPHA:-16}"
REBUILD_DATA="${ORACLE_FINETUNE_REBUILD_DATA:-0}"
INSTALL_ONLY="0"

if [[ "${1:-}" == "--install-only" ]]; then
  INSTALL_ONLY="1"
  shift
fi

ensure_venv
ensure_base_tools
ensure_torch
ensure_finetune_requirements

if [[ "$INSTALL_ONLY" == "1" ]]; then
  printf '[finetune] install-only complete\n'
  exit 0
fi

current_rows="$(dataset_row_count "$DATASET_PATH")"
if [[ "$REBUILD_DATA" == "1" || "$current_rows" -lt "$TARGET_COUNT" ]]; then
  printf '[finetune] building dataset: target=%s current=%s\n' "$TARGET_COUNT" "$current_rows"
  "$(venv_python)" "$ROOT_DIR/scripts/finetune/build_korean_cute_dataset.py" \
    --allow-network \
    --output "$DATASET_PATH" \
    --metadata-output "$METADATA_PATH" \
    --target-count "$TARGET_COUNT" \
    --examples-per-topic "$EXAMPLES_PER_TOPIC" \
    --limit-per-term "$LIMIT_PER_TERM" \
    --pause-seconds "$PAUSE_SECONDS"
else
  printf '[finetune] dataset already has %s rows; skipping crawl\n' "$current_rows"
fi

train_args=(
  "$ROOT_DIR/scripts/finetune/train_qlora.py"
  --dataset "$DATASET_PATH"
  --output-dir "$OUTPUT_DIR"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --batch-size "$BATCH_SIZE"
  --grad-accumulation "$GRAD_ACCUMULATION"
  --num-train-epochs "$NUM_TRAIN_EPOCHS"
  --lora-rank "$LORA_RANK"
  --lora-alpha "$LORA_ALPHA"
  --offload-embedding
)

if [[ -n "${ORACLE_FINETUNE_MAX_STEPS:-}" ]]; then
  train_args+=(--max-steps "$ORACLE_FINETUNE_MAX_STEPS")
fi
if [[ "${ORACLE_FINETUNE_ALLOW_LOW_VRAM:-0}" == "1" ]]; then
  train_args+=(--allow-low-vram)
fi
if [[ -n "${ORACLE_FINETUNE_MODEL_ID:-}" ]]; then
  train_args+=(--model-id "$ORACLE_FINETUNE_MODEL_ID")
fi

printf '[finetune] starting QLoRA training\n'
"$(venv_python)" "${train_args[@]}" "$@"

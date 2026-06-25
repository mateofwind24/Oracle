#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LLAMA_CPP_DIR="${ORACLE_LLAMA_CPP_DIR:-$ROOT_DIR/llama.cpp}"
MODEL_PATH="${1:-${ORACLE_LLAMA_MODEL_PATH:-$ROOT_DIR/models/gemma-3-1b-it-Q4_0.gguf}}"
if [[ ! -f "$MODEL_PATH" ]]; then
  printf '[llama][error] model not found: %s\n' "$MODEL_PATH" >&2
  printf 'usage: scripts/run_llama_server.sh /path/to/model.gguf\n' >&2
  exit 1
fi

if [[ -n "${ORACLE_LLAMA_SERVER_BIN:-}" ]]; then
  LLAMA_SERVER_BIN="$ORACLE_LLAMA_SERVER_BIN"
elif command -v llama-server >/dev/null 2>&1; then
  LLAMA_SERVER_BIN="$(command -v llama-server)"
elif [[ -x "$LLAMA_CPP_DIR/build/bin/llama-server" ]]; then
  LLAMA_SERVER_BIN="$LLAMA_CPP_DIR/build/bin/llama-server"
else
  printf '[llama][error] llama-server not found; run ./build.sh first\n' >&2
  exit 1
fi
"$LLAMA_SERVER_BIN" \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 \
  --port 8080 \
  -c "${LLAMA_CONTEXT_SIZE:-4096}" \
  -fa off \
  -ctk q4_0 \
  --reasoning-format none

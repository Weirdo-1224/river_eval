#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PATH="$HOME/.local/bin:$PATH"

python3 -m river_eval.runners.run_eval \
  --config configs/retro_memory_gpt.yaml \
  "$@"

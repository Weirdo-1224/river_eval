#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTHONPATH="${PYTHONPATH:-}:$(pwd)"
export PATH="$HOME/.local/bin:$PATH"

python3 -m benchmark.runners.run_eval \
  --config configs/experiments/river_retro_qwen_strict.yaml \
  "$@"

#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python}
CONFIG=${SPECTRAL_TRIGGER_CONFIG:-$ROOT/experiments/configs/spectral_event_trigger.yaml}
OUT_DIR=${1:-$ROOT/remote_results/controlled_spectral_trigger}

mkdir -p "$OUT_DIR"
exec "$PYTHON_BIN" "$ROOT/experiments/probes/run_spectral_event_trigger.py" \
  --config "$CONFIG" \
  --out-dir "$OUT_DIR"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
PYTHON="${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}"
GPU_INDEX="${GPU_INDEX:-2}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/support_manifold_oracle_f81_registered_v1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
CAPTURE_INDEX="${CAPTURE_INDEX:-$ROOT/results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv}"
PROTOCOL="${PROTOCOL:-$ROOT/configs/support_manifold_oracle_f81_v1.json}"

mkdir -p "$LOG_DIR"
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to reuse existing output path: $OUTPUT_DIR" >&2
  exit 2
fi

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONHASHSEED=0

"$PYTHON" scripts/probe_support_manifold_oracle.py \
  --capture-index "$CAPTURE_INDEX" \
  --protocol-config "$PROTOCOL" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --run-kind registered \
  --capture-hash-mode sha256 \
  --execution-resource-note "exclusive visible H200 GPU_INDEX=$GPU_INDEX numerical oracle; no measured latency claim" \
  2>&1 | tee "$LOG_DIR/support_manifold_oracle_f81_registered_v1.log"

#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
PYTHON="${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}"
GPU_INDEX="${GPU_INDEX:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/results/restricted_rotation_oracle_f81_refinement_v1}"
ANALYSIS_DIR="${ANALYSIS_DIR:-$ROOT/results/restricted_rotation_oracle_f81_refinement_analysis_v2}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
CAPTURE_INDEX="${CAPTURE_INDEX:-$ROOT/results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv}"
PROTOCOL="${PROTOCOL:-$ROOT/configs/restricted_rotation_oracle_f81_refinement_v1.json}"

mkdir -p "$LOG_DIR"
for path in "$OUTPUT_DIR" "$ANALYSIS_DIR"; do
  if [[ -e "$path" ]]; then
    echo "refusing to reuse existing output path: $path" >&2
    exit 2
  fi
done

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONHASHSEED=0

"$PYTHON" scripts/probe_restricted_rotation_oracle.py \
  --capture-index "$CAPTURE_INDEX" \
  --protocol-config "$PROTOCOL" \
  --output-dir "$OUTPUT_DIR" \
  --device cuda:0 \
  --run-kind registered \
  --capture-hash-mode sha256 \
  --execution-resource-note "exclusive visible RTX4090 GPU_INDEX=$GPU_INDEX convergence refinement; no latency claim" \
  2>&1 | tee "$LOG_DIR/restricted_rotation_oracle_f81_refinement_v1.log"

"$PYTHON" scripts/analyze_restricted_rotation_oracle.py \
  --input-dir "$OUTPUT_DIR" \
  --output-dir "$ANALYSIS_DIR" \
  2>&1 | tee "$LOG_DIR/restricted_rotation_oracle_f81_refinement_analysis_v2.log"

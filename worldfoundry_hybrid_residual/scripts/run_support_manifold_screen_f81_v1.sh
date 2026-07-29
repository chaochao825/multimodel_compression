#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
PYTHON="${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}"
GPU_A="${GPU_A:-2}"
GPU_B="${GPU_B:-3}"
BASE="${BASE:-$ROOT/results/support_manifold_oracle_f81_screen_v1}"
SHARD_A="${SHARD_A:-${BASE}_shard0}"
SHARD_B="${SHARD_B:-${BASE}_shard1}"
MERGED="${MERGED:-${BASE}_merged}"
ANALYSIS="${ANALYSIS:-${BASE}_analysis_v1}"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
CAPTURE_INDEX="${CAPTURE_INDEX:-$ROOT/results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv}"
PROTOCOL="${PROTOCOL:-$ROOT/configs/support_manifold_oracle_f81_screen_v1.json}"

mkdir -p "$LOG_DIR"
for path in "$SHARD_A" "$SHARD_B" "$MERGED" "$ANALYSIS"; do
  if [[ -e "$path" ]]; then
    echo "refusing to reuse existing path: $path" >&2
    exit 2
  fi
done

cd "$ROOT"
export PYTHONHASHSEED=0

run_shard() {
  local gpu="$1"
  local index="$2"
  local output="$3"
  local log="$4"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" scripts/probe_support_manifold_oracle.py \
    --capture-index "$CAPTURE_INDEX" \
    --protocol-config "$PROTOCOL" \
    --output-dir "$output" \
    --device cuda:0 \
    --run-kind registered \
    --capture-hash-mode sha256 \
    --sample-shard-index "$index" \
    --sample-shard-count 2 \
    --execution-resource-note "exclusive visible H200 physical GPU_INDEX=$gpu, sample shard=$index/2; no latency claim" \
    >"$log" 2>&1
}

run_shard "$GPU_A" 0 "$SHARD_A" "$LOG_DIR/support_manifold_screen_shard0.log" &
pid_a=$!
run_shard "$GPU_B" 1 "$SHARD_B" "$LOG_DIR/support_manifold_screen_shard1.log" &
pid_b=$!

cleanup() {
  kill -TERM "$pid_a" "$pid_b" 2>/dev/null || true
}
trap cleanup INT TERM

status=0
wait "$pid_a" || status=$?
wait "$pid_b" || status=$?
trap - INT TERM
if [[ "$status" -ne 0 ]]; then
  echo "support manifold shard failed with status $status" >&2
  exit "$status"
fi

"$PYTHON" scripts/merge_support_manifold_shards.py \
  --input-dir "$SHARD_A" \
  --input-dir "$SHARD_B" \
  --protocol-config "$PROTOCOL" \
  --output-dir "$MERGED"

"$PYTHON" scripts/analyze_support_manifold_oracle.py \
  --input-dir "$MERGED" \
  --output-dir "$ANALYSIS"

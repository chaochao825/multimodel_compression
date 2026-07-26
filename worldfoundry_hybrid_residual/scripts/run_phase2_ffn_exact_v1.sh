#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
BASE_ROOT="${BASE_ROOT:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723}"
PY="${PY:-$BASE_ROOT/.venv/bin/python}"
WAN_SOURCE="${WAN_SOURCE:-$BASE_ROOT/wan_runtime/MonarchRT}"
CHECKPOINT="${CHECKPOINT:-$WAN_SOURCE/wan_models/Wan2.1-T2V-1.3B}"
OUT="${OUT:-$PROBE_ROOT/results/ffn_exact_h200_v1}"
GPU="${GPU:-2}"
LOCK_PATH="${LOCK_PATH:-/tmp/codex_phase2_strict_h200_v1.lock}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
IDLE_POLLS="${IDLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MONITOR_SECONDS="${MONITOR_SECONDS:-10}"
COMPILE_MODES="${COMPILE_MODES:-default,reduce-overhead,max-autotune}"

mkdir -p "$OUT/logs"

gpu_process_count() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    --id="$GPU" 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
}

wait_for_idle_gpu() {
  if [[ "$WAIT_FOR_IDLE" != "1" ]]; then
    return
  fi
  local consecutive=0
  while (( consecutive < IDLE_POLLS )); do
    local count
    count="$(gpu_process_count)"
    if [[ "$count" == "0" ]]; then
      consecutive=$((consecutive + 1))
      printf '[ffn-exact] GPU %s idle poll %d/%d\n' "$GPU" "$consecutive" "$IDLE_POLLS"
    else
      consecutive=0
      printf '[ffn-exact] waiting: %s compute processes still use GPU %s\n' "$count" "$GPU"
    fi
    if (( consecutive < IDLE_POLLS )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

monitor_gpu() {
  local stop_file="$1"
  local telemetry_file="$2"
  while [[ ! -e "$stop_file" ]]; do
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)" >> "$telemetry_file"
    nvidia-smi --id="$GPU" \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    nvidia-smi --id="$GPU" \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    sleep "$MONITOR_SECONDS"
  done
}

# The same advisory lock serializes this runner behind the strict phase-2 queue.
exec 9>"$LOCK_PATH"
printf '[ffn-exact] waiting for lock %s\n' "$LOCK_PATH"
flock 9
printf '[ffn-exact] acquired lock %s\n' "$LOCK_PATH"
wait_for_idle_gpu

STOP_FILE="$OUT/logs/gpu-monitor-stop.$BASHPID.$RANDOM"
TELEMETRY_FILE="$OUT/logs/gpu-telemetry.log"
monitor_gpu "$STOP_FILE" "$TELEMETRY_FILE" &
MONITOR_PID=$!

cleanup_monitor() {
  touch "$STOP_FILE"
  wait "$MONITOR_PID" 2>/dev/null || true
}
trap cleanup_monitor EXIT

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
  "$PROBE_ROOT/scripts/benchmark_wan_ffn_exact_paths.py" \
  --wan-source "$WAN_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT" \
  --device cuda:0 \
  --layers 0,14,29 \
  --cases F17:7800,F81:32760 \
  --compile-modes "$COMPILE_MODES" \
  --warmup 10 \
  --repeats 50 \
  --graph-capture-warmup 3 \
  --amortization-calls 40 \
  2>&1 | tee "$OUT/logs/benchmark.log"
BENCHMARK_STATUS=${PIPESTATUS[0]}
set -e
if (( BENCHMARK_STATUS != 0 )); then
  exit "$BENCHMARK_STATUS"
fi

"$PY" "$PROBE_ROOT/scripts/summarize_wan_ffn_exact_paths.py" \
  --input "$OUT/wan_ffn_exact_paths.csv" \
  --output-dir "$OUT" \
  --min-median-speedup 1.10 \
  --min-p95-speedup 1.00 \
  --min-amortized-speedup 1.00 \
  --max-incremental-memory-gib 4.0 \
  2>&1 | tee "$OUT/logs/summary.log"

printf '[ffn-exact] completed %s\n' "$OUT"

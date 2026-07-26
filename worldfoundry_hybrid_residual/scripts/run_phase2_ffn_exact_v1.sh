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
MONITOR_SECONDS="${MONITOR_SECONDS:-5}"
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

is_descendant_of() {
  local current="$1"
  local ancestor="$2"
  while [[ "$current" =~ ^[0-9]+$ ]] && (( current > 1 )); do
    if [[ "$current" == "$ancestor" ]]; then
      return 0
    fi
    if [[ ! -r "/proc/$current/status" ]]; then
      return 1
    fi
    current="$(awk '/^PPid:/ {print $2}' "/proc/$current/status")"
  done
  return 1
}

monitor_gpu() {
  local stop_file="$1"
  local telemetry_file="$2"
  local contamination_file="$3"
  local owner_pid="$4"
  local command_pid="$5"
  while [[ ! -e "$stop_file" ]]; do
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)" >> "$telemetry_file"
    nvidia-smi --id="$GPU" \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    local compute_rows
    compute_rows="$(nvidia-smi --id="$GPU" \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits 2>&1 || true)"
    printf '%s\n' "$compute_rows" >> "$telemetry_file"
    while IFS=',' read -r gpu_uuid process_pid process_name used_memory; do
      process_pid="${process_pid//[[:space:]]/}"
      if [[ ! "$process_pid" =~ ^[0-9]+$ ]]; then
        continue
      fi
      if ! is_descendant_of "$process_pid" "$owner_pid"; then
        printf 'timestamp=%s gpu_uuid=%s pid=%s process=%s memory=%s\n' \
          "$(date --iso-8601=seconds)" "$gpu_uuid" "$process_pid" \
          "$process_name" "$used_memory" >> "$contamination_file"
        if kill -0 "$command_pid" 2>/dev/null; then
          kill -TERM "$command_pid"
        fi
      fi
    done <<< "$compute_rows"
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
RUN_ID="$(date +%s)"
TELEMETRY_FILE="$OUT/logs/gpu-telemetry.$RUN_ID.log"
CONTAMINATION_FILE="$OUT/logs/gpu-contamination.$RUN_ID.log"
AUDIT_FILE="$OUT/gpu_exclusivity_audit.json"
OWNER_PID=$BASHPID

cleanup_monitor() {
  touch "$STOP_FILE"
  if [[ -n "${MONITOR_PID:-}" ]]; then
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
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
  > >(tee "$OUT/logs/benchmark.log") 2>&1 &
BENCHMARK_PID=$!
monitor_gpu "$STOP_FILE" "$TELEMETRY_FILE" "$CONTAMINATION_FILE" \
  "$OWNER_PID" "$BENCHMARK_PID" &
MONITOR_PID=$!
wait "$BENCHMARK_PID"
BENCHMARK_STATUS=$?
touch "$STOP_FILE"
wait "$MONITOR_PID"
GPU_UUID="$(nvidia-smi --id="$GPU" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')"
AUDIT_STATUS=0
"$PY" "$PROBE_ROOT/scripts/audit_gpu_telemetry.py" \
  --telemetry "$TELEMETRY_FILE" \
  --output "$AUDIT_FILE" \
  --stage ffn_exact_h200 \
  --used-gpu-uuid "$GPU_UUID" \
  --allowed-process-prefix "$PY" \
  --foreign-events "$CONTAMINATION_FILE" \
  --require-exclusive || AUDIT_STATUS=$?
set -e
if (( AUDIT_STATUS != 0 )); then
  printf '[ffn-exact] invalid timing: foreign process overlap detected\n'
  exit 86
fi
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

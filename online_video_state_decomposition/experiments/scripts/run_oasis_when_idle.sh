#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf 'usage: %s RUN_NAME [smoke|formal] [GPU_INDEX]\n' "$0" >&2
  exit 2
fi

unset PREFIX

RUN_NAME="$1"
SCOPE="${2:-smoke}"
GPU_INDEX="${3:-3}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
OUTPUT_DIR="$RUNTIME_ROOT/remote_results/oasis_streamingbench/$RUN_NAME"
BUILD_DIR="$RUNTIME_ROOT/remote_results/downloads/oasis_env_flashattn_source_build_sm80_20260719"
POLL_SECONDS="${POLL_SECONDS:-60}"
MAX_IDLE_MEMORY_MIB="${MAX_IDLE_MEMORY_MIB:-4096}"
MAX_IDLE_UTILIZATION="${MAX_IDLE_UTILIZATION:-20}"

mkdir -p "$OUTPUT_DIR"
printf 'waiting_for_flash_attn\n' > "$OUTPUT_DIR/queue_status"

while true; do
  build_status="missing"
  if [[ -f "$BUILD_DIR/status" ]]; then
    build_status="$(cat "$BUILD_DIR/status")"
  fi
  case "$build_status" in
    completed)
      break
      ;;
    failed)
      printf 'failed_dependency\n' > "$OUTPUT_DIR/queue_status"
      printf '%s flash-attn build failed\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$OUTPUT_DIR/queue.log"
      exit 1
      ;;
    *)
      printf '%s flash-attn status=%s\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$build_status" \
        >> "$OUTPUT_DIR/queue.log"
      sleep "$POLL_SECONDS"
      ;;
  esac
done

while true; do
  IFS=',' read -r memory_used utilization < <(
    nvidia-smi --id="$GPU_INDEX" \
      --query-gpu=memory.used,utilization.gpu \
      --format=csv,noheader,nounits | tr -d ' '
  )
  if (( memory_used <= MAX_IDLE_MEMORY_MIB && utilization <= MAX_IDLE_UTILIZATION )); then
    attempt="$(date -u +%Y%m%dT%H%M%SZ)"
    attempt_log="$OUTPUT_DIR/queue_attempt_${attempt}.log"
    printf 'launching\n' > "$OUTPUT_DIR/queue_status"
    set +e
    MAX_IDLE_MEMORY_MIB="$MAX_IDLE_MEMORY_MIB" \
      MAX_IDLE_UTILIZATION="$MAX_IDLE_UTILIZATION" \
      bash "$PROJECT_ROOT/experiments/scripts/run_oasis_streamingbench.sh" \
        "$RUN_NAME" "$SCOPE" "$GPU_INDEX" > "$attempt_log" 2>&1
    return_code=$?
    set -e
    printf '%s\n' "$return_code" > "$OUTPUT_DIR/queue_attempt_${attempt}.exit_code"
    if [[ "$return_code" -eq 0 ]]; then
      printf 'completed\n' > "$OUTPUT_DIR/queue_status"
      exit 0
    fi
    if grep -Eq 'GPU idle gate failed|GPU became busy|GPU lock is already held' \
      "$attempt_log"; then
      printf 'waiting_for_idle\n' > "$OUTPUT_DIR/queue_status"
      printf '%s launch race; retrying\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        >> "$OUTPUT_DIR/queue.log"
      sleep "$POLL_SECONDS"
      continue
    fi
    printf 'failed\n' > "$OUTPUT_DIR/queue_status"
    exit "$return_code"
  fi
  printf 'waiting_for_idle\n' > "$OUTPUT_DIR/queue_status"
  printf '%s gpu=%s memory=%sMiB utilization=%s%%\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$GPU_INDEX" "$memory_used" "$utilization" \
    >> "$OUTPUT_DIR/queue.log"
  sleep "$POLL_SECONDS"
done

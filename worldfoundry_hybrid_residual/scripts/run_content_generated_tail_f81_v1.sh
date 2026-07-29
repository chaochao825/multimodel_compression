#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}
PYTHON=${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}
CAPTURE_INDEX=${CAPTURE_INDEX:-$ROOT/results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv}
PROTOCOL=${PROTOCOL:-$ROOT/configs/content_generated_tail_f81_v1.json}
LOG_DIR=${LOG_DIR:-$ROOT/logs}

cd "$ROOT"
mkdir -p "$LOG_DIR" results trash

for required in "$PYTHON" "$CAPTURE_INDEX" "$PROTOCOL"; do
  if [[ ! -e "$required" ]]; then
    printf 'missing required input: %s\n' "$required" >&2
    exit 2
  fi
done

for rank in 16 32 48 64; do
  output="results/content_generated_tail_f81_rank${rank}_v1"
  if [[ -e "$output" ]]; then
    printf 'refusing to reuse output: %s\n' "$output" >&2
    exit 2
  fi
done

PYTHONPATH=scripts "$PYTHON" -m unittest -v test_content_generated_tail

exec 8>/tmp/codex-gpu2.lock
exec 9>/tmp/codex-gpu3.lock
if ! flock -n 8; then
  printf 'physical GPU2 lock is busy\n' >&2
  exit 75
fi
if ! flock -n 9; then
  printf 'physical GPU3 lock is busy\n' >&2
  exit 75
fi

wait_idle() {
  local gpu=$1 waited=0 memory utilization
  while true; do
    IFS=, read -r memory utilization < <(
      nvidia-smi --id="$gpu" \
        --query-gpu=memory.used,utilization.gpu \
        --format=csv,noheader,nounits
    )
    memory=${memory//[[:space:]]/}
    utilization=${utilization//[[:space:]]/}
    if (( memory <= 2048 && utilization <= 5 )); then
      sleep 10
      IFS=, read -r memory utilization < <(
        nvidia-smi --id="$gpu" \
          --query-gpu=memory.used,utilization.gpu \
          --format=csv,noheader,nounits
      )
      memory=${memory//[[:space:]]/}
      utilization=${utilization//[[:space:]]/}
      if (( memory <= 2048 && utilization <= 5 )); then
        return 0
      fi
    fi
    if (( waited >= 21600 )); then
      printf 'timed out waiting for physical GPU%s\n' "$gpu" >&2
      return 75
    fi
    printf 'waiting for physical GPU%s: memory=%s MiB utilization=%s%%\n' \
      "$gpu" "$memory" "$utilization"
    sleep 30
    waited=$((waited + 30))
  done
}

run_rank() {
  local gpu=$1 rank=$2
  local output="results/content_generated_tail_f81_rank${rank}_v1"
  local log="$LOG_DIR/content_generated_tail_f81_rank${rank}_v1.log"
  printf 'state=RUNNING\nrank=%s\ngpu=%s\nstarted_utc=%s\n' \
    "$rank" "$gpu" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$LOG_DIR/content_generated_tail_f81_rank${rank}_v1.status"
  CUDA_VISIBLE_DEVICES="$gpu" PYTHONHASHSEED=0 "$PYTHON" \
    scripts/probe_content_generated_tail.py \
      --capture-index "$CAPTURE_INDEX" \
      --protocol-config "$PROTOCOL" \
      --rank "$rank" \
      --output-dir "$output" \
      --device cuda:0 \
      --run-kind diagnostic \
      --execution-resource-note \
        "exclusive H200 physical GPU${gpu}; numerical probe only; no measured latency claim" \
      >"$log" 2>&1
  printf 'state=SUCCESS\nrank=%s\ngpu=%s\nfinished_utc=%s\n' \
    "$rank" "$gpu" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >"$LOG_DIR/content_generated_tail_f81_rank${rank}_v1.status"
}

(
  wait_idle 2
  run_rank 2 16
  run_rank 2 32
) &
worker_a=$!

(
  wait_idle 3
  run_rank 3 48
  run_rank 3 64
) &
worker_b=$!

status=0
wait "$worker_a" || status=$?
wait "$worker_b" || status=$?
if (( status != 0 )); then
  printf 'content-tail worker failed with status %s\n' "$status" >&2
  exit "$status"
fi

printf 'all content-generated tail ranks completed\n'

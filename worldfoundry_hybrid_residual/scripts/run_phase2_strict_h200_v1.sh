#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
BASE_ROOT="${BASE_ROOT:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723}"
PY="${PY:-$BASE_ROOT/.venv/bin/python}"
WAN_SOURCE="${WAN_SOURCE:-$BASE_ROOT/wan_runtime/MonarchRT}"
CHECKPOINT="${CHECKPOINT:-$WAN_SOURCE/wan_models/Wan2.1-T2V-1.3B}"
PROMPT_FILE="${PROMPT_FILE:-$BASE_ROOT/scripts/prompts_pilot8.txt}"
OUT="${OUT:-$PROBE_ROOT/results/strict_phase2_h200_v1}"
GPU_PAIR="${GPU_PAIR:-2,3}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
IDLE_POLLS="${IDLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MONITOR_SECONDS="${MONITOR_SECONDS:-10}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p "$OUT/logs" "$OUT/cfg_f81_matrix"

gpu_process_count() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    --id="${GPU_PAIR}" 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
}

wait_for_idle_pair() {
  if [[ "$WAIT_FOR_IDLE" != "1" ]]; then
    return
  fi
  local consecutive=0
  while (( consecutive < IDLE_POLLS )); do
    local count
    count="$(gpu_process_count)"
    if [[ "$count" == "0" ]]; then
      consecutive=$((consecutive + 1))
      printf '[phase2] H200 pair idle poll %d/%d\n' "$consecutive" "$IDLE_POLLS"
    else
      consecutive=0
      printf '[phase2] waiting: %s compute processes still use GPU %s\n' "$count" "$GPU_PAIR"
    fi
    if (( consecutive < IDLE_POLLS )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

monitor_gpu_pair() {
  local stop_file="$1"
  local telemetry_file="$2"
  while [[ ! -e "$stop_file" ]]; do
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)" >> "$telemetry_file"
    nvidia-smi --id="$GPU_PAIR" \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    nvidia-smi --id="$GPU_PAIR" \
      --query-compute-apps=gpu_uuid,pid,process_name,used_memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    sleep "$MONITOR_SECONDS"
  done
}

run_cfg_stage() {
  local label="$1"
  local seed="$2"
  local stage_out="$OUT/$label"
  local stop_file="$OUT/logs/${label}.monitor-stop.$BASHPID.$RANDOM"
  local telemetry_file="$OUT/logs/${label}.gpu-telemetry.log"
  mkdir -p "$stage_out"
  wait_for_idle_pair
  monitor_gpu_pair "$stop_file" "$telemetry_file" &
  local monitor_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PY" -m torch.distributed.run \
    --standalone --nproc-per-node=2 \
    "$PROBE_ROOT/scripts/generate_wan_cfg_parallel.py" \
    --wan-source "$WAN_SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$stage_out" \
    --prompt-file "$PROMPT_FILE" \
    --max-prompts 4 \
    --methods sequential,cfg_parallel \
    --frame-num 81 \
    --sampling-steps 20 \
    --warmup-steps 1 \
    --repeats 2 \
    --seed "$seed" \
    --alternate-method-order \
    2>&1 | tee "$OUT/logs/${label}.log"
  local command_status=${PIPESTATUS[0]}
  set -e
  touch "$stop_file"
  wait "$monitor_pid"
  if (( command_status != 0 )); then
    return "$command_status"
  fi
  "$PY" "$PROBE_ROOT/scripts/summarize_cfg_parallel.py" \
    --run-dir "$stage_out" \
    --out-dir "$stage_out" \
    --require-exact \
    2>&1 | tee "$OUT/logs/${label}.summary.log"
}

run_geometry_stage() {
  local stage_out="$OUT/geometry_attention"
  mkdir -p "$stage_out"
  wait_for_idle_pair
  CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
    "$PROBE_ROOT/scripts/probe_geometry_sparse_attention.py" \
    --replay "$BASE_ROOT/results/qkv_replay_f17_t1000_l0/f17_t1000_cond_l0_self.pt" \
    --replay "$BASE_ROOT/results/qkv_replay_f81_t1000_l0_v1/f81_t1000_cond_l0_self.pt" \
    --output-dir "$stage_out" \
    --device cuda:0 \
    --query-samples 256 \
    --tail-ranks 0,4,8,16 \
    --error-targets 0.02,0.05 \
    2>&1 | tee "$OUT/logs/geometry_attention.log"
  "$PY" "$PROBE_ROOT/scripts/plot_geometry_sparse_attention.py" \
    --input-dir "$stage_out" \
    --output-dir "$stage_out" \
    2>&1 | tee "$OUT/logs/geometry_attention.plot.log"
}

run_crossattn_cache_stage() {
  local stage_out="$OUT/crossattn_cache_f17"
  local stop_file="$OUT/logs/crossattn_cache_f17.monitor-stop.$BASHPID.$RANDOM"
  local telemetry_file="$OUT/logs/crossattn_cache_f17.gpu-telemetry.log"
  mkdir -p "$stage_out"
  wait_for_idle_pair
  monitor_gpu_pair "$stop_file" "$telemetry_file" &
  local monitor_pid=$!
  set +e
  CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
    "$PROBE_ROOT/scripts/generate_wan_crossattn_cache.py" \
    --wan-source "$WAN_SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$stage_out" \
    --prompt-file "$PROMPT_FILE" \
    --max-prompts 2 \
    --methods baseline,crossattn_kv_cache \
    --frame-num 17 \
    --sampling-steps 20 \
    --warmup-steps 1 \
    --repeats 2 \
    --seed 20260738 \
    --alternate-method-order \
    --device cuda:0 \
    2>&1 | tee "$OUT/logs/crossattn_cache_f17.log"
  local command_status=${PIPESTATUS[0]}
  set -e
  touch "$stop_file"
  wait "$monitor_pid"
  if (( command_status != 0 )); then
    return "$command_status"
  fi
  "$PY" "$PROBE_ROOT/scripts/summarize_crossattn_cache.py" \
    --run-dir "$stage_out" \
    --out-dir "$stage_out" \
    --require-exact \
    2>&1 | tee "$OUT/logs/crossattn_cache_f17.summary.log"
}

run_geometry_stage
run_crossattn_cache_stage
run_cfg_stage cfg_f81_seed20260730 20260730
run_cfg_stage cfg_f81_seed20260734 20260734

"$PY" "$PROBE_ROOT/scripts/summarize_cfg_parallel.py" \
  --run-dir "$OUT/cfg_f81_seed20260730" \
  --run-dir "$OUT/cfg_f81_seed20260734" \
  --out-dir "$OUT/cfg_f81_matrix" \
  --require-exact \
  2>&1 | tee "$OUT/logs/cfg_f81_matrix.summary.log"

printf '[phase2] completed %s\n' "$OUT"

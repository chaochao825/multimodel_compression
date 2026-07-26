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
MONITOR_SECONDS="${MONITOR_SECONDS:-5}"
RUN_GEOMETRY_STAGE="${RUN_GEOMETRY_STAGE:-1}"
RUN_CROSSATTN_CACHE_STAGE="${RUN_CROSSATTN_CACHE_STAGE:-1}"
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

monitor_gpu_pair() {
  local stop_file="$1"
  local telemetry_file="$2"
  local contamination_file="$3"
  local owner_pid="$4"
  local command_pid="$5"
  local gpu_ids="$6"
  while [[ ! -e "$stop_file" ]]; do
    printf 'timestamp=%s\n' "$(date --iso-8601=seconds)" >> "$telemetry_file"
    nvidia-smi --id="$gpu_ids" \
      --query-gpu=index,name,memory.used,memory.total,utilization.gpu,utilization.memory \
      --format=csv,noheader,nounits >> "$telemetry_file" 2>&1
    local compute_rows
    compute_rows="$(nvidia-smi --id="$gpu_ids" \
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

audit_gpu_stage() {
  local stage="$1"
  local telemetry_file="$2"
  local audit_file="$3"
  local gpu_ids="$4"
  local contamination_file="$5"
  local audit_args=(
    --telemetry "$telemetry_file"
    --output "$audit_file"
    --stage "$stage"
    --allowed-process-prefix "$PY"
    --foreign-events "$contamination_file"
    --require-exclusive
  )
  while IFS= read -r gpu_uuid; do
    gpu_uuid="${gpu_uuid//[[:space:]]/}"
    if [[ -n "$gpu_uuid" ]]; then
      audit_args+=(--used-gpu-uuid "$gpu_uuid")
    fi
  done < <(nvidia-smi --id="$gpu_ids" --query-gpu=uuid --format=csv,noheader,nounits)
  "$PY" "$PROBE_ROOT/scripts/audit_gpu_telemetry.py" "${audit_args[@]}"
}

run_cfg_stage() {
  local label="$1"
  local seed="$2"
  local stage_out="$OUT/$label"
  local run_id="$(date +%s)"
  local stop_file="$OUT/logs/${label}.monitor-stop.$BASHPID.$RANDOM"
  local telemetry_file="$OUT/logs/${label}.gpu-telemetry.${run_id}.log"
  local contamination_file="$OUT/logs/${label}.contamination.${run_id}.log"
  local audit_file="$stage_out/gpu_exclusivity_audit.json"
  local owner_pid=$BASHPID
  mkdir -p "$stage_out"
  wait_for_idle_pair
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
    > >(tee "$OUT/logs/${label}.log") 2>&1 &
  local command_pid=$!
  monitor_gpu_pair "$stop_file" "$telemetry_file" "$contamination_file" \
    "$owner_pid" "$command_pid" "$GPU_PAIR" &
  local monitor_pid=$!
  wait "$command_pid"
  local command_status=$?
  touch "$stop_file"
  wait "$monitor_pid"
  local audit_status=0
  audit_gpu_stage "$label" "$telemetry_file" "$audit_file" "$GPU_PAIR" \
    "$contamination_file" || audit_status=$?
  set -e
  if (( audit_status != 0 )); then
    printf '[phase2] invalid timing: foreign process overlap detected for %s\n' "$label"
    return 86
  fi
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
  local run_id="$(date +%s)"
  local stop_file="$OUT/logs/crossattn_cache_f17.monitor-stop.$BASHPID.$RANDOM"
  local telemetry_file="$OUT/logs/crossattn_cache_f17.gpu-telemetry.${run_id}.log"
  local contamination_file="$OUT/logs/crossattn_cache_f17.contamination.${run_id}.log"
  local audit_file="$stage_out/gpu_exclusivity_audit.json"
  local used_gpu="${GPU_PAIR%%,*}"
  local owner_pid=$BASHPID
  mkdir -p "$stage_out"
  wait_for_idle_pair
  set +e
  CUDA_VISIBLE_DEVICES="$used_gpu" "$PY" \
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
    > >(tee "$OUT/logs/crossattn_cache_f17.log") 2>&1 &
  local command_pid=$!
  monitor_gpu_pair "$stop_file" "$telemetry_file" "$contamination_file" \
    "$owner_pid" "$command_pid" "$used_gpu" &
  local monitor_pid=$!
  wait "$command_pid"
  local command_status=$?
  touch "$stop_file"
  wait "$monitor_pid"
  local audit_status=0
  audit_gpu_stage crossattn_cache_f17 "$telemetry_file" "$audit_file" \
    "$used_gpu" "$contamination_file" || audit_status=$?
  set -e
  if (( audit_status != 0 )); then
    printf '[phase2] invalid timing: foreign process overlap detected for crossattn_cache_f17\n'
    return 86
  fi
  if (( command_status != 0 )); then
    return "$command_status"
  fi
  "$PY" "$PROBE_ROOT/scripts/summarize_crossattn_cache.py" \
    --run-dir "$stage_out" \
    --out-dir "$stage_out" \
    --require-exact \
    2>&1 | tee "$OUT/logs/crossattn_cache_f17.summary.log"
}

if [[ "$RUN_GEOMETRY_STAGE" == "1" ]]; then
  run_geometry_stage
fi
if [[ "$RUN_CROSSATTN_CACHE_STAGE" == "1" ]]; then
  run_crossattn_cache_stage
fi
run_cfg_stage cfg_f81_seed20260730 20260730
run_cfg_stage cfg_f81_seed20260734 20260734

"$PY" "$PROBE_ROOT/scripts/summarize_cfg_parallel.py" \
  --run-dir "$OUT/cfg_f81_seed20260730" \
  --run-dir "$OUT/cfg_f81_seed20260734" \
  --out-dir "$OUT/cfg_f81_matrix" \
  --require-exact \
  2>&1 | tee "$OUT/logs/cfg_f81_matrix.summary.log"

printf '[phase2] completed %s\n' "$OUT"

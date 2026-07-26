#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
BASE_ROOT="${BASE_ROOT:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723}"
PY="${PY:-$BASE_ROOT/.venv/bin/python}"
WAN_SOURCE="${WAN_SOURCE:-$BASE_ROOT/wan_runtime/MonarchRT}"
CHECKPOINT="${CHECKPOINT:-$WAN_SOURCE/wan_models/Wan2.1-T2V-1.3B}"
OUT="${OUT:-$PROBE_ROOT/results/acceleration_frontier_h200_v1}"
GPU_PAIR="${GPU_PAIR:-2,3}"
WAIT_FOR_IDLE="${WAIT_FOR_IDLE:-1}"
IDLE_POLLS="${IDLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

mkdir -p \
  "$OUT/logs" \
  "$OUT/speculative_batch" \
  "$OUT/full_model_batch" \
  "$OUT/defect_rmt" \
  "$OUT/cfg_smoke" \
  "$OUT/cfg_f17"

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
      printf '[frontier] H200 pair idle poll %d/%d\n' "$consecutive" "$IDLE_POLLS"
    else
      consecutive=0
      printf '[frontier] waiting: %s compute processes still use GPU %s\n' "$count" "$GPU_PAIR"
    fi
    if (( consecutive < IDLE_POLLS )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

wait_for_idle_pair

CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
  "$PROBE_ROOT/scripts/benchmark_h200_speculative_batch.py" \
  --replay "$BASE_ROOT/results/qkv_replay_f17_t1000_l0/f17_t1000_cond_l0_self.pt" \
  --replay "$BASE_ROOT/results/qkv_replay_f81_t1000_l0_v1/f81_t1000_cond_l0_self.pt" \
  --output-dir "$OUT/speculative_batch" \
  --device cuda:0 \
  --batches 1,2,4 \
  --warmup 5 \
  --repetitions 12 \
  2>&1 | tee "$OUT/logs/speculative_batch.log"

wait_for_idle_pair

CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
  "$PROBE_ROOT/scripts/benchmark_wan_target_batch.py" \
  --wan-source "$WAN_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --output-dir "$OUT/full_model_batch" \
  --device cuda:0 \
  --frame-nums 17,81 \
  --batches 1,2,4 \
  --sampling-steps 20 \
  --warmup 1 \
  --repetitions 4 \
  2>&1 | tee "$OUT/logs/full_model_batch.log"

wait_for_idle_pair

CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
  "$PROBE_ROOT/scripts/probe_defect_rmt.py" \
  --samples \
    "$PROBE_ROOT/results/activation_defect_subspace_f17_v1/data/activation_defect_samples.pt" \
  --out-dir "$OUT/defect_rmt" \
  --device cuda:0 \
  --max-rows 8192 \
  --null-repeats 4 \
  --stability-rank 16 \
  2>&1 | tee "$OUT/logs/defect_rmt.log"

wait_for_idle_pair

CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  "$PROBE_ROOT/scripts/generate_wan_cfg_parallel.py" \
  --wan-source "$WAN_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT/cfg_smoke" \
  --methods cfg_parallel \
  --frame-num 17 \
  --sampling-steps 1 \
  --warmup-steps 1 \
  --repeats 1 \
  2>&1 | tee "$OUT/logs/cfg_smoke.log"

wait_for_idle_pair

CUDA_VISIBLE_DEVICES="$GPU_PAIR" "$PY" -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  "$PROBE_ROOT/scripts/generate_wan_cfg_parallel.py" \
  --wan-source "$WAN_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT/cfg_f17" \
  --methods sequential,cfg_parallel \
  --frame-num 17 \
  --sampling-steps 20 \
  --warmup-steps 1 \
  --repeats 2 \
  --alternate-method-order \
  2>&1 | tee "$OUT/logs/cfg_f17.log"

"$PY" "$PROBE_ROOT/scripts/summarize_cfg_parallel.py" \
  --run-dir "$OUT/cfg_f17" \
  --out-dir "$OUT/cfg_f17" \
  2>&1 | tee "$OUT/logs/cfg_f17_summary.log"

printf '[frontier] completed %s\n' "$OUT"

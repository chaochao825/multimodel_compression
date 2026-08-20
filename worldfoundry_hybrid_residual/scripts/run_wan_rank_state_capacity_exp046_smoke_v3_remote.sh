#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
WAN_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/wan_runtime/MonarchRT
PY=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python
GPU=3
SHARED_LOCK=/tmp/learnable_logic_gpu_3.lock
OWN_LOCK=/tmp/codex-exp046-rankstate-gpu3.lock
OUT="$ROOT/results/rank_state_capacity_exp046_smoke_v3"
ANALYSIS="$ROOT/results/rank_state_capacity_exp046_smoke_v3_analysis"
LOG="$ROOT/logs/exp046_rank_state_capacity_smoke_v3.log"
EXIT_FILE="$ROOT/logs/exp046_rank_state_capacity_smoke_v3.exit"

exec 8>"$OWN_LOCK"
flock 8
exec 9>"$SHARED_LOCK"

printf 'QUEUE_START=%s\n' "$(date --iso-8601=seconds)" >>"$LOG"
while true; do
  if flock -n 9; then
    memory_used=$(nvidia-smi -i "$GPU" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')
    utilization=$(nvidia-smi -i "$GPU" --query-gpu=utilization.gpu --format=csv,noheader,nounits | tr -d ' ')
    process_count=$(nvidia-smi -i "$GPU" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^$/d' | wc -l)
    if [ "$memory_used" -lt 1024 ] && [ "$utilization" -lt 10 ] && [ "$process_count" -eq 0 ]; then
      break
    fi
    flock -u 9
  fi
  sleep 30
done

printf 'GPU_ACQUIRED=%s gpu=%s\n' "$(date --iso-8601=seconds)" "$GPU" >>"$LOG"
if [ -e "$OUT" ] || [ -e "$ANALYSIS" ]; then
  printf 'refusing to overwrite an existing EXP-046 smoke v3 artifact\n' >>"$LOG"
  printf '2\n' >"$EXIT_FILE"
  exit 2
fi

set +e
CUDA_VISIBLE_DEVICES="$GPU" "$PY" \
  "$ROOT/worldfoundry_hybrid_residual/scripts/run_wan_rank_state_capacity.py" \
  --wan-source "$WAN_ROOT" \
  --checkpoint "$WAN_ROOT/wan_models/Wan2.1-T2V-1.3B" \
  --prompt-file "$ROOT/worldfoundry_hybrid_residual/configs/wan_rank_state_capacity_prompts_exp046.txt" \
  --config "$ROOT/worldfoundry_hybrid_residual/configs/wan_rank_state_capacity_exp046_smoke_v1.json" \
  --sample-indices 0 \
  --out-dir "$OUT" \
  --device cuda:0 \
  --verify-equivalence >>"$LOG" 2>&1
runner_status=$?
if [ "$runner_status" -eq 0 ]; then
  "$PY" "$ROOT/worldfoundry_hybrid_residual/scripts/analyze_wan_rank_state_capacity.py" \
    --input-dir "$OUT" \
    --config "$ROOT/worldfoundry_hybrid_residual/configs/wan_rank_state_capacity_exp046_smoke_v1.json" \
    --split calibration \
    --out-dir "$ANALYSIS" >>"$LOG" 2>&1
  status=$?
else
  status=$runner_status
fi
set -e

printf '%s\n' "$status" >"$EXIT_FILE"
printf 'QUEUE_END=%s status=%s\n' "$(date --iso-8601=seconds)" "$status" >>"$LOG"
exit "$status"

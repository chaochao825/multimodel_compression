#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
BASE_ROOT="${BASE_ROOT:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723}"
PY="${PY:-$BASE_ROOT/.venv/bin/python}"
WAN_SOURCE="${WAN_SOURCE:-$BASE_ROOT/wan_runtime/MonarchRT}"
CHECKPOINT="${CHECKPOINT:-$WAN_SOURCE/wan_models/Wan2.1-T2V-1.3B}"
PROMPT_FILE="${PROMPT_FILE:-$BASE_ROOT/scripts/prompts_pilot8.txt}"
OUT="${OUT:-$PROBE_ROOT/results/geometry_generalization_h200_v1}"
GPU_PAIR="${GPU_PAIR:-2,3}"
IDLE_POLLS="${IDLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"

mkdir -p "$OUT/logs" "$OUT/qkv_replays" "$OUT/geometry_analysis"

gpu_process_count() {
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    --id="${GPU_PAIR}" 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
}

wait_for_idle_pair() {
  local consecutive=0
  while (( consecutive < IDLE_POLLS )); do
    local count
    count="$(gpu_process_count)"
    if [[ "$count" == "0" ]]; then
      consecutive=$((consecutive + 1))
      printf '[geometry-generalization] H200 pair idle poll %d/%d\n' "$consecutive" "$IDLE_POLLS"
    else
      consecutive=0
      printf '[geometry-generalization] waiting: %s compute processes use GPU %s\n' "$count" "$GPU_PAIR"
    fi
    if (( consecutive < IDLE_POLLS )); then
      sleep "$POLL_SECONDS"
    fi
  done
}

wait_for_idle_pair
CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
  "$PROBE_ROOT/scripts/capture_wan_qkv_trajectory.py" \
  --wan-source "$WAN_SOURCE" \
  --checkpoint "$CHECKPOINT" \
  --out-dir "$OUT/qkv_replays" \
  --prompt-file "$PROMPT_FILE" \
  --max-prompts 3 \
  --sample-plan 0:20260740,1:20260740,0:20260741,2:20260741 \
  --frame-num 81 \
  --sampling-steps 20 \
  --capture-steps 0 \
  --capture-layers 0 \
  --stop-after-last-capture \
  --seed 20260740 \
  --device cuda:0 \
  2>&1 | tee "$OUT/logs/qkv_capture.log"

wait_for_idle_pair
CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}" "$PY" \
  "$PROBE_ROOT/scripts/probe_geometry_sparse_attention.py" \
  --replay-index "$OUT/qkv_replays/capture_index.csv" \
  --output-dir "$OUT/geometry_analysis" \
  --device cuda:0 \
  --query-samples 128 \
  --tail-ranks 0,8,16 \
  --error-targets 0.02,0.05 \
  2>&1 | tee "$OUT/logs/geometry_analysis.log"

"$PY" "$PROBE_ROOT/scripts/summarize_geometry_generalization.py" \
  --heads-csv "$OUT/geometry_analysis/geometry_attention_heads.csv" \
  --out-dir "$OUT/geometry_analysis" \
  --calibration-sample-id s00_p00_seed20260740 \
  --validation-sample-id s01_p01_seed20260740 \
  --test-sample-id s02_p00_seed20260741 \
  --test-sample-id s03_p02_seed20260741 \
  2>&1 | tee "$OUT/logs/geometry_generalization.summary.log"

"$PY" "$PROBE_ROOT/scripts/plot_geometry_generalization.py" \
  --cells-csv "$OUT/geometry_analysis/geometry_generalization_cells.csv" \
  --out-dir "$OUT/geometry_analysis" \
  2>&1 | tee "$OUT/logs/geometry_generalization.plot.log"

printf '[geometry-generalization] pilot completed %s\n' "$OUT"

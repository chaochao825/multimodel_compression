#!/usr/bin/env bash
set -euo pipefail

STEP_START=${1:?usage: run_tri_mode_oracle_f17_cache_refine_v1.sh START END GPU TAG}
STEP_END=${2:?usage: run_tri_mode_oracle_f17_cache_refine_v1.sh START END GPU TAG}
GPU_INDEX=${3:?usage: run_tri_mode_oracle_f17_cache_refine_v1.sh START END GPU TAG}
TAG=${4:?usage: run_tri_mode_oracle_f17_cache_refine_v1.sh START END GPU TAG}

BASE_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723
PROBE_ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
OUT="$PROBE_ROOT/results/tri_mode_oracle_f17_cache_refine_${TAG}_v1"
mkdir -p "$OUT"

finish() {
    rc=$?
    if [[ $rc -eq 0 ]]; then
        printf 'PASS\n' >"$OUT/DONE"
    else
        printf '%s\n' "$rc" >"$OUT/FAILED"
    fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$BASE_ROOT/src/WorldFoundry-bc062d7-core:$BASE_ROOT/src/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:$BASE_ROOT/src/flash-attention-b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7/hopper${PYTHONPATH:+:$PYTHONPATH}"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/build_tri_mode_oracle_schedules.py" \
  --output "$OUT/schedules.json" --actions C \
  --step-group-size 1 --block-group-size 3 \
  --probe-step-start "$STEP_START" --probe-step-end "$STEP_END"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/generate_wan_tri_mode_oracle.py" \
  --wan-source "$BASE_ROOT/wan_runtime/MonarchRT" \
  --worldfoundry-source "$BASE_ROOT/src/WorldFoundry-bc062d7-core" \
  --checkpoint "$BASE_ROOT/wan_runtime/MonarchRT/wan_models/Wan2.1-T2V-1.3B" \
  --out-dir "$OUT/data" --schedule-file "$OUT/schedules.json" \
  --prompt-file "$BASE_ROOT/scripts/prompts_pilot8.txt" --max-prompts 1 \
  --seeds 20260723 --repeats 1 --frame-num 17 --sampling-steps 20 \
  --calibration-steps 20 --precision-warmup-steps 1 \
  >"$OUT/run.log" 2>&1

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/summarize_tri_mode_oracle.py" \
  --run-dir "$OUT/data" --out-dir "$OUT/analysis" \
  >"$OUT/analysis.log" 2>&1

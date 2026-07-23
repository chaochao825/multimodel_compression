#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 <name> <physical_gpu> <residual_rank> <budget_rank>" >&2
    exit 2
fi

NAME=$1
PHYSICAL_GPU=$2
RESIDUAL_RANK=$3
BUDGET_RANK=$4

BASE_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723
PROBE_ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
OUT="$PROBE_ROOT/results/worldfoundry_ffn_f17_components_${NAME}_v1"
mkdir -p "$OUT/data" "$OUT/analysis"

finish() {
    rc=$?
    if [[ $rc -eq 0 ]]; then
        printf 'PASS\n' >"$OUT/DONE"
    else
        printf '%s\n' "$rc" >"$OUT/FAILED"
    fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"
export PYTHONPATH="$BASE_ROOT/src/WorldFoundry-bc062d7-core:$BASE_ROOT/src/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:$BASE_ROOT/src/flash-attention-b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7/hopper${PYTHONPATH:+:$PYTHONPATH}"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/generate_wan_hybrid_residual.py" \
  --wan-source "$BASE_ROOT/wan_runtime/MonarchRT" \
  --worldfoundry-source "$BASE_ROOT/src/WorldFoundry-bc062d7-core" \
  --checkpoint "$BASE_ROOT/wan_runtime/MonarchRT/wan_models/Wan2.1-T2V-1.3B" \
  --out-dir "$OUT/data" \
  --prompt-file "$BASE_ROOT/scripts/prompts_pilot8.txt" \
  --max-prompts 2 --seeds 20260723 \
  --methods dense,fp8_middle1,hybrid_middle1 \
  --frame-num 17 --height 480 --width 832 --sampling-steps 20 \
  --linear-scope ffn --residual-targets up,down --residual-blocks all \
  --residual-rank "$RESIDUAL_RANK" --budget-rank "$BUDGET_RANK" \
  --row-block-size 8 --static-scale-margin 1.05 \
  --precision-boundary-steps 2 --warmup-steps 20 --precision-warmup-steps 1 \
  --alternate-method-order \
  >"$OUT/run.log" 2>&1

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/summarize_hybrid_residual.py" \
  --run-dir "$OUT/data" --out-dir "$OUT/analysis" \
  --primary-method hybrid_middle1 \
  >"$OUT/analysis.log" 2>&1

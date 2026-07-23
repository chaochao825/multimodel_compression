#!/usr/bin/env bash
set -euo pipefail

BASE_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723
PROBE_ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
OUT="$PROBE_ROOT/results/activation_defect_subspace_f17_v1"
mkdir -p "$OUT"

finish() {
    rc=$?
    if [[ $rc -eq 0 ]]; then printf 'PASS\n' >"$OUT/DONE"; else printf '%s\n' "$rc" >"$OUT/FAILED"; fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="$BASE_ROOT/src/WorldFoundry-bc062d7-core:$BASE_ROOT/src/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:$BASE_ROOT/src/flash-attention-b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7/hopper${PYTHONPATH:+:$PYTHONPATH}"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/probe_wan_activation_defects.py" \
  --wan-source "$BASE_ROOT/wan_runtime/MonarchRT" \
  --worldfoundry-source "$BASE_ROOT/src/WorldFoundry-bc062d7-core" \
  --checkpoint "$BASE_ROOT/wan_runtime/MonarchRT/wan_models/Wan2.1-T2V-1.3B" \
  --out-dir "$OUT/data" --prompt-file "$BASE_ROOT/scripts/prompts_pilot8.txt" \
  --max-prompts 2 --seeds 20260723,20260724 \
  --steps 14,16,19 --blocks 6,12,24 --branches 0,1 \
  --forecast-scales 0.5,0.75,1.0 --sample-rows 256 \
  --frame-num 17 --sampling-steps 20 --calibration-steps 20 \
  >"$OUT/run.log" 2>&1

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/analyze_activation_defect_spectrum.py" \
  --samples "$OUT/data/activation_defect_samples.pt" \
  --out-dir "$OUT/analysis" --device cuda:0 \
  >"$OUT/analysis.log" 2>&1

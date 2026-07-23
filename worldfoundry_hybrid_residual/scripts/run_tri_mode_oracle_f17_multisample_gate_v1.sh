#!/usr/bin/env bash
set -euo pipefail

GPU_INDEX=${1:-3}
BASE_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723
PROBE_ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
OUT="$PROBE_ROOT/results/tri_mode_oracle_f17_multisample_gate_v1"
mkdir -p "$OUT"

finish() {
    rc=$?
    if [[ $rc -eq 0 ]]; then printf 'PASS\n' >"$OUT/DONE"; else printf '%s\n' "$rc" >"$OUT/FAILED"; fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$BASE_ROOT/src/WorldFoundry-bc062d7-core:$BASE_ROOT/src/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:$BASE_ROOT/src/flash-attention-b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7/hopper${PYTHONPATH:+:$PYTHONPATH}"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/generate_wan_tri_mode_oracle.py" \
  --wan-source "$BASE_ROOT/wan_runtime/MonarchRT" \
  --worldfoundry-source "$BASE_ROOT/src/WorldFoundry-bc062d7-core" \
  --checkpoint "$BASE_ROOT/wan_runtime/MonarchRT/wan_models/Wan2.1-T2V-1.3B" \
  --out-dir "$OUT/data" \
  --schedule-file "$PROBE_ROOT/configs/tri_mode_oracle_f17_multisample_gate_v1.json" \
  --prompt-file "$BASE_ROOT/scripts/prompts_pilot8.txt" --max-prompts 2 \
  --seeds 20260723,20260724 --repeats 1 --frame-num 17 --sampling-steps 20 \
  --calibration-steps 20 --precision-warmup-steps 1 --alternate-schedule-order \
  >"$OUT/run.log" 2>&1

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/summarize_tri_mode_oracle.py" \
  --run-dir "$OUT/data" --out-dir "$OUT/analysis" \
  >"$OUT/analysis.log" 2>&1

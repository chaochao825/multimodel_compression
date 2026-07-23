#!/usr/bin/env bash
set -euo pipefail

BRANCH=${1:?usage: run_tri_mode_oracle_f17_branch_probe_v1.sh BRANCH GPU}
GPU_INDEX=${2:?usage: run_tri_mode_oracle_f17_branch_probe_v1.sh BRANCH GPU}
if [[ "$BRANCH" != 0 && "$BRANCH" != 1 ]]; then
    printf 'BRANCH must be 0 or 1\n' >&2
    exit 2
fi

BASE_ROOT=/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723
PROBE_ROOT=/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723
OUT="$PROBE_ROOT/results/tri_mode_oracle_f17_cache_branch${BRANCH}_probe_v1"
mkdir -p "$OUT"

finish() {
    rc=$?
    if [[ $rc -eq 0 ]]; then printf 'PASS\n' >"$OUT/DONE"; else printf '%s\n' "$rc" >"$OUT/FAILED"; fi
}
trap finish EXIT

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$BASE_ROOT/src/WorldFoundry-bc062d7-core:$BASE_ROOT/src/SageAttention-d1a57a546c3d395b1ffcbeecc66d81db76f3b4b5:$BASE_ROOT/src/flash-attention-b7d29fb3b79f0b78b1c369a52aaa6628dabfb0d7/hopper${PYTHONPATH:+:$PYTHONPATH}"

"$BASE_ROOT/.venv/bin/python" "$PROBE_ROOT/scripts/build_tri_mode_oracle_schedules.py" \
  --output "$OUT/schedules.json" --actions C --skip-global-anchors \
  --probe-steps 12,14,16,19 --probe-blocks 0,6,12,18,24 --branches "$BRANCH"

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

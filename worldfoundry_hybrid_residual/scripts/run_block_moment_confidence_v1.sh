#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}
PYTHON=${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}
GPU_ID=${GPU_ID:-0}
OUTPUT=${OUTPUT:-results/block_moment_marginal_f81_confidence_v1}

cd "$PROJECT_ROOT"
mkdir -p logs results trash

wait_for_idle_gpu() {
    local consecutive=0
    while (( consecutive < 3 )); do
        IFS=, read -r memory utilization < <(
            nvidia-smi --id="$GPU_ID" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ' '
        )
        if (( memory <= 2048 && utilization <= 10 )); then
            consecutive=$((consecutive + 1))
        else
            consecutive=0
        fi
        printf '[idle-gate] gpu=%s memory_mib=%s utilization=%s consecutive=%s/3\n' \
            "$GPU_ID" "$memory" "$utilization" "$consecutive"
        if (( consecutive < 3 )); then sleep 10; fi
    done
}

wait_for_idle_gpu
exec 9>"/tmp/codex-gpu${GPU_ID}.lock"
flock 9
wait_for_idle_gpu

if [[ -e "$OUTPUT" ]]; then
    stamp=$(date +%Y%m%d-%H%M%S)
    staged="trash/${stamp}-block-moment-confidence/${OUTPUT}"
    mkdir -p "$(dirname "$staged")"
    mv "$OUTPUT" "$staged"
fi

"$PYTHON" -m unittest -v \
    scripts/test_block_moment_marginal.py \
    scripts/test_block_moment_confidence.py

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/probe_block_moment_marginal.py \
    --capture-index results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv \
    --head-stats-index results/attention_head_factorial_f81_v1/head_stats/head_stats_index.csv \
    --output-dir "$OUTPUT" \
    --layers 0,14,29 \
    --steps 0,9,19 \
    --branches cond \
    --query-tile-size 64 \
    --query-tiles 1 \
    --block-size 64 \
    --tail-group-sizes 16,8,4 \
    --densities 0.125,0.25,0.375 \
    --methods centroid \
    --routers moment,oracle_mass \
    --max-work-ratio 0.50 \
    --device cuda:0

"$PYTHON" scripts/summarize_block_moment_confidence.py \
    --heads-csv "$OUTPUT/block_moment_marginal_heads.csv" \
    --output-dir "$OUTPUT" \
    --error-target 0.02 \
    --aggregate-target 0.01 \
    --minimum-validation-coverage 0.05 \
    --attention-speed-target 1.5

printf '[block-moment-confidence] completed gpu=%s\n' "$GPU_ID"

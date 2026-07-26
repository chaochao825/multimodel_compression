#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}
PYTHON=${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}
GPU_ID=${GPU_ID:-1}
MAX_MEMORY_MIB=${MAX_MEMORY_MIB:-2048}
MAX_UTILIZATION=${MAX_UTILIZATION:-10}
IDLE_CHECKS=${IDLE_CHECKS:-3}
IDLE_INTERVAL_SECONDS=${IDLE_INTERVAL_SECONDS:-10}

cd "$PROJECT_ROOT"
mkdir -p logs results trash

CAPTURE_INDEX=results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv
HEAD_STATS_INDEX=results/attention_head_factorial_f81_v1/head_stats/head_stats_index.csv
BLOCK_OUTPUT=results/block_moment_marginal_f81_pilot_v1
DISPLACEMENT_OUTPUT=results/local_displacement_mixture_f81_pilot_v1
PLOT_OUTPUT=results/content_structure_gates_f81_pilot_v1

stage_existing() {
    local target=$1
    if [[ -e "$target" ]]; then
        local stamp
        stamp=$(date +%Y%m%d-%H%M%S)
        local staged="trash/${stamp}-content-structure-pilot/${target}"
        mkdir -p "$(dirname "$staged")"
        mv "$target" "$staged"
    fi
}

wait_for_idle_gpu() {
    local consecutive=0
    while (( consecutive < IDLE_CHECKS )); do
        IFS=, read -r memory utilization < <(
            nvidia-smi --id="$GPU_ID" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits | tr -d ' '
        )
        if (( memory <= MAX_MEMORY_MIB && utilization <= MAX_UTILIZATION )); then
            consecutive=$((consecutive + 1))
        else
            consecutive=0
        fi
        printf '[idle-gate] gpu=%s memory_mib=%s utilization=%s consecutive=%s/%s\n' \
            "$GPU_ID" "$memory" "$utilization" "$consecutive" "$IDLE_CHECKS"
        if (( consecutive < IDLE_CHECKS )); then
            sleep "$IDLE_INTERVAL_SECONDS"
        fi
    done
}

wait_for_idle_gpu
exec 9>"/tmp/codex-gpu${GPU_ID}.lock"
flock 9
wait_for_idle_gpu

stage_existing "$BLOCK_OUTPUT"
stage_existing "$DISPLACEMENT_OUTPUT"
stage_existing "$PLOT_OUTPUT"

"$PYTHON" -m unittest -v \
    scripts/test_block_moment_marginal.py \
    scripts/test_local_displacement_mixture.py

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/probe_block_moment_marginal.py \
    --capture-index "$CAPTURE_INDEX" \
    --head-stats-index "$HEAD_STATS_INDEX" \
    --output-dir "$BLOCK_OUTPUT" \
    --layers 0,14,29 \
    --steps 0,9,19 \
    --branches cond \
    --query-tile-size 64 \
    --query-tiles 1 \
    --block-size 64 \
    --tail-group-sizes 64,16 \
    --densities 0.0625,0.125 \
    --methods centroid,diag_gaussian \
    --routers moment,oracle_mass \
    --device cuda:0

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/probe_local_displacement_mixture.py \
    --capture-index "$CAPTURE_INDEX" \
    --head-stats-index "$HEAD_STATS_INDEX" \
    --output-dir "$DISPLACEMENT_OUTPUT" \
    --layers 0,14,29 \
    --steps 0,9,19 \
    --branches cond \
    --query-samples 32 \
    --radius 2x4x4 \
    --ranks 0,2,4,8 \
    --ridge-lambda 1.0 \
    --device cuda:0

if ! "$PYTHON" -c 'import matplotlib' >/dev/null 2>&1; then
    "$(dirname "$PYTHON")/pip" install matplotlib
fi
"$PYTHON" scripts/plot_content_structure_probes.py \
    --block-dir "$BLOCK_OUTPUT" \
    --displacement-dir "$DISPLACEMENT_OUTPUT" \
    --output-dir "$PLOT_OUTPUT" \
    --attention-share 0.5388

printf '[content-structure] completed gpu=%s\n' "$GPU_ID"

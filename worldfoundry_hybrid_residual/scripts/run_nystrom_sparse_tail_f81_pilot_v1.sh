#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}
PYTHON=${PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}
MAX_MEMORY_MIB=${MAX_MEMORY_MIB:-2048}
MAX_UTILIZATION=${MAX_UTILIZATION:-5}
IDLE_CONFIRM_SECONDS=${IDLE_CONFIRM_SECONDS:-10}
POLL_SECONDS=${POLL_SECONDS:-30}
MAX_WAIT_SECONDS=${MAX_WAIT_SECONDS:-21600}
RESOURCE_MODE=${RESOURCE_MODE:-dedicated_idle_numerical_probe}
GPU_ALLOWLIST=${GPU_ALLOWLIST:-2,3}

cd "$PROJECT_ROOT"
mkdir -p logs results trash

CAPTURE_INDEX=results/attention_head_factorial_f81_v1/qkv_replays/capture_index.csv
HEAD_STATS_INDEX=results/attention_head_factorial_f81_v1/head_stats/head_stats_index.csv
SPLIT_CONFIG=configs/nystrom_split_protocols_f81_v1.json
PROBE_OUTPUT=results/nystrom_sparse_tail_f81_pilot_v1
SELECTION_OUTPUT=results/nystrom_sparse_tail_f81_pilot_selection_v1
PLOT_OUTPUT=results/nystrom_sparse_tail_f81_pilot_plots_v1
REPORT_OUTPUT=results/nystrom_sparse_tail_f81_pilot_report_v1
STATUS_PATH=logs/nystrom_sparse_tail_f81_pilot_v1.status

stage_existing() {
    local target=$1
    if [[ -e "$target" ]]; then
        local stamp staged
        stamp=$(date +%Y%m%d-%H%M%S)
        staged="trash/${stamp}-nystrom-sparse-tail-pilot/${target}"
        mkdir -p "$(dirname "$staged")"
        mv "$target" "$staged"
    fi
}

write_status() {
    local state=$1
    local detail=${2:-}
    local temporary="${STATUS_PATH}.tmp.$$"
    printf 'state=%s\ntime_utc=%s\ndetail=%s\n' \
        "$state" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$detail" >"$temporary"
    mv "$temporary" "$STATUS_PATH"
}

on_exit() {
    local code=$?
    if (( code != 0 )); then
        local current_state=""
        if [[ -f "$STATUS_PATH" ]]; then
            current_state=$(sed -n 's/^state=//p' "$STATUS_PATH" | head -n 1)
        fi
        case "$current_state" in
            ALREADY_QUEUED|INVALID_INPUT|NO_IDLE_GPU)
                ;;
            *)
                write_status FAILED "exit_code=${code}"
                ;;
        esac
    fi
}
trap on_exit EXIT

exec 8>/tmp/codex-nystrom-sparse-tail-pilot.lock
if ! flock -n 8; then
    write_status ALREADY_QUEUED "another pilot runner owns the queue lock"
    exit 76
fi

for required in "$PYTHON" "$CAPTURE_INDEX" "$HEAD_STATS_INDEX" "$SPLIT_CONFIG"; do
    if [[ ! -e "$required" ]]; then
        write_status INVALID_INPUT "missing=${required}"
        exit 2
    fi
done

write_status WAITING_FOR_H200 "max_wait_seconds=${MAX_WAIT_SECONDS}"
started=$(date +%s)
GPU_ID=""
while [[ -z "$GPU_ID" ]]; do
    now=$(date +%s)
    if (( now - started >= MAX_WAIT_SECONDS )); then
        write_status NO_IDLE_GPU "waited_seconds=$((now - started))"
        exit 75
    fi

    while IFS=, read -r index name memory utilization; do
        index=${index//[[:space:]]/}
        name=${name//[[:space:]]/}
        memory=${memory//[[:space:]]/}
        utilization=${utilization//[[:space:]]/}
        if [[ "$name" != *H200* ]]; then
            continue
        fi
        case ",${GPU_ALLOWLIST}," in
            *",${index},"*)
                ;;
            *)
                continue
                ;;
        esac
        if (( memory > MAX_MEMORY_MIB || utilization > MAX_UTILIZATION )); then
            continue
        fi

        exec 9>"/tmp/codex-gpu${index}.lock"
        if ! flock -n 9; then
            exec 9>&-
            continue
        fi
        sleep "$IDLE_CONFIRM_SECONDS"
        IFS=, read -r confirmed_memory confirmed_utilization < <(
            nvidia-smi --id="$index" \
                --query-gpu=memory.used,utilization.gpu \
                --format=csv,noheader,nounits
        )
        confirmed_memory=${confirmed_memory//[[:space:]]/}
        confirmed_utilization=${confirmed_utilization//[[:space:]]/}
        if (( confirmed_memory <= MAX_MEMORY_MIB \
              && confirmed_utilization <= MAX_UTILIZATION )); then
            GPU_ID=$index
            break
        fi
        flock -u 9
        exec 9>&-
    done < <(
        nvidia-smi \
            --query-gpu=index,name,memory.used,utilization.gpu \
            --format=csv,noheader,nounits
    )

    if [[ -z "$GPU_ID" ]]; then
        printf '[idle-gate] no idle H200; sleeping %ss\n' "$POLL_SECONDS"
        sleep "$POLL_SECONDS"
    fi
done

write_status RUNNING "gpu=${GPU_ID}"
stage_existing "$PROBE_OUTPUT"
stage_existing "$SELECTION_OUTPUT"
stage_existing "$PLOT_OUTPUT"
stage_existing "$REPORT_OUTPUT"

PYTHONPATH=scripts "$PYTHON" -m unittest -v \
    test_experiment_artifacts \
    test_nystrom_sparse_tail \
    test_select_nystrom_sparse_tail

CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON" scripts/probe_nystrom_sparse_tail.py \
    --capture-index "$CAPTURE_INDEX" \
    --head-stats-index "$HEAD_STATS_INDEX" \
    --split-config "$SPLIT_CONFIG" \
    --output-dir "$PROBE_OUTPUT" \
    --layers 14 \
    --steps 9 \
    --branches cond \
    --query-tile-size 64 \
    --query-tiles 1 \
    --block-size 64 \
    --landmarks 32,64,128 \
    --landmark-modes segment \
    --pinv-rtols 1e-4 \
    --densities 0.125,0.25 \
    --execution-resource-note "$RESOURCE_MODE" \
    --device cuda:0

"$PYTHON" scripts/select_nystrom_sparse_tail.py \
    --probe-dir "$PROBE_OUTPUT" \
    --split-config "$SPLIT_CONFIG" \
    --output-dir "$SELECTION_OUTPUT" \
    --aggregate-target 0.01 \
    --record-target 0.02 \
    --speed-target 1.5 \
    --max-work-ratio 0.5

if ! "$PYTHON" -c 'import matplotlib' >/dev/null 2>&1; then
    "$(dirname "$PYTHON")/pip" install matplotlib
fi
"$PYTHON" scripts/plot_nystrom_sparse_tail.py \
    --selection-dir "$SELECTION_OUTPUT" \
    --output-dir "$PLOT_OUTPUT"

"$PYTHON" scripts/report_nystrom_sparse_tail.py \
    --selection-dir "$SELECTION_OUTPUT" \
    --output-dir "$REPORT_OUTPUT" \
    --experiment-label 'F81 Nystrom/Landmark Sparse-Tail Pilot' \
    --run-kind pilot

numerical_gate=$(
    "$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["numerical_gate"])' \
        "$SELECTION_OUTPUT/SUCCESS.json"
)
scientific_gate=$(
    "$PYTHON" -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["scientific_gate"])' \
        "$SELECTION_OUTPUT/SUCCESS.json"
)
write_status SUCCESS \
    "gpu=${GPU_ID};numerical_gate=${numerical_gate};scientific_gate=${scientific_gate}"
trap - EXIT
printf '[nystrom-pilot] completed gpu=%s numerical_gate=%s scientific_gate=%s\n' \
    "$GPU_ID" "$numerical_gate" "$scientific_gate"

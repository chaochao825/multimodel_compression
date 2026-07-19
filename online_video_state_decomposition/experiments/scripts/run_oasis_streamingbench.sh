#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 3 ]]; then
  printf 'usage: %s RUN_NAME [smoke|formal] [GPU_INDEX]\n' "$0" >&2
  exit 2
fi

unset PREFIX

RUN_NAME="$1"
SCOPE="${2:-smoke}"
GPU_INDEX="${3:-3}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
PYTHON_BIN="${OASIS_PYTHON:-$RUNTIME_ROOT/.conda/oasis-py312/bin/python}"
DATA_ROOT="$RUNTIME_ROOT/third_party/OASIS-StreamingBench-RT-1-50-v1"
OUTPUT_DIR="$RUNTIME_ROOT/remote_results/oasis_streamingbench/$RUN_NAME"

case "$SCOPE" in
  smoke)
    METADATA="$DATA_ROOT/metadata/rtu_1.json"
    DATASET_MANIFEST="$DATA_ROOT/manifests/rtu_1_mapping.json"
    SCOPE_ARGS=(--allow-smoke-subset)
    ;;
  formal)
    METADATA="$DATA_ROOT/metadata/rtu_1_50.json"
    DATASET_MANIFEST="$DATA_ROOT/manifests/rtu_1_50_mapping.json"
    SCOPE_ARGS=()
    ;;
  *)
    printf 'unknown scope: %s\n' "$SCOPE" >&2
    exit 2
    ;;
esac

exec "$PYTHON_BIN" "$PROJECT_ROOT/experiments/probes/run_oasis_streamingbench.py" \
  --source-root "$RUNTIME_ROOT/external_baselines/OASIS" \
  --metadata "$METADATA" \
  --dataset-root "$DATA_ROOT/dataset" \
  --dataset-manifest "$DATASET_MANIFEST" \
  --mllm-path "$RUNTIME_ROOT/third_party/Qwen3-VL-8B-Instruct-modelscope" \
  --embedding-path "$RUNTIME_ROOT/third_party/Qwen3-Embedding-0.6B-modelscope" \
  --output-dir "$OUTPUT_DIR" \
  --python-bin "$PYTHON_BIN" \
  --gpu-index "$GPU_INDEX" \
  --max-idle-memory-mib "${MAX_IDLE_MEMORY_MIB:-4096}" \
  --max-idle-utilization "${MAX_IDLE_UTILIZATION:-20}" \
  "${SCOPE_ARGS[@]}"

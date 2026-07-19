#!/usr/bin/env bash
set -euo pipefail

unset PREFIX

PROJECT_ROOT="${PROJECT_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
RUNTIME_ROOT="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
PYTHON_BIN="${OASIS_PYTHON:-$RUNTIME_ROOT/.conda/oasis-py312/bin/python}"
SOURCE_METADATA="$RUNTIME_ROOT/external_baselines/OASIS/metadata/StreamingBench/Real_Time_Visual_Understanding.json"
SOURCE_DATA="$RUNTIME_ROOT/third_party/StreamingBench-RT-1-50/prepared"
TARGET="$RUNTIME_ROOT/third_party/OASIS-StreamingBench-RT-1-50-v1"

mkdir -p "$TARGET/metadata" "$TARGET/manifests" "$TARGET/dataset"

prepare_scope() {
  local stem="$1"
  local first_video="$2"
  local last_video="$3"
  "$PYTHON_BIN" "$PROJECT_ROOT/experiments/probes/prepare_oasis_streamingbench_subset.py" \
    --oasis-unified-json "$SOURCE_METADATA" \
    --streamingbench-csv "$SOURCE_DATA/Real_Time_Visual_Understanding_1-50.csv" \
    --prepared-video-root "$SOURCE_DATA/realtime_video" \
    --upstream-manifest "$SOURCE_DATA/manifest.json" \
    --subset-json "$TARGET/metadata/$stem.json" \
    --mapping-manifest "$TARGET/manifests/${stem}_mapping.json" \
    --first-video "$first_video" \
    --last-video "$last_video"
}

prepare_scope rtu_1_50 1 50 > "$TARGET/manifests/prepare_formal_stdout.json"
prepare_scope rtu_1 1 1 > "$TARGET/manifests/prepare_smoke_stdout.json"

"$PYTHON_BIN" "$PROJECT_ROOT/experiments/probes/materialize_oasis_dataset_links.py" \
  --mapping-manifest "$TARGET/manifests/rtu_1_50_mapping.json" \
  --dataset-root "$TARGET/dataset" \
  --output-manifest "$TARGET/manifests/materialized_links.json" \
  > "$TARGET/manifests/materialize_stdout.json"

printf 'prepared OASIS dataset at %s\n' "$TARGET"

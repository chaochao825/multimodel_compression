#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.conda/oasis-py312/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
DATASET_ROOT="${DATASET_ROOT:-/home/wangmeiqi/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-24}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/reader_risk}"

mkdir -p "${OUT_DIR}"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" experiments/probes/probe_vsi_onevision_reader_risk_stage_a.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --jsonl-path "${DATASET_ROOT}/test.jsonl" \
  --pruned-ids-path "${DATASET_ROOT}/pruned_ids.txt" \
  --video-root "${RESULT_ROOT}/videos" \
  --feature-dir "${RESULT_ROOT}/features" \
  --model-dir "${MODEL_DIR}" \
  --source-codec "${PROJECT_ROOT}/remote_results/onevision_rank_support_allocation_20_20260825_v1/codec/onevision_feature_pca_rank456.pt" \
  --spectral-artifact "${PROJECT_ROOT}/remote_results/onevision_reader_quotient_equal_budget_stage_a_20260830_v1/spectrum/spectral_artifacts.pt" \
  --out-dir "${OUT_DIR}" \
  --sample-offset "${SAMPLE_OFFSET}" \
  --sample-count "${SAMPLE_COUNT}" \
  --frame-budget 8 \
  --margin-floor 0.05 \
  --device "${DEVICE}"

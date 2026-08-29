#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.conda/oasis-py312/bin/python}"
DEVICE="${DEVICE:-cuda:0}"
RESULT_ROOT="${RESULT_ROOT:-${PROJECT_ROOT}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
DATASET_ROOT="${DATASET_ROOT:-/home/wangmeiqi/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_ROOT}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
OUT_DIR="${OUT_DIR:-${RESULT_ROOT}/cmrq_stage_b}"
RISK_DIR="${RISK_DIR:-${RESULT_ROOT}/reader_risk}"
RISK_FIT_RANGES="${RISK_FIT_RANGES:-}"
RISK_FIT_OFFSET="${RISK_FIT_OFFSET:-0}"
RISK_FIT_COUNT="${RISK_FIT_COUNT:-24}"
EVALUATION_OFFSET="${EVALUATION_OFFSET:-24}"
ATOM_COUNTS="${ATOM_COUNTS:-16,32,64,96}"
NULL_ATOM_COUNTS="${NULL_ATOM_COUNTS:-16,32,64,96}"
MIX_ATOM_COUNT="${MIX_ATOM_COUNT:-32}"
MIX_WEIGHTS="${MIX_WEIGHTS:-0.03,0.1,0.3,1,3,10}"
METHOD_NAMES="${METHOD_NAMES:-}"

mkdir -p "${OUT_DIR}" "${RESULT_ROOT}/logs"
cd "${PROJECT_ROOT}"

"${PYTHON_BIN}" experiments/probes/probe_vsi_onevision_cmrq_stage_b.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --jsonl-path "${DATASET_ROOT}/test.jsonl" \
  --pruned-ids-path "${DATASET_ROOT}/pruned_ids.txt" \
  --video-root "${RESULT_ROOT}/videos" \
  --feature-dir "${RESULT_ROOT}/features" \
  --model-dir "${MODEL_DIR}" \
  --spectral-artifact "${PROJECT_ROOT}/remote_results/onevision_reader_quotient_equal_budget_stage_a_20260830_v1/spectrum/spectral_artifacts.pt" \
  --reader-risk-artifact "${RISK_DIR}/reader_risk_artifact.pt" \
  --reader-risk-summary "${RISK_DIR}/summary.json" \
  --out-dir "${OUT_DIR}" \
  --risk-fit-ranges "${RISK_FIT_RANGES}" \
  --risk-fit-offset "${RISK_FIT_OFFSET}" \
  --risk-fit-count "${RISK_FIT_COUNT}" \
  --evaluation-offset "${EVALUATION_OFFSET}" \
  --evaluation-count 24 \
  --rank 456 \
  --atom-counts "${ATOM_COUNTS}" \
  --null-atom-counts "${NULL_ATOM_COUNTS}" \
  --mix-atom-count "${MIX_ATOM_COUNT}" \
  --mix-weights "${MIX_WEIGHTS}" \
  --method-names "${METHOD_NAMES}" \
  --seed 20260830 \
  --device "${DEVICE}"

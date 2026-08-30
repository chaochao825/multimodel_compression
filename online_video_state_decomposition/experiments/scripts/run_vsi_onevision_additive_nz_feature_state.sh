#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${project_root}/.conda/oasis-py312/bin/python}"
device="${DEVICE:-cuda:1}"
dataset_root="${DATASET_ROOT:-/home/wangmeiqi/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d}"
result_root="${RESULT_ROOT:-${project_root}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
model_dir="${MODEL_DIR:-${project_root}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
capture_dir="${CAPTURE_DIR:-${result_root}/additive_nz_capture_calibration_dev96_v1}"
out_dir="${OUT_DIR:-${result_root}/additive_nz_feature_state_dev_v1}"

cd "${project_root}"
"${python_bin}" experiments/probes/capture_vsi_onevision_additive_nz_dataset.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --jsonl-path "${dataset_root}/test.jsonl" \
  --pruned-ids-path "${dataset_root}/pruned_ids.txt" \
  --video-root "${result_root}/videos" \
  --feature-dir "${result_root}/features" \
  --model-dir "${model_dir}" \
  --out-dir "${capture_dir}" \
  --sample-offset 0 \
  --sample-count "${CAPTURE_COUNT:-96}" \
  --frame-budget 8 \
  --device "${device}" \
  ${CAPTURE_SMOKE:+--smoke}

if [[ "${CAPTURE_SMOKE:-}" == "" ]]; then
  "${python_bin}" experiments/probes/train_vsi_onevision_additive_nz_feature_state.py \
    --capture-dir "${capture_dir}" \
    --out-dir "${out_dir}" \
    --feature-width 32 \
    --steps 1000 \
    --batch-size 2 \
    --learning-rate 0.001 \
    --evaluation-interval 100 \
    --seed 20260830 \
    --device "${device}"
fi

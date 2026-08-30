#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${project_root}/.conda/oasis-py312/bin/python}"
device="${DEVICE:-cuda:1}"
dataset_root="${DATASET_ROOT:-/home/wangmeiqi/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d}"
result_root="${RESULT_ROOT:-${project_root}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
model_dir="${MODEL_DIR:-${project_root}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
selection_feature_dir="${result_root}/features_selection"
merged_risk_dir="${result_root}/reader_risk_merged72"
selection_out_dir="${result_root}/cmrq_selection_frozen_v1"

mkdir -p "${result_root}/logs"
cd "${project_root}"

"${python_bin}" experiments/probes/materialize_vsi_role_videos.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --archive-root "${dataset_root}" \
  --out-dir "${result_root}/videos" \
  --role selection

"${python_bin}" experiments/probes/capture_vsi_onevision_stage_a_features.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --video-root "${result_root}/videos" \
  --model-dir "${model_dir}" \
  --out-dir "${selection_feature_dir}" \
  --role selection \
  --feature-pool-frames 16 \
  --device "${device}"

"${python_bin}" experiments/probes/merge_onevision_reader_risk_artifacts.py \
  --input-dir "${result_root}/reader_risk" \
  --input-dir "${result_root}/reader_risk_fold1" \
  --input-dir "${result_root}/reader_risk_fold2" \
  --out-dir "${merged_risk_dir}" \
  --device "${device}"

"${python_bin}" experiments/probes/vsi_onevision_progressive_cmrq_selection.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --jsonl-path "${dataset_root}/test.jsonl" \
  --pruned-ids-path "${dataset_root}/pruned_ids.txt" \
  --video-root "${result_root}/videos" \
  --calibration-feature-dir "${result_root}/features" \
  --evaluation-feature-dir "${selection_feature_dir}" \
  --model-dir "${model_dir}" \
  --spectral-artifact "${project_root}/remote_results/onevision_reader_quotient_equal_budget_stage_a_20260830_v1/spectrum/spectral_artifacts.pt" \
  --reader-risk-artifact "${merged_risk_dir}/reader_risk_artifact.pt" \
  --reader-risk-summary "${merged_risk_dir}/summary.json" \
  --out-dir "${selection_out_dir}" \
  --frame-budget 8 \
  --rank 456 \
  --margin-threshold 0 \
  --seed 20260830 \
  --device "${device}"

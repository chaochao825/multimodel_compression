#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${project_root}/.conda/oasis-py312/bin/python}"
device="${DEVICE:-cuda:2}"
result_root="${RESULT_ROOT:-${project_root}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
capture_dir="${CAPTURE_DIR:-${result_root}/additive_nz_capture_calibration_dev96_v2_exact_pages}"
checkpoint_dir="${CHECKPOINT_DIR:-${result_root}/additive_nz_feature_state_dev_v1}"
out_dir="${OUT_DIR:-${result_root}/joint_control_variate_support_state_capacity_dev_v1}"

cd "${project_root}"
"${python_bin}" experiments/probes/train_vsi_onevision_joint_control_variate_support_state.py \
  --capture-dir "${capture_dir}" \
  --checkpoint-dir "${checkpoint_dir}" \
  --out-dir "${out_dir}" \
  --page-size 4 \
  --exact-fraction 0.25 \
  --feature-width 32 \
  --steps 1000 \
  --batch-size 2 \
  --learning-rate 0.0003 \
  --log-interval 100 \
  --greedy-round-size 14 \
  --bootstrap-repetitions 10000 \
  --seed 20260901 \
  --device "${device}"

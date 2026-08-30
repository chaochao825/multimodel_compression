#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${project_root}/.conda/oasis-py312/bin/python}"
device="${DEVICE:-cuda:1}"
result_root="${RESULT_ROOT:-${project_root}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
capture_dir="${CAPTURE_DIR:-${result_root}/additive_nz_capture_calibration_dev96_v1}"
out_dir="${OUT_DIR:-${result_root}/exact_boundary_additive_tail_dev_v1}"

cd "${project_root}"
"${python_bin}" experiments/probes/train_vsi_onevision_exact_boundary_additive_tail.py \
  --capture-dir "${capture_dir}" \
  --out-dir "${out_dir}" \
  --exact-fraction 0.25 \
  --feature-width 32 \
  --steps 1000 \
  --batch-size 2 \
  --learning-rate 0.001 \
  --evaluation-interval 100 \
  --seed 20260830 \
  --device "${device}"

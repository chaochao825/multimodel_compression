#!/usr/bin/env bash
set -euo pipefail

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${project_root}/.conda/oasis-py312/bin/python}"
device="${DEVICE:-cuda:1}"
dataset_root="${DATASET_ROOT:-/home/wangmeiqi/.cache/huggingface/hub/datasets--nyu-visionx--VSI-Bench/snapshots/d7cb1a3960b79dd3e20d4990b83005e96e1bcd9d}"
result_root="${RESULT_ROOT:-${project_root}/remote_results/vsi_onevision_reader_quotient_stage_a_20260830_v1}"
model_dir="${MODEL_DIR:-${project_root}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
out_dir="${OUT_DIR:-${result_root}/query_fixed_local_secant_exposed_v1}"

cd "${project_root}"
"${python_bin}" experiments/probes/probe_vsi_onevision_query_fixed_local_secant_measure.py \
  --split-path configs/vsi/onevision_reader_quotient_stage_a_20260830.json \
  --jsonl-path "${dataset_root}/test.jsonl" \
  --pruned-ids-path "${dataset_root}/pruned_ids.txt" \
  --video-root "${result_root}/videos" \
  --feature-dir "${result_root}/features" \
  --model-dir "${model_dir}" \
  --prototype-summary "${result_root}/query_fixed_prototype_mixture_exposed_v1/summary.json" \
  --ppe-summary "${result_root}/true_2x2_ppe_exposed_v1/summary.json" \
  --out-dir "${out_dir}" \
  --sample-offset 72 \
  --sample-count "${SAMPLE_COUNT:-24}" \
  --frame-budget 8 \
  --device "${device}" \
  ${SMOKE:+--smoke}

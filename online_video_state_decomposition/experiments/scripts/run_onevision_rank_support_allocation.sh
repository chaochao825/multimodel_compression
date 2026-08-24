#!/usr/bin/env bash
set -euo pipefail

mode="$1"
gpu="${2:-0}"
shard="${3:-0}"
shard_count="${4:-4}"

project_root=/home/spco/online_video_state_decomposition
dataset_root=/home/wangmeiqi/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/a776e554280b99b70f00cc3eacd69a65e0727efc
model_dir=/home/spco/online_video_state_decomposition/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope
out="${project_root}/remote_results/onevision_rank_support_allocation_20_20260825_v1"
python="${project_root}/.conda/oasis-py312/bin/python"

mkdir -p "${out}/logs"
cd "${project_root}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false

if [[ "${mode}" == fit ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" "${python}" -u \
    experiments/probes/fit_mvbench_onevision_feature_pca.py \
    --dataset-root "${dataset_root}" \
    --model-dir "${model_dir}" \
    --out-dir "${out}/codec" \
    --rank 456 \
    --residual-tokens 0 \
    --device cuda:0 \
    >"${out}/logs/fit.log" 2>&1
  exit 0
fi

if [[ "${mode}" != evaluate ]]; then
  printf 'unknown mode: %s\n' "${mode}" >&2
  exit 2
fi

printf '125\n' >"${out}/logs/shard_${shard}.exit"
set +e
CUDA_VISIBLE_DEVICES="${gpu}" "${python}" -u \
  experiments/probes/mvbench_onevision_rank_support_allocation.py \
  --dataset-root "${dataset_root}" \
  --model-dir "${model_dir}" \
  --codec-path "${out}/codec/onevision_feature_pca_rank456.pt" \
  --out-dir "${out}/evaluation" \
  --device cuda:0 \
  --shard-index "${shard}" \
  --shard-count "${shard_count}" \
  --fail-fast \
  >"${out}/logs/shard_${shard}.log" 2>&1
status="$?"
set -e
printf '%s\n' "${status}" >"${out}/logs/shard_${shard}.exit"
exit "${status}"

#!/usr/bin/env bash
set -euo pipefail

gpu="${1:-0}"
shard="${2:-0}"
shard_count="${3:-1}"

project_root=/home/spco/online_video_state_decomposition
dataset_snapshot=/home/wangmeiqi/.cache/huggingface/hub/datasets--lmms-lab--Video-MME/snapshots/ead1408f75b618502df9a1d8e0950166bf0a2a0b
parquet_path="${dataset_snapshot}/videomme/test-00000-of-00001.parquet"
video_root=/home/wangmeiqi/.cache/huggingface/videomme/data
model_dir="${project_root}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope"
codec_path="${project_root}/remote_results/onevision_rank_support_allocation_20_20260825_v1/codec/onevision_feature_pca_rank456.pt"
split_path="${project_root}/configs/videomme/onevision_pca_cross_domain_600_20260829.json"
out="${project_root}/remote_results/videomme_onevision_pca_replication_600_20260829_v1"
python="${project_root}/.conda/oasis-py312/bin/python"

mkdir -p "${out}/logs" "${project_root}/.gpu_locks"
cd "${project_root}"
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false

exec 9>"${project_root}/.gpu_locks/gpu_${gpu}.lock"
if ! flock -n 9; then
  printf 'GPU %s is locked by another project job\n' "${gpu}" >&2
  exit 75
fi

printf '125\n' >"${out}/logs/shard_${shard}.exit"
set +e
CUDA_VISIBLE_DEVICES="${gpu}" "${python}" -u \
  experiments/probes/videomme_onevision_pca_replication.py \
  --parquet-path "${parquet_path}" \
  --video-root "${video_root}" \
  --split-path "${split_path}" \
  --model-dir "${model_dir}" \
  --codec-path "${codec_path}" \
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

#!/usr/bin/env bash
set -u

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
  printf 'usage: %s GPU SHARD_INDEX SHARD_COUNT OUT_DIR SPLIT_MANIFEST [SPLITS]\n' "$0"
  exit 2
fi

gpu="$1"
shard_index="$2"
shard_count="$3"
out_dir="$4"
split_manifest="$5"
splits="${6:-calibration,evaluation}"
project_root="/home/spco/online_video_state_decomposition"
dataset_root="/home/wangmeiqi/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/snapshots/a776e554280b99b70f00cc3eacd69a65e0727efc"
model_dir="${project_root}/third_party/clip-vit-large-patch14-336-full"
log_dir="${out_dir}/logs"
log_path="${log_dir}/shard_${shard_index}.log"
exit_path="${log_dir}/shard_${shard_index}.exit"
lock_path="/tmp/online_video_state_gpu_${gpu}.lock"

mkdir -p "${log_dir}"
gpu_state="$(
  nvidia-smi \
    --id="${gpu}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits |
    head -n 1
)"
IFS=',' read -r memory_used utilization <<<"${gpu_state}"
memory_used="${memory_used//[[:space:]]/}"
utilization="${utilization//[[:space:]]/}"
if [ "${memory_used}" -gt 12000 ] || [ "${utilization}" -gt 10 ]; then
  printf 'gpu %s is busy: memory_used=%s MiB utilization=%s%%\n' \
    "${gpu}" "${memory_used}" "${utilization}" >"${log_path}"
  printf '75\n' >"${exit_path}"
  exit 75
fi

unset PREFIX
source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate Qwen3
export TRANSFORMERS_NO_TF=1
export USE_TF=0
export USE_FLAX=0
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="${gpu}"

set +e
flock -n "${lock_path}" python -u \
  "${project_root}/experiments/probes/mvbench_query_cache.py" \
  --dataset-root "${dataset_root}" \
  --model-dir "${model_dir}" \
  --split-manifest "${split_manifest}" \
  --out-dir "${out_dir}" \
  --splits "${splits}" \
  --frames 32 \
  --image-batch-size 16 \
  --device cuda:0 \
  --shard-index "${shard_index}" \
  --shard-count "${shard_count}" \
  >"${log_path}" 2>&1
status="$?"
printf '%s\n' "${status}" >"${exit_path}"
exit "${status}"

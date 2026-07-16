#!/usr/bin/env bash
set -u

if [ "$#" -lt 3 ] || [ "$#" -gt 4 ]; then
  printf 'usage: %s GPU EXTRACTION_DIR RANK [SHARD_COUNT]\n' "$0"
  exit 2
fi

gpu="$1"
extraction_dir="$2"
rank="$3"
shard_count="${4:-3}"
project_root="/home/spco/online_video_state_decomposition"
log_dir="${extraction_dir}/logs"
log_path="${log_dir}/fit_rank${rank}.log"
exit_path="${log_dir}/fit_rank${rank}.exit"
lock_path="/tmp/online_video_state_gpu_${gpu}.lock"

mkdir -p "${log_dir}"
for ((shard = 0; shard < shard_count; shard++)); do
  shard_exit="${log_dir}/extract_shard_${shard}.exit"
  if [ ! -f "${shard_exit}" ] || [ "$(cat "${shard_exit}")" != "0" ]; then
    printf 'extraction shard %s is incomplete or failed\n' "${shard}" \
      >"${log_path}"
    printf '76\n' >"${exit_path}"
    exit 76
  fi
done

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
export CUDA_VISIBLE_DEVICES="${gpu}"

set +e
flock -n "${lock_path}" python -u \
  "${project_root}/experiments/probes/fit_llava_feature_pca.py" \
  --token-dir "${extraction_dir}/tokens" \
  --out-dir "${extraction_dir}/codec_rank${rank}" \
  --rank "${rank}" \
  --expected-files 100 \
  --seed 20260718 \
  --niter 4 \
  >"${log_path}" 2>&1
status="$?"
printf '%s\n' "${status}" >"${exit_path}"
exit "${status}"

#!/usr/bin/env bash
set -u

if [ "$#" -lt 5 ] || [ "$#" -gt 8 ]; then
  printf 'usage: %s GPU EXTRACTION_DIR SELECTION_MANIFEST OUT_PREFIX SAMPLES_PER_TASK [RANKS] [POLICIES] [RESIDUAL_TOKENS]\n' "$0"
  exit 2
fi

gpu="$1"
extraction_dir="$2"
selection_manifest="$3"
out_prefix="$4"
samples_per_task="$5"
ranks="${6:-64,128,256}"
policies="${7:-exact_recent,learned_recent_query_topk}"
residual_tokens="${8:-0,1,2,4}"
project_root="/home/spco/online_video_state_decomposition"
sweep_log="${out_prefix}_sweep.log"
sweep_exit="${out_prefix}_sweep.exit"

mkdir -p "$(dirname "${out_prefix}")"
: >"${sweep_log}"

IFS=',' read -ra rank_values <<<"${ranks}"
status=0
for rank in "${rank_values[@]}"; do
  printf 'rank=%s stage=fit\n' "${rank}" | tee -a "${sweep_log}"
  bash "${project_root}/experiments/scripts/run_llava_feature_pca_fit.sh" \
    "${gpu}" "${extraction_dir}" "${rank}" 1
  status="$?"
  if [ "${status}" -ne 0 ]; then
    printf 'rank=%s stage=fit status=%s\n' \
      "${rank}" "${status}" | tee -a "${sweep_log}"
    break
  fi

  out_dir="${out_prefix}_rank${rank}"
  codec_path="$(
    realpath \
      "${extraction_dir}/codec_rank${rank}/llava_feature_pca_rank${rank}.pt"
  )"
  printf 'rank=%s stage=evaluate out=%s\n' \
    "${rank}" "${out_dir}" | tee -a "${sweep_log}"
  bash \
    "${project_root}/experiments/scripts/run_mvbench_compressed_feature_memory_shard.sh" \
    "${gpu}" 0 1 "${out_dir}" "${selection_manifest}" \
    "${codec_path}" "${samples_per_task}" "${policies}" \
    "${residual_tokens}"
  status="$?"
  if [ "${status}" -ne 0 ]; then
    printf 'rank=%s stage=evaluate status=%s\n' \
      "${rank}" "${status}" | tee -a "${sweep_log}"
    break
  fi

  unset PREFIX
  source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
  conda activate Qwen3
  python \
    "${project_root}/experiments/probes/aggregate_compressed_feature_memory.py" \
    --run-dir "${out_dir}" \
    --out-dir "${out_dir}/aggregate"
  status="$?"
  if [ "${status}" -ne 0 ]; then
    printf 'rank=%s stage=aggregate status=%s\n' \
      "${rank}" "${status}" | tee -a "${sweep_log}"
    break
  fi
  printf 'rank=%s stage=complete status=0\n' \
    "${rank}" | tee -a "${sweep_log}"
done

printf '%s\n' "${status}" >"${sweep_exit}"
exit "${status}"

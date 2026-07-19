#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  printf 'usage: %s METHOD RUN_NAME [GPU_ID]\n' "$0" >&2
  printf 'METHOD must be causal_mem\n' >&2
  exit 2
fi

method="$1"
run_name="$2"
gpu_id="${3:-0}"
if [ "${method}" != "causal_mem" ]; then
  printf 'unsupported method: %s\n' "${method}" >&2
  printf '%s\n' \
    'The pinned upstream baseline imports missing llava_arch_baseline.py.' >&2
  exit 2
fi
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'GPU_ID must be a non-negative integer: %s\n' "${gpu_id}" >&2
  exit 2
fi

runtime_root="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
code_root="${CODE_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-${runtime_root}/.conda/causalmem-py310/bin/python}"
source_root="${CAUSALMEM_SOURCE_ROOT:-${runtime_root}/external_baselines/CausalMem}"
model_path="${MODEL_PATH:-${runtime_root}/third_party/llava-onevision-qwen2-7b-ov-original-hf}"
hf_hub_cache="${HF_HUB_CACHE:-${runtime_root}/third_party/hf_cache}"
prepared_root="${STREAMINGBENCH_PREPARED_ROOT:-${runtime_root}/third_party/StreamingBench-RT-1-50/prepared}"
video_dir="${VIDEO_DIR:-${prepared_root}/realtime_video}"
gt_file="${GT_FILE:-${prepared_root}/Real_Time_Visual_Understanding_1-50.csv}"
dataset_manifest="${DATASET_MANIFEST:-${prepared_root}/manifest.json}"
output_dir="${OUTPUT_DIR:-${runtime_root}/remote_results/causalmem_streamingbench/${run_name}}"
runner="${code_root}/experiments/probes/run_causalmem_streamingbench.py"

for required in "${python_bin}" "${runner}" "${gt_file}" "${dataset_manifest}"; do
  if [ ! -f "${required}" ]; then
    printf 'required file not found: %s\n' "${required}" >&2
    exit 2
  fi
done
if [ ! -d "${video_dir}" ]; then
  printf 'prepared video directory not found: %s\n' "${video_dir}" >&2
  exit 2
fi
if [ ! -d "${hf_hub_cache}" ]; then
  printf 'offline Hugging Face cache not found: %s\n' "${hf_hub_cache}" >&2
  exit 2
fi

unset PREFIX
export PYTHONDONTWRITEBYTECODE=1
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

lock_path="${GPU_LOCK_PATH:-/tmp/online-video-state-gpu-${gpu_id}.lock}"
exec 9>"${lock_path}"
if ! flock -n 9; then
  printf 'GPU lock is already held: %s\n' "${lock_path}" >&2
  exit 75
fi

cd "${code_root}"
extra_args=()
if [ "${ALLOW_SMOKE_SUBSET:-0}" = "1" ]; then
  extra_args+=(--allow-smoke-subset)
fi
"${python_bin}" "${runner}" \
  --source-root "${source_root}" \
  --model-path "${model_path}" \
  --hf-hub-cache "${hf_hub_cache}" \
  --video-dir "${video_dir}" \
  --gt-file "${gt_file}" \
  --dataset-manifest "${dataset_manifest}" \
  --output-dir "${output_dir}" \
  --python-bin "${python_bin}" \
  --method "${method}" \
  --gpu-index "${gpu_id}" \
  --foss-budget "${FOSS_BUDGET:-12000}" \
  --foss-decay "${FOSS_DECAY:-0.9}" \
  --foss-k-max "${FOSS_K_MAX:-64}" \
  --foss-max-new-basis "${FOSS_MAX_NEW_BASIS:-8}" \
  --foss-time-weight "${FOSS_TIME_WEIGHT:-0.8}" \
  --foss-update-ratio "${FOSS_UPDATE_RATIO:-0.1}" \
  --foss-time-power "${FOSS_TIME_POWER:-1.0}" \
  --max-idle-memory-mib "${MAX_IDLE_MEMORY_MIB:-4096}" \
  --max-idle-utilization "${MAX_IDLE_UTILIZATION:-20}" \
  "${extra_args[@]}"

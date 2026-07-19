#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  printf 'usage: %s MODE RUN_NAME [GPU_ID]\n' "$0" >&2
  printf 'MODE must be rekv or rekv_stc\n' >&2
  exit 2
fi

mode="$1"
run_name="$2"
gpu_id="${3:-0}"
case "${mode}" in
  rekv|rekv_stc) ;;
  *)
    printf 'unsupported mode: %s\n' "${mode}" >&2
    exit 2
    ;;
esac
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'GPU_ID must be a non-negative integer: %s\n' "${gpu_id}" >&2
  exit 2
fi

runtime_root="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
code_root="${CODE_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/anaconda3/envs/Qwen3/bin/python}"
source_root="${STC_SOURCE_ROOT:-${runtime_root}/external_baselines/STC}"
model_path="${MODEL_PATH:-${runtime_root}/third_party/llava-onevision-qwen2-7b-ov-chat-hf-modelscope}"
output_dir="${OUTPUT_DIR:-${runtime_root}/remote_results/stc_rekv_official/${run_name}/${mode}}"
runner="${code_root}/experiments/probes/run_stc_rekv_official.py"

for required in "${python_bin}" "${runner}"; do
  if [ ! -f "${required}" ]; then
    printf 'required file not found: %s\n' "${required}" >&2
    exit 2
  fi
done
for required_dir in "${source_root}" "${model_path}"; do
  if [ ! -d "${required_dir}" ]; then
    printf 'required directory not found: %s\n' "${required_dir}" >&2
    exit 2
  fi
done

unset PREFIX
export PYTHONDONTWRITEBYTECODE=1
cd "${code_root}"
extra_args=()
if [ "${DRY_RUN:-0}" = "1" ]; then
  extra_args+=(--dry-run)
fi
if [ "${CHECK_RUNTIME:-0}" = "1" ]; then
  extra_args+=(--check-runtime)
fi
if [ -n "${VIDEO:-}" ]; then
  extra_args+=(--video "${VIDEO}" --sample-fps "${SAMPLE_FPS:-0.5}")
fi
if [ -n "${GPU_LOCK_PATH:-}" ]; then
  extra_args+=(--gpu-lock-path "${GPU_LOCK_PATH}")
fi

"${python_bin}" "${runner}" \
  --source-root "${source_root}" \
  --model-path "${model_path}" \
  --output-dir "${output_dir}" \
  --python-bin "${python_bin}" \
  --mode "${mode}" \
  --gpu-index "${gpu_id}" \
  --num-frames "${NUM_FRAMES:-64}" \
  --image-size "${IMAGE_SIZE:-384}" \
  --repeats "${REPEATS:-20}" \
  --warmup "${WARMUP:-5}" \
  --max-idle-memory-mib "${MAX_IDLE_MEMORY_MIB:-4096}" \
  --max-idle-utilization "${MAX_IDLE_UTILIZATION:-20}" \
  "${extra_args[@]}"

#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ] || [ "$#" -gt 3 ]; then
  printf 'usage: %s METHOD RUN_NAME [GPU_ID]\n' "$0" >&2
  exit 2
fi

method="$1"
run_name="$2"
gpu_id="${3:-0}"
runtime_root="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
code_root="${CODE_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
external_root="${EXTERNAL_ROOT:-${runtime_root}/external_baselines}"
case "${method}" in
  streamingtom_*)
    python_bin="${PYTHON_BIN:-${runtime_root}/.conda/streamingtom-py310/bin/python}"
    ;;
  stc_pruner)
    python_bin="${PYTHON_BIN:-/home/wangmeiqi/anaconda3/envs/Qwen3/bin/python}"
    ;;
  *)
    printf 'unsupported method: %s\n' "${method}" >&2
    exit 2
    ;;
esac

if [ -n "${FRAMES:-}" ]; then
  frames="${FRAMES}"
elif [ "${method}" = "streamingtom_oqm_select" ]; then
  frames=256
elif [ "${method}" = "streamingtom_ctr" ] || \
     [ "${method}" = "streamingtom_oqm_write" ]; then
  frames=64
else
  frames=32
fi

runner="${code_root}/experiments/probes/benchmark_official_streaming_kernels.py"
out_dir="${OUT_DIR:-${runtime_root}/remote_results/official_streaming_kernels/${run_name}}"
for required in "${python_bin}" "${runner}"; do
  if [ ! -f "${required}" ]; then
    printf 'required file not found: %s\n' "${required}" >&2
    exit 2
  fi
done

unset PREFIX
export PYTHONDONTWRITEBYTECODE=1
if [[ "${method}" == streamingtom_* ]]; then
  "${python_bin}" -c 'import flash_attn, torch, transformers'
else
  "${python_bin}" -c 'import torch; assert tuple(map(int, torch.__version__.split("+")[0].split(".")[:2])) >= (2, 1)'
fi
lock_path="${GPU_LOCK_PATH:-/tmp/online-video-state-gpu-${gpu_id}.lock}"
exec 9>"${lock_path}"
if ! flock -n 9; then
  printf 'GPU lock is already held: %s\n' "${lock_path}" >&2
  exit 75
fi

cd "${code_root}"
"${python_bin}" "${runner}" \
  --method "${method}" \
  --external-root "${external_root}" \
  --out-dir "${out_dir}" \
  --gpu-index "${gpu_id}" \
  --frames "${frames}" \
  --layers "${LAYERS:-28}" \
  --warmup "${WARMUP:-20}" \
  --repeat "${REPEAT:-200}" \
  --dtype "${DTYPE:-float16}" \
  --stc-tokens-per-frame "${STC_TOKENS_PER_FRAME:-64}" \
  --max-idle-memory-mib "${MAX_IDLE_MEMORY_MIB:-4096}" \
  --max-idle-utilization "${MAX_IDLE_UTILIZATION:-20}"

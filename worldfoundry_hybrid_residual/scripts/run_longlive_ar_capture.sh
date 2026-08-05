#!/usr/bin/env bash
set -euo pipefail

if (( $# < 3 )); then
  echo "usage: $0 <cuda-index> <output-dir> <prompt-id> [prompt-id ...]" >&2
  exit 2
fi

cuda_index="$1"
output_dir="$2"
shift 2

project="${AR_VIDEO_PROJECT:-/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805}"
source_root="${LONGLIVE_ROOT:-${project}/external/LongLive-v1}"
python_bin="${LONGLIVE_PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}"
overlay="${project}/.venv/lib/python3.11/site-packages"

export PYTHONPATH="${overlay}:${source_root}:${project}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
cd "${source_root}"

prompt_args=()
for prompt_id in "$@"; do
  prompt_args+=(--prompt-id "${prompt_id}")
done

exec "${python_bin}" "${project}/scripts/capture_longlive_causal_qkv.py" \
  --longlive-root "${source_root}" \
  --runtime-config "${project}/configs/longlive_capture_f21.yaml" \
  --protocol "${project}/configs/ar_video_residual_memory_longlive_v1.json" \
  --output-dir "${output_dir}" \
  --device "cuda:${cuda_index}" \
  "${prompt_args[@]}"

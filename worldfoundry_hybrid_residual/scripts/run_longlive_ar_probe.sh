#!/usr/bin/env bash
set -euo pipefail

if (( $# != 3 )); then
  echo "usage: $0 <cuda-index> <capture-dir> <result-dir>" >&2
  exit 2
fi

cuda_index="$1"
capture_dir="$2"
result_dir="$3"
project="${AR_VIDEO_PROJECT:-/home/wangmeiqi/codex_runs/ar_video_multiresidual_20260805}"
python_bin="${LONGLIVE_PYTHON:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723/.venv/bin/python}"
overlay="${project}/.venv/lib/python3.11/site-packages"
source_root="${LONGLIVE_ROOT:-${project}/external/LongLive-v1}"

export PYTHONPATH="${overlay}:${source_root}:${project}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
exec "${python_bin}" "${project}/scripts/probe_ar_video_residual_memory.py" \
  --protocol "${project}/configs/ar_video_residual_memory_longlive_v1.json" \
  --capture-dir "${capture_dir}" \
  --output-dir "${result_dir}" \
  --device "cuda:${cuda_index}"

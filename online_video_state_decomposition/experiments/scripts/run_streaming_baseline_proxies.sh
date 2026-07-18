#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 3 ]; then
  printf 'usage: %s OUT_DIR [CACHE_DIR] [LEARNED_RANKER]\n' "$0"
  exit 2
fi

project_root="${PROJECT_ROOT:-/home/spco/online_video_state_decomposition}"
out_dir="$1"
cache_dir="${2:-${project_root}/remote_results/mvbench_query_confirmation_20260718_v1/cache}"
learned_ranker="${3:-${project_root}/remote_results/mvbench_query_confirmation_20260718_v1/aggregate/learned_feature_ranker.json}"
python_bin="${PYTHON_BIN:-/home/wangmeiqi/anaconda3/envs/Qwen3/bin/python}"

if [ -e "${out_dir}" ]; then
  printf 'refusing to overwrite existing output: %s\n' "${out_dir}" >&2
  exit 17
fi
if [ ! -d "${cache_dir}" ]; then
  printf 'cache directory not found: %s\n' "${cache_dir}" >&2
  exit 2
fi
if [ ! -f "${learned_ranker}" ]; then
  printf 'learned ranker not found: %s\n' "${learned_ranker}" >&2
  exit 2
fi

unset PREFIX
export MPLBACKEND=Agg
export PYTHONHASHSEED=0

cd "${project_root}"
"${python_bin}" experiments/probes/evaluate_streaming_baseline_proxies.py \
  --cache-dir "${cache_dir}" \
  --learned-ranker "${learned_ranker}" \
  --out-dir "${out_dir}" \
  --bootstrap-samples "${BOOTSTRAP_SAMPLES:-5000}" \
  --seed "${SEED:-20260719}"

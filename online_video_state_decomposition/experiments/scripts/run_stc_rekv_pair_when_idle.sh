#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  printf 'usage: %s RUN_NAME [GPU_ID]\n' "$0" >&2
  exit 2
fi

run_name="$1"
gpu_id="${2:-0}"
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'GPU_ID must be a non-negative integer: %s\n' "${gpu_id}" >&2
  exit 2
fi

runtime_root="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
code_root="${CODE_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
launcher="${code_root}/experiments/scripts/run_stc_rekv_official.sh"
run_root="${RUN_ROOT:-${runtime_root}/remote_results/stc_rekv_official/${run_name}}"
max_memory_mib="${MAX_IDLE_MEMORY_MIB:-4096}"
max_utilization="${MAX_IDLE_UTILIZATION:-20}"
poll_seconds="${POLL_SECONDS:-60}"
lock_path="${GPU_LOCK_PATH:-/tmp/online-video-state-gpu-${gpu_id}.lock}"
inner_lock_path="${lock_path}.stc-pair-${run_name}"

if [ ! -f "${launcher}" ]; then
  printf 'launcher not found: %s\n' "${launcher}" >&2
  exit 2
fi
mkdir -p "${run_root}"
printf 'waiting_for_idle\n' > "${run_root}/queue_status"
trap 'rc=$?; if [ "${rc}" -ne 0 ]; then printf "failed:%s\n" "${rc}" > "${run_root}/queue_status"; fi' EXIT

query_gpu() {
  nvidia-smi \
    --id="${gpu_id}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, ""); print $1, $2}'
}

exec 9>"${lock_path}"
while true; do
  read -r memory_mib utilization < <(query_gpu)
  printf '%s memory_mib=%s utilization=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${memory_mib}" "${utilization}" \
    >> "${run_root}/idle_samples.log"
  if [ "${memory_mib}" -le "${max_memory_mib}" ] && \
     [ "${utilization}" -le "${max_utilization}" ] && flock -n 9; then
    read -r locked_memory_mib locked_utilization < <(query_gpu)
    if [ "${locked_memory_mib}" -le "${max_memory_mib}" ] && \
       [ "${locked_utilization}" -le "${max_utilization}" ]; then
      break
    fi
    flock -u 9
  fi
  sleep "${poll_seconds}"
done

printf 'running_rekv\n' > "${run_root}/queue_status"
GPU_LOCK_PATH="${inner_lock_path}" \
OUTPUT_DIR="${run_root}/rekv" \
NUM_FRAMES="${NUM_FRAMES:-64}" \
REPEATS="${REPEATS:-20}" \
WARMUP="${WARMUP:-5}" \
MAX_IDLE_MEMORY_MIB="${max_memory_mib}" \
MAX_IDLE_UTILIZATION="${max_utilization}" \
bash "${launcher}" rekv "${run_name}" "${gpu_id}"

while true; do
  read -r memory_mib utilization < <(query_gpu)
  if [ "${memory_mib}" -le "${max_memory_mib}" ] && \
     [ "${utilization}" -le "${max_utilization}" ]; then
    break
  fi
  sleep "${poll_seconds}"
done

printf 'running_rekv_stc\n' > "${run_root}/queue_status"
GPU_LOCK_PATH="${inner_lock_path}" \
OUTPUT_DIR="${run_root}/rekv_stc" \
NUM_FRAMES="${NUM_FRAMES:-64}" \
REPEATS="${REPEATS:-20}" \
WARMUP="${WARMUP:-5}" \
MAX_IDLE_MEMORY_MIB="${max_memory_mib}" \
MAX_IDLE_UTILIZATION="${max_utilization}" \
bash "${launcher}" rekv_stc "${run_name}" "${gpu_id}"

printf 'complete\n' > "${run_root}/queue_status"

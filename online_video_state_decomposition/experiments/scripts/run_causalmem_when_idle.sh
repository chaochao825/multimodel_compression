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
launcher="${code_root}/experiments/scripts/run_causalmem_streamingbench.sh"
run_root="${RUN_ROOT:-${runtime_root}/remote_results/causalmem_streamingbench/${run_name}}"
max_memory_mib="${MAX_IDLE_MEMORY_MIB:-4096}"
max_utilization="${MAX_IDLE_UTILIZATION:-20}"
poll_seconds="${POLL_SECONDS:-60}"
lock_path="${GPU_LOCK_PATH:-/tmp/online-video-state-gpu-${gpu_id}.lock}"
inner_lock_path="${lock_path}.causalmem-${run_name}"
wait_status_file="${WAIT_STATUS_FILE:-}"
wait_status_value="${WAIT_STATUS_VALUE:-complete}"
wait_status_allow_failed="${WAIT_STATUS_ALLOW_FAILED:-0}"

if [ ! -f "${launcher}" ]; then
  printf 'launcher not found: %s\n' "${launcher}" >&2
  exit 2
fi
mkdir -p "${run_root}"
trap 'rc=$?; if [ "${rc}" -ne 0 ]; then printf "failed:%s\n" "${rc}" > "${run_root}/queue_status"; fi' EXIT

if [ -n "${wait_status_file}" ]; then
  printf 'waiting_for_dependency\n' > "${run_root}/queue_status"
  while true; do
    if [ -f "${wait_status_file}" ]; then
      dependency_status="$(tr -d '\r\n' < "${wait_status_file}")"
      if [ "${dependency_status}" = "${wait_status_value}" ]; then
        break
      fi
      case "${dependency_status}" in
        failed:*)
          if [ "${wait_status_allow_failed}" = "1" ]; then
            printf 'dependency reached failed terminal state; continuing: %s\n' \
              "${dependency_status}" >> "${run_root}/dependency.log"
            break
          fi
          printf 'dependency failed: %s\n' "${dependency_status}" >&2
          exit 1
          ;;
      esac
    fi
    sleep "${poll_seconds}"
  done
fi

query_gpu() {
  nvidia-smi \
    --id="${gpu_id}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, ""); print $1, $2}'
}

printf 'waiting_for_idle\n' > "${run_root}/queue_status"
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

printf 'running_causal_mem\n' > "${run_root}/queue_status"
GPU_LOCK_PATH="${inner_lock_path}" \
OUTPUT_DIR="${run_root}/official" \
MAX_IDLE_MEMORY_MIB="${max_memory_mib}" \
MAX_IDLE_UTILIZATION="${max_utilization}" \
bash "${launcher}" causal_mem "${run_name}" "${gpu_id}"

printf 'complete\n' > "${run_root}/queue_status"

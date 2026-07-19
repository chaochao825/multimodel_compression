#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  printf 'usage: %s RUN_NAME [GPU_ID]\n' "$0" >&2
  exit 2
fi

run_name="$1"
gpu_id="${2:-0}"
if ! [[ "${run_name}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  printf 'RUN_NAME contains unsafe characters: %s\n' "${run_name}" >&2
  exit 2
fi
if ! [[ "${gpu_id}" =~ ^[0-9]+$ ]]; then
  printf 'GPU_ID must be a non-negative integer: %s\n' "${gpu_id}" >&2
  exit 2
fi

runtime_root="${RUNTIME_ROOT:-/home/spco/online_video_state_decomposition}"
code_root="${CODE_ROOT:-/home/spco/sow_linear/multimodel_compression/online_video_state_decomposition}"
launcher="${code_root}/experiments/scripts/run_official_streaming_kernel_benchmark.sh"
run_root="${RUN_ROOT:-${runtime_root}/remote_results/official_streaming_kernels/${run_name}}"
max_memory_mib="${MAX_IDLE_MEMORY_MIB:-4096}"
max_utilization="${MAX_IDLE_UTILIZATION:-20}"
poll_seconds="${POLL_SECONDS:-60}"
lock_path="${GPU_LOCK_PATH:-/tmp/online-video-state-gpu-${gpu_id}.lock}"
inner_lock_path="${lock_path}.streamingtom-kernels-${run_name}"
methods=(
  streamingtom_ctr
  streamingtom_oqm_write
  streamingtom_oqm_select
)

if [ ! -f "${launcher}" ]; then
  printf 'launcher not found: %s\n' "${launcher}" >&2
  exit 2
fi
if [ -e "${run_root}" ]; then
  printf 'refusing to reuse StreamingTOM queue directory: %s\n' "${run_root}" >&2
  exit 2
fi
mkdir -p "${run_root}"
printf '%s\n' "$$" > "${run_root}/pid"
printf 'waiting_for_idle\n' > "${run_root}/queue_status"
trap 'rc=$?; if [ "${rc}" -ne 0 ]; then printf "failed:%s\n" "${rc}" > "${run_root}/queue_status"; fi' EXIT

query_gpu() {
  nvidia-smi \
    --id="${gpu_id}" \
    --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader,nounits | awk -F',' '{gsub(/ /, ""); print $1, $2}'
}

wait_for_idle_state() {
  while true; do
    read -r memory_mib utilization < <(query_gpu)
    printf '%s memory_mib=%s utilization=%s\n' \
      "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${memory_mib}" "${utilization}" \
      >> "${run_root}/idle_samples.log"
    if [ "${memory_mib}" -le "${max_memory_mib}" ] && \
       [ "${utilization}" -le "${max_utilization}" ]; then
      return
    fi
    sleep "${poll_seconds}"
  done
}

exec 9>"${lock_path}"
while true; do
  wait_for_idle_state
  if flock -n 9; then
    read -r locked_memory_mib locked_utilization < <(query_gpu)
    if [ "${locked_memory_mib}" -le "${max_memory_mib}" ] && \
       [ "${locked_utilization}" -le "${max_utilization}" ]; then
      break
    fi
    flock -u 9
  fi
  sleep "${poll_seconds}"
done

for method in "${methods[@]}"; do
  wait_for_idle_state
  suffix="${method#streamingtom_}"
  case "${method}" in
    streamingtom_ctr|streamingtom_oqm_write)
      frames=64
      ;;
    streamingtom_oqm_select)
      frames=256
      ;;
  esac
  printf 'running_%s\n' "${suffix}" > "${run_root}/queue_status"
  RUNTIME_ROOT="${runtime_root}" \
  CODE_ROOT="${code_root}" \
  PYTHON_BIN="${runtime_root}/.conda/streamingtom-py310/bin/python" \
  GPU_LOCK_PATH="${inner_lock_path}" \
  OUT_DIR="${run_root}/${suffix}" \
  FRAMES="${frames}" \
  LAYERS=28 \
  WARMUP=20 \
  REPEAT=200 \
  DTYPE=float16 \
  MAX_IDLE_MEMORY_MIB="${max_memory_mib}" \
  MAX_IDLE_UTILIZATION="${max_utilization}" \
  bash "${launcher}" "${method}" "${run_name}_${suffix}" "${gpu_id}" \
    > "${run_root}/${suffix}.log" 2>&1
done

python3 - "${run_root}" <<'PY'
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
expected = {
    "ctr": "streamingtom_ctr",
    "oqm_write": "streamingtom_oqm_write",
    "oqm_select": "streamingtom_oqm_select",
}
for directory, method in expected.items():
    path = root / directory / "summary.json"
    if not path.is_file():
        raise SystemExit(f"missing StreamingTOM summary: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("method") != method:
        raise SystemExit(f"method mismatch in {path}: {payload.get('method')}")
    quality = payload.get("quality_gate")
    if not isinstance(quality, dict) or quality.get("passed") is not True:
        raise SystemExit(f"quality gate did not pass in {path}")
    for metric in ("wall_ms", "cuda_event_ms"):
        summary = payload.get(metric)
        if not isinstance(summary, dict) or summary.get("count") != 200.0:
            raise SystemExit(f"invalid {metric} summary in {path}")
print("validated StreamingTOM CTR/OQM result triplet")
PY

printf 'complete\n' > "${run_root}/queue_status"

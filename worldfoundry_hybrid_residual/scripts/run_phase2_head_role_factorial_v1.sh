#!/usr/bin/env bash
set -euo pipefail

PROBE_ROOT="${PROBE_ROOT:-/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723}"
BASE_ROOT="${BASE_ROOT:-/home/wangmeiqi/codex_runs/base2_h200_fp8_20260723}"
PY="${PY:-$BASE_ROOT/.venv/bin/python}"
WAN_SOURCE="${WAN_SOURCE:-$BASE_ROOT/wan_runtime/MonarchRT}"
CHECKPOINT="${CHECKPOINT:-$WAN_SOURCE/wan_models/Wan2.1-T2V-1.3B}"
PROMPT_FILE="${PROMPT_FILE:-$BASE_ROOT/scripts/prompts_pilot8.txt}"
OUT="${OUT:-$PROBE_ROOT/results/attention_head_factorial_f81_v1}"
GPU="${GPU:-auto}"
GPU_CANDIDATES="${GPU_CANDIDATES:-2,3}"
LOCK_PATH="${LOCK_PATH:-/tmp/codex_phase2_strict_h200_v1.lock}"
IDLE_POLLS="${IDLE_POLLS:-3}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MIN_FREE_BYTES="${MIN_FREE_BYTES:-32212254720}"

mkdir -p "$OUT/logs" "$OUT/qkv_replays" "$OUT/head_stats" "$OUT/summary"

gpu_process_count() {
  local gpu="$1"
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    --id="$gpu" 2>/dev/null | sed '/^[[:space:]]*$/d' | wc -l
}

select_gpu() {
  if [[ "$GPU" != "auto" ]]; then
    GPU_CANDIDATES="$GPU"
  fi
  local candidates=()
  IFS=',' read -r -a candidates <<< "$GPU_CANDIDATES"
  declare -A idle_counts=()
  while true; do
    local candidate
    for candidate in "${candidates[@]}"; do
      candidate="${candidate//[[:space:]]/}"
      if [[ ! "$candidate" =~ ^[0-9]+$ ]]; then
        printf '[head-factorial] invalid GPU candidate: %s\n' "$candidate" >&2
        return 2
      fi
      local count
      count="$(gpu_process_count "$candidate")"
      if [[ "$count" == "0" ]]; then
        idle_counts[$candidate]=$(( ${idle_counts[$candidate]:-0} + 1 ))
        printf '[head-factorial] GPU %s idle poll %d/%d\n' \
          "$candidate" "${idle_counts[$candidate]}" "$IDLE_POLLS"
        if (( idle_counts[$candidate] >= IDLE_POLLS )); then
          GPU="$candidate"
          printf '[head-factorial] selected first-idle GPU %s\n' "$GPU"
          return
        fi
      else
        idle_counts[$candidate]=0
        printf '[head-factorial] waiting: %s compute processes use GPU %s\n' "$count" "$candidate"
      fi
    done
    sleep "$POLL_SECONDS"
  done
}

capture_complete() {
  [[ -f "$OUT/qkv_replays/capture_manifest.json" ]] && "$PY" - "$OUT/qkv_replays/capture_manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as handle:
    payload = json.load(handle)
raise SystemExit(0 if payload.get("capture_complete") else 1)
PY
}

free_bytes="$(df --output=avail -B1 "$OUT" | tail -n 1 | tr -d '[:space:]')"
if (( free_bytes < MIN_FREE_BYTES )); then
  printf '[head-factorial] insufficient free bytes: %s < %s\n' "$free_bytes" "$MIN_FREE_BYTES" >&2
  exit 3
fi

exec 9>"$LOCK_PATH"
printf '[head-factorial] waiting for shared H200 lock %s\n' "$LOCK_PATH"
flock 9
printf '[head-factorial] acquired shared H200 lock\n'
select_gpu

if capture_complete; then
  printf '[head-factorial] reusing complete QKV capture\n'
else
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PROBE_ROOT/scripts/capture_wan_qkv_trajectory.py" \
    --wan-source "$WAN_SOURCE" \
    --checkpoint "$CHECKPOINT" \
    --out-dir "$OUT/qkv_replays" \
    --prompt-file "$PROMPT_FILE" \
    --max-prompts 2 \
    --sample-plan 0:20260740,1:20260740,0:20260741,1:20260741 \
    --frame-num 81 \
    --sampling-steps 20 \
    --capture-steps 0,9,19 \
    --capture-layers 0,14,29 \
    --seed 20260740 \
    --device cuda:0 \
    2>&1 | tee "$OUT/logs/qkv_capture.log"
fi

index="$OUT/head_stats/head_stats_index.csv"
printf '%s\n' 'label,sample_id,prompt_index,seed,sampling_step,timestep,branch,layer,head_csv' > "$index"
tail -n +2 "$OUT/qkv_replays/capture_index.csv" | while IFS=, read -r sample_id prompt_index seed sampling_step timestep branch layer replay bytes; do
  branch="${branch//$'\r'/}"
  layer="${layer//$'\r'/}"
  replay="${replay//$'\r'/}"
  label="${sample_id}_step$(printf '%02d' "$sampling_step")_${branch}_l$(printf '%02d' "$layer")"
  stat_dir="$OUT/head_stats/$label"
  if [[ ! -s "$stat_dir/attention_rmt_entropy_heads.csv" ]]; then
    CUDA_VISIBLE_DEVICES="$GPU" "$PY" "$PROBE_ROOT/scripts/probe_attention_rmt_entropy.py" \
      --replay "$replay" \
      --output-dir "$stat_dir" \
      --device cuda:0 \
      --query-samples 128 \
      --geometry-mask s3_temporal_pm2 \
      --seed 20260726 \
      > "$OUT/logs/${label}.log" 2>&1
  fi
  printf '%s,%s,%s,%s,%s,%s,%s,%s,%s\n' \
    "$label" "$sample_id" "$prompt_index" "$seed" "$sampling_step" "$timestep" \
    "$branch" "$layer" "$stat_dir/attention_rmt_entropy_heads.csv" >> "$index"
done

flock -u 9
printf '[head-factorial] released shared H200 lock\n'

"$PY" "$PROBE_ROOT/scripts/summarize_attention_head_factorial.py" \
  --index "$index" \
  --output-dir "$OUT/summary" \
  2>&1 | tee "$OUT/logs/factorial_summary.log"

printf '[head-factorial] completed %s\n' "$OUT"

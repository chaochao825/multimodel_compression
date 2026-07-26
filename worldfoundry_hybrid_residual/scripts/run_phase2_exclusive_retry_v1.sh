#!/usr/bin/env bash
set -euo pipefail

TARGET_SCRIPT="${TARGET_SCRIPT:?TARGET_SCRIPT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
MAX_ATTEMPTS="${MAX_ATTEMPTS:-12}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"
USE_OUTER_LOCK="${USE_OUTER_LOCK:-0}"
LOCK_PATH="${LOCK_PATH:-/tmp/codex_phase2_strict_h200_v1.lock}"

mkdir -p "$OUT_ROOT/logs"
ATTEMPTS_CSV="$OUT_ROOT/attempts.csv"
if [[ ! -e "$ATTEMPTS_CSV" ]]; then
  printf 'attempt,started_at,finished_at,status,output_dir\n' > "$ATTEMPTS_CSV"
fi

for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
  started_at="$(date --iso-8601=seconds)"
  attempt_out="$(printf '%s/attempt_%02d' "$OUT_ROOT" "$attempt")"
  mkdir -p "$attempt_out"
  printf '[exclusive-retry] attempt %d/%d target=%s out=%s\n' \
    "$attempt" "$MAX_ATTEMPTS" "$TARGET_SCRIPT" "$attempt_out"
  set +e
  if [[ "$USE_OUTER_LOCK" == "1" ]]; then
    OUT="$attempt_out" flock "$LOCK_PATH" bash "$TARGET_SCRIPT"
  else
    OUT="$attempt_out" bash "$TARGET_SCRIPT"
  fi
  status=$?
  set -e
  finished_at="$(date --iso-8601=seconds)"
  printf '%d,%s,%s,%d,%s\n' \
    "$attempt" "$started_at" "$finished_at" "$status" "$attempt_out" \
    >> "$ATTEMPTS_CSV"
  if (( status == 0 )); then
    printf '%s\n' "$attempt_out" > "$OUT_ROOT/successful_attempt.txt"
    printf '[exclusive-retry] completed on attempt %d\n' "$attempt"
    exit 0
  fi
  if (( status != 86 )); then
    printf '[exclusive-retry] non-retryable status=%d on attempt %d\n' \
      "$status" "$attempt"
    exit "$status"
  fi
  printf '[exclusive-retry] timing contamination on attempt %d; retrying in %ss\n' \
    "$attempt" "$RETRY_SECONDS"
  sleep "$RETRY_SECONDS"
done

printf '[exclusive-retry] exhausted %d attempts\n' "$MAX_ATTEMPTS"
exit 86

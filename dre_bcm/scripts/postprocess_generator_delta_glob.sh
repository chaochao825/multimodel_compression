#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-/home/spco/diff_bitnet/dre_bcm}
UPSTREAM_JOB=${2:?upstream job name required}
CHECKPOINT_GLOB=${3:?checkpoint glob required}
RUN_PREFIX=${4:?run prefix required}
CONDA_ENV=${5:-/home/wangmeiqi/zhuoziying/env/bitnet}

PID_FILE="$ROOT_DIR/pids/${UPSTREAM_JOB}.pid"

if [ ! -f "$PID_FILE" ]; then
  echo "missing pid file: $PID_FILE"
  exit 1
fi

pid=$(cat "$PID_FILE")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

. /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"
cd "$ROOT_DIR"

shopt -s nullglob
for ckpt in $CHECKPOINT_GLOB; do
  run_name=$(basename "$ckpt" .pt)
  PYTHONPATH=. python -m src.analysis.generator_delta_stats \
    --checkpoint "$ckpt" \
    --run-name "${RUN_PREFIX}_${run_name}"
done

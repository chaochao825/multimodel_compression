#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-/home/spco/diff_bitnet/dre_bcm}
PID_FILE="$ROOT_DIR/pids/llama2_qproj_retry.pid"
SUMMARY_FILE="$ROOT_DIR/results/matrix_fit/llama2_qproj_retry/summary_metrics.csv"

if [ ! -f "$PID_FILE" ]; then
  echo "missing pid file: $PID_FILE"
  exit 1
fi

pid=$(cat "$PID_FILE")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

. /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate /home/wangmeiqi/anaconda3/envs/srlm
cd "$ROOT_DIR"

PYTHONPATH=. python - <<'PY'
import csv
import json
from pathlib import Path

path = Path("results/matrix_fit/llama2_qproj_retry/summary_metrics.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
rows.sort(key=lambda row: float(row["relative_fro_error"]))
best = rows[0]
output = path.parent / "best_metrics.json"
output.write_text(json.dumps(best, indent=2), encoding="utf-8")
print(json.dumps(best, indent=2))
PY

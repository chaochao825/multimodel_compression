#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-/home/spco/diff_bitnet/dre_bcm}
PID_FILE="$ROOT_DIR/pids/qwen3_downproj_retry.pid"
CKPT_DIR="$ROOT_DIR/results/matrix_fit/qwen3_downproj_retry/model__layers__0__mlp__down_proj__weight"

if [ ! -f "$PID_FILE" ]; then
  echo "missing pid file: $PID_FILE"
  exit 1
fi

pid=$(cat "$PID_FILE")
while kill -0 "$pid" 2>/dev/null; do
  sleep 30
done

. /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate /home/wangmeiqi/zhuoziying/env/bitnet
cd "$ROOT_DIR"

for ckpt in "$CKPT_DIR"/generator_delta_bcm__bs*__r0__basis1.pt; do
  [ -e "$ckpt" ] || continue
  run_name=$(basename "$ckpt" .pt)
  PYTHONPATH=. python -m src.analysis.generator_delta_stats \
    --checkpoint "$ckpt" \
    --run-name "qwen3_downproj_retry_${run_name}"
done

PYTHONPATH=. python - <<'PY'
import csv
import json
from pathlib import Path

path = Path("results/matrix_fit/qwen3_downproj_retry/summary_metrics.csv")
rows = list(csv.DictReader(path.open(encoding="utf-8")))
rows.sort(key=lambda row: float(row["relative_fro_error"]))
best = rows[0]
output = path.parent / "best_metrics.json"
output.write_text(json.dumps(best, indent=2), encoding="utf-8")
print(json.dumps(best, indent=2))
PY

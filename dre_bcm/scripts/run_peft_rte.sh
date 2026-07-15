#!/usr/bin/env bash
set -euo pipefail

MODEL=${MODEL:-roberta-base}
METHOD=${METHOD:-dre_bcm}
CONFIG=${CONFIG:-configs/peft/${METHOD}.yaml}
OUTPUT_DIR=${OUTPUT_DIR:-results/peft/${METHOD}_rte}
LOCAL_DATA_DIR=${LOCAL_DATA_DIR:-}

EXTRA_ARGS=()
if [[ -n "$LOCAL_DATA_DIR" ]]; then
  EXTRA_ARGS+=(--local-data-dir "$LOCAL_DATA_DIR")
fi

PYTHONPATH=. python -m src.train.train_peft \
  --config "$CONFIG" \
  --model-name-or-path "$MODEL" \
  --task-name rte \
  --method "$METHOD" \
  --output-dir "$OUTPUT_DIR" \
  "${EXTRA_ARGS[@]}"

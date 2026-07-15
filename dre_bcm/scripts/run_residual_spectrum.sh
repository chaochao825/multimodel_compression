#!/usr/bin/env bash
set -euo pipefail

WEIGHT_FILE=${1:-results/matrix_fit/raw_weights/qwen3_0_6b/model__layers__0__self_attn__q_proj__weight.pt}

PYTHONPATH=. python -m src.analysis.residual_spectrum \
  --weight-file "$WEIGHT_FILE" \
  --block-size 32

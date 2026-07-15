#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:-results/matrix_fit/qwen3_0_6b/model__layers__0__self_attn__q_proj__weight/generator_delta_bcm__bs32__r0__basis1.pt}
RUN_NAME=${RUN_NAME:-qwen3_qproj_delta}

PYTHONPATH=. python -m src.analysis.generator_delta_stats \
  --checkpoint "$CHECKPOINT" \
  --run-name "$RUN_NAME"

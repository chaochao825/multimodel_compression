#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR=${MODEL_DIR:-/home/wangmeiqi/ZHuan/model/Qwen3-0.6B}
MODEL_NAME=${MODEL_NAME:-qwen3_0_6b}
LAYERS=${LAYERS:-q_proj k_proj v_proj o_proj gate_proj up_proj down_proj}

python src/utils/matrix_extract.py \
  --model-dir "$MODEL_DIR" \
  --model-name "$MODEL_NAME" \
  --layers $LAYERS

PYTHONPATH=. python -m src.analysis.fit_matrix \
  --input-dir "results/matrix_fit/raw_weights/$MODEL_NAME" \
  --methods lowrank_svd bcm_only bcm_plus_lowrank generator_delta_bcm \
  --block-sizes 16 32 64 \
  --ranks 0 4 8 16

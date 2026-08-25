#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CUDA_DEVICE SHARD" >&2
  exit 2
fi

CUDA_DEVICE="$1"
SHARD="$2"
if [[ "$SHARD" != "0" && "$SHARD" != "1" ]]; then
  echo "SHARD must be 0 or 1" >&2
  exit 2
fi

source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate /home/wangmeiqi/zhuoziying/env/diffusion_policy

PROJECT=/home/spco/action_dit_structured_probe_20260826
DP=/home/wangmeiqi/zhuoziying/diffusion_policy/diffusion_policy
RESULT_ROOT="$PROJECT/results/action_dit_noise_response_bridge_20260826"
mkdir -p "$RESULT_ROOT/logs"
cd "$PROJECT"
export PYTHONPATH="$PROJECT/src:$DP"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

CHECKPOINTS=(
  "$DP/pusht_experiments_lowdim/diffusion_transformer/train_0/checkpoints/epoch=0850-test_mean_score=0.967.ckpt"
  "$DP/pusht_experiments_lowdim/diffusion_transformer/train_1/checkpoints/epoch=0950-test_mean_score=0.952.ckpt"
  "$DP/pusht_experiments_lowdim/diffusion_transformer/train_2/checkpoints/epoch=0750-test_mean_score=0.934.ckpt"
)

for train_id in 0 1 2; do
  if (( train_id % 2 == SHARD )); then
    run_name="train${train_id}_m8_v1"
    output_dir="$RESULT_ROOT/$run_name"
    if [[ -e "$output_dir" ]]; then
      echo "refusing to overwrite $output_dir" >&2
      exit 1
    fi
    python scripts/probe_action_dit_noise_response_bridge.py \
      --diffusion-policy-root "$DP" \
      --checkpoint "${CHECKPOINTS[$train_id]}" \
      --output-dir "$output_dir" \
      --device cuda:0 \
      --control-offset 8 \
      --calibration-transitions 96 \
      --evaluation-transitions 48 \
      --flow-points 10 \
      --bucket-count 5 \
      --rank 8 \
      --late-flow-count 3 \
      --seed 20260827 \
      >"$RESULT_ROOT/logs/$run_name.log" 2>&1
    echo "completed $run_name"
  fi
done

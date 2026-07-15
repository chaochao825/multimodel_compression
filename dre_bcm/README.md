# DRE-BCM

Structured PEFT playground for validating

`Delta W = W_BCM + U V^T`

with optional generator differential coding on top of block-circulant adapters.

## Method Family To Cite / Compare

Primary PEFT baselines:

- LoRA
- AdaLoRA
- VeRA
- DoRA
- IA3
- BitFit

Closest structured adapter baselines:

- BCA / Block Circulant Adapter
- C3A / Circular Convolution Adaptation
- FourierFT

Theory / structured matrix related work:

- CirCNN
- Structured Transforms for Small-Footprint Deep Learning
- Learning Compressed Transforms with Low Displacement Rank
- Kaleidoscope / K-matrices
- RP-BCM

Bit / Boolean / BitNet related work:

- MBOK
- BitNet b1.58
- bitnet.cpp
- T-MAC
- TOM

## What Is Implemented

- `BlockCirculantLinear` with direct and FFT multiply
- `LowRankResidual` in LoRA-style form
- `DREBCMLinear` with `bcm_only`, `lowrank_only`, `bcm_plus_lowrank`
- `BCALinear` as a tracked block-circulant PEFT baseline; under the current codebase it still shares the same block-circulant parameterization as the structure-only BCM adapter rather than a more paper-faithful standalone reimplementation
- Optional generator differential reconstruction: `horizontal`, `vertical`, `2d`
- Matrix extraction from local BitNet / Hugging Face checkpoints
- Matrix fitting, residual spectrum, generator delta statistics, FLOP / bit proxy analysis
- Minimal PEFT injection utilities for `nn.Linear` and BitNet `BitLinear`

## Current Practical Status

- Matrix-fit and weight analysis can run immediately on the provided BitNet checkpoints.
- The provided `/home/wangmeiqi/zhuoziying/env/bitnet` environment currently has CPU-only PyTorch.
- The provided `/home/wangmeiqi/zhuoziying/env/bitnet-a4.8` environment currently lacks PyTorch, so GPU fine-tuning needs dependency repair before running PEFT training.
- The remote `srlm` environment is currently the practical path for GPU PEFT runs and now has `matplotlib`, `pandas`, `transformers`, `datasets`, `evaluate`, and `accelerate` available.

## Suggested MVP

1. Extract BitNet attention / MLP weights into `results/matrix_fit/raw_weights`.
2. Run `bcm_only`, `bcm_plus_lowrank`, and `lowrank_svd`.
3. Inspect residual spectrum.
4. Add generator-delta statistics on the best BCM checkpoint.
5. Repair the GPU environment and then run PEFT classification experiments.

## Current PEFT MVP Entry Points

```bash
cd /home/spco/diff_bitnet/dre_bcm
source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate /home/wangmeiqi/anaconda3/envs/srlm

bash scripts/run_peft_sst2.sh
METHOD=bca bash scripts/run_peft_sst2.sh
METHOD=lora bash scripts/run_peft_sst2.sh
bash scripts/run_peft_rte.sh
bash scripts/run_peft_boolq.sh
```

If Hugging Face dataset downloads are unavailable, prepare GLUE TSV files locally and point training at them:

```bash
bash scripts/download_glue_data.sh
LOCAL_DATA_DIR=data/glue_data MODEL=/path/to/gpt2 bash scripts/run_peft_sst2.sh
LOCAL_DATA_DIR=data/glue_data MODEL=/path/to/gpt2 bash scripts/run_peft_rte.sh
```

## Quick Start On The Remote Server

```bash
cd /home/spco/diff_bitnet/dre_bcm
source /home/wangmeiqi/zhuoziying/env/bitnet/bin/activate
python src/utils/matrix_extract.py \
  --model-dir /home/wangmeiqi/zhuoziying/BitBitforward/bitnet_b1_58-large \
  --model-name bitnet_b1_58_large \
  --layers q_proj v_proj o_proj gate_proj up_proj down_proj

PYTHONPATH=. python -m src.analysis.fit_matrix \
  --input-dir results/matrix_fit/raw_weights/bitnet_b1_58_large \
  --methods lowrank_svd bcm_only bcm_plus_lowrank generator_delta_bcm \
  --block-sizes 32 64 \
  --ranks 0 4 8 16
```

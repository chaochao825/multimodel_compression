# DRE-BCM Preliminary Report

## Scope

This report summarizes the experiments that have already been executed on the server under `/home/spco/diff_bitnet/dre_bcm`.

Completed so far:

- Project scaffold and core modules for BCM, low-rank residual, and DRE-BCM
- Weight extraction from existing local checkpoints
- Matrix-fit experiments on:
  - `bert-base-uncased` layer 0 `attention.self.query`
  - `Qwen3-0.6B` layer 0 `self_attn.q_proj`
  - `Qwen3-0.6B` layer 0 `mlp.down_proj`
  - `Llama-2-7b-hf` layer 0 `self_attn.q_proj`
- Qwen generator-delta statistics
- Residual spectrum analysis for Qwen and LLaMA `q_proj`

Not completed yet:

- PEFT downstream fine-tuning on SST-2 / RTE / BoolQ
- Full BCA / FourierFT / C3A paper-faithful baselines
- Plot PNG generation
- End-to-end latency benchmarking

## Existing Model Assets Reused

Normal-precision local checkpoints found on the server:

- `/home/wangmeiqi/zjh/meta-llama/Llama-2-7b-hf`
- `/home/wangmeiqi/zjh/mistralai/Mistral-7B-v0.1`
- `/home/wangmeiqi/ZHuan/model/Qwen2.5-1.5B`
- `/home/wangmeiqi/ZHuan/model/Qwen3-0.6B`
- `/home/wangmeiqi/ZHuan/model/bert-base-uncased`

Bit / BitNet local checkpoints reused:

- `/home/wangmeiqi/zhuoziying/BitBitforward/bitnet_b1_58-large`
- `/home/wangmeiqi/zhuoziying/BitBitforward/bitnet_b1_58-xl`
- `/home/wangmeiqi/zhuoziying/BitBitforward/bitnet_b1_58-3B`

## Key Findings

### 1. Does BCM-only show a clear expressivity bottleneck?

Yes.

Observed relative Frobenius errors:

| Model / Layer | BCM-only |
| --- | ---: |
| BERT-base `query` (`bs=32`) | `0.9983` |
| Qwen3-0.6B `q_proj` (`bs=32`) | `0.9874` |
| Qwen3-0.6B `down_proj` (`bs=32`) | `0.9846` |
| Llama-2-7b-hf `q_proj` (`bs=32`) | `1.0813` |

Interpretation:

- On both Qwen and LLaMA, plain BCM is a weak approximator for the tested dense weights.
- On LLaMA `q_proj`, BCM-only is especially poor and even exceeds `1.0` relative Fro error in the current setting.

### 2. Does BCM + low-rank residual improve over BCM-only?

Yes, consistently in the tested runs.

Qwen3-0.6B layer 0 `q_proj`:

| Method | Setting | Relative Fro Error |
| --- | --- | ---: |
| BCM-only | `bs=32` | `0.9874` |
| BCM + LR | `bs=32, r=4` | `0.9443` |
| BCM + LR | `bs=32, r=8` | `0.9298` |
| BCM + LR | `bs=32, r=16` | `0.9126` |

Qwen3-0.6B layer 0 `down_proj`:

| Method | Setting | Relative Fro Error |
| --- | --- | ---: |
| BCM-only | `bs=32` | `0.9846` |
| BCM + LR | `bs=32, r=4` | `0.9732` |
| BCM + LR | `bs=32, r=8` | `0.9663` |

BERT-base layer 0 `query`:

| Method | Setting | Relative Fro Error |
| --- | --- | ---: |
| BCM-only | `bs=32` | `0.9983` |
| BCM + LR | `bs=32, r=4` | `0.9648` |
| BCM + LR | `bs=32, r=8` | `0.9376` |

Llama-2-7b-hf layer 0 `q_proj`:

| Method | Setting | Relative Fro Error |
| --- | --- | ---: |
| BCM-only | `bs=32` | `1.0813` |
| BCM + LR | `bs=32, r=4` | `0.9374` |
| BCM + LR | `bs=32, r=8` | `0.8727` |

Interpretation:

- The low-rank residual is clearly doing useful corrective work.
- The DRE pattern is much stronger than BCM-only even before downstream fine-tuning.

### 3. Is the BCM residual approximately low-rank?

Mixed, and model-dependent.

Qwen3-0.6B layer 0 `q_proj` residual energy after BCM-only:

- `R_8 = 0.1077`
- `R_16 = 0.1398`
- `R_32 = 0.1870`
- `R_64 = 0.2659`

Llama-2-7b-hf layer 0 `q_proj` residual energy after BCM-only:

- `R_8 = 0.3785`
- `R_16 = 0.4931`
- `R_32 = 0.6046`
- `R_64 = 0.7023`

Interpretation:

- For Qwen `q_proj`, the residual is not strongly low-rank under the current BCM fit; a small-rank correction helps, but it does not capture most residual energy.
- For LLaMA `q_proj`, the residual is much more compressible, which supports the DRE intuition more strongly.

### 4. Does DRE-BCM beat low-rank-only?

Current answer: depends on the model / layer.

Qwen3-0.6B `q_proj`:

- `lowrank_svd r=8`: `0.9438`
- `bcm_plus_lowrank bs=32 r=8`: `0.9298`

Qwen3-0.6B `down_proj`:

- `lowrank_svd r=8`: `0.9851`
- `bcm_plus_lowrank bs=32 r=8`: `0.9663`

BERT-base `query`:

- `lowrank_svd r=8`: `0.9394`
- `bcm_plus_lowrank bs=32 r=8`: `0.9376`

Llama-2-7b-hf `q_proj`:

- `lowrank_svd r=8`: `0.7362`
- `bcm_plus_lowrank bs=32 r=8`: `0.8727`

Interpretation:

- On Qwen, DRE-BCM is better than the same tested low-rank baseline.
- On BERT, DRE-BCM is marginally better than the same tested low-rank baseline.
- On LLaMA `q_proj`, the current DRE-BCM setting is still worse than low-rank-only.
- So the method is promising but not yet uniformly dominant.

### 5. Does generator differential coding reduce entropy / storage?

Yes, clearly.

Qwen3-0.6B layer 0 `q_proj`:

- Generator entropy: `6.9672`
- Delta entropy: `5.3221`
- Fraction of deltas with `|delta| < 0.01`: `96.82%`

Qwen3-0.6B layer 0 `down_proj`:

- Generator entropy: `6.9121`
- Delta entropy: `3.6473`
- Fraction of deltas with `|delta| < 0.01`: `99.52%`

Quantization error for delta coding remains small:

- Qwen `q_proj` int4 delta MSE: `2.47e-06`
- Qwen `down_proj` int4 delta MSE: `1.02e-06`

Interpretation:

- Differential coding is genuinely making the learned structural signal more compressible.
- The down-projection case is especially sparse and low-entropy.

### 6. Best tested block size and rank

For Qwen3-0.6B layer 0 `q_proj`, the best tested setting so far is:

- `block_size = 16`
- `rank = 16`
- Relative Fro error: `0.9051`

The tested trend is:

- Smaller block size helps approximation quality.
- Higher residual rank helps consistently.
- There is a clear quality / parameter trade-off.

## Current Positioning

What the current evidence already supports:

- BCM-only has a clear structure-induced bottleneck.
- Adding a low-rank residual is useful and often substantially improves fit.
- Generator differential coding is promising for storage reduction.

What is not supported yet:

- A blanket claim that DRE-BCM always beats LoRA-like low-rank baselines.
- A strong claim that BCM residuals are always low-rank.

More defensible wording:

> DRE-BCM is a hybrid structured adapter that improves over plain BCM and can outperform low-rank-only fitting on some normal-precision layers, while also exposing a promising entropy-reduction path via generator differential coding. Its effectiveness appears model- and layer-dependent, which motivates broader PEFT evaluation and better BCM optimization.

## Environment Status

User-specified environments:

- `/home/wangmeiqi/zhuoziying/env/bitnet`: available, but `torch` is CPU-only
- `/home/wangmeiqi/zhuoziying/env/bitnet-a4.8`: available, but currently lacks `torch`

Additional reusable GPU environments discovered:

- `srlm`: `torch 2.7.1+cu118`, CUDA available
- `qwen_omni`: `torch 2.2.2+cu121`, CUDA available

These GPU environments were used to finish the LLaMA matrix-fit and residual-spectrum runs faster.

## Next Steps

1. Add a PEFT training path that uses a working GPU environment while preserving the user-requested BitNet envs for extraction and analysis.
2. Add proper `max-files` / `single-weight` flags to the fitting scripts so experiments can be scheduled more cleanly.
3. Add paper-faithful FourierFT / BCA / C3A wrappers or reuse a mature PEFT implementation.
4. Run downstream SST-2 / RTE / BoolQ on a normal-precision model first, then port to BitNet.
5. Generate plots from the saved CSVs and append them to this report.

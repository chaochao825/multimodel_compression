# DRE-BCM Plan Audit

## Runtime status

- Current remote batch is finished normally; no active `fit_matrix.py`, `residual_spectrum.py`, `generator_delta_stats.py`, `train_peft.py`, or `eval_peft.py` process was found during the latest check.
- Existing `pids/*.pid` are now mostly stale bookkeeping artifacts rather than live jobs.

## Phase-by-phase audit

### Phase 0

- `done`: project scaffold exists with `src/`, `configs/`, `scripts/`, `results/`.
- `done`: requested core files for methods/modules/analysis/utils/train were created.

### Phase 1

- `done`: `circulant.py`, `block_circulant_linear.py`, `low_rank_linear.py`, `dre_bcm.py` are implemented.
- `done`: `DREBCMLinear` supports `bcm_only`, `lowrank_only`, `bcm_plus_lowrank`.
- `partial`: no dedicated automated unit-test file was added for the direct-vs-dense / fft-vs-direct checks.

### Phase 2

- `done`: generator differential coding is implemented in `BlockCirculantLinear`.
- `done`: `horizontal`, `vertical`, `2d`, and L1 regularization hooks exist in code.
- `done`: `generator_delta_stats.py` outputs entropy, summary stats, threshold sparsity, quantization MSE, and compressed-bit estimates.
- `partial`: experiments so far are almost entirely `horizontal` mode; `vertical` and `2d` are not yet benchmarked.

### Phase 3

- `done`: matrix extraction and fitting pipeline works.
- `done`: fitted methods include `lowrank_svd`, `bcm_only`, `bcm_plus_lowrank`, `multi_bcm`, `bcm_plus_sparse_delta`, `generator_delta_bcm`.
- `done`: reused existing checkpoints on the server, including normal-precision models and BitNet weights.
- `partial`: the full planned sweep over all listed models/layers/parameter grids is not finished.

### Phase 4

- `done`: residual spectrum script exists and was run for BERT, LLaMA, and Qwen-family layers.
- `partial`: the original phase description expected per-layer PNG generation directly from `residual_spectrum.py`; the script itself still writes CSV only.
- `done`: a separate overview plotting script now produces `residual_spectrum_by_layer.png`.

### Phase 5

- `missing`: downstream PEFT training is not yet implemented end-to-end.
- `partial`: `train_peft.py` and `eval_peft.py` are scaffolds only.
- `missing`: SST-2 / RTE / BoolQ runs have not started.
- `missing`: fair-budget settings A/B/C are not wired into a true training/eval loop.

### Phase 5 baselines

- `done (matrix-fit only)`: `lowrank_svd` acts as a LoRA-like low-rank approximation baseline.
- `partial`: `fourierft_wrapper.py` exists, but no downstream PEFT experiment has used it yet.
- `partial`: `c3a_wrapper.py` is only a proxy based on `MultiBCMLinear`, not a paper-faithful implementation.
- `missing`: no explicit `bca_wrapper.py` implementation yet.
- `missing`: AdaLoRA / VeRA / DoRA / IA3 / BitFit / full finetune / linear probe are not implemented yet.

### Phase 6

- `partial`: block-size and rank ablations exist at matrix-fit level and are now visualized.
- `missing`: no PEFT-accuracy ablations yet.
- `missing`: `multi_bcm` has code support but has not been seriously benchmarked.
- `missing`: `generator delta` ablations beyond `horizontal` are absent.

### Phase 7

- `done`: `param_flop_counter.py` provides a latency/storage proxy.
- `partial`: current latency numbers are proxy-only.
- `missing`: no FFT-vs-direct wall-clock benchmark, no microbenchmark, no real BitNet inference baseline comparison yet.

### Phase 8

- `done`: overview plots now exist for:
  - `matrix_fit_error_vs_params`
  - `matrix_fit_error_vs_bits`
  - `residual_spectrum_by_layer`
  - `block_size_ablation`
  - `rank_ablation`
  - `generator_delta_entropy`
  - `method_latency_proxy`
- `done`: raw CSV was saved beside each plot.
- `missing`: `peft_accuracy_vs_trainable_params` and `peft_accuracy_vs_estimated_bits` are absent because PEFT results do not exist yet.

### Phase 9

- `partial`: `results/report.md` exists and already answers several core questions.
- `missing`: the report cannot yet support claims against LoRA / BCA / FourierFT / C3A on downstream PEFT tasks.
- `missing`: the "adapter vs pure compression" conclusion is still preliminary because current evidence is mostly matrix-fit.

## Current result interpretation

- `strongly supported`: BCM-only has a clear expressivity bottleneck.
- `strongly supported`: BCM + low-rank consistently improves over BCM-only.
- `mixed`: DRE-BCM beats the tested low-rank baseline on BERT, BitNet q-proj/down-proj, and earlier Qwen3 runs, but loses on LLaMA, Mistral, Qwen2.5 q-proj, and Qwen3 down-proj retry.
- `mixed`: BCM residual low-rankness is model/layer dependent; LLaMA looks much more compressible than BERT/Qwen in the current spectra.
- `promising but not stable`: generator differential coding can lower entropy a lot, but standalone `generator_delta_bcm` fitting quality is currently weak and should not yet be treated as a strong approximation baseline.

## Most important missing pieces

1. Replace the PEFT training/eval scaffolds with a runnable downstream fine-tuning path.
2. Add a real BCA baseline and decide whether FourierFT / C3A wrappers will be paper-faithful or explicitly labeled as proxies.
3. Run the MVP downstream task set: `RoBERTa-base` or `GPT-2` on `SST-2`, `RTE`, and `BoolQ`.
4. Separate "generator delta as storage coding" from "generator_delta_bcm as standalone fitting method" in the writeup and experiments.

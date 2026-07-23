# Low-Rank + Static Row-Block Sparse + Cache-Aware Refresh

## Scope

This experiment tests a training-free residual path inside the existing Wan
1.3B and World Foundry H200 workflow. It deliberately excludes BCM from the
runtime path because the previous H200 measurements showed that input FFT,
complex contraction, inverse FFT, and launch overhead dominate its cost.

The tested linear approximation is

```text
W_hat = Q_fp8(W) + U V^T + P_Omega S,
```

where `U V^T` is a rank-8 approximation of the FP8 weight error and
`P_Omega S` stores fixed contiguous output-row blocks from the remaining
error. The total high-precision value budget is matched to rank-16 low rank.
World Foundry TeaCache controls full-model refreshes. A cache hit bypasses the
entire stack, including the low-rank and sparse branches.

## What Is Theoretically Complete

Let `E = W - Q_fp8(W)` and let `L_r` be its best rank-r approximation. For a
fixed `L_r` and a mask containing `k` row blocks, selecting the blocks with the
largest residual Frobenius energy minimizes

```text
||E - L_r - P_Omega(E - L_r)||_F.
```

This makes the implemented row-block selection optimal for the sequential
Frobenius subproblem. It does not make the finite-budget model universally
complete. Exact representation of every matrix requires either full rank or a
row mask covering every output row. Joint low-rank plus sparse optimization can
also differ from the sequential solution.

For a residual operator `R(x) = Delta_W x`, reusing a result from input `x_s`
at `x_t` obeys

```text
||R(x_t) - R(x_s)|| <= ||Delta_W||_2 ||x_t - x_s||.
```

For the full denoiser, the same idea holds with a local Lipschitz constant.
TeaCache's embedding-distance polynomial is an empirical proxy for this bound,
not a certificate. Dense warmup, bounded refresh intervals, boundary refresh,
and paired quality measurements remain necessary.

The combined local error is bounded schematically by

```text
quantization_error
+ residual_representation_error
+ cache_hit * residual_operator_norm * input_drift.
```

Across denoising steps these errors can be amplified by sampler Jacobians, so
single-layer matrix error alone is not a complete quality argument.

## Evidence Reused

- Nine Wan weights: pure rank-16 low rank had held-out activation error
  `0.48305`; matched-budget BCM b128 + rank-10 had `0.48619`.
- FP8 main error was already small: mean activation relative L2 `0.02701`;
  rank-16 reduced it to `0.02638`, while rank-8 + row sparse reached `0.02645`.
- H200 q projection at 32,760 rows: BF16 was `0.2194 ms`; dynamic FP8 main was
  `0.9311 ms`; static-input FP8 was a `0.1366 ms` lower bound. Activation
  quantization and launch fusion are therefore first-order issues.
- Cached generator FFT changed BCM latency by only approximately 0-5%.
- F81 TeaCache threshold 0.08 cached 2 of 40 model forwards, gave mean frame
  SSIM `0.9621` and mean PSNR `35.66 dB` over eight videos, but only about
  `1.039x` speedup.
- F17 threshold 0.08 cached 8 of 40 forwards and gave SSIM `0.8697`; threshold
  0.09 dropped to `0.5863`. The cache threshold is a sharp quality boundary.
- Naive NFE reduction was not acceptable: step 12 versus step 20 had frame
  SSIM about `0.314`.

## Selected Integration

The first end-to-end stage replaces `self_attn.q` and `self_attn.o` in all 30
Wan blocks. Every replacement supports runtime switching among:

1. `dense`: original BF16 linear.
2. `fp8`: World Foundry tensorwise FP8 weight plus calibrated static input
   scale.
3. `hybrid`: FP8 main plus rank-8 BF16 residual and fixed 8-row blocks, within
   the rank-16 value budget.

The paired methods are `dense`, `dense_cache008`, `fp8`, `hybrid`, and
`hybrid_cache008`. FA3 BF16 self-attention and dense SDPA cross-attention are
kept fixed. This isolates the linear residual and cache effects.

Stage 1 uses four prompts, two seeds, 17 frames, 20 sampling steps, and all five
methods. Stage 2 uses 81 frames only after Stage 1 passes.

## Acceptance Gates

- Correctness: every run succeeds, outputs are finite, and all compared pairs
  use the same prompt, seed, sampler, dimensions, and checkpoint.
- Hybrid representation: non-cache hybrid mean frame SSIM versus dense at
  least `0.98` and mean PSNR at least `30 dB`.
- F17 cache screen: hybrid cache SSIM at least `0.85`, no more than `0.02`
  below dense cache, and no failed pairs.
- Runtime: hybrid without cache no more than 15% slower than dense; hybrid
  cache must improve over hybrid without cache. A speedup claim versus dense
  requires geometric-mean paired speedup above `1.0`.
- F81 confirmation: hybrid cache mean frame SSIM at least `0.95` and paired
  speedup reported with confidence intervals or all individual pairs.

Passing correctness but failing runtime is still a useful result: it proves
the integration and rejects the current eager FP8/residual kernel path. Passing
runtime but failing quality rejects the refresh threshold. Only passing both
supports a no-distillation acceleration claim.

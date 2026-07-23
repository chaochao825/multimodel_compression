# RobuQ, Structured Residuals, and World Foundry Follow-up

This report reuses the completed World Foundry and H200 records under
`remote_snapshot/` and the remote workspace
`/home/wangmeiqi/codex_runs/robuq_structured_probe_20260723`. No completed
20-step video generation was repeated.

The paper and public implementation are [RobuQ on arXiv](https://arxiv.org/abs/2509.23582)
and [the RobuQ repository](https://github.com/racoonykc/RobuQ).

## 1. Method comparison

RobuQ uses a quantized DiT weight branch plus a high-precision low-rank SVD
correction initialized from the transformed quantization residual. Our current
candidate is a block-circulant residual plus a low-rank residual, optionally
followed by sparse correction:

```text
W_hat = W_quant + Delta_W_BCM + U V^T + Delta_W_sparse
```

The probe uses nine Wan 1.3B weights at early, middle, and late blocks. It is a
post-training representation and latency probe, not a final FID or video-quality
comparison. The held-out activation metric is relative output L2.

| method | held-out activation error | relative change vs ternary | budget |
| --- | ---: | ---: | ---: |
| ternary main | 0.51093 | baseline | 0 |
| Frobenius low-rank r16 | 0.48305 | 5.46% lower | 1.00x |
| Frobenius BCM b128 + LR r10 | 0.48619 | 4.84% lower | 1.00x |
| activation-aware low-rank r16 | 0.50083 | 1.98% lower | 1.00x |
| activation-aware BCM b128 + LR r10 | 0.50699 | 0.77% lower | 1.00x |
| activation LR r8 + row sparse | 0.50093 | 1.96% lower | 0.98x |

The direct conclusion is that RobuQ-style dense low-rank correction is the
stronger current baseline for correcting arbitrary quantization error. At the
same residual parameter budget, BCM plus low rank is close but does not improve
over pure SVD low rank. The earlier alternating probe is consistent with this:
after three updates, BCM, row-sparse, and tile-sparse candidates all lost to
pure rank-16 SVD on all nine weights.

BCM is not useless, but its present role is different. The Frobenius BCM energy
capture is only 1.56% at block size 64, 0.78% at block size 128, and 0.39% at
block size 256. The activation-aware fit captures 6.18% and 3.11% on the fit
split for block sizes 64 and 128, but the held-out captures are -3.10% and
-1.51%. This is a generalization failure on the current calibration size, not
evidence for a useful inference branch.

The v1 activation-aware run overfit badly: held-out error rose to 14 to 23.
The v2 relative-ridge and covariance-SVD implementation reduced this to about
0.50. This correction is important, but it still does not beat ordinary SVD.

## 2. H200 evidence

The new benchmark uses an H200 NVL, a 1536 by 1536 Wan q projection, and the
largest recorded activation batch of 32,760 rows. Dynamic rows include input
quantization. Static-input rows are a lower bound because quantization is done
before the timed region.

| path | latency | speedup vs BF16 | output error | scope |
| --- | ---: | ---: | ---: | --- |
| BF16 dense | 0.2194 ms | 1.00x | 0 | measured |
| FP8 dynamic main | 0.9311 ms | 0.236x | 0.03931 | measured |
| FP8 static-input main | 0.1366 ms | 1.607x | 0.03931 | lower bound |
| FP8 dynamic + LR16 | 1.1015 ms | 0.199x | 0.03851 | measured |
| FP8 static-input + LR16 | 0.2971 ms | 0.739x | 0.03851 | lower bound |
| FP8 dynamic + cached BCM b128 + LR10 | 2.6756 ms | 0.082x | 0.03870 | measured |
| FP8 static-input + cached BCM b128 + LR10 | 1.8842 ms | 0.116x | 0.03870 | lower bound |
| FP8 dynamic + LR8 row sparse | 1.2594 ms | 0.174x | 0.03866 | measured |
| FP8 static-input + LR8 row sparse | 0.4647 ms | 0.472x | 0.03866 | lower bound |

Precomputing the generator FFT changed BCM latency by approximately 0 to 5%
across 512 to 32,760 rows. Therefore the dominant cost is the input FFT,
complex contraction, inverse FFT, and extra kernel launches. The current eager
BCM path is not GPU-competitive, even though its storage count is small.

The row-sparse branch is materially cheaper than BCM in this probe, but it is
still not faster than BF16 once the full residual path is included. It should
be implemented as a contiguous selected-row or grouped-GEMM kernel rather than
general `index_add_` if it is promoted to a runtime path.

## 3. World Foundry bottlenecks

The existing one-prompt, one-seed 20-step record has 1,200 self-attention and
1,200 cross-attention calls. Its end-to-end times are 71.46 s for SDPA, 58.35 s
for FA3 BF16, 53.48 s for the every-two-step hybrid FP8 schedule, and 47.62 s
for full FP8. This makes attention backend, NFE, and repeated model calls the
first-order system levers. A standalone residual GEMM is a second-order lever
until it is fused into the full pipeline.

The existing NFE sweep shows why naive step reduction is unsafe: relative to a
20-step reference, step 12 has frame SSIM 0.314 and step 4 has frame SSIM
0.214. The existing eight-video TeaCache pilot is much more promising for a
no-distillation route: mean frame SSIM is 0.9621 and mean PSNR is 35.66 dB,
but measured speedup is only 1.039x with a cached-forward fraction of 0.05.
This is a good quality-preserving cache signal, not yet a turbo result.

The World Foundry scorecard in the snapshot is an existing-results artifact and
does not pass the official/full-suite quality gate. The numbers above should be
used for engineering direction, not as a final benchmark claim.

## 4. Recommended hybrid

The current all-branches-at-once design spends compute redundantly. The next
version should use a static, offline-calibrated router:

```text
Y = FP8_main(X)
  + refresh_flag * LowRankResidual(X)
  + refresh_flag * RowBlockResidual(X)
  + optional_selected_BCM(X)
```

Recommended policy:

1. Fit low rank first because it is the best current approximation per value
   budget for the measured quantization residual.
2. Add only a small number of contiguous output-row or output-tile blocks when
   held-out activation sensitivity justifies them.
3. Keep BCM only for layers or blocks whose held-out capture exceeds a measured
   threshold. Current b128 results do not meet that threshold.
4. Use a static mask and grouped GEMM. Avoid per-token top-k routing, scatter,
   and `index_add_` in the hot path.
5. Fuse activation quantization, scale calculation, and FP8 scaled GEMM. The
   H200 static-input lower bound shows that dynamic quantization overhead is
   large.
6. Refresh residual branches only on cache misses. Reuse the residual output
   on steps where the World Foundry adaptive cache accepts the approximation.

This keeps the structured branch available for storage or ASIC exploration while
making the default GPU path low-rank plus static row/tile sparse. A useful
parameterization is:

```text
Delta_W = U V^T + M_rows * S_rows + M_bcm * G_BCM
```

where `M_rows` and `M_bcm` are static masks learned from held-out activation
energy. The masks should be selected per layer and refresh schedule, not per
token.

## 5. Can this become no-distillation turbo diffusion?

Yes as a systems direction, but the current evidence supports a staged
inference accelerator rather than a claim of a new turbo model. The lowest-risk
path is:

1. Integrate adaptive residual caching into the existing World Foundry cache
   path and measure 8 to 16 prompts over multiple seeds.
2. Compare fixed refresh intervals, cache-threshold refresh, and late-step
   refresh at fixed 20 NFE.
3. Add static FP8 scales and fused quantization before changing the weight
   representation.
4. Insert low-rank plus static row-block residuals only at layers selected by
   activation sensitivity.
5. Measure end-to-end seconds, cached forward fraction, frame SSIM, PSNR,
   temporal delta MAE, peak memory, and kernel launch counts.
6. Only after the cache path is stable, test whether selected BCM blocks help.

This can produce a no-distillation or low-cost-adaptation accelerator. It cannot
currently be described as turbo diffusion based on BCM: the existing NFE sweep
rejects naive step dropping, and the H200 BCM branch is slower than BF16.

## 6. Files and remaining gaps

New probe and benchmark code:

- `scripts/probe_activation_aware.py`
- `scripts/benchmark_h200.py`
- `scripts/plot_followup.py`

New results:

- `results/activation_aware_v1/` records the original overfit calibration.
- `results/activation_aware_v2/` records the stabilized held-out probe.
- `results/h200_bcm_cached_v1/` records the H200 cached-FFT benchmark.
- `figures/followup_v1/` contains PNG figures and the CSV used by each figure.

Not yet demonstrated: final Wan video quality with the proposed residual router,
paper-faithful BCA/FourierFT/C3A comparison, fused BCM CUDA performance, and a
multi-prompt end-to-end turbo scorecard. These are the next required gates
before making a method-level superiority claim.

# Causal BCM/BCCB Probe on LongLive-1.3B

Status: frozen on 2026-08-05 before any BCCB evaluation of the existing
LongLive captures. The captures predate this protocol, so this is a frozen
post-capture representation study rather than a capture-level preregistration.

## Question

Test whether content-generated spatial BCCB kernels, causal temporal sharing,
query-tile modulation, nonperiodic Toeplitz boundaries, exact event residuals,
and a low-rank output tail can approximate LongLive causal attention at a cost
that could plausibly exceed 1.5x arithmetic reduction.

## Structured operator

For query frame `t`, key frame `s`, and spatial displacement `delta`, the
record-generated BCCB kernel is the least-squares projection of captured QK
logits onto displacement buckets:

\[
g_{t,s,h}(\delta)=
\frac{1}{|\mathcal P|\sqrt d}
\sum_{p\in\mathcal P}q_{t,p,h}^{\top}k_{s,p+\delta,h}.
\]

`global` uses all four captured query tiles as `P`; `capture_tiles` fits one
kernel per 64-query tile. Periodic BCCB uses modulo spatial displacement.
Nonperiodic Toeplitz uses signed displacement and never wraps an image edge.
Relative-time BCCB additionally shares kernels between frame pairs with equal
absolute frame offset. Time itself is never circular.

The hybrid keeps three sink and three recent frames exact. Event tokens from
the remaining six frames overwrite structured logits with true QK logits.
All exact and structured logits enter one shared softmax, followed by the
original dense V. This is a representation/cost screen; a passing candidate
would still require an FFT/sparse fused numerator and denominator kernel.

## Important distinction from earlier negative results

Earlier Wan probes fitted calibration-frozen offset tables. The dynamic BCCB
here is generated from each record's Q/K, following the key idea of
[Circulant Attention](https://arxiv.org/abs/2512.21542). It can therefore
change its spectrum with content. Query-tile kernels further relax global
stationarity. The matrix remains spatially structured, but it is not a fixed
Fourier table shared by every sample.

This does not test weight-space BCA or C3A. Those are trained PEFT methods,
whereas this protocol tests attention/KV inference structure.

## Data and leakage

- Reuse all 96 manifest-validated captures from
  `ar-video-residual-memory-longlive-v1`.
- Calibration: four fixed prompt/seed records. Validation and test each contain
  one prompt with two seeds.
- Current Q/K may generate a dynamic kernel at runtime. Dense output may not
  select a kernel, support, event budget, temporal sharing rule, or rank.
- Frozen BCCB kernels and output bases use calibration records only.
- Because only four query tiles were captured, dynamic projection is an
  optimistic sampled-query screen. A pass requires full-grid Q recapture.

## Gates

- Capture parity: aggregate and worst dense-reference error below 0.5%.
- Adaptive rank-16: aggregate <=0.5%, worst <=1%.
- Calibration-frozen output basis: aggregate <=1%, worst <=2%.
- Primary arithmetic FFT proxy >=1.5x.
- No measured speed claim without a full-grid fused H200 kernel reaching at
  least 1.5x after routing and fallback.

The primary candidate was frozen as query-tile, frame-pair BCCB with exact
sink/recent frames, 5% train-free event tokens, and rank-16 diagnostics.

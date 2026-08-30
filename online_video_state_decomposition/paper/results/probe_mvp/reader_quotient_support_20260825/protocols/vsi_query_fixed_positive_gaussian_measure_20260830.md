# VSI Query-Fixed Positive Gaussian Measure Gate

Date frozen: 2026-08-30

Role: exposed calibration-only function-class and active-read diagnostic

## Decision question

Can a regular video K/V block be represented by a positive, query-conditioned
Gaussian measure state, while a bounded number of exact blocks repairs the
remaining heavy-tail innovation?

For one attention head and one block, the compact state stores

\[
(\mu_k,\mu_v,P_r,\lambda_r,C_{vk}P_r).
\]

For the current scaled query \(\bar q\), it evaluates

\[
\log \widehat Z_g
=\log |g|+\bar q^\top\mu_k
+\tfrac12\sum_{i=1}^r\lambda_i(P_i^\top\bar q)^2,
\]

\[
\widehat y_g
=\mu_v+(C_{vk}P_r)(P_r^\top\bar q),
\qquad
\widehat N_g=\widehat Z_g\widehat y_g.
\]

Exact and compact groups share one numerator and denominator. Retrieved groups
replace, rather than add to, their compact contribution.
Exact and non-visual measures are already stabilized by the dense global-logit
maximum. The implementation therefore chooses any secondary online-softmax
shift only from currently active compact groups; inactive compact groups cannot
rescale an all-exact result, whose secondary shift is exactly zero.

## Frozen scope

- LLaVA-OneVision Qwen2-7B, eager attention replay.
- Calibration positions 73--96 only; selection and formal roles stay unread.
- Final generated-token query at language layers 0, 13, and 27.
- Eight frames with the native `8 x 14 x 14` visual-token geometry.
- Two regular group families:
  - `spatial_7x7`: 32 groups of 49 tokens;
  - `temporal2_spatial_7x7`: 16 groups of 98 tokens.
- Gaussian ranks `0, 2, 4, 8, 16`.
- Exact fractions `0, 12.5%, 25%, 50%, 100%`, applied independently per
  attention head.
- Deployable selector: compact predicted group mass.
- Capacity selector: target-visible local output defect. It is an oracle and
  cannot support a runtime claim.

The active-read ratio counts all compact moments plus exact K/V leaves read for
the current query. Cold exact storage is retained and is not claimed to be
compressed. FLOP estimates and active-read ratios are proxies, not latency.

## Guards

- Captured Q/K/V must reproduce the registered native-BF16 eager attention
  projection input with relative error at most `1e-4`. The subsequent measure
  reference promotes those captured tensors to FP32 and is not compared
  numerically to the BF16 AV matmul.
- Every topology must partition all 1,568 visual tokens exactly once.
- Compact group masses must be finite and non-negative.
- Full exact replacement must reproduce the dense FP32 visual/full measure
  reference.
- No benchmark answer, teacher margin, selection example, or formal example may
  choose rank, topology, support, or threshold.

## Decision rule

Only candidates with exact fraction at most `25%` and conservative active-read
ratio at least `2x` are eligible.

`POSITIVE_GAUSSIAN_COMPACT_PATH` requires a compact-mass candidate with:

- visual relative L2 mean at most `1%`;
- visual P95 at most `2%`;
- visual worst cell at most `5%`;
- full-output mean/P95 at most `0.5%/1%`.

`POSITIVE_GAUSSIAN_CAPACITY_ONLY` requires an oracle-local candidate with the
stricter visual mean/P95/worst thresholds `0.5%/1%/2%` and full-output
mean/P95 thresholds `0.25%/0.5%`.

If neither condition holds, the result is
`NO_POSITIVE_GAUSSIAN_MEASURE_PATH`. Do not train a selector or read a hidden
role after a null. A capacity-only result may justify a separate, freshly
registered compact-selector Gate; it does not authorize selection/formal use.

## Claim boundary

This Gate tests one fixed-query, single-layer attention function class and its
active-read arithmetic on exposed calibration examples. It is not a complete
reader, long-video task, TTFT, memory-capacity, kernel, or deployment result.

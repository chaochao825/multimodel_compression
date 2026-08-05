# Causal Butterfly-Lifting Memory Screen

Date: 2026-08-05

Status: frozen exploratory gate on development-exposed LongLive captures.

## Decision question

Can a causal, invertible multiresolution transform preserve local detail and
long-range history more faithfully than temporal means or BCCB attention while
using at most two thirds of the original K/V storage?

The candidate is deliberately not another sparse-attention mask. It transforms
the stored K/V history and then evaluates the original dense softmax attention.

## Recovered negative evidence

- Dynamic query-tile BCCB plus adaptive rank-16 reached only `16.12%/41.06%`
  aggregate/worst held-out error. More displacement capacity cannot repair the
  shared-eigenvector mismatch.
- A nine-stage token-Butterfly product reached about `57.8%` test error because
  repeated local softmax factors change one-shot attention into path diffusion.
- Fixed temporal summaries plus adaptive rank-16 reached `11.46%/34.13%`; a
  temporal mean is not a sufficient statistic for query-dependent softmax.
- Exact-support residual-width routing improved the strongest mass/value oracle
  by only `3.15%` relative at 10% budget. Selecting more omitted tokens does not
  create a stable low-rank tail.

These results close fixed BCCB output, repeated-softmax Butterfly, and pure
support selection. They do not test an invertible predictive transform of K/V.

## Candidate

For adjacent historical states, use a lifting step

\[
d=x_1-P_\delta x_0,\qquad
c=x_0+\tfrac12P_\delta^{-1}d,
\]

with exact inverse

\[
\hat x_0=c-\tfrac12P_\delta^{-1}\hat d,\qquad
\hat x_1=P_\delta\hat x_0+\hat d.
\]

`P_delta` is an invertible 2D cyclic shift selected only from current K/V
prediction energy. Dyadic recursion gives one low-temporal-rank coarse map and
five detail maps for the six middle frames. Three sink and three recent frames
remain exact. Only the highest-energy regular 64-token detail tiles are stored;
they form a sparse high-rank event branch. K is optionally transformed before
3D RoPE and restored at its original frame positions after decoding.

The cyclic operator is a predictor, not an attention matrix. Wrap-around errors
remain in `d`, so dropping a detail is an explicit approximation rather than an
implicit periodic semantic assumption. Reconstructed tokens enter the original
single softmax; there is no branch-normalization ambiguity.

## Frozen comparisons

- identity lifting in canonical pre-RoPE K space;
- shared-shift lifting with one shift per merge;
- per-head shifts as a payload/capacity control;
- post-RoPE shared shifts as a RoPE-domain ablation;
- 5%, 10%, and 20% retained detail-tile fractions where registered.

The primary candidate is canonical shared-shift lifting with 20% detail. The
same-budget identity transform is the mechanism baseline.

## Gates

- Dense-reference parity: at most `0.5%`.
- Cache compression: at least `1.5x`, including sparse indices and shift IDs.
- Direct reconstructed output: aggregate at most `1%`, worst head at most `2%`.
- Adaptive rank-16 capacity: aggregate at most `0.5%`, worst head at most `1%`.
- Shared shift must improve adaptive aggregate error over identity by at least
  10% relative to justify the circulant component.

An adaptive-only pass is a boundary result and permits only a bounded
content-generated residual study. Missing adaptive capacity is a valid null and
stops predictor, kernel, and rollout work. This screen makes no attention-speed
claim because it reconstructs all tokens before dense replay.

## Scope and leakage

The screen uses the same eight registered hard captures as the residual-width
study: four prompt classes and two cells (`layer14/frame15/call1` and
`layer29/frame18/call1`). K/V may select shifts and detail energy at write time.
Dense attention, dense output, and adaptive SVD are evaluator-only. Existing
captures have been repeatedly explored, so any positive result requires new
prompt/seed confirmation before updating a paper claim.

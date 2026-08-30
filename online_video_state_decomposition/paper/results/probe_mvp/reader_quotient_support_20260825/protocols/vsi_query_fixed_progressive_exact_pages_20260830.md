# Query-fixed progressive exact-page Gate

Date: 2026-08-30
Role: exposed calibration diagnostic only

## Decision question

After the positive Gaussian measure closure fails, can compact page metadata be
used only to rank or certify exact K/V leaves, rather than directly entering the
attention numerator and denominator?

For page `g`, the Quest-style coordinate box bound is

\[
u_g(q)=\sum_i
\begin{cases}
q_i k^{\max}_{g,i}, & q_i\ge0,\\
q_i k^{\min}_{g,i}, & q_i<0,
\end{cases}
\]

so that every member score is at most `u_g(q)` and

\[
Z_g(q)\le m_g\exp(u_g(q)).
\]

Selected pages are read exactly and share one numerator and denominator.
Unselected visual pages contribute nothing to the output; their upper mass is
used only as a diagnostic fallback certificate. This Gate does not fit or emit
an approximate bulk value.

## Frozen scope

- LLaVA-OneVision Qwen2-7B with eager attention replay.
- Calibration positions 73--96 only; selection and formal roles stay unread.
- Final generated-token query at language layers 0, 13, and 27.
- Eight frames with native `8 x 14 x 14` visual-token geometry.
- Page topologies: per-frame `7 x 7`, and two-frame `2 x 7 x 7`.
- Exact page fractions: `12.5%, 25%, 50%, 62.5%, 75%, 100%`.
- Deployment baselines: query/centroid score and Quest min/max box bound.
- Capacity ceilings: exact page mass and target-visible local output effect.

## Cost accounting

The active-read proxy charges exact K/V leaves plus one mean-key vector per page
for the centroid path or min/max key vectors per page for the Quest path. Oracle
selectors are charged as dense and cannot support a runtime claim. A separate
leaf-only ratio records their capacity ceiling. Cold exact storage is retained.

## Guards

- Captured Q/K/V replay error is at most `1e-4`.
- Every page family partitions all 1,568 visual tokens exactly once.
- The Quest page mass upper bound covers every exact page mass.
- The induced output error certificate covers every measured per-head error.
- Full exact pages reproduce the dense FP32 visual/full measure reference.
- No selection/formal example, answer, or reader margin chooses a selector or
  fraction.

## Decision rule

Only fractions at most `25%` are eligible. A deployable selector also requires
active-read ratio at least `2x` and visual mean/P95/worst at most `1%/2%/5%`,
with full mean/P95 at most `0.5%/1%`.

An oracle capacity pass uses the stricter visual `0.5%/1%/2%` and full
`0.25%/0.5%` thresholds, with leaf-only ratio at least `2x`.

The Gate can return a deployable path, capacity-only path, or a narrow no-go for
these page families. It cannot establish reader accuracy, TTFT, or latency.

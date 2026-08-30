# VSI query-fixed visual-measure remainder Gate

Date: 2026-08-30
Status: frozen before execution

## Decision question

After the frozen-reader topology/PPE line closes, does a single Qwen2 attention
layer with a fixed current query expose a genuinely compressible and
certifiable visual-memory measure?

This Gate tests the mathematical interface required by a future external
cross-attention memory. It does not replace the current self-attention stack,
train a model, evaluate final reader logits, measure latency, or claim an
end-to-end method.

## Data and identity

- Require valid M0, M1, true-2x2, and PPE summaries, ending in
  `NO_PPE_HEADROOM`.
- Use only already exposed VSI calibration positions 73--96, all 24 samples.
- Keep positions 97--120, selection, and formal endpoints unread.
- Freeze OneVision, eight dense frames, original 1,568 visual tokens, eager
  attention, and standard scalar Qwen2 RoPE.
- Inspect language layers `0`, `13`, and `27`, all 28 heads, and only the final
  prompt-token query.
- Reconstruct each selected attention output from captured post-RoPE Q/K and V;
  relative error after the layer output projection must be at most `1e-4`, or
  the run is invalid.

## Fixed-query measure

For one head and fixed query `q`, the dense visual memory is

\[
Z=\sum_j e^{s_j},\qquad
N=\sum_j e^{s_j}v_j,\qquad
Y=N/Z,\qquad
s_j=q^\top k_j/\sqrt d.
\]

Group each frame into true non-overlapping `2x2` nodes. A coarse node stores
mass four, post-RoPE key centroid, and value centroid:

\[
\hat Z_g=4e^{\bar s_g},\qquad
\hat N_g=\hat Z_g\bar v_g.
\]

An exact split replaces `hat Z_g, hat N_g` by the four exact leaf
contributions. Budgets are `0,49,98,147,196,392` split nodes, corresponding to
visual token retention `25%,34.375%,43.75%,53.125%,62.5%,100%`.

Evaluate both visual-only normalized output and the full final-query attention
output with all non-visual tokens kept exact.

## Local analytic remainder

For each head/node define

\[
\delta_g=\max_{j\in g}|s_j-\bar s_g|,\qquad
r^v_g=\max_{j\in g}\|v_j-\bar v_g\|_2,
\]

\[
\epsilon^Z_g=\hat Z_g(e^{\delta_g}-1),
\]

\[
\epsilon^N_g=
4e^{\bar s_g+\delta_g}r^v_g
+\|\bar v_g\|_2\epsilon^Z_g.
\]

These bound the local denominator and numerator defects. The implementation
must also compute exact local defects to separate approximation capacity from
analytic-bound looseness. A constant all-coarse denominator floor and numerator
upper envelope are used to form non-negative per-node output bounds. Removing
an exact node must make the registered aggregate certificate non-increasing by
construction; any empirical violation invalidates the implementation.

## Equal-budget selectors

All selectors share one node set across heads within a sample/layer:

1. `analytic_remainder`: greedy reduction of the aggregate analytic
   certificate;
2. `attention_mass`: descending exact visual attention mass, a target-visible
   but standard importance baseline;
3. `exact_local_oracle`: descending first-order local output defect using the
   dense query/normalizer, a non-deployable capacity ceiling;
4. `fixed_random`: registered seed `20260830`.

No selector may use reader logits, answers, unexposed samples, or later-layer
states.

## Frozen endpoints and outcomes

Primary errors are aggregate relative L2 over all heads for the visual-only
output. Also record worst-head error, full-attention relative L2, certificate
value/tightness, finite-certificate rate, and every non-monotone actual-error
transition.

At 196 exact nodes, `QUERY_FIXED_CERTIFIED_HEADROOM` requires:

- analytic certificates finite for at least 95% of heads, with zero empirical
  bound violations and zero certificate-increase transitions;
- `analytic_remainder` visual-only mean/P95 relative error at most `1%/2%` and
  worst sample-layer error at most `5%`;
- full-attention mean/P95 relative error at most `0.5%/1%`;
- mean visual-only error at least 10% below `attention_mass` at two or more of
  budgets 98, 147, and 196.

`QUERY_FIXED_CAPACITY_BOUND_LOOSE` applies when `exact_local_oracle` satisfies
the same actual-error thresholds but the analytic candidate fails a
certificate or accuracy condition.

`NO_QUERY_FIXED_MEASURE_HEADROOM` applies when the exact-local oracle fails the
actual-error thresholds. `INVALID_QUERY_FIXED_MEASURE_GATE` applies to capture,
reconstruction, identity, or certificate-validity implementation failures.

A certified result authorizes a query-fixed external-memory prototype with
actual reader-risk calibration. A bound-loose result authorizes only improving
the remainder estimator. A null parks train-free hierarchical measure memory
and leaves low-cost path-consistency adaptation as the remaining route.

## Cost and stop rule

- One isolated A800 on server 210, at most 30 GPU-minutes.
- One implementation repair is allowed.
- Stop after one valid outcome or after the repair allowance is exhausted.

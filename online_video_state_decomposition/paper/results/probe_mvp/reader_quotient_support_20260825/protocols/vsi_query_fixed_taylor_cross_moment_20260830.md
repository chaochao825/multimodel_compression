# VSI query-fixed Taylor cross-moment capacity Gate

Date: 2026-08-30
Status: frozen before engineering smoke

## Decision question

At the same per-head target-visible support and exact-node budget as the closed
headwise-support ceiling, can a low-order query-conditioned Taylor moment state
remove the residual error left by a true-`2x2` centroid?

This Gate tests representation capacity and mechanism only. Computing moments
from all four leaves is permitted here and is charged as non-deployable oracle
information. No speed or compressed-state claim is allowed.

## Frozen identity and boundary

- Same OneVision checkpoint, VSI split, exposed positions 73--96, layers
  `0/13/27`, final-token query, 28 heads, eight frames, 1,568 visual tokens,
  392 true-`2x2` groups, and budgets `0/49/98/147/196/392`.
- Positions 97--120, selection, formal, reader logits, and answer correctness
  remain inaccessible.
- `taylor_order0` must reproduce every previous `headwise_exact_local`
  sample-layer-budget row before the Gate is valid.

## Registered states

For member score residual `a_j = s_j - mean_j(s_j)`, use

`p_P(a) = sum_{n=0}^P a^n / n!`, for `P in {0,1,2,3}`.

The coarse group measure is

`Z_g^P = exp(s_bar) sum_j p_P(a_j)` and
`N_g^P = exp(s_bar) sum_j p_P(a_j) v_j`.

Each order independently uses the same target-visible per-head exact-local
score to select exact groups at each registered budget. This isolates state
order from shared-support error. `P=0` is the previous centroid state;
`P=1` contains the `Sigma_vk q` cross term; higher orders add score variance
and score-value moments.

## Endpoint and outcomes

Primary endpoint at `k=196` is visual-output relative L2 across 72 cells.
Secondary endpoint is full-attention relative L2. A state passes when:

- visual mean `<=1%`, P95 `<=2%`, worst cell `<=5%`;
- full mean `<=0.5%`, P95 `<=1%`.

Classify by the lowest passing order:

- `TAYLOR_CROSS_MOMENT_ORDER1_PASS`;
- `TAYLOR_CROSS_MOMENT_ORDER2_PASS`;
- `TAYLOR_CROSS_MOMENT_ORDER3_ONLY`;
- `NO_TAYLOR_CROSS_MOMENT_CAPACITY`;
- `INVALID_TAYLOR_CROSS_MOMENT_GATE` for identity or implementation failure.

Only order 1 or 2 authorizes a later larger-node low-rank moment/cost Gate.
Order 3 only is a mechanism observation but does not authorize engineering,
because its state/application cost is unlikely to beat reading four leaves.

## Cost and stop rule

One isolated A800, at most 30 GPU-minutes, one position-73 smoke, one exposed
decision run, and one implementation repair. Preserve failures and stop after
classification. No training, reader endpoint, or timing.

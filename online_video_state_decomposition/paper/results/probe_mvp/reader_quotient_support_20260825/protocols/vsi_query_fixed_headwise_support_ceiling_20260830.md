# VSI query-fixed headwise-support ceiling Gate

Date: 2026-08-30
Status: frozen before engineering smoke

## Decision question

At the same true-`2x2` coarse state and exact-node budgets as the closed
query-fixed measure Gate, does allowing a different exact support for each
attention head recover the registered `1%/2%` visual-output capacity target?

This is a target-visible capacity diagnostic. It is not a deployable router,
latency result, reader endpoint, or claim about every cardinality subset.

## Frozen identity and data boundary

- OneVision Qwen2-7B checkpoint and VSI split identity are unchanged.
- Read exposed positions 73--96 only. Positions 97--120, selection, formal,
  reader logits, and answer correctness remain inaccessible.
- Use layers `0/13/27`, final-token query, all 28 heads, eight frames, 1,568
  visual tokens, and 392 non-overlapping true-`2x2` groups.
- Budgets remain `0/49/98/147/196/392`, corresponding to
  `25/34.375/43.75/53.125/62.5/100%` retained visual tokens.
- The previous repaired query-fixed result must reproduce exactly through the
  shared `exact_local_score` control before this Gate can be valid.

## Methods

1. `shared_exact_local`: previous target-visible one-shot score with one support
   shared by all heads;
2. `headwise_attention_mass`: exact mass ordering independently per head;
3. `headwise_exact_local`: exact first-order output-defect ordering per head;
4. `headwise_exact_greedy`: independently per head, recompute exact current
   visual-output error after every split and select the best remaining node.

The registered headwise envelope selects the best of methods 2--4 separately
for every exposed sample-layer-budget cell. It is deliberately stronger than a
single deployable policy, but it is still not a global subset optimum.

## Endpoints and outcome mapping

Primary endpoint at `k=196` is visual-output relative L2 across 72
sample-layer cells. Secondary endpoint is full-attention relative L2 with the
non-visual measure exact.

Capacity thresholds:

- visual mean `<=1%`, P95 `<=2%`, worst sample-layer `<=5%`;
- full mean `<=0.5%`, P95 `<=1%`.

Outcomes:

- `HEADWISE_SUPPORT_CAPACITY_PASS`: the registered headwise envelope satisfies
  every capacity threshold;
- `HEADWISE_SUPPORT_PARTIAL`: it fails a threshold but lowers visual mean by at
  least 25% relative to `shared_exact_local`;
- `NO_HEADWISE_SUPPORT_CAPACITY`: it fails and improves visual mean by less than
  25%;
- `INVALID_HEADWISE_SUPPORT_GATE`: identity, replay, row-count, or control
  reproduction failure.

Only `HEADWISE_SUPPORT_CAPACITY_PASS` authorizes a later low-bandwidth
head/group support-router Gate. `PARTIAL` or `NO` parks selector work and moves
the unique uncertainty to query-conditioned cross-moment state capacity.

## Cost and stop rule

One isolated A800, at most 30 GPU-minutes, one position-73 engineering smoke,
one 24-sample decision run, and at most one implementation repair. Preserve all
failed outputs. Stop immediately after classification; no training or timing.

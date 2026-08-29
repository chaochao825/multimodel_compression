# RDR-032: Accept a target-separated conditional rate-distortion probe

- Status: accepted
- Date: 2026-08-29
- Decider: researcher
- Supersedes: none

## Context

EXP-002 and EXP-003 showed that reuse or first-order forecasting of Wan
self-attention and FFN outputs removed too little work at strict local-error
thresholds. EXP-045 later showed that current block-input drift recovers 87.7%
of a coefficient oracle gap in a subset of late layers, but it evaluated only a
whole-block residual target and failed breadth and open-loop requirements.
EXP-046 and EXP-048 then ruled out increasing post-hoc whole-block state rank.

These results leave one high-information question unresolved: whether the same
current-state information is useful when attached to the module boundary where
it is naturally observed, and whether that gain survives suffix propagation at
a measured H200 cost. The researcher explicitly requested a conditional
rate-distortion frontier for attention, FFN, and whole block, with fixed quality,
oracle-recovery, and end-to-end coverage thresholds.

## Decision

Accept L-028 / EXP-049 / G-028 as one bounded side probe while L-026 remains the
sole mainline. Compare target-separated conditional innovation interfaces for
self-attention, FFN, and whole block under identical split, history, error, and
cost rules.

The probe is staged:

1. Reuse exposed EXP-003 sampled rows only as a non-claiming function-class
   screen.
2. Promote only locally viable targets to fresh full-token intervention capture
   and suffix-risk evaluation.
3. Benchmark the complete observable, cache read, and renderer boundary on one
   isolated H200 before making any speed or frontier claim.

## Scientific boundary

- The deterministic UniPC endpoint is evaluated by suffix intervention, not by
  an SDE path-KL formula.
- A sampled-row local result is not suffix risk and cannot pass G-028.
- Target-visible fitting is a capacity ceiling only. It cannot select a
  deployable method, layer, step, fallback, or threshold.
- Attention uses the current block input as its interface. FFN uses the exact
  post-attention hidden state available before FFN. Whole block uses the block
  input. Any extra projection or probe is charged to the candidate.
- EXP-047 methods, prompt-seed identities, thresholds, outputs, and H200 result
  root remain unchanged.

## Consequences

- A target that misses the local capacity screen is not promoted to expensive
  suffix capture.
- A target can authorize training or kernel work only by passing every G-028
  quality, oracle-recovery, H200-cost, and runtime-coverage condition.
- A null closes this post-hoc conditional-interface family. It does not refute
  same-step sparse attention, released few-step students, training-native
  state/render separation, or physical-time long-video memory.

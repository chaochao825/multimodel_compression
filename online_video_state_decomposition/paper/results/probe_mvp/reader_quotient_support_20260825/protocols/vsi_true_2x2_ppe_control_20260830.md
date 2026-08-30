# VSI true-2x2 paper-faithful PPE control

Date: 2026-08-30
Status: frozen before execution

## Decision question

Does the remaining error of the decision-positive `spatial_2x2 + group_mass`
quotient primarily come from assigning every merged token one representative
RoPE position, and can paper-faithful multi-position PPE recover that error?

This control isolates the positional representation of one fixed quotient. It
does not test support routing, progressive refinement, selection, formal,
latency, or a trained adapter.

## Preconditions and identity

- Require M0 decision `SAME_KERNEL_MASS_VALID`.
- Require M1 decision `NO_BATCHED_CURRENT_SUPPORT_PATH`.
- Require the topology-control decision `TRUE_2X2_DECISION_HEADROOM`.
- Use only already exposed VSI calibration positions 73--96, all 24 samples.
- Keep positions 97--120, selection, and formal endpoints unread.
- Freeze OneVision, eight frames, PCA rank 456 reconstruction, true
  non-overlapping `2x2` groups, group mass four, 392 quotient tokens, eager
  attention, prompts, candidate tokens, and dense endpoint.
- The baseline must reproduce the recorded `spatial_2x2 + group_mass`
  prediction and candidate KL within `1e-6`.

## Candidate

The incumbent assigns every merged token the constituent at offset one as its
single one-dimensional Qwen2 RoPE position.

The PPE candidate uses `K=4`, equal to the four members of each spatial group.
For each quotient group:

1. compute squared L2 distance from each reconstructed member token to the
   quotient mean;
2. sort the four member positions from nearest to farthest with stable
   row-major tie breaking;
3. split the 64 Qwen2 rotary frequency pairs into four contiguous groups of 16;
4. bind each frequency group to one sorted original position;
5. duplicate each pair's angle over the two Qwen2 rotary halves exactly as the
   standard implementation does.

All non-visual tokens retain their ordinary scalar position. Token values,
group mass, causal mask, attention implementation, and every model parameter
remain unchanged. Expanding an ordinary scalar position over all 64 frequency
pairs must reproduce standard Qwen2 RoPE numerically before the candidate is
evaluated.

## Frozen endpoints and outcome mapping

Compare PPE directly with the registered spatial-2x2 group-mass incumbent.

`PPE_STRICT_HEADROOM` requires all four:

- at least two fewer dense-decision mismatches;
- no increase in harmful flips;
- mean candidate KL at most `0.8x` the incumbent;
- P95 candidate KL at most `0.8x` the incumbent.

`PPE_DECISION_HEADROOM` applies when strict headroom fails but PPE has at least
one fewer mismatch, no harmful increase, and both mean and P95 KL no greater
than the incumbent.

`NO_PPE_HEADROOM` applies otherwise.

Strict headroom authorizes one bounded true-2x2/PPE current-support
monotonicity control on the same exposed samples. Decision-only headroom does
not authorize that expensive path; it keeps PPE only as a component candidate
for low-cost path-consistency adaptation. A null parks further train-free
self-attention position/topology tuning and moves the main candidate to a
query-fixed external-memory remainder bound or training-native quotient path.

## Cost and stop rule

- One isolated A800 on server 210, at most 30 GPU-minutes.
- One implementation repair is allowed.
- Stop after one valid outcome or after the repair allowance is exhausted.

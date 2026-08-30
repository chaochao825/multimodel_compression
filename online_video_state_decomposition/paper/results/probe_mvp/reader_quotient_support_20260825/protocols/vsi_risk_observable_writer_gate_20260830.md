# VSI risk-observable writer Gate

Date: 2026-08-30

Status: frozen before reading calibration positions 73--96

## Decision question

Can a query-independent learned writer expose task-sensitive innovation through a
small stored key, and does a joint writer-controller outperform learning only the
writer or only the controller at the same exact-group budget?

This Gate changes the memory interface after the fixed 39-scalar metadata
controller produced a valid `NO_GO`. It does not tune that controller, reuse its
prospective endpoint for a new claim, or read selection/formal data.

## Data roles

The existing ordered 120-question calibration role is partitioned before capture:

- positions 1--48: training;
- positions 49--72: validation and epoch selection;
- positions 73--96: one-shot prospective endpoint, unread at freeze time;
- positions 97--120: reserved for a later certificate Gate only after a positive
  prospective result;
- selection and formal roles remain unread.

The rank-456 PCA bulk was frozen by the earlier calibration-only protocol. Its
unsupervised basis is unchanged and is not a candidate in this Gate.

## Common memory and reader budget

- Frozen OneVision reader and tokenizer.
- Eight uniformly sampled frames, 1,568 visual tokens.
- Four contiguous visual tokens per regular group, 392 groups total.
- One quotient mean resident per group.
- Exactly 98 groups restored to four exact tokens.
- No-fallback reader input: 686 tokens, or 43.75% visual-token retention.
- No scalar-margin fallback is used in this Gate.

The full-reader first-order adverse margin risk remains the calibration teacher:

\[
r_g^*=\max_c
\frac{[-\langle\nabla_{X_g}m_c,\delta X_g\rangle]_+}
{\max(m_c,0.05)}.
\]

## Writer interface

For each group, form one quotient mode and four orthonormal signed residual modes
using the fixed 4x4 Hadamard transform. Normalize each hidden-channel mode and keep
its log RMS as an explicit scalar.

The learned writer is frozen to:

1. a shared `Linear(H, 8, bias=False)` applied to all five modes;
2. concatenation of the five 8D outputs;
3. `Linear(40, 32)` to produce one 32D stored key per group.

The question encoder is `Linear(H, 32, bias=False)`. The writer is query-independent;
only the group scorer sees the question.

Writer arithmetic is recorded as a one-time per-video proxy. Stored-key bytes and
per-question scorer MACs are reported separately. Neither is measured latency.

## Frozen methods

1. `residual_energy`: query-free L2 proxy.
2. `query_cosine`: fixed quotient/question cosine.
3. `fixed_controller`: the prior 39D metadata width-32 MLP, retrained on the new
   48/24 development partition for fairness.
4. `writer_dot`: learned 32D writer key, learned 32D question key, their scaled dot
   product, and a linear head over ten frozen magnitude/position scalars.
5. `joint_writer_controller`: the same writer/query key family plus a width-32 MLP
   over writer key, question key, elementwise interaction, and the same scalars.
6. `target_gradient_risk`: target-visible capacity ceiling only, never deployable.

All learned methods use weighted BCE on the exact top-98 teacher labels, AdamW
`lr=1e-3`, `weight_decay=1e-4`, 100 full-batch epochs, seed 20260830, and earliest
maximum validation top-98 recall. Widths, features, labels, optimizer, and epoch count
cannot change after prospective capture begins.

## Primary endpoints

For all methods on positions 73--96:

- mean and minimum top-98 teacher-risk recall;
- mean captured teacher-risk mass;
- full-reader agreement, harmful count, candidate KL, and task accuracy;
- stored-key bytes, writer MAC proxy, and per-question scorer MAC proxy.

Random top-98 recall is 25%.

## Outcome mapping

`JOINT_GO` requires all of:

- joint prospective top-98 recall at least 45%;
- joint prospective risk-mass capture at least 50%;
- joint recall at least 3 percentage points above both `fixed_controller` and
  `writer_dot`;
- joint full-reader agreement at least 91.67% (22/24);
- joint harmful count at most 1;
- joint mean candidate KL at most 0.05;
- no task-accuracy loss relative to the full reader;
- stored key exactly 32 FP16 scalars per group and scorer proxy below 2M MACs.

`WRITER_ONLY_GO` requires the same coverage and reader guards for `writer_dot`, while
joint fails only the 3-point synergy condition. This supports a learned memory
interface but rejects the joint-composition claim.

Any other valid result is `NO_GO` and closes this query-independent 32D writer family.
An engineering crash before prospective read may receive at most one narrow repair;
all failed attempts remain preserved.

## Claim boundary

A positive result is only a calibration prospective observability result. It does not
establish a calibrated risk certificate, measured latency, selection/formal
generalization, end-to-end task improvement, or superiority to LongVU, FrameFusion,
StreamingTOM, or FlexMem. A positive result only authorizes a separately frozen Gate
on calibration positions 97--120 for omitted-risk calibration and progressive exact
fallback.

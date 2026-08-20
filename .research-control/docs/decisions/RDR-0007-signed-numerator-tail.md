# RDR-0007: Authorize a signed numerator-tail capacity and transfer probe

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the explicit request to continue theoretical and
  methodological exploration rather than stop after EXP-006
- Supersedes: none

## Context

EXP-006 localized the latest attention-approximation bottleneck. At 50% native
work, a query-specific target-leaking proposal still produced 1.666% aggregate
local AV error and inflated the equivalent 64x64 execution union to 95.44%.
Replacing only the partition estimate with the exact value changed aggregate
error from 1.666% to 1.661%, whereas an exact numerator reduced the
denominator-only error to 0.164%. The dominant error is therefore the vector
numerator, not scalar normalization.

Earlier output-space diagnostics found that adaptive rank-16 corrections can
capture a substantial part of selected defects, while frozen low-rank bases,
fixed BCM/BCCB bases, and positive separable linear tails do not transfer. This
supports one new function-class test: generate the numerator basis from the
current K/V content, permit signed cancellation, model the partition with a
separate positive branch, and retain regular exact 64x64 blocks.

## Options

1. Hold the closed portfolio and perform no further numerical work.
2. Reopen fixed BCM, larger Nyström rank, or block sampling despite their
   registered null results.
3. Run one split-frozen signed-numerator-tail probe that explicitly separates
   capacity, transfer, and partition errors.
4. Relax the local quality objective or begin rollout/kernel work before the
   new function class has a numerical ceiling.

## Decision

Authorize option 3 as one bounded `side_probe`. Original Wan QKV weights remain
frozen. The mutable surface is limited to small numerator and partition feature
maps plus an optional rank-8 output adapter. The exact support is fixed to a
25% regular contiguous-64 proxy selected without dense held-out targets.

The probe may train on the existing calibration captures, select one registered
architecture on one validation capture, evaluate the chosen architecture once
on one test capture, and run one separately labelled transductive capacity
diagnostic. It may not add a model family, rank, support density, query tile, or
loss after outcomes are read.

This decision does not change the stopped temporal-transition mainline, revive
fixed BCM/BCCB output reconstruction, or authorize new captures, full rollout,
CUDA kernels, measured speed claims, or paper-level generalization.

## Consequences

- A pass permits a fresh deployment-generator Gate, not rollout or a speed
  claim.
- An exact-partition-only pass narrows future work to normalization rather than
  numerator capacity.
- A transductive-only pass identifies transfer as the bottleneck and permits at
  most a separately authorized low-cost conditioning probe.
- Failure of the transductive exact-partition capacity diagnostic stops this
  signed separable numerator family on the registered cell.
- The method is related to sparse-linear attention; no novelty claim is made
  from merely combining sparse and linear branches. The tested distinction is
  asymmetric signed numerator modeling with a separate positive partition and
  exact-block replacement.

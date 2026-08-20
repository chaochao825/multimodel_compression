# PLAN-009: Test signed value-aware latent pooling

- Status: completed
- Owner: researcher and Agent
- Gate: G-009
- Claim: C-009
- Candidate: L-009
- Lane: explore
- Resource cap: existing four captures and checkpoint, at most two H200-hours,
  one valid execution, and one pre-outcome engineering repair

## Decision to unlock

Determine whether a staged O(rNd) content mechanism can repair the output
defect left by 25% sparse-only attention, or whether current K/V content still
fails to expose the required directions.

## Milestones

1. Freeze latent families, split, ranks, optimizer, restart count, selection,
   cost model, gates, and leakage labels.
2. Implement and test signed/unsigned pooling, fixed V sketch, staged basis
   refinement, projection accounting, and arithmetic lower bounds.
3. Select one family on validation and evaluate it once on test.
4. Visualize quality versus rank, adaptive-energy recovery, head tails, and
   optimistic arithmetic fraction.
5. Close G-009 before any transferable generator or kernel work.

## Stop rules

- Do not add a family, rank, restart, step, support, tile, or identity after
  reading outcomes.
- All latent optimization and coefficients are target-exposed oracle access.
- Stop after one valid terminal outcome or one allowed pre-outcome repair.
- Do not infer transfer, rollout quality, or H200 speed from this Gate.

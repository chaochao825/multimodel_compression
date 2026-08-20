# PLAN-020: Test the temporal heterogeneous-rank frontier

- Status: completed
- Owner: researcher and Agent
- Gate: G-020
- Experiment: EXP-041
- Claim: C-020
- Candidate: L-020
- Lane: explore

## Outcome

Completed as `strong-temporal-frontier`. The validation-frozen table passed the
unopened `s03` at 0.486% / 0.874%, 65.8% improvement over matched static, all
three targets passing, and 46.02% local optimistic work.

## Completed milestones

1. Locked rank/action/cost/staging semantics and passed seven remote tests plus
   a complete synthetic CPU staging smoke.
2. Verified local/remote code hashes and EXP-040's `s03_loaded=False` provenance.
3. Waited for an exclusive H200 window, obtained dual locks, passed six idle
   checks, and completed one formal execution with zero PID overlap.
4. Downloaded and hash-verified all artifacts, generated bound CSV/PDF/PNG
   visualizations, and separated representation evidence from speed claims.

## Stop rules carried forward

- Do not treat oracle support/coefficient results as deployment evidence.
- Do not open rollout or kernel work before target-free coefficients and
  refresh-aware anchor horizon pass a separate Gate.
- Do not add higher ranks or reopen fixed BCM/BCCB/Butterfly reconstruction.

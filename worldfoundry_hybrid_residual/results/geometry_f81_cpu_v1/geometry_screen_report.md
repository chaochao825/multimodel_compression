# F81 Geometry Sparse Attention Screen

## Decision

- Strict 2% gate: `NO-GO`.
- Lowest density at 2%: `s3_temporal_pm2`, rank `16`, density `0.6132`, dense fallback `7/12`.
- Relaxed 5% oracle gate: `GO`.
- Lowest density at 5%: `s3_temporal_pm2`, rank `16`, density `0.0718`, dense fallback `0/12`.
- Rank-0 heads passing the static local gate across all masks: `0`.

## Interpretation

The fixed geometry masks do not satisfy the strict local error budget. Rank-16 can explain a large fraction of the current replay's activation defect and reaches the relaxed target for one candidate, but this is an input-specific SVD oracle. It does not establish a reusable low-rank basis, fused-kernel speedup, or end-to-end video quality.

# RDR-0009: Authorize a signed value-aware latent-pooling ceiling

- Status: accepted
- Date: 2026-08-11
- Decider: researcher through the explicit request to continue bounded theory
  and method exploration after informative failures
- Supersedes: none

## Context

EXP-008 separated low-rank capacity from basis provenance. On its frozen
positive rank-64 bulk plus 25% regular support, target-exposed adaptive SVD
reached 0.803% aggregate / 1.298% worst-head error at rank 32, while the best
current-QKV moment basis remained at 5.022% / 10.218% even with oracle
coefficients. The missing directions are therefore not the dominant V-variance
axes exposed by PCA or simple K/V cross-moments.

A pre-execution arithmetic audit found that carrying that positive rank-64
bulk into a runtime candidate would already cost approximately one dense QK+AV
in optimistic MACs before exact support or residual correction. It remains a
useful provenance diagnostic but is not a viable acceleration base.

The remaining high-information question is whether current content can expose
those low-variance, high-leverage directions through signed value pooling. A
latent query can select a contrast of value sets rather than a covariance
principal axis, and a staged rank-8 pursuit can test whether successive
contrasts approach the adaptive rank-32 boundary.

## Decision

Authorize one target-exposed `side_probe` on the same frozen captures and 25%
regular support. The deployment-relevant target is dense AV minus sparse-only
attention normalized over that exact support. Compare four preregistered latent
pooling families that isolate signedness and value-aware features. Latent
queries may optimize directly against each validation or test defect because
this Gate tests only the function-class ceiling. All such access must be
labelled oracle and may not be interpreted as transfer or deployment.

Select one family on the validation identity and execute that family once on
the test identity. Compare every registered rank with the adaptive SVD ceiling
and report a lower-bound arithmetic cost that includes sparse QK+AV plus latent
basis construction but omits the unknown coefficient predictor. The
implementation may use two frozen optimization restarts and one pre-outcome
engineering repair.

This decision does not authorize a calibration-frozen latent-query generator,
new capture, QKV tuning, rollout, CUDA kernel, measured speed claim, or revival
of fixed BCM/BCCB/Butterfly output reconstruction.

## Consequences

- A rank-16 quality and cost pass permits a separate transferable-generator
  Gate.
- A rank-32 quality pass with an arithmetic lower bound above 2/3 dense work is
  a representation-only boundary; support/rank co-design must precede a
  generator.
- A rank-16 match to adaptive rank 16 without absolute quality is evidence for
  staged basis generation, not deployment readiness.
- Failure to recover 90% of adaptive captured defect energy at rank 32 parks
  forward K/V-generated low-rank bases on this target. Future work should move
  to learned sparse support, a different residual target, or dense FP8.

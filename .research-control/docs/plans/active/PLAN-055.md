# PLAN-055: Select the post-EXP-046 successor

- Status: active decision-only plan
- Owner: researcher
- Current administrative mainline: L-025 (decided; no further runs)
- Active experiment/Gate: none
- Resource scope: zero GPU hours and no data access until an accepted decision

## Objective

Choose one successor after the valid EXP-046 whole-block rank-state null without
silently reviving rank growth, opening final identities, or conflating a
released few-step baseline with a new state-model claim.

## Decision surface

1. Benchmark a released Wan-compatible full-observability few-step student on
   the existing H200 stack before designing another student.
2. Register a training-native shared-state architecture with a different
   decoder, multi-step own-state loss, and fresh data as a new claim.
3. Stop algorithmic state work and retain only independent exact/low-precision
   system optimizations.

## Recommendation

Option 1 first. It establishes the quality/latency target that any new state
student must beat and has lower scientific and engineering risk. It does not
authorize using EXP-046 target-visible factors as training labels.

## Stop rule

No implementation, model download, final-split access, training, rollout, or GPU
run until the researcher accepts one option through a new RDR. Close this plan
when that decision is recorded.

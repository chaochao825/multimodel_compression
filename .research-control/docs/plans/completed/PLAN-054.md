# PLAN-054: Test the whole-block rank-state representation ceiling

- Status: completed
- Owner: researcher
- Gate: G-025
- Claim: C-025
- Candidate: L-025
- Lane: explore
- Resource scope: one H200, one smoke identity, four fresh selection identities

## Objective

Separate state representation capacity from state-coordinate observability by
testing a target-visible rank-state oracle on the complete Wan block defect.

## Execution order

1. Freeze new identities, rank grid, randomized SVD, exact-anchor semantics,
   arithmetic accounting, and G-025.
2. Add synthetic low-rank recovery, monotonicity, open-loop leakage, and Gate
   aggregation tests.
3. Run one calibration smoke with dense-recorder equivalence.
4. Run the four-identity selection Gate once.
5. Produce per-layer/step/horizon tables and rank/error/cost visualizations.
6. Open no coordinate predictor, recurrent student, final split, or kernel stage
   unless G-025 passes.

## Stop rule

Stop on one valid result, any dense-equivalence or target-leakage failure,
foreign GPU overlap, incomplete selection scope, or resource exhaustion. Rank 96
may classify a boundary but cannot rescue C-025.

## Decision effect

A pass authorizes only a distinct current-h state-coordinate observability Gate.
A null returns the program to a released full-observability few-step baseline or
a training-native architecture whose state is created during training.

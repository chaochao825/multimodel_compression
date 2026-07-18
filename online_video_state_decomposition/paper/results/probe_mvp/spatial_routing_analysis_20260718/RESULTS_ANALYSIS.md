# Spatial Routing Exploratory Analysis

This is a post-hoc mechanism-selection analysis on a reused 200-sample set. It is not an independent confirmation result.

## Overall

| Method | Correct | Accuracy | Better / worse vs full | Steady state | Cold start | Mean error |
|---|---:|---:|---:|---:|---:|---:|
| Full state | 102 / 200 | 51.0% | 0 / 0 | 8.024 MiB | 8.024 MiB | 0.00% |
| Low-rank only | 100 / 200 | 50.0% | 1 / 3 | 0.524 MiB | 2.531 MiB | 6.00% |
| Fixed sparse s4 | 101 / 200 | 50.5% | 0 / 1 | 1.024 MiB | 3.032 MiB | 5.34% |
| Spatial grid 2x2 | 101 / 200 | 50.5% | 0 / 1 | 1.024 MiB | 3.031 MiB | 5.15% |
| Routed grid/sparse | 102 / 200 | 51.0% | 0 / 0 | 1.024 MiB | 3.031 MiB | 5.13% |

## Interpretation

- Full state reaches 102/200; fixed-s4 and grid reach 101 and 101 respectively.
- Routed memory reaches 102/200 with 0 better and 0 worse samples relative to full state.
- The routed writer uses grid mode on 76.5% of frames and sparse mode on 23.5%.
- The amortized steady-state ratio is 7.84x; including the shared codec at cold start it is 2.65x.
- Reconstruction error is diagnostic only: the fixed and grid methods can have similar aggregate accuracy while failing on complementary evidence.

## Routing By Task

| Task | Accuracy | Grid frames | Sparse frames | All-grid / all-sparse / mixed samples |
|---|---:|---:|---:|---:|
| action_sequence | 47.5% | 92.2% | 7.8% | 25 / 1 / 14 |
| moving_direction | 40.0% | 57.7% | 42.3% | 8 / 2 / 30 |
| object_existence | 57.5% | 53.1% | 46.9% | 4 / 3 / 33 |
| scene_transition | 67.5% | 93.4% | 6.6% | 24 / 0 / 16 |
| state_change | 42.5% | 86.2% | 13.8% | 25 / 2 / 13 |

## Evidence Boundary

The route was designed after inspecting this same sample set. A paper claim requires a frozen implementation and a fresh paired reserve set, together with a matched-budget recent-only baseline and cold-start/runtime accounting.

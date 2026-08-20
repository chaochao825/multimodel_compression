# RESULT-EXP-046: Full-observability whole-block rank-state capacity

- Status: complete
- Validity: valid prospective target-visible capacity null
- Date: 2026-08-20
- Gate: G-025
- Claim: C-025
- Candidate: L-025

## Registered outcome

`null`. Rank 64 passed `0/10` layers for every registered step/horizon, rather
than the required `6/10`. Diagnostic rank 96 also passed `0/10` everywhere, so
the outcome is not a rank-96 boundary.

The four selection identities produced 2,880 complete finite metric rows. All
four final identities remained unopened.

## Principal measurements

| Rank | Aggregate block-output L2 | Worst identity/branch | Passing cells | Render/exact MAC proxy |
|---:|---:|---:|---:|---:|
| 0 | 42.957% | 392.024% | 0/60 | 0.000% |
| 8 | 7.455% | 15.279% | 0/60 | 0.040% |
| 16 | 6.541% | 13.504% | 0/60 | 0.081% |
| 32 | 5.678% | 12.424% | 0/60 | 0.161% |
| 64 | 4.833% | 11.093% | 0/60 | 0.323% |
| 96 | 4.345% | 10.210% | 0/60 | 0.484% |

Rank-64 H1 output L2 was 3.595% at step 4 and 4.104% at step 6, so the result
does not depend only on the unstable H3 diagonal rollout. The easiest rank-96
cell, L29 step 4 H1, remained at 1.189% aggregate and 1.716% worst, above both
registered quality limits.

## Supports

- The exact-history diagonal defect contains correlated low-dimensional energy:
  rank 8 reduced the overall output L2 substantially, and later layers were
  consistently easier than early layers in the registered late-layer window.
- The optimistic state renderer itself is arithmetically cheap. Representation
  quality, rather than the frozen `2*N*D*r` proxy, closed G-025.

## Does not support

- It does not support a rank-64 whole-block endpoint state under the frozen
  causal diagonal renderer.
- It does not support training a current-h coordinate observer or recurrent
  transition for this representation.
- Rank growth from 64 to 96 does not show a near-threshold rescue.

## Unknown

- Whether training can create a different shared decoder and stable recurrent
  state rather than infer the target-specific EXP-046 factors.
- The quality/latency Pareto of a released full few-step video student on this
  exact H200 stack.
- End-to-end approximate rollout quality and speed; the protocol prohibited
  both after a capacity null.

## Interpretation

EXP-046 used an independently fitted target-visible SVD at every endpoint. It
therefore removed the observer problem and gave the representation a generous
per-sample ceiling. Failure of that ceiling means an observer cannot rescue the
same rank-64 state definition.

This is not a proof that denoising has no compact sufficient state. The run did
not require a shared basis, did not recursively apply the correction at
intermediate steps, and evaluated exact dense-trajectory block inputs. A
training-native state with a different renderer is a different claim.

## Decision

Reject C-025 within its registered F17 boundary and park L-025. Do not open the
final split, train a coordinate observer, extend the rank sweep, run approximate
rollout, or build a renderer kernel for this candidate. Any training-native
state or released few-step integration requires a new accepted decision.

## Integrity

- One calibration smoke established exact dense-recorder final-latent equality
  with zero relative L2 and produced all 36 expected cells.
- Two earlier smoke attempts failed before Wan import because the wrong Python
  environment lacked dependencies; both logs were preserved and contributed no
  scientific rows.
- The successful smoke and formal selection used the prior Wan `.venv`, PyTorch
  2.9.1+cu128, CUDA 12.8, and one NVIDIA H200 NVL under the shared GPU lock.
- Sixteen numerical, Gate, leakage, and recorder tests passed in the execution
  environment.
- The formal run exited 0; all selection identities and only selection
  identities were present.

## Evidence

- `worldfoundry_hybrid_residual/results/WAN_RANK_STATE_CAPACITY_EXP046_20260820.zh-CN.md`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1/all_cell_metrics.csv`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1/run_manifest.json`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1_analysis/rank_summary.csv`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1_analysis/cell_rank_metrics.csv`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1_analysis/coverage_gate.csv`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1_analysis/rank_capacity_frontier.png`
- `worldfoundry_hybrid_residual/results/rank_state_capacity_exp046_gate_v1_analysis/rank64_layer_map.png`

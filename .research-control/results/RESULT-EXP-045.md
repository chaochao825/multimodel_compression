# RESULT-EXP-045: Current-input denoising observability Gate

- Status: complete
- Validity: valid prospective no-training null
- Date: 2026-08-20
- Gate: G-024
- Claim: C-024
- Candidate: L-024

## Registered outcome

`observability-null`. No registered causal method passed G-024 on the four
selection identities:

- the strongest horizon-1 diagnostic, DPLR-16, passed only 3/10 layers at each
  target step rather than 6/10;
- DPLR-16 recovered 86.2% and 85.9% of the target-visible oracle gap, but its
  horizons 2/3 were not registered and therefore it was ineligible to pass;
- Broyden-2 was the strongest complete exact-history candidate, but passed only
  2/10 layers at each step, recovered 45.4%/57.5% of the oracle gap, and reached
  a worst open-loop risk ratio of 1.530 versus matched AR(2);
- no width-32 router was trained and all four final identities remained unopened.

## Principal measurements

Matched horizon-1 residual risks at steps 4/6 were:

| Method | Step 4 | Step 6 | Block-output L2, aggregate | Mean registered MACs |
|---|---:|---:|---:|---:|
| calibration-frozen AR(2) | 35.082% | 27.040% | 14.124% | 0.036G |
| diagonal current-input | 26.194% | 19.489% | 10.415% | 0.048G |
| Broyden-2 | 26.121% | 18.526% | 10.222% | 0.132G |
| DPLR-8 | 18.394% | 14.588% | 7.483% | 0.216G |
| DPLR-16 | 18.062% | 14.324% | 7.348% | 0.407G |
| target-visible 75-shift oracle | 15.342% | 12.239% | 6.86% / 5.53% by step | oracle only |

DPLR-16 exceeded the 2x layer criterion only at L21, L24, and L25 at both
steps. L26--L29 achieved only 1.02x--1.22x, so current-input observability did
not increase monotonically with depth. Rank 8 to 16 yielded only a small
increment, and Broyden rank 2 to 4 was effectively flat.

The nonperiodic transport branches did not improve over the diagonal map.
History-selected transport remained near 10.46% aggregate block-output L2;
the fixed-row Q/K selector remained near 10.44% while increasing mean arithmetic
from 0.129G to 0.321G MACs. The cheap sketch therefore did not reveal the target
spacetime coordinate in this experiment.

## Interpretation

The positive result is local and mechanistic: current block-input drift exposes
a useful channel-diagonal Jacobian component, and a sample-local low-rank
channel correction captures additional one-step structure in three layers.

The negative result is the required deployment conclusion. The useful component
is not broad across the ten contiguous late layers, the complete secant methods
become unstable over three skipped steps, and even the target-visible oracle is
far outside a 1%--2% block-output fidelity range. This is not a router problem:
the tested causal function class lacks sufficient stable capacity before router
training.

## Does not support

- It does not test physical state prediction from one video second to the next.
- It does not refute sparse video attention, physical-time transport, or
  autoregressive long-video memory.
- It does not refute a train-native hidden-state model, full few-step student,
  consistency/DMD training, or learned sparse attention.
- It does not establish an end-to-end quality or H200 speed result; the protocol
  prohibited rollout and timing claims after a failed observability Gate.

## Decision

Refute C-024 within its registered F17 denoising-time boundary and park L-024.
Do not train the width-32 router, open the final split, run approximate rollout,
or build a kernel for this candidate. Successor selection must use a different
observation/function class rather than another rank, shift, or secant sweep.

## Integrity

- Four calibration and four selection identities completed; 9,760 metric rows
  were finite and the four final identities were absent.
- The matched AR(2) coefficients were fit from calibration only and frozen before
  the first selection identity.
- Nine synthetic/runtime tests passed, including exact-history leakage,
  nonperiodic token transport, linear oracle recovery, output-error aggregation,
  and dense recorder preservation.
- The formal process waited for the prior foreign GPU process to exit, then ran
  alone on one NVIDIA H200 NVL and exited 0.
- A prior smoke run established exact dense-recorder latent equivalence with zero
  relative L2; no end-to-end timing evidence was accepted from either run.

## Evidence

- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/report.zh-CN.md`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/method_gate.csv`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/layer_gate.csv`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/source_all_cell_metrics.csv`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/risk_reduction_by_layer.png`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/open_loop_stability.png`
- `worldfoundry_hybrid_residual/results/current_input_observability_exp045_gate_v1_analysis/quality_cost.png`

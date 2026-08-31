# RESULT-EXP-050: Fixed-state control-variate support interaction

- Experiment: EXP-050
- Gate: G-029 Stage 0 as originally registered
- Date: 2026-09-01
- Validity: valid for fixed-state interaction; insufficient to close jointly
  trained support-state capacity
- Outcome: boundary

## Scope and validity

The valid run used VSI positions 73--96, layers 0/13/27, four-token pages,
25% exact tokens, and the calibration-trained width-32 whole-measure state.
Positions 97--120, official selection, formal data, task intervention, and H200
timing remained unread/unrun.

The initial capture lacked exact per-page FP32 mass, so an all-page recovery
guard correctly rejected reconstructed BF16 Q/K contributions. A repaired
capture stored the original visual exponentials. The final run contained 288
finite method rows and passed:

- maximum learned-state replay absolute error: `5.2452e-6`;
- maximum all-page dense-recovery absolute error: `1.9550e-5`;
- maximum all-page dense-recovery relative error: `1.1587e-6`;
- exact page budget for every sample, layer, and head.

Failed pre-guard and guard-triggering attempts are preserved in the remote
result root and were not used for the decision.

## Results

| Method | Visual mean / P95 / worst | Full mean / P95 | Visual risk |
|---|---:|---:|---:|
| exact-only mass | 13.728% / 21.363% / 23.325% | 2.948% / 6.632% | 0.020674 |
| state-only | 31.509% / 52.518% / 65.178% | 2.863% / 7.453% | 0.110991 |
| independent mass correction | 11.124% / 19.106% / 21.330% | 1.534% / 2.652% | 0.014068 |
| residual-aware support oracle | 6.759% / 12.566% / 19.383% | 0.823% / 1.445% | 0.005710 |

Residual-aware support improved paired visual risk by `59.414%` over mass
support. The position-bootstrap 95% interval was `[50.014%, 65.915%]`. The
analytic active-state ratio was `3.842x`.

## Interpretation

This supports one narrow mechanism claim: at identical payload, exact support
should target the residual of the bulk state rather than attention mass alone.

It does not pass the registered local capacity threshold. More importantly,
the state was frozen from whole-measure training. Therefore this run is not an
upper bound on the explicitly requested jointly trained support-state model.
Calling it a no-go for C-029 would conflate support selection with state
adaptation.

## Decision

Close EXP-050 as a fixed-state boundary. Preserve the positive interaction and
the capacity miss. RDR-034 / EXP-051 prospectively repairs the missing factorial
cell by training independent and residual-aware states from the same
initialization, data, steps, and payload. No deployable router or later split is
authorized by this result.

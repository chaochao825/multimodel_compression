# Stronger Train-Free Residual-Tail Oracle Report

Run kind: `registered`.

Every result below is a post-hoc function-class capacity diagnostic. Dense
attention selects the critical mask, and each sample/head envelope may choose
a different candidate. It is not a frozen test or deployment result.

## Fixed-Configuration Diagnostics

| Family | Registered-set post-hoc best fixed configuration | Aggregate | Worst |
|---|---|---:|---:|
| lowrank_covariance | centroid, rank=0, components=2, density=25.0% | 7.876% | 23.380% |
| residual_tail_polynomial | mean, order=4, density=25.0% | 4.952% | 10.134% |
| value_aware_coreset | value_aware_kv_thw, m=128, density=25.0% | 5.426% | 14.617% |

## Per-Record Oracle Envelope

| Family | Aggregate | Worst | Oracle gate | Query-work proxy |
|---|---:|---:|---:|---:|
| lowrank_covariance | 6.719% | 21.938% | FAIL | 0.306 |
| residual_tail_polynomial | 4.952% | 10.134% | FAIL | not implemented |
| value_aware_coreset | 5.409% | 14.617% | FAIL | 0.254 |

## Sample-Wise Post-Hoc Envelope

| Sample | Coreset | Polynomial | Covariance |
|---|---:|---:|---:|
| s00_p00_seed20260740 | 5.261% | 4.451% | 6.426% |
| s01_p01_seed20260740 | 5.141% | 5.135% | 6.406% |
| s02_p00_seed20260741 | 5.225% | 4.801% | 7.268% |
| s03_p01_seed20260741 | 5.966% | 5.343% | 6.864% |

## Failure Concentration

| Family | Easiest head (aggregate) | Hardest head (aggregate / worst) | Records <= 1% |
|---|---:|---:|---:|
| value_aware_coreset | h0: 1.391% | h6: 13.253% / 14.617% | 2/48 |
| residual_tail_polynomial | h4: 1.143% | h6: 9.072% / 9.830% | 4/48 |
| lowrank_covariance | h0: 1.320% | h6: 15.637% / 21.938% | 4/48 |

## Decision Boundary

Probe decision: `STOP_TRAINFREE_TAIL`.

The smallest recorded mean residual-tail score range is `18.523`.
The residual score interval is therefore not narrow after removing the
critical blocks; fourth-order Taylor approximation is still far outside
the oracle quality gate. Odd midrange expansions also create signed tail
weights and occasional non-positive shared denominators.

The full-covariance Gaussian variants are numerically finite and their
covariance products are covered by exact reconstruction tests, but they
are usually worse than centroid/diagonal moments. This supports model
mismatch rather than a missing covariance rank as the failure mechanism.

Polynomial arithmetic speed is intentionally unreported because no
TensorSketch/random-Maclaurin feature realization was implemented.
Coreset query work omits dense-oracle leverage and clustering; covariance
query work omits online moment formation and SVD. No H200 latency claim is
authorized by this experiment.

If all envelopes fail the 0.5% aggregate and 1% worst gate, stop the
train-free tail family and move to a small learned Q/K-conditioned
sparse-linear tail with frozen base QKV.

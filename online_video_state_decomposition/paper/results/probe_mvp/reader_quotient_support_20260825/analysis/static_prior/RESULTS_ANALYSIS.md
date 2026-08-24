# Reader-Quotient Static Prior Transfer

Decision: `ADVERSE`; samples: `40`.

| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Top-1 delta | W/T/L | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| euclidean_s4 | +0.00% | +0.00% | [+0.00%, +0.00%] | 1.000 | +0.00% | 0/40/0 | REFERENCE |
| position_s4 | -189.82% | -189.62% | [-555.39%, -15.89%] | 3.297 | +0.00% | 13/1/26 | ABLATION |
| channel_s4 | +2.47% | +2.49% | [-2.96%, +8.91%] | 0.919 | +0.00% | 8/25/7 | ABLATION |
| separable_s4 | -186.46% | -186.24% | [-558.74%, -13.55%] | 3.046 | +0.00% | 11/2/27 | ABLATION |
| static_fisher_s4 | -190.65% | -190.45% | [-565.06%, -13.86%] | 3.297 | +0.00% | 15/0/25 | ADVERSE |
| mixed_static_s4 | -122.56% | -122.57% | [-511.54%, +20.65%] | 1.328 | +0.00% | 14/4/22 | ADVERSE |

Calibration-only frozen prior evaluated on disjoint tasks and samples.

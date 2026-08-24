# Reader-Quotient Sparse-Support Oracle

Decision: `GO`; samples: `40`.

| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Top-1 delta | W/T/L | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| pca_only | -472.96% | -475.42% | [-1089.07%, -146.75%] | 3.667 | -2.50% | 8/5/27 | ABLATION |
| euclidean_s4 | +0.00% | +0.00% | [+0.00%, +0.00%] | 1.000 | +0.00% | 0/40/0 | REFERENCE |
| fisher_s4 | +72.07% | +71.95% | [+51.58%, +83.25%] | 0.197 | +2.50% | 26/2/12 | GO |
| mixed_s4 | +75.93% | +75.82% | [+60.10%, +84.87%] | 0.136 | +2.50% | 25/3/12 | GO |

The result is a transductive support oracle. It cannot be read as an online writer, task-accuracy result, strong-reader replication, or latency claim.

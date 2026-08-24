# LLaVA-OneVision Reader-Quotient Capacity Replication

Decision: `BOUNDARY`; samples: `20`.

| Variant | Candidate KL reduction | Vocab KL reduction | 95% CI | P95 ratio | Top-1 delta | W/T/L | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| pca_only | -27.42% | -27.42% | [-79.33%, +34.89%] | 1.761 | +0.00% | 8/4/8 | ABLATION |
| euclidean_s4 | +0.00% | +0.00% | [+0.00%, +0.00%] | 1.000 | +0.00% | 0/20/0 | REFERENCE |
| fisher_s4 | +44.24% | +44.24% | [-93.59%, +88.03%] | 0.613 | -5.00% | 10/2/8 | BOUNDARY |
| mixed_s4 | +54.34% | +54.34% | [-44.12%, +88.79%] | 0.577 | -5.00% | 13/1/6 | BOUNDARY |

Fresh five-task, 20-sample strong-reader capacity replication; transductive Fisher is not deployable.

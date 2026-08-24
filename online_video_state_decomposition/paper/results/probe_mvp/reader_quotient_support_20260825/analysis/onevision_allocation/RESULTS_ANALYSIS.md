# OneVision Equal-Budget Rank/Support Allocation

Decision: `BOUNDARY`; selected diagnostic endpoint: `fisher_r438_s1`; fresh confirmation authorized: `False`.

| Variant | Bytes | KL sum | Paired reduction | Min LOO | 95% CI | P95 ratio | Absolute/anchor | Positive tasks | Top-1 delta | Accuracy delta | Feature L2 | Decision |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| euclidean_r384_s4 | 2,867,328 | 0.123267 | +0.00% | +0.00% | [+0.00%,+0.00%] | 1.000 | 1.000 | 0/5 | +0.00% | +0.00% | 15.85% | REFERENCE |
| fisher_r384_s4 | 2,867,328 | 0.049616 | +59.75% | -2.11% | [-45.17%,+86.85%] | 0.463 | 0.403 | 2/5 | +0.00% | +0.00% | 16.32% | BOUNDARY |
| mixed_r384_s4 | 2,867,328 | 0.055772 | +54.76% | -16.17% | [-89.62%,+86.71%] | 0.757 | 0.452 | 3/5 | +0.00% | +0.00% | 16.20% | BOUNDARY |
| euclidean_r402_s3 | 2,865,504 | 0.092289 | +0.00% | +0.00% | [+0.00%,+0.00%] | 1.000 | 0.749 | 0/5 | +0.00% | +0.00% | 15.64% | REFERENCE |
| fisher_r402_s3 | 2,865,504 | 0.058773 | +36.32% | -22.07% | [-70.30%,+82.18%] | 0.526 | 0.477 | 3/5 | +0.00% | +0.00% | 16.01% | BOUNDARY |
| mixed_r402_s3 | 2,865,504 | 0.071683 | +22.33% | -51.34% | [-155.32%,+79.46%] | 0.990 | 0.582 | 2/5 | +0.00% | +0.00% | 15.92% | BOUNDARY |
| euclidean_r420_s2 | 2,863,680 | 0.069919 | +0.00% | +0.00% | [+0.00%,+0.00%] | 1.000 | 0.567 | 0/5 | +0.00% | +0.00% | 15.46% | REFERENCE |
| fisher_r420_s2 | 2,863,680 | 0.053202 | +23.91% | -16.29% | [-39.13%,+75.85%] | 0.387 | 0.432 | 4/5 | -5.00% | -5.00% | 15.73% | ADVERSE |
| mixed_r420_s2 | 2,863,680 | 0.055251 | +20.98% | -21.00% | [-54.17%,+75.97%] | 0.484 | 0.448 | 4/5 | -5.00% | -5.00% | 15.67% | ADVERSE |
| euclidean_r438_s1 | 2,861,856 | 0.049421 | +0.00% | +0.00% | [+0.00%,+0.00%] | 1.000 | 0.401 | 0/5 | +0.00% | +0.00% | 15.32% | REFERENCE |
| fisher_r438_s1 | 2,861,856 | 0.037465 | +24.19% | +9.78% | [-86.25%,+55.53%] | 0.972 | 0.304 | 4/5 | +0.00% | +0.00% | 15.47% | BOUNDARY |
| mixed_r438_s1 | 2,861,856 | 0.040751 | +17.54% | +0.23% | [-124.22%,+49.05%] | 0.972 | 0.331 | 3/5 | +0.00% | +0.00% | 15.44% | BOUNDARY |
| euclidean_r456_s0 | 2,860,032 | 0.043739 | +0.00% | +0.00% | [+0.00%,+0.00%] | 1.000 | 0.355 | 0/5 | +0.00% | +0.00% | 15.25% | REFERENCE |

This is an allocation-selection result on observed samples. Only a GO may enter a separately frozen, untouched-task confirmation; no scorer or deployment claim is authorized here.

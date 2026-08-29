# Video-MME OneVision Same-Rank Domain-Residual Selection

Decision: **CAPACITY_ONLY**

Selected candidate: `target_pca_r456`

| Candidate | Decision | KL ratio | P95 ratio | L2 ratio | Mismatch | Harmful | Correct |
|---|---|---:|---:|---:|---:|---:|---:|
| source_r456 | BASELINE | 1.000 | 1.000 | 1.000 | 6 | 1 | 102 |
| target_mean_source_r456 | NO_GO | 0.986 | 1.215 | 0.996 | 8 | 2 | 101 |
| residual_swap_r16 | NO_GO | 0.862 | 0.996 | 0.977 | 5 | 0 | 103 |
| residual_swap_r32 | NO_GO | 1.024 | 1.572 | 0.968 | 5 | 0 | 103 |
| residual_swap_r64 | NO_GO | 0.898 | 1.413 | 0.954 | 8 | 2 | 102 |
| residual_swap_r96 | NO_GO | 0.782 | 1.100 | 0.943 | 5 | 1 | 101 |
| residual_swap_r128 | NO_GO | 0.840 | 1.109 | 0.934 | 4 | 0 | 102 |
| target_pca_r456 | CAPACITY_ONLY | 0.521 | 0.605 | 0.852 | 8 | 1 | 102 |

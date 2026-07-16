# Streaming Hybrid State V0 Result Summary

This is a representation-level causal probe. It does not claim end-to-end Video-LLM accuracy, PPA, or encoder speedup.

## Selected Predictors

| Layer | Validation-selected predictor | Test NMSE | Test cosine |
|---:|---|---:|---:|
| 15 | ema_050 | 0.329062 | 0.811522 |
| 22 | ema_075 | 0.325020 | 0.824608 |

## Best VQ Points

| Layer | Budget | Method | Codec | Stream bps | Effective bps | Cosine |
|---:|---:|---|---|---:|---:|---:|
| 15 | 0.50 | residual_pq | pq_0p5b_k16_d8@1.00 | 0.7197 | 0.7205 | 0.770384 |
| 15 | 1.00 | residual_pq | pq_1p5b_k64_d4@0.50 | 1.1885 | 1.1900 | 0.899029 |
| 15 | 1.58 | residual_pq | pq_1p5b_k64_d4@1.00 | 1.6572 | 1.6588 | 0.936733 |
| 15 | 2.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.1260 | 2.1322 | 0.964815 |
| 15 | 4.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.1260 | 2.1322 | 0.964815 |
| 22 | 0.50 | residual_pq | pq_0p5b_k16_d8@1.00 | 0.7197 | 0.7205 | 0.738998 |
| 22 | 1.00 | residual_pq | pq_1p5b_k64_d4@0.50 | 1.1885 | 1.1900 | 0.876307 |
| 22 | 1.58 | residual_pq | pq_1p5b_k64_d4@1.00 | 1.6572 | 1.6588 | 0.924064 |
| 22 | 2.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.1260 | 2.1322 | 0.959285 |
| 22 | 4.00 | residual_pq | pq_4b_k256_d2@1.00 | 4.0010 | 4.0041 | 0.986831 |

## Controller And Combined Policy

Torch controllers enabled: `True`.

| Layer | Budget | Controller | Action accuracy | Combined cosine | Refresh rate | Effective bps |
|---:|---:|---|---:|---:|---:|---:|
| 15 | 0.50 | threshold | 0.6933 | 0.903686 | 0.8125 | 3.2636 |
| 15 | 0.50 | decision_tree | 0.8867 | 0.888525 | 0.5750 | 2.3099 |
| 15 | 0.50 | mlp | 0.8133 | 0.880669 | 0.5563 | 2.2415 |
| 15 | 0.50 | dlgn | 0.8467 | 0.890392 | 0.6062 | 2.4362 |
| 15 | 1.00 | threshold | 0.6933 | 0.909654 | 0.4625 | 2.1277 |
| 15 | 1.00 | decision_tree | 0.5867 | 0.917068 | 0.3187 | 1.7567 |
| 15 | 1.00 | mlp | 0.7200 | 0.902870 | 0.3563 | 1.7610 |
| 15 | 1.00 | dlgn | 0.6800 | 0.884537 | 0.6188 | 2.4872 |
| 15 | 1.58 | threshold | 0.5600 | 0.929614 | 0.3000 | 1.9751 |
| 15 | 1.58 | decision_tree | 0.6200 | 0.914382 | 0.5312 | 2.5288 |
| 15 | 1.58 | mlp | 0.7200 | 0.911744 | 0.2750 | 1.6816 |
| 15 | 1.58 | dlgn | 0.7133 | 0.908982 | 0.2687 | 1.6442 |
| 15 | 2.00 | threshold | 0.6867 | 0.945711 | 0.1875 | 2.0093 |
| 15 | 2.00 | decision_tree | 0.7000 | 0.944111 | 0.2062 | 2.0097 |
| 15 | 2.00 | mlp | 0.7800 | 0.905284 | 0.2000 | 1.5882 |
| 15 | 2.00 | dlgn | 0.7267 | 0.917981 | 0.0625 | 1.3332 |
| 15 | 4.00 | threshold | 0.6867 | 0.945711 | 0.1875 | 2.0093 |
| 15 | 4.00 | decision_tree | 0.7000 | 0.944111 | 0.2062 | 2.0097 |
| 15 | 4.00 | mlp | 0.7600 | 0.930005 | 0.1938 | 1.7881 |
| 15 | 4.00 | dlgn | 0.6667 | 0.901105 | 0.0625 | 1.1957 |
| 22 | 0.50 | threshold | 0.7467 | 0.936653 | 0.8125 | 3.2636 |
| 22 | 0.50 | decision_tree | 0.7667 | 0.936376 | 0.7937 | 3.1883 |
| 22 | 0.50 | mlp | 0.8667 | 0.923828 | 0.6188 | 2.4893 |
| 22 | 0.50 | dlgn | 0.8200 | 0.930621 | 0.7188 | 2.8880 |
| 22 | 1.00 | threshold | 0.5667 | 0.930092 | 0.6625 | 2.8121 |
| 22 | 1.00 | decision_tree | 0.6733 | 0.938303 | 0.4437 | 2.1336 |
| 22 | 1.00 | mlp | 0.8000 | 0.929701 | 0.4125 | 1.9306 |
| 22 | 1.00 | dlgn | 0.7467 | 0.924401 | 0.6125 | 2.4621 |
| 22 | 1.58 | threshold | 0.6133 | 0.944645 | 0.3750 | 2.1638 |
| 22 | 1.58 | decision_tree | 0.6800 | 0.945169 | 0.3438 | 2.0571 |
| 22 | 1.58 | mlp | 0.7333 | 0.933184 | 0.3000 | 1.8101 |
| 22 | 1.58 | dlgn | 0.6067 | 0.936624 | 0.2500 | 1.8502 |
| 22 | 2.00 | threshold | 0.6933 | 0.955278 | 0.2313 | 2.0975 |
| 22 | 2.00 | decision_tree | 0.7133 | 0.954182 | 0.2062 | 2.0097 |
| 22 | 2.00 | mlp | 0.8133 | 0.940217 | 0.2250 | 1.8011 |
| 22 | 2.00 | dlgn | 0.6933 | 0.955100 | 0.2062 | 2.0480 |
| 22 | 4.00 | threshold | 0.7400 | 0.970282 | 0.0625 | 3.2542 |
| 22 | 4.00 | decision_tree | 0.7600 | 0.970014 | 0.0625 | 3.1793 |
| 22 | 4.00 | mlp | 0.8400 | 0.951084 | 0.0625 | 2.5580 |
| 22 | 4.00 | dlgn | 0.7400 | 0.970282 | 0.0625 | 3.2551 |

## Claim Boundary

- Predictor, VQ, and controller rows use disjoint clip-level splits.
- Codebook, bitmap, scale, action, and controller metadata are counted.
- `payload_bps` excludes one-time static codebook/controller bits; `effective_bps` amortizes them over the test corpus.
- DLGN policy results use the final hard gate network. Soft accuracy and discretization gap are reported separately.
- The RGB controller is a pre-encoder proxy, but innovation coding still requires current hidden features. `encoder_required_rate` therefore counts both innovation and refresh actions. No ViT skip or measured-compute claim is made.

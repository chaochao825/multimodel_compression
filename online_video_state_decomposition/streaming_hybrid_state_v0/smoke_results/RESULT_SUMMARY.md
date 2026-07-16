# Streaming Hybrid State V0 Result Summary

This is a representation-level causal probe. It does not claim end-to-end Video-LLM accuracy, PPA, or encoder speedup.

## Selected Predictors

| Layer | Validation-selected predictor | Test NMSE | Test cosine |
|---:|---|---:|---:|
| 22 | ema_075 | 0.325020 | 0.824608 |

## Best VQ Points

| Layer | Budget | Method | Codec | Stream bps | Effective bps | Cosine |
|---:|---:|---|---|---:|---:|---:|
| 22 | 0.50 | residual_pq | pq_0p5b_k16_d8@1.00 | 0.7197 | 0.7205 | 0.703311 |
| 22 | 1.00 | residual_pq | pq_1p5b_k64_d4@0.50 | 1.1885 | 1.1900 | 0.842148 |
| 22 | 1.58 | residual_pq | pq_2b_k256_d4@0.50 | 1.4229 | 1.4291 | 0.896630 |
| 22 | 2.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.1260 | 2.1322 | 0.942713 |
| 22 | 4.00 | residual_pq | pq_4b_k256_d2@1.00 | 4.0010 | 4.0041 | 0.972137 |

## Controller And Combined Policy

Torch controllers enabled: `False`.

| Layer | Budget | Controller | Action accuracy | Combined cosine | Refresh rate | Effective bps |
|---:|---:|---|---:|---:|---:|---:|
| 22 | 0.50 | threshold | 0.7533 | 0.936656 | 0.8125 | 3.2636 |
| 22 | 0.50 | decision_tree | 0.7733 | 0.936379 | 0.7937 | 3.1883 |
| 22 | 1.00 | threshold | 0.6333 | 0.936001 | 0.7500 | 3.0759 |
| 22 | 1.00 | decision_tree | 0.7200 | 0.938854 | 0.5188 | 2.3598 |
| 22 | 1.58 | threshold | 0.5867 | 0.932027 | 0.6625 | 2.8542 |
| 22 | 1.58 | decision_tree | 0.6667 | 0.941025 | 0.4437 | 2.2258 |
| 22 | 2.00 | threshold | 0.6067 | 0.948129 | 0.3000 | 2.2361 |
| 22 | 2.00 | decision_tree | 0.6267 | 0.948081 | 0.3125 | 2.2238 |
| 22 | 4.00 | threshold | 0.7000 | 0.959783 | 0.1875 | 3.2562 |
| 22 | 4.00 | decision_tree | 0.7133 | 0.959472 | 0.2062 | 3.1815 |

## Claim Boundary

- Predictor, VQ, and controller rows use disjoint clip-level splits.
- Codebook, bitmap, scale, action, and controller metadata are counted.
- `payload_bps` excludes one-time static codebook/controller bits; `effective_bps` amortizes them over the test corpus.
- DLGN policy results use the final hard gate network. Soft accuracy and discretization gap are reported separately.
- The RGB controller is a pre-encoder proxy, but innovation coding still requires current hidden features. No ViT skip claim is made.

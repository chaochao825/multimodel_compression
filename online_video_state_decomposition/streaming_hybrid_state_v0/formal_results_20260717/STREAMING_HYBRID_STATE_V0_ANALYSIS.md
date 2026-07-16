# Streaming Hybrid State V0 Analysis

## Executive Summary

- Spectral/Fourier predictor: **Negative**.
- Prediction-residual VQ: **Mixed**.
- Hardened logic controller: **Negative**.
- Open-loop memory representation: **Mixed**.
- Conditional visual compute: **Negative**.
- End-to-end streaming Video-LLM competitiveness: **UNVALIDATED**.
- No encoder speedup, task-accuracy, or hardware PPA claim is made.

## 1. Predictor Ablation

| Layer | Best simple | Simple NMSE | Best Fourier | Fourier NMSE | Fourier change | Raw->residual entropy | Winner |
|---|---|---|---|---|---|---|---|
| 15 | ema_050 | 0.3291 | fourier_h8_k2 | 1.2851 | +290.5% | -78.7% | simple |
| 22 | ema_075 | 0.3250 | fourier_h8_k2 | 1.2361 | +280.3% | -91.8% | simple |

Fourier is positive only when validation-selected Fourier beats the validation-selected simple causal predictor on test NMSE.

## 2. Multi-Bit Rate-Quality Comparison

| Layer | Nominal budget | Method | Point | Actual bps | Cosine | NMSE |
|---|---|---|---|---|---|---|
| 15 | 0.50 | raw_pq | pq_0p5b_k16_d8 | 0.501 | 0.6483 | 0.5835 |
| 15 | 0.50 | residual_pq | pq_0p5b_k16_d8@1.00 | 0.721 | 0.7704 | 0.4302 |
| 15 | 1.00 | raw_pq | pq_1b_k16_d4 | 1.000 | 0.7540 | 0.4361 |
| 15 | 1.00 | residual_pq | pq_1p5b_k64_d4@0.50 | 1.190 | 0.8990 | 0.1926 |
| 15 | 1.58 | raw_pq | pq_1p5b_k64_d4 | 1.502 | 0.9045 | 0.1860 |
| 15 | 1.58 | residual_pq | pq_1p5b_k64_d4@1.00 | 1.659 | 0.9367 | 0.1243 |
| 15 | 2.00 | raw_pq | pq_2b_k256_d4 | 2.006 | 0.9464 | 0.1093 |
| 15 | 2.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.132 | 0.9648 | 0.0700 |
| 15 | 2.00 | scalar_quant | int2 | 2.016 | 0.5341 | 0.7084 |
| 15 | 4.00 | raw_pq | pq_4b_k256_d2 | 4.003 | 0.9835 | 0.0381 |
| 15 | 4.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.132 | 0.9648 | 0.0700 |
| 15 | 4.00 | scalar_quant | int4 | 4.016 | 0.9037 | 0.2038 |
| 22 | 0.50 | raw_pq | pq_0p5b_k16_d8 | 0.501 | 0.6105 | 0.6266 |
| 22 | 0.50 | residual_pq | pq_0p5b_k16_d8@1.00 | 0.721 | 0.7390 | 0.4843 |
| 22 | 1.00 | raw_pq | pq_1b_k16_d4 | 1.000 | 0.7808 | 0.3908 |
| 22 | 1.00 | residual_pq | pq_1p5b_k64_d4@0.50 | 1.190 | 0.8763 | 0.2252 |
| 22 | 1.58 | raw_pq | pq_1p5b_k64_d4 | 1.502 | 0.8845 | 0.2162 |
| 22 | 1.58 | residual_pq | pq_1p5b_k64_d4@1.00 | 1.659 | 0.9240 | 0.1407 |
| 22 | 2.00 | raw_pq | pq_2b_k256_d4 | 2.006 | 0.9467 | 0.1054 |
| 22 | 2.00 | residual_pq | pq_2b_k256_d4@1.00 | 2.132 | 0.9593 | 0.0779 |
| 22 | 2.00 | scalar_quant | int2 | 2.016 | 0.3783 | 0.8614 |
| 22 | 4.00 | raw_pq | pq_4b_k256_d2 | 4.003 | 0.9887 | 0.0244 |
| 22 | 4.00 | residual_pq | pq_4b_k256_d2@1.00 | 4.004 | 0.9868 | 0.0266 |
| 22 | 4.00 | scalar_quant | int4 | 4.016 | 0.9368 | 0.1344 |

### Residual Entropy

| Layer | Codec | Raw H bps | Residual H bps | Reduction | Raw cosine | Residual cosine |
|---|---|---|---|---|---|---|
| 15 | pq_0p5b_k16_d8 | 0.467 | 0.315 | 32.4% | 0.6483 | 0.7704 |
| 15 | pq_1b_k16_d4 | 0.926 | 0.696 | 24.8% | 0.7540 | 0.8384 |
| 15 | pq_1p5b_k64_d4 | 1.430 | 1.219 | 14.8% | 0.9045 | 0.9367 |
| 15 | pq_2b_k256_d4 | 1.967 | 1.811 | 7.9% | 0.9464 | 0.9648 |
| 15 | pq_4b_k256_d2 | 3.956 | 3.750 | 5.2% | 0.9835 | 0.9536 |
| 22 | pq_0p5b_k16_d8 | 0.480 | 0.294 | 38.7% | 0.6105 | 0.7390 |
| 22 | pq_1b_k16_d4 | 0.959 | 0.689 | 28.2% | 0.7808 | 0.8584 |
| 22 | pq_1p5b_k64_d4 | 1.460 | 1.292 | 11.5% | 0.8845 | 0.9240 |
| 22 | pq_2b_k256_d4 | 1.963 | 1.847 | 5.9% | 0.9467 | 0.9593 |
| 22 | pq_4b_k256_d2 | 3.966 | 3.754 | 5.4% | 0.9887 | 0.9868 |

Residual VQ is retained when matched full-PQ index entropy falls by more than 30%, or when it improves held-out distortion at a comparable reported stream rate. This probe does not test task accuracy.

## 3. Logic Controller

| Layer | Budget | MLP bAcc | DLGN bAcc | Soft-hard gap | DLGN near MLP |
|---|---|---|---|---|---|
| 15 | 0.50 | 0.432 | 0.448 | -0.240 | False |
| 15 | 1.00 | 0.554 | 0.419 | -0.267 | False |
| 15 | 1.58 | 0.553 | 0.549 | -0.227 | False |
| 15 | 2.00 | 0.597 | 0.434 | -0.227 | False |
| 15 | 4.00 | 0.581 | 0.395 | -0.240 | False |
| 22 | 0.50 | 0.451 | 0.373 | +0.000 | False |
| 22 | 1.00 | 0.626 | 0.467 | -0.027 | False |
| 22 | 1.58 | 0.567 | 0.463 | -0.227 | False |
| 22 | 2.00 | 0.620 | 0.539 | -0.347 | False |
| 22 | 4.00 | 0.439 | 0.373 | +0.000 | False |

DLGN enters the provisional Pareto set only when test balanced accuracy is within 0.02 of MLP, its static description is smaller, and validation soft-hard gap is at most 0.02.

## 4. Combined Open-Loop Policy

| Layer | Budget | Controller | Cosine | Effective bps | Refresh | Innovation | Encoder required | Encoder skip | State-update proxy | RQ dominated | Compute candidate |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 22 | 2.00 | threshold | 0.9553 | 2.098 | 0.231 | 0.581 | 0.812 | 0.188 | 0.377 | False | False |
| 22 | 2.00 | decision_tree | 0.9542 | 2.010 | 0.206 | 0.588 | 0.794 | 0.206 | 0.353 | False | False |
| 22 | 2.00 | dlgn | 0.9549 | 2.048 | 0.206 | 0.606 | 0.812 | 0.188 | 0.358 | False | False |
| 22 | 4.00 | threshold | 0.9703 | 3.254 | 0.062 | 0.750 | 0.812 | 0.188 | 0.250 | False | False |
| 22 | 4.00 | decision_tree | 0.9700 | 3.179 | 0.062 | 0.731 | 0.794 | 0.206 | 0.245 | False | False |
| 22 | 4.00 | dlgn | 0.9703 | 3.255 | 0.062 | 0.750 | 0.812 | 0.188 | 0.250 | False | False |

A memory-representation candidate needs mean cosine >=0.95 and lower effective bps than always-INT4 refresh on the same held-out layer, and it must not be rate-quality dominated by standalone raw PQ, residual PQ, or scalar quantization. It is not automatically a compute-saving point.

A conditional-compute candidate additionally needs encoder_required_rate = innovation_rate + refresh_rate <0.30. Innovation coding requires the current hidden state and therefore does not skip the visual encoder.

The learned policies selected the cheap predictor action at a maximum rate of 0.000. A zero value means the nominal four-way controller collapsed to reuse/innovation/refresh on this corpus.

## 5. Scientific Verdict

The components are not automatically stronger when stacked. A weak Fourier result removes Fourier from the preferred path; a useful residual code or controller can still be retained independently. The combined result is a bounded-state latent codec/controller probe, not a direct comparison with published Video-LLM task accuracy or measured latency.

The next experiment should connect the qualified 4-bit policy to a task-level streaming benchmark and first reduce the true encoder-required rate. RTL work is justified only after that gate.

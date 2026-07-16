# Streaming Hybrid State V0

This package evaluates three proposed components on the same frozen visual
hidden-state corpus:

1. causal temporal predictors, including simple and Fourier-basis models;
2. raw-state, prediction-residual, and multi-bit quantization;
3. threshold, decision-tree, MLP, and hardened differentiable-logic
   controllers.

It then evaluates their open-loop combination with four actions:

```text
0: reuse the decoded previous state
1: run the causal predictor
2: run the predictor and transmit a PQ innovation
3: refresh with INT4 scalar quantization
```

The controller sees only low-cost current/previous RGB statistics. Codebooks
and predictors are selected on train/validation clips; all reported policy
quality is measured on disjoint test clips. The experiment is a
representation-level causal probe. It does not establish end-to-end
Video-LLM accuracy, encoder skipping, PPA, area, power, or latency.

Action 2 still needs the current hidden state to form its residual. The
compute-facing metric is therefore:

```text
encoder_required_rate = innovation_rate + refresh_rate
encoder_skip_rate     = reuse_rate + predict_rate
```

The lower state-update proxy assigned to innovation is not a ViT cost model.

Predictor rows include raw-state and prediction-residual temporal spectral
entropy, so a frequency model is retained only when it beats the simple
causal baselines rather than merely producing a low-frequency residual.

## Run

```bash
python -m streaming_hybrid_state.evaluate \
  --root /path/to/clip_corpus \
  --out-dir /path/to/results \
  --layers 15,22

python -m streaming_hybrid_state.analyze \
  --results-dir /path/to/results

python -m streaming_hybrid_state.review \
  --results-dir /path/to/results \
  --source-root /path/to/repository
```

Use `--no-torch` to omit the optional MLP and DLGN controllers. The primary
rate sweep includes nominal `0.5`, `1.0`, `1.58`, `2.0`, and `4.0`
bit/scalar innovation budgets. Actual stream and amortized effective rates
are always reported separately.

## Outputs

- `predictor_results.csv`
- `vq_results.csv`
- `controller_results.csv`
- `combined_results.csv`
- `split_manifest.csv`
- `summary.json`
- `RESULT_SUMMARY.md`
- `review_report.md`
- `review_findings.json`

## Regression

```bash
python -m unittest discover \
  -s streaming_hybrid_state/tests \
  -p "test_*.py"
```

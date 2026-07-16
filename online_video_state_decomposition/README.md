# Online Video State Decomposition

Probe-first research code and audited evidence for bounded online-video
understanding. The project tests whether exact recent evidence,
query-conditioned bounded history, compressed native visual state, and sparse
event residuals can form a useful algorithm-system trade-off.

The current evidence does **not** support replacing video attention with a
global BCCB operator. Block-circulant and low-rank modules remain optional
components for a later writer, router, or projection accelerator.

## Current Result

The strongest confirmed configuration uses a frozen LLaVA-1.5-7B visual
feature cache, a rank-256 calibration-only PCA codec, and four sparse residual
tokens per frame:

| Result | Full cache | Compressed |
|---|---:|---:|
| Per-stream persistent state | 8.024 MiB | 1.024 MiB |
| Learned-selector accuracy | 51.0% | 50.5% |
| Exact prediction agreement | - | 99.0% |
| Full-correct/compressed-wrong | - | 1 / 200 |
| One-sided 95% loss-rate upper bound | - | 2.35% |

The state is 7.84x smaller, but the strict 2% non-inferiority gate is not
passed. At matched compressed state, query-conditioned learned selection
reaches 50.5% versus 46.5% exact recent: +4.0 points, 95% interval
[+1.0, +7.0], 9 better / 1 worse, McNemar p=0.0215.

See
[FEATURE_MEMORY_COMPRESSION_ANALYSIS_20260718.md](paper/results/probe_mvp/FEATURE_MEMORY_COMPRESSION_ANALYSIS_20260718.md)
for the complete protocol, state accounting, paired tests, and failure
localization.

## Repository Layout

- `configs/`: frozen MVBench split and experiment configuration.
- `experiments/memory/`: fixed-capacity and query-conditioned memory modules.
- `experiments/probes/`: extraction, evaluation, aggregation, and validation.
- `experiments/scripts/`: resumable GPU runners with GPU locks and safety
  checks.
- `experiments/tests/`: unit and accounting regression tests.
- `figures/`: reproducible plotting code.
- `streaming_hybrid_state_v0/`: parallel representation-level probe for
  causal predictors, residual product quantization, and low-cost controllers.
- `paper/notes/innovation/`: literature matrix, decision log, and claim
  boundaries.
- `paper/results/probe_mvp/`: selected aggregate CSV, JSON, reports, and plots.

## Environment

The confirmed remote run used Python from the existing `Qwen3` conda
environment on server 210:

```bash
unset PREFIX
source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate Qwen3
pip install -r experiments/requirements.txt
```

The code was verified with PyTorch 2.6.0, Transformers 4.51.0, NumPy 2.2.6,
and Matplotlib 3.10.9. Missing Python packages may be installed into the
selected environment; create a new conda environment only when dependency
conflicts make reuse unsafe.

## Reproduction

Run the complete unit suite:

```bash
python -m unittest discover -s experiments/tests -v
```

Extract calibration features, fit codecs, and run the rank gate:

```bash
bash experiments/scripts/run_llava_feature_pca_extract_shard.sh ...
bash experiments/scripts/run_llava_feature_pca_fit.sh ...
bash experiments/scripts/run_compressed_feature_rank_sweep.sh ...
```

Run formal compressed native-memory confirmation:

```bash
bash experiments/scripts/run_mvbench_compressed_feature_memory_shard.sh \
  1 0 1 \
  remote_results/mvbench_compressed_feature_confirmation \
  remote_results/mvbench_query_confirmation/aggregate/llava_selection_manifest.json \
  remote_results/llava_feature_pca_calibration/codec_rank256/llava_feature_pca_rank256.pt \
  40 \
  exact_recent,learned_recent_query_topk \
  0,4
```

Aggregate and validate:

```bash
python experiments/probes/aggregate_compressed_feature_memory.py \
  --run-dir remote_results/mvbench_compressed_feature_confirmation

python experiments/probes/validate_compressed_feature_memory.py \
  --run-dir remote_results/mvbench_compressed_feature_confirmation \
  --selection-manifest remote_results/mvbench_query_confirmation/aggregate/llava_selection_manifest.json \
  --split-manifest configs/mvbench/query_memory_split_20260717.json \
  --fit-summary remote_results/llava_feature_pca_calibration/codec_rank256/fit_summary.json \
  --reference-run remote_results/mvbench_feature_memory_confirmation \
  --expected-samples 200 \
  --expected-variants full,pca_r256_s0,pca_r256_s4
```

Full operational details are in
[experiments/README.md](experiments/README.md).

## Data and Model Assets

Model weights, MVBench videos, raw checkpoints, projected-feature tensors, and
development dumps are intentionally excluded. The scripts expect local assets
on the execution server; paths are configurable in the runners. Selected
aggregate evidence is committed so the reported claims can be audited without
redistributing third-party models or datasets.

## Claim Boundary

- PCA, low-rank coding, sparse residuals, BCCB, and their combinations are
  established tools; this repository does not claim these primitives as new.
- The current evidence supports query-conditioned access to bounded history,
  not the optimality of the frozen four-feature ridge selector.
- The compressed state is promising but has not passed the strict finite-
  sample non-inferiority gate.
- Latency measurements are unfused Python/CUDA measurements, not production
  serving or hardware-kernel claims.
- The next mechanism gate is adaptive event allocation on disjoint training
  and confirmation data, followed by replication on a second encoder or
  benchmark.

The parallel `streaming_hybrid_state_v0` probe reaches a similarly bounded
verdict: simple EMA predictors beat its Fourier predictors; residual product
quantization is useful at selected rate-quality points; hardened logic and
conditional-compute controllers fail their promotion gates. It is
representation-level evidence only and does not establish Video-LLM accuracy,
encoder skipping, latency, or hardware PPA.

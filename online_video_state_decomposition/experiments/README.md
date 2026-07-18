# Online Video State Experiments

This directory contains executable probes for budgeted online-video memory.
The current validated path uses LLaVA-1.5-7B projected visual features on a
frozen MVBench split. It separates:

1. online visual-state writing;
2. query-conditioned frame selection;
3. native feature-cache reading; and
4. low-rank feature-state compression with optional sparse token residuals.

The experiments are mechanism and accounting probes, not a claim that PCA,
sparse residual coding, or low-rank adaptation is new.

## Runtime

The verified remote workspace is:

```text
/home/spco/online_video_state_decomposition
```

The verified Python environment on server 210 is:

```bash
unset PREFIX
source /home/wangmeiqi/anaconda3/etc/profile.d/conda.sh
conda activate /home/wangmeiqi/anaconda3/envs/Qwen3
```

The scripts reuse local LLaVA and MVBench assets. No model or dataset download
is required for the recorded runs:

```text
third_party/llava-v1.5-7b-local
third_party/LLaVA
/home/wangmeiqi/.cache/huggingface/hub/datasets--OpenGVLab--MVBench/
```

## Tests

```bash
python -m unittest discover -s experiments/tests -v
```

The remote suite currently covers circulant probes, selector accounting,
native feature memory, PCA/sparse-residual coding, aggregation, and exact
finite-sample non-inferiority bounds.

## Frozen Data Contract

`configs/mvbench/query_memory_split_20260717.json` contains mutually disjoint
sets:

- 100 calibration videos used to fit selectors/codecs;
- 200 evaluation videos used for formal confirmation;
- 500 untouched reserve videos;
- 200 examples excluded because they were used in earlier formal work.

Codec fitting is label-free. Token checkpoints contain features, indices, and
timing metadata but no questions, options, labels, or answers.

## Native Feature-Memory Anchor

The anchor writes a 16-frame by 64-token by 4096-dimensional FP16 projected
feature cache, then reads eight selected frames without replaying the source
video:

```bash
bash experiments/scripts/run_mvbench_feature_memory_llava_shard.sh ...
python experiments/probes/aggregate_mvbench_llava.py ...
```

See `paper/results/probe_mvp/mvbench_feature_memory_confirmation_20260718_v1/`
for the selected formal aggregate.

## Feature Codec Calibration

Extract label-free calibration tokens:

```bash
bash experiments/scripts/run_llava_feature_pca_extract_shard.sh \
  1 0 1 remote_results/llava_feature_pca_calibration_20260718_v1
```

Fit a codec:

```bash
bash experiments/scripts/run_llava_feature_pca_fit.sh \
  1 remote_results/llava_feature_pca_calibration_20260718_v1 256 1
```

Run the configuration gate over several ranks:

```bash
bash experiments/scripts/run_compressed_feature_rank_sweep.sh \
  1 \
  remote_results/llava_feature_pca_calibration_20260718_v1 \
  remote_results/mvbench_query_confirmation_20260718_v1/aggregate/llava_selection_manifest.json \
  remote_results/mvbench_compressed_feature_smoke_rank_sweep \
  1 \
  64,128,256,512
```

Generate the bound rank-sweep CSV and figure:

```bash
python figures/plot_feature_codec_rank_sweep.py \
  --rank-run 64=remote_results/RUN_R64 \
  --rank-run 128=remote_results/RUN_R128 \
  --rank-run 256=remote_results/RUN_R256 \
  --rank-run 512=remote_results/RUN_R512 \
  --fit-root remote_results/llava_feature_pca_calibration_20260718_v1 \
  --out-dir remote_results/feature_codec_rank_sweep_analysis
```

## Formal Compressed-Memory Confirmation

Evaluate full cache and selected compressed variants:

```bash
bash experiments/scripts/run_mvbench_compressed_feature_memory_shard.sh \
  1 0 1 \
  remote_results/mvbench_compressed_feature_confirmation \
  remote_results/mvbench_query_confirmation_20260718_v1/aggregate/llava_selection_manifest.json \
  remote_results/llava_feature_pca_calibration_20260718_v1/codec_rank256/llava_feature_pca_rank256.pt \
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
  --selection-manifest remote_results/mvbench_query_confirmation_20260718_v1/aggregate/llava_selection_manifest.json \
  --split-manifest configs/mvbench/query_memory_split_20260717.json \
  --fit-summary remote_results/llava_feature_pca_calibration_20260718_v1/codec_rank256/fit_summary.json \
  --reference-run remote_results/mvbench_feature_memory_confirmation_20260718_v1 \
  --expected-samples 200 \
  --expected-variants full,pca_r256_s0,pca_r256_s4
```

The aggregate reports steady-state per-stream bytes and cold-start bytes
including shared codec parameters separately. The 2-point non-inferiority gate
uses a one-sided 95% Clopper-Pearson upper bound on
full-correct/compressed-wrong outcomes; a degenerate bootstrap interval is not
used to promote a configuration. `selector_gain_by_variant.csv` separately
tests query-conditioned versus exact-recent selection within each matched
memory variant.

## Streaming Baseline Mechanism Proxies

Audit pinned external checkouts without vendoring their code:

```bash
python experiments/probes/audit_streaming_baseline_sources.py \
  --manifest experiments/configs/streaming_baseline_sources.json \
  --external-root external_baselines \
  --out remote_results/streaming_source_checkout_audit.json
```

Replay the frozen CLIP cache through CausalMem, StreamingTOM, STC,
SelectStream, OASIS, and StateKV mechanism proxies plus project controls:

```bash
bash experiments/scripts/run_streaming_baseline_proxies.sh \
  remote_results/streaming_baseline_proxy_200
```

This command writes paired statistics, component-level state accounting, and
PNG/PDF plots. CPU proxy timings are not official GPU or end-to-end latency.
The external methods operate at different state layers, so `reproduction_tier`
and the complete active/archive/detailed byte breakdown must remain attached
to every comparison.

## Resource Safety

GPU runners check memory and utilization before launch and take a per-GPU
`flock`. On shared servers, inspect all compute processes first and never stop
or move unrelated jobs.

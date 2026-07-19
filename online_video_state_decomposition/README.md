# Online Video State Decomposition

Probe-first research code and audited evidence for bounded online-video
understanding. The project tests whether exact recent evidence,
query-conditioned bounded history, compressed native visual state, and sparse
event residuals can form a useful algorithm-system trade-off.

The current evidence does **not** support replacing video attention with a
global BCCB operator. Block-circulant and low-rank modules remain optional
components for a later writer, router, or projection accelerator.

## Current Result

The strongest current evidence is a post-hoc mechanism regression on a reused
200-sample MVBench set. With a frozen LLaVA-1.5-7B feature cache and a
calibration-only rank-256 PCA codec, fixed-s4 and spatial grid-2x2 each score
101/200 versus 102/200 for full state, but fail on different samples. A causal
error-oracle route between the two representations scores 102/200 in the
audited v2 scan:

| Result | Full cache | Routed compressed |
|---|---:|---:|
| Steady per-stream tensor payload | 8.024 MiB | 1.024 MiB |
| Cold start including shared codec | 8.024 MiB | 3.031 MiB |
| Learned-selector accuracy | 51.0% | 51.0% |
| Exact prediction agreement | - | 98.5% |
| Full-correct/compressed-wrong | - | 0 / 200 |

The amortized state ratio is 7.84x; the cold-start ratio is 2.65x. The route
computes both candidates and was designed after inspecting the same set, so it
is not a deployable router or independent confirmation. Separately, the
option-aware learned reader shows an exploratory 9-better/1-worse paired
signal over recent-only access at the compressed state (unadjusted McNemar
p=0.0215). Both findings require a frozen, previously unseen reserve set.

See
[FEATURE_MEMORY_COMPRESSION_ANALYSIS_20260718.md](paper/results/probe_mvp/FEATURE_MEMORY_COMPRESSION_ANALYSIS_20260718.md)
for the original protocol and failure localization. The stricter evidence
audit and routed redesign are summarized in
[COMPETITIVENESS_LOSS_REDESIGN_ANALYSIS_20260718.md](paper/results/probe_mvp/COMPETITIVENESS_LOSS_REDESIGN_ANALYSIS_20260718.md).

A separate six-baseline audit now distinguishes official source availability,
paper-to-code mismatches, and unified feature-level mechanism proxies. On the
reused 200-sample development cache, the dev-fitted bounded selector scores
53.0% versus 50.0% for exact recent, but this is post-hoc and not an
independent confirmation. StateKV, OASIS, and StreamingTOM additionally retain
growing detailed or archive state in the audited accounting. See
[STREAMING_BASELINE_REPRODUCTION_AUDIT_20260719.md](paper/results/probe_mvp/STREAMING_BASELINE_REPRODUCTION_AUDIT_20260719.md).
Targeted official-module smokes now also cover CausalMem `FOSSCache`,
StreamingTOM `OQM`, STC core imports, and OASIS `ShortMemory`; these remain
mechanism checks rather than model-level latency or quality reproductions.
Strict machine-readable preflights now additionally pass for CausalMem's
official 50-video/250-question evaluator and both modes of STC's official ReKV
latency benchmark. Their official jobs are waiting in the safe idle-GPU queue;
queue placement is not a quality or latency result. OASIS now also has an
exact-50x5 data adapter, no-copy dataset materializer, and official evaluator
wrapper. Its static source/model/data preflight passes. The server-210
FlashAttention 2.8.3 source build and CPU-side import/ELF audit also pass, with
only `sm_80` cubins and maximum GLIBC requirement 2.14. The CUDA BF16 kernel
preflight and one-video model inference remain incomplete.
OASIS is retained as a slow event-archive quality baseline; a future `pace=0`
whole-run wall time is not request TTFT or SLO latency. See
[CAUSALMEM_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md](paper/results/probe_mvp/CAUSALMEM_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md)
and
[STC_REKV_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md](paper/results/probe_mvp/STC_REKV_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md).
The OASIS contract is
[OASIS_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md](paper/results/probe_mvp/OASIS_OFFICIAL_REPRODUCTION_PROTOCOL_20260719.md).
Its completed static evidence is
[oasis_official_static_preflight_20260719.json](paper/results/probe_mvp/oasis_official_static_preflight_20260719.json);
the intentionally null runtime field means it is not a CUDA launch result.
The separate source-build/import evidence is
[oasis_flash_attn_build_audit_20260719.json](paper/results/probe_mvp/oasis_flash_attn_build_audit_20260719.json),
which likewise is not a CUDA kernel result.

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

Run the audited official CausalMem quality evaluator or one STC ReKV latency
mode. Both launchers enforce pinned sources, complete local models, output
fingerprints, a per-GPU lock, and an idle-GPU gate.

```bash
bash experiments/scripts/run_causalmem_streamingbench.sh \
  causal_mem RUN_NAME 0

bash experiments/scripts/run_stc_rekv_official.sh \
  rekv RUN_NAME 0
bash experiments/scripts/run_stc_rekv_official.sh \
  rekv_stc RUN_NAME 0
```

Prepare and run the audited OASIS one-video smoke before scheduling its formal
50-video quality evaluation:

```bash
bash experiments/scripts/prepare_oasis_streamingbench.sh
bash experiments/scripts/run_oasis_when_idle.sh \
  oasis_smoke_1video_v1 smoke 3
```

The CausalMem and STC jobs are currently waiting through these queue helpers;
no official quality or CUDA timing result has been produced:

```bash
bash experiments/scripts/run_stc_rekv_pair_when_idle.sh RUN_NAME 0
bash experiments/scripts/run_causalmem_when_idle.sh RUN_NAME 0
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

Run the equal-value-budget spatial/sparse router by setting environment
variables on the same shard runner. The reported state sizes are logical
tensor payload bytes; a serialized archive additionally contains format and
layout metadata.

```bash
ROUTED_RESIDUAL_GRIDS=2 \
ROUTED_GRID_ERROR_RATIO=1.0 \
bash experiments/scripts/run_mvbench_compressed_feature_memory_shard.sh \
  1 0 1 \
  remote_results/mvbench_routed_feature_memory \
  remote_results/mvbench_query_confirmation/aggregate/llava_selection_manifest.json \
  remote_results/llava_feature_pca_calibration/codec_rank256/llava_feature_pca_rank256.pt \
  40 \
  learned_recent_query_topk \
  4
```

The router computes both a 2x2 coarse spatial candidate and a top-4 sparse
candidate from the current frame, quantizes both to the configured storage
dtype, and stores the lower-error candidate. It is causal but not yet a
low-cost learned router; matched-value-vector comparisons do not imply
matched writer FLOPs.

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
- The routed result is a post-hoc regression on a reused set. It does not
  replace the failed strict finite-sample gate for fixed-s4 or establish
  generalization.
- Latency measurements are unfused Python/CUDA measurements, not production
  serving or hardware-kernel claims.
- The next mechanism gate is a cheap causal router trained on disjoint data,
  followed by a frozen paired run on at least 400 unseen samples and
  replication on a second encoder or benchmark.

The parallel `streaming_hybrid_state_v0` probe reaches a similarly bounded
verdict: simple EMA predictors beat its Fourier predictors; residual product
quantization is useful at selected rate-quality points; hardened logic and
conditional-compute controllers fail their promotion gates. It is
representation-level evidence only and does not establish Video-LLM accuracy,
encoder skipping, latency, or hardware PPA.

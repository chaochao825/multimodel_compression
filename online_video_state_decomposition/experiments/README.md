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

After method-specific dependencies are installed, run executable module
smokes without writing bytecode into the third-party checkouts:

```bash
python experiments/probes/smoke_external_baseline_modules.py \
  --external-root external_baselines \
  --methods causalmem,streamingtom,stc,oasis \
  --oasis-python .conda/oasis-py312/bin/python \
  --out remote_results/official_module_smoke.json
```

This validates a synthetic CausalMem cache update, StreamingTOM OQM 4-bit
round trip, STC core imports, and OASIS `ShortMemory` window/event behavior.
It is dependency and mechanism evidence only, not a model-level quality or
latency reproduction.

The OASIS static preflight, BF16 CUDA-kernel check, and
one-video/five-question smoke passed on server 210. The audited formal
50-video/250-question run subsequently completed; the launch remains resumable
behind the idle-GPU gate with:

```bash
bash experiments/scripts/prepare_oasis_streamingbench.sh
bash experiments/scripts/run_oasis_when_idle.sh \
  oasis_formal_50x5_v1 formal 1
```

The adapter cross-checks OASIS JSON against the validated CSV and upstream
archive manifest, then creates audited symbolic links instead of copying
videos. The runner hashes source and model assets, enforces official arguments,
and validates exact-prefix resume/output integrity. OASIS is the slow
event-archive quality baseline. Its whole-run `pace=0` wall time covers the
full offline evaluator and is not request TTFT or SLO latency.

While an audited OASIS job is still running, validate its atomic output prefix
and generate a diagnostic progress snapshot with:

```bash
python experiments/probes/analyze_oasis_progress.py \
  --output remote_results/oasis_streamingbench/<run>/rtu_1_50_output.json \
  --metadata third_party/OASIS-StreamingBench-RT-1-50-v1/metadata/rtu_1_50.json \
  --preflight remote_results/oasis_streamingbench/<run>/preflight.json \
  --out-dir remote_results/oasis_progress/<run>/<snapshot>
```

The snapshot writes per-video and per-task CSVs, a machine-readable summary,
and PNG/PDF progress figures. Partial accuracy uses completed questions only,
includes a 95% Wilson interval, and is always marked
`formal_comparison_eligible=false`; completion-order bias is not removed. The
linear remaining-time estimate is diagnostic only. A completed run must still
pass the strict official-result aggregator before entering a method comparison.

Separately, CausalMem and STC model-level jobs and the pinned StreamingTOM
CTR/OQM core triplet can wait in strict idle-GPU queues. The StreamingTOM queue
fixes CTR/OQM-write/OQM-select at 64/64/256 frames, 28 layers, 20 warmups, 200
repeats, and FP16, while holding one project GPU lock across the full triplet:

```bash
bash experiments/scripts/run_streamingtom_kernels_when_idle.sh \
  streamingtom_core_formal_v1 2
```

Waiting in a queue is not an official quality or timing result.

Create a read-only runtime snapshot before rebuilding the evidence matrix:

```bash
python experiments/probes/collect_streaming_runtime_status.py \
  --oasis-run remote_results/oasis_streamingbench/<run> \
  --oasis-metadata third_party/OASIS-StreamingBench-RT-1-50-v1/metadata/rtu_1_50.json \
  --stc-run remote_results/stc_rekv_official/<pair-run> \
  --causalmem-run remote_results/causalmem_streamingbench/<run> \
  --streamingtom-run remote_results/official_streaming_kernels/<run> \
  --streamingtom-preflight-dir remote_results/official_streaming_kernels/<preflight> \
  --out remote_results/runtime_status/<snapshot>.json
```

The collector checks queue vocabulary and PID liveness for nonterminal jobs,
then applies the same strict artifact parsers used by the official aggregator.
A `complete` or `completed` status cannot become `PASS` without a valid final
artifact.

Once audited runs complete, aggregate only their final model-level artifacts:

```bash
python experiments/probes/aggregate_official_streaming_results.py \
  --causalmem-metrics remote_results/causalmem_streamingbench/<run>/metrics.json \
  --stc-result remote_results/stc_rekv_official/<rekv-run>/result.json \
  --stc-result remote_results/stc_rekv_official/<stc-run>/result.json \
  --oasis-result remote_results/oasis_streamingbench/<run>/result.json \
  --streamingtom-summary remote_results/official_streaming_kernels/<run>/ctr/summary.json \
  --streamingtom-summary remote_results/official_streaming_kernels/<run>/oqm_write/summary.json \
  --streamingtom-summary remote_results/official_streaming_kernels/<run>/oqm_select/summary.json \
  --out-dir remote_results/official_streaming_aggregate
```

The aggregator rejects incomplete or internally inconsistent results and writes
CSV, JSON, PNG, and PDF artifacts. Formal 50-video/250-question quality is kept
separate from smoke quality. STC P50/P95/P99 values cover only the instrumented
ViT-encode and visual-token-prefill stages; they are not request-tail latency,
TTFT, or decode latency. StreamingTOM core P50/P95/P99 values are written to a
separate CSV/figure with CUDA-event and synchronized-host-wall timing bases;
they cover only CTR compression, OQM write, and OQM select, whose input scopes
also differ. They are not end-to-end Video-LLM latency or a same-workload speed
ranking. OASIS `pace=0` wall time and method-specific memory fields are likewise
retained only with their original semantics. Proxy results are intentionally
excluded.

When OASIS and CausalMem cover the same question IDs, generate a paired
benchmark-system diagnostic with:

```bash
python experiments/probes/compare_official_streaming_quality.py \
  --oasis-output remote_results/oasis_streamingbench/<run>/rtu_1_50_output.json \
  --oasis-preflight remote_results/oasis_streamingbench/<run>/preflight.json \
  --causalmem-predictions remote_results/causalmem_streamingbench/<run>/official/pred.json \
  --causalmem-manifest remote_results/causalmem_streamingbench/<run>/official/run_manifest.json \
  --out-dir remote_results/official_streaming_aggregate
```

This writes question-level outcomes, task deltas, exact McNemar tests, source
hashes, and PNG/PDF figures. It explicitly records the different official VLM
backbones, so the result is not presented as a memory-module ablation.

Plot and cross-check the per-run GPU monitor trace against its audited result:

```bash
python experiments/probes/plot_official_gpu_trace.py \
  --samples remote_results/<method>/<run>/gpu_samples_<attempt>.csv \
  --result remote_results/<method>/<run>/result.json \
  --out-dir remote_results/<method>/<run>/gpu_trace
```

This writes a normalized CSV, a JSON resource summary, and PNG/PDF memory and
utilization traces. The command rejects timestamp, sample-count, peak-memory,
or peak-utilization disagreement between the raw monitor and `result.json`.

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

Run the pinned StreamingTOM CTR/OQM core microbenchmarks only after the strict
GPU idle gate admits the job:

```bash
bash experiments/scripts/run_streamingtom_kernels_when_idle.sh \
  streamingtom_core_formal_v1 2
```

The queue holds the project GPU lock across CTR, OQM write, and OQM select,
while each child runner rechecks at most 4096 MiB allocated and at most 20%
utilization. It writes `complete` only after all three official-core summaries
pass their quality gates and contain 200 latency samples. These measurements
are core-module P50/P95/P99 latency, not encoder, prefill, TTFT, or end-to-end
request latency.

Build a single evidence-tiered completion matrix without mixing official,
paper-only, smoke, proxy, or post-hoc results:

```bash
python experiments/probes/build_streaming_evidence_matrix.py \
  --runtime-status remote_results/streaming_runtime_status.json \
  --out-dir remote_results/streaming_evidence_matrix
```

The optional runtime-status JSON uses `format_version=1`, an `observed_at`
timestamp, and records containing `method_id`, `stage`, `status`, `detail`, and
`source_path`. The builder re-reads the frozen query-memory, spectral-trigger,
codec, BCCB/event-residual, unified-proxy, and OASIS-smoke artifacts. It writes
nested JSON, flattened evidence and completion CSVs, a Markdown analysis, and
PNG/PDF matrix figures. Every metric retains a `comparability_group`; rows from
different groups must not be ranked together.

## Controlled Dual-Timescale Trigger

Run the matched-rank synthetic trigger gate on static, camera-motion,
lighting, periodic-motion, object-change, scene-cut, OCR, and brief-action
streams:

```bash
bash experiments/scripts/run_spectral_event_trigger.sh \
  remote_results/controlled_spectral_trigger
```

Single-state baselines receive the full basis-rank budget; the dual state
splits the same basis payload between fast and slow channels. The runner fits
thresholds only on calibration seeds, evaluates disjoint seeds, and writes
scenario-level metrics plus paired seed-bootstrap false-trigger intervals.
This is controlled trigger evidence only. It is not Video-LLM task quality,
an official CausalMem reproduction, or end-to-end GPU latency evidence. A
nonzero exit code is expected when any preregistered gate fails; retain and
report the artifacts rather than relaxing the gate post hoc.

## Resource Safety

GPU runners check memory and utilization before launch and take a per-GPU
`flock`. On shared servers, inspect all compute processes first and never stop
or move unrelated jobs.

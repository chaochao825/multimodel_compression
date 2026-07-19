# STC ReKV Official Latency Reproduction Protocol

Date: 2026-07-19

## Status

Both audited preflights pass. The CUDA benchmark remains pending because GPU 0
has not yet satisfied the formal idle gate of at most 4096 MiB allocated and at
most 20% utilization. No latency value is reported from preflight evidence.

| Mode | Preflight | Run fingerprint | Current evidence |
|---|---:|---|---|
| ReKV baseline | pass | `d986af8a7fd988fbdf5297126f506469fb86ef161463909af0b87980cd996b8b` | source, model, runtime, and official imports validated |
| ReKV + STC | pass | `020b8b0c941618aaa6dbb8dff3a19814f54269f4327ebb0e4e0908b0178709f3` | source, model, runtime, and official imports validated |

Machine-readable preflights:

- `stc_rekv_official_rekv_preflight_20260719.json`
- `stc_rekv_official_stc_preflight_20260719.json`

## Evidence Boundary

The runner executes the pinned upstream
`speed_benchmark/benchmark_rekv.py` without modifying the STC checkout. It
measures CUDA-event time for two instrumented stages:

1. the complete vision-tower forward used for video encoding; and
2. language-model forwards with `inputs_embeds`, labeled visual-token LLM
   prefill by the upstream benchmark.

These measurements are not end-to-end TTFT, decode latency, task accuracy, or
SLO throughput. The sum of the two sampled stages is reported only as
`instrumented_stage_sum_ms`; it is not relabeled as end-to-end latency.

## Pinned Inputs

| Item | Pin |
|---|---|
| STC source | commit `cf53f781d8740df5c07d7924756acc429641ffd0` |
| Upstream benchmark SHA-256 | `296a97fabdfffab01fe2578e5849c6d69264f687772c7e9cf5a7654aba433d73` |
| Audited wrapper SHA-256 | `d5d8f9107c319ff9b71cd01832f6c9b5ab92ecf75041edcb60fafcfeb7ee3ce5` |
| Model architecture | `LlavaOnevisionForConditionalGeneration` |
| Model type | `llava_onevision` with `siglip_vision_model` vision tower |
| Model index SHA-256 | `e389b969d3b9b7120f136fbf6592cbbc1c07326157cae6f863fb64559261dae9` |
| Indexed shards | exactly four, 16,061,717,016 total bytes |
| Python runtime | Python 3.10.0, Torch 2.6.0+cu124, Transformers 4.51.0 |

The four shard SHA-256 values, metadata hashes, exact command, and complete
offline model path are stored in each machine-readable preflight. The runner
rejects missing, empty, extra, nested, or unindexed safetensor shards.

The runtime check imports `stc`, the official ReKV model loader, and the
official benchmark module; it also resolves the local model path and constructs
a two-frame synthetic video. This is stronger than a package-version check but
does not load model weights or execute CUDA kernels.

## Matched Configurations

The two modes use the exact environment settings in upstream `run_rekv.sh` and
must run in separate processes because STC configuration is global.

| Setting | ReKV | ReKV + STC |
|---|---:|---:|
| `STC_PATCH_VISION` | 0 | 1 |
| `STC_TOKEN_PER_FRAME` | 196 | 64 |
| `STC_UPDATE_TOKEN_RATIO` | 1.0 | 0.25 |
| `STC_CACHE_INTERVAL` | 4 | 4 |

The formal MVP uses 64 temporally coherent synthetic frames at source size
384, five warmups, and twenty measured repetitions. Both modes use the same
model, FP16 loading path, generated clip, physical GPU, offline cache, and
allocator settings.

## Launch And Validation

```bash
bash experiments/scripts/run_stc_rekv_official.sh rekv RUN_NAME 0
bash experiments/scripts/run_stc_rekv_official.sh rekv_stc RUN_NAME 0
```

Before launching, the wrapper requires:

- the pinned, code-clean upstream checkout, allowing only Python cache files;
- exact model architecture, index, shard set, sizes, and full hashes;
- successful CUDA runtime and official-module imports;
- the per-GPU nonblocking lock; and
- GPU memory/utilization below the formal idle thresholds.

After a run, the wrapper checks mode label, frame and repeat counts, all four
STC configuration fields, peak memory, finite positive samples, and consistency
of upstream min/median/mean/std/max fields with raw samples. It derives observed
P50/P95/P99 without replacing the official raw JSON and writes sampled GPU
memory/utilization separately.

Existing output is never silently overwritten. A mismatched fingerprint is a
hard failure; valid official raw output can be recovered into a wrapper result
after interruption.

## Reporting Contract

Report per mode and per stage:

- all twenty raw samples;
- min, P50, P95, P99, mean, population standard deviation, and max;
- official peak allocated memory and sampled process/total GPU peaks; and
- STC-to-ReKV reduction and speedup calculated from matched statistics.

Do not cite the upstream paper's percentage reduction as a reproduced result.
Do not mix the separate `stc_pruner` core benchmark with this end-to-end model
stage benchmark.

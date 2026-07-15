# Phase 5 MVP Summary

Date: 2026-04-25 local workspace, remote runs executed on 2026-04-27 CST.

## What now runs end-to-end

- `train_peft.py` for `lora`, `bca`, `dre_bcm`
- `eval_peft.py` reloading saved checkpoints
- local GLUE TSV loading for `SST-2` and `RTE`
- offline Arrow loading for `BoolQ`

## Important scope notes

- The current `bca` code path is a tracked block-circulant baseline label built on the same block-circulant parameterization as the structure-only `bcm_only` adapter; it is not yet a more paper-faithful standalone BCA reimplementation.
- This published snapshot keeps lightweight per-run JSON artifacts such as `run_config.json`, `train_metrics.json`, `eval_metrics.json`, and `injection_summary.json`, but omits large PEFT checkpoints like `peft_state.pt`.
- The reported downstream runs depend on server-210-local model and dataset assets. Those source paths are preserved in each published `run_config.json`, but the assets themselves are not bundled into the repository snapshot.

## Remote model / data used

- Model: local GPT-2 snapshot at `/home/spco/base-2-bitnet/.hf_cache/hub/models--gpt2/snapshots/607a30d783dfa663caf39e06633721c8d4cfcd7e`
- GLUE data: `/home/spco/diff_bitnet/dre_bcm/data/glue_data`

## Key runs

| Run | Task | Method | Train samples | Eval samples | Accuracy | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `lora_gpt2_boolq_smoke` | BoolQ | LoRA | 32 | 64 | 0.28125 | prior offline Arrow smoke |
| `lora_gpt2_sst2_smoke` | SST-2 | LoRA | 32 | 64 | 0.515625 | first local GLUE smoke |
| `bca_gpt2_sst2_smoke_fft` | SST-2 | BCA | 8 | 16 | 0.500000 | FFT path, very slow but runnable |
| `dre_gpt2_sst2_smoke_fft` | SST-2 | DRE-BCM | 8 | 16 | 0.500000 | FFT path, runnable on GPU1 |
| `lora_gpt2_rte_smoke` | RTE | LoRA | 32 | 64 | 0.500000 | local GLUE loader verified |
| `lora_gpt2_sst2_mvp` | SST-2 | LoRA | 1024 | 256 | 0.609375 | first more informative MVP run |

## Current takeaways

- Phase 5 is no longer scaffold-only. We now have real downstream PEFT runs on `BoolQ`, `SST-2`, and `RTE`.
- `BCA` is now a runnable baseline in the current code path, but it is still a shared block-circulant parameterization rather than a more paper-faithful reimplementation.
- `BCA` and `DRE-BCM` both currently need `--use-fft` and very small smoke settings on GPT-2 because the structured path is much slower than LoRA in this environment.
- The first informative downstream number is `LoRA + GPT-2 + SST-2 = 0.609375` on a `1024/256` split after `1.0` epoch.
- The next most valuable step is to run matched-budget `LoRA / BCA / DRE-BCM` on the same `SST-2` split and then extend to `RTE`.

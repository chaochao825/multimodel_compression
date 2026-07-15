# Publishing Notes

This directory is a curated publish snapshot of the active DRE-BCM project from:

- `/home/spco/diff_bitnet/dre_bcm`

Included here:

- implementation code under `src/`
- configs and helper scripts
- lightweight experiment summaries, plots, metrics JSON/CSV, and markdown reports

Deliberately omitted from git because they are large intermediate artifacts:

- `results_snapshot/`
- `results/matrix_fit/raw_weights/`
- PEFT checkpoints such as `peft_state.pt`
- tokenizer dumps copied into per-run directories

Important caveats for readers:

- The current `bca` path is a tracked block-circulant baseline label built on the same block-circulant parameterization as the structure-only BCM adapter. It is useful for internal comparison runs, but it is not yet a more paper-faithful standalone BCA reimplementation.
- Published PEFT run folders keep small machine-readable artifacts such as `run_config.json`, `train_metrics.json`, `eval_metrics.json`, `eval_metrics_recomputed.json`, and `injection_summary.json`, but omit the full saved checkpoints needed for complete rerun-by-checkpoint verification.
- Several reported runs depend on server-210-local model caches and local dataset files. Those resolved paths are kept in the published `run_config.json` files for provenance, but the underlying large assets are not mirrored into git.

The source experiment directory on server 210 retains the full raw artifacts.

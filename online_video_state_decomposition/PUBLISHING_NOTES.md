# Publishing Notes

This directory is published as a research snapshot inside
`chaochao825/multimodel_compression`.

## Included

- experiment source, configurations, safety wrappers, and tests;
- the audited `streaming_hybrid_state_v0` representation-level probe and its
  selected aggregate results;
- literature matrix and claim-boundary notes;
- selected aggregate CSV/JSON evidence;
- publication-ready PNG/PDF figures;
- protocol and result-analysis Markdown files.

## Excluded

- model weights and Hugging Face caches;
- MVBench and other source videos;
- raw per-sample checkpoints under `remote_results/`;
- projected-feature tensors and PCA codec checkpoints;
- development-only Qwen/CLIP dumps;
- large per-query metric traces and packaged tar archives;
- logs, lock files, Python caches, and local trash.

The exclusions are enforced by `.gitignore` and by the server-side staging
copy. Fit summaries include ranks, dimensions, hashes, and calibration IDs so
the omitted codec files can still be audited against a local rerun.

## Reproduction Boundary

The committed aggregate outputs are sufficient to reproduce tables and inspect
the statistical decisions. Re-running model inference requires locally
licensed/downloaded model and dataset assets. No credentials, passwords,
private keys, tokens, or server authentication files belong in this snapshot.

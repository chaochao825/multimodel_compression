# TileLogic-RVQ Formal Result Bundle

This directory is the sanitized public evidence bundle for the formal
TileLogic-RVQ experiment. The method combines tile-local DCT, scaled base VQ,
sequential residual VQ, calibration-only MLP and discrete-logic routing,
fixed slots, and a fully charged sparse FP16 fallback.

## Frozen Decisions

- Q1: **FAIL** - Base VQ extends the full-overhead frontier beyond INT4
- Q2: **FAIL** - Fisher RVQ improves on base-only and unweighted RVQ
- Q3: **FAIL** - MLP routing beats energy and cosine-risk heuristics
- Q4: **INCONCLUSIVE** - Discrete logic retains the MLP routing benefit
- Q5: **FAIL** - Fixed slots reduce layout, decoder, and TTFT cost
- Q6: **FAIL** - Fully charged exact fallback improves the frontier

Aggregate positive claim allowed: **NO**.

## Included

- Aggregate feature, rate, Fisher, quality, router, latency, TTFT, and memory tables.
- Six frozen-rule decisions, the independent machine audit, and the sole
  Review Agent's initial and final verdicts.
- The cache-provenance migration record and artifact-precision audit.
- PDF/PNG figures, environment summaries, per-sample latency rows, and the
  during-run GPU co-residency snapshot.
- `PUBLICATION_MANIFEST.json` with SHA256 and byte count for every public file.

## Deliberately Excluded

- Model checkpoints, images, cached visual tensors, and private dataset paths.
- Trained `.pt` codebooks/router payloads and build caches.
- Per-example questions, answers, predictions, and 20 MiB feature JSONL.
- Credentials, private server paths, and unrelated experiment logs.

The private run was audited against those excluded records before this bundle
was produced. The public bundle supports result and claim inspection, but does
not by itself reproduce cache extraction or model inference.

The rate ledger charges FP32 for scalar scales, VQ metric weights,
MLP/normalizer state, logic leaves, and curvature priors; FP16 is charged only
for codewords, scale tables, logic thresholds, and exact residual values that
are explicitly serialized or exactly round-tripped at that precision.
Per-entry cache provenance was
backfilled without changing any cached tensor payload and is hash-linked in the
included migration record.

Hashes named `source_*` describe the private pre-sanitization evidence used by
the machine audit. Hashes named `public_*` are recomputed from the actual files
in this bundle after path and whitespace sanitization. Both scopes are retained
so the public provenance chain does not reinterpret private source hashes.

The quality path reconstructs all 1,280 visual tokens. Latency is a PyTorch
implementation diagnostic, not native compact-prefill evidence, isolated-GPU
signoff, kernel-fusion evidence, PPA, or post-layout hardware evidence.

# TileSpec-Ex Minimal Feasibility Experiment

## Decision Contract

This experiment asks three predeclared questions before any fused-kernel or
full benchmark investment:

The novelty and comparison boundary is summarized in
[`TILESPEC_EX_RELATED_WORK.md`](TILESPEC_EX_RELATED_WORK.md).

1. Does tile-local low-pass outperform one global low-pass transform?
2. Does task-risk exception selection outperform residual-energy selection?
3. Does a structured 2x2 exception layout deliver measured GPU latency gains?

All three gates must pass before proceeding to a fused kernel. A negative gate
is a useful stop or scope-reduction result, not an implementation failure.

## Fixed Scope

- Model: `Qwen2.5-VL-3B-Instruct`.
- Multi-tile adapter: one 448x448 global thumbnail plus four independently
  encoded 448x448 TL/TR/BL/BR crops.
- Datasets: 200 GQA validation samples, 200 TextVQA validation samples, and
  200 ChartQA test samples.
- Crop-token retention: 12.5% and 25%. The thumbnail is always retained.
- Headline methods:
  1. no compression;
  2. spatial average pooling;
  3. global DCT low-pass;
  4. per-tile DCT low-pass;
  5. per-tile low-pass plus energy-selected 2x2 residual blocks;
  6. per-tile low-pass plus risk-selected 2x2 residual blocks.
- Structural ablation: arbitrary risk tokens, dynamic 2x2 risk blocks, and
  fixed per-tile block slots. These are not additional headline baselines.

The adapter is a controlled five-image Qwen input, not a claim that the base
checkpoint has a native LLaVA-NeXT-style AnyRes cropper. The result is scoped
to this declared topology.

## Representation

The four crop feature grids are `X_t in R^(16x16xC)`. Tile-local low-pass uses
an orthonormal DCT independently in each crop. Global low-pass stitches the
four crops into one `32x32` grid and retains the same total coefficient count.

Exception methods spend 75% of the retained crop-token budget on the tile
low-frequency base and 25% on exact residual tokens. Energy and risk methods
therefore use identical base, block count, and compact vector count. Risk is

```text
risk(block, query) = residual_energy(block) * query_relevance(block, query)
```

where relevance is cosine similarity after the Qwen visual merger, mapped to
`[0,1]`. Oracle sensitivity is independently measured as
`abs(<dL/dX_block, residual_block>)` using teacher-forced answer NLL.

## Evidence Boundaries

The task-quality path injects reconstructed embeddings with the original 1280
visual-token length. It tests whether information survives compression; it
does not save prefill work and is not latency evidence.

The compact representation has 256 thumbnail vectors plus 128 or 256 retained
crop vectors. The latency probe measures selector, gather/layout, and the real
first-layer Qwen `q_proj` on A800. Its payload exactly matches the quality
methods: 96+32 or 192+64 base/exception tokens, where the exceptions are 8 or
16 selected 2x2 blocks. The compact prefill diagnostic includes 256 thumbnail
tokens, 128 or 256 crop tokens, 64 text tokens, the full language decoder, and
first-token logits. It reports P50/P95/P99 for preblocked and layout-included
paths. It still omits the visual encoder, native multimodal position
construction, and end-to-end TTFT, so it cannot decide the structured gate.

## Completed Outcome

The 2026-07-17 run used all 600 declared samples and 48 oracle-gradient
samples. A review-driven correction aligned the latency payload, fixed the
fixed-slot selector path, refreshed all answer-derived fields, and upgraded
the audit to recompute gate inputs from raw records. The corrected audit passes.

- Tile-local passed at 25% retention but failed at 12.5% because GQA dropped
  by 2.0 percentage points versus global low-pass.
- Risk exceptions beat energy exceptions by only 0.25 percentage points on
  TextVQA+ChartQA at each rate, below the declared 2-point threshold.
- The structured gate is **INCONCLUSIVE**, not FAIL or PASS. In aligned
  diagnostics, layout-included gather was 217%-279% slower, the combined
  selector+gather+q_proj path was 6.8%-15.4% slower, and compact prefill plus
  logits ranged from a 0.9% improvement to a 5.8% regression. These negative
  diagnostics do not replace the missing end-to-end multimodal TTFT test.

The result therefore says not to invest in a fused kernel: the first two gates
already failed, and the third is not validated. Reproducible summaries and the
independent raw-data review are tracked in
[`remote_logs/tilespec_ex_minimal_20260717/`](remote_logs/tilespec_ex_minimal_20260717/).

The snapshot uses the first deterministic 200 viewer rows per dataset (and one
joined question per selected GQA image), not a random or stratified sample.
This is appropriate for a feasibility screen but limits population-level
claims.

## Reproduction

Download the deterministic dataset snapshot:

```powershell
python scripts/download_tilespec_ex_dataset.py `
  --samples-per-dataset 200 `
  --output-dir data/tilespec_ex_minimal
```

Run task quality and oracle sensitivity on the model server:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
python scripts/run_tilespec_ex_quality.py \
  --manifest data/tilespec_ex_minimal/manifest.jsonl \
  --data-root data/tilespec_ex_minimal \
  --output-dir results/tilespec_ex_minimal_20260717 \
  --samples-per-dataset 200 \
  --oracle-samples-per-dataset 16
```

Run the isolated latency probe after the quality process exits:

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=$PWD \
python scripts/benchmark_tilespec_ex_latency.py \
  --output-dir results/tilespec_ex_minimal_20260717 \
  --batch-sizes 1 4 8 16 \
  --no-compile
```

The recorded run used uniform eager execution because TorchInductor failed on
this Python 3.10 environment with a `PY_SSIZE_T_CLEAN` extension error. This
keeps layout comparisons on one execution backend; the failure is not hidden
or counted as a method failure.

If answer-normalization code changes after inference, preserve the original
records and recompute scores before analysis:

```bash
PYTHONPATH=$PWD python scripts/rescore_tilespec_ex_quality.py \
  --results-dir results/tilespec_ex_minimal_20260717 \
  --trash-root trash
```

Apply gates and run the independent audit:

```bash
PYTHONPATH=$PWD python scripts/analyze_tilespec_ex.py \
  --results-dir results/tilespec_ex_minimal_20260717
PYTHONPATH=$PWD python scripts/review_tilespec_ex.py \
  --results-dir results/tilespec_ex_minimal_20260717
```

Unit tests cover DCT round-trip, exact compact budgets, exception restoration,
answer metrics, and structural budget equality:

```bash
python -m pytest -q
```

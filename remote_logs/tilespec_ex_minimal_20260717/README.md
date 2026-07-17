# TileSpec-Ex Minimal Result Bundle

This is the compact, Git-tracked evidence bundle for the corrected 2026-07-17
feasibility run. The experiment used Qwen2.5-VL-3B-Instruct, 200 samples each
from GQA, TextVQA, and ChartQA, crop-token retention rates of 12.5% and 25%,
and the six methods declared in the experiment contract.

## Scientific Outcome

- Tile-local better than global: **FAIL** overall (25% PASS, 12.5% FAIL).
- Risk exceptions better than energy exceptions: **FAIL** at both rates.
- Structured blocks have a real latency benefit: **INCONCLUSIVE**.
- All-three fused-kernel investment gate: **FAIL**.
- Decision: do not start fused-kernel work for the current design.

The aligned structured diagnostic uses the exact method budgets: 96+32 tokens
at 12.5% and 192+64 tokens at 25%, with 8/16 exception blocks. Once
raster-to-block layout is included, gather is 217%-279% slower than arbitrary
token gathering and selector+gather+q_proj is 6.8%-15.4% slower. Compact
prefill plus first-token logits has no stable gain. These are useful negative
diagnostics, but they omit the visual encoder and native multimodal TTFT, so
the structured scientific gate is not declared FAIL or PASS.

`TILESPEC_EX_MINIMAL_REPORT.md` is the primary summary. `review_report.md` and
`review_findings.json` show the corrected audit, which independently
recomputes answer fields, quality/oracle gate inputs, latency reductions, and
claim status from raw records.

## Provenance

- Remote workspace: `/home/spco/sow_linear/tilespec_ex_minimal_20260717`
- Remote result directory: `results/tilespec_ex_minimal_20260717`
- Python: 3.10; PyTorch: 2.6.0+cu124; Transformers: 4.51.0
- GPU: NVIDIA A800 80GB PCIe
- Quality records: 600 samples / 9,000 method variants
- Oracle records: 48 samples (16 per dataset)
- Latency backend: uniform eager execution
- Dataset manifest SHA-256:
  `2dd118b849e920b94b6dffe859dd7cfe464ebbcc52589e53b0e001325c858d5a`

`initial_metric_rescore_summary.json` records the first metric correction,
which changed 96 saved scores. `rescore_summary.json` records the later
review-driven refresh: zero score changes, 632 normalized-prediction updates,
and two agreement-field updates. The scientific quality gates were unchanged.

## Verification

The final 210-server tree was rechecked with:

```bash
/home/wangmeiqi/anaconda3/envs/Qwen3/bin/python -m pytest -q
PYTHONPATH=. /home/wangmeiqi/anaconda3/envs/Qwen3/bin/python \
  scripts/review_tilespec_ex.py \
  --results-dir results/tilespec_ex_minimal_20260717
```

The test result was `34 passed in 1.40s`. The raw-data reviewer returned
`overall: PASS`, no major or minor issues, and the scientific outcome
`tile_local=false`, `risk_exception=false`,
`structured_latency=INCONCLUSIVE`.

`raw_records.tar.gz` contains the two raw audit inputs:
`quality_samples.jsonl` (5.6 MB) and `oracle_blocks.jsonl` (2.2 MB). Its
SHA-256 is
`dc7a006531746d1be715626d3a588c6e988ff626f5eaee10bdd77fc7dbe518de`.
Extract it with `tar -xzf raw_records.tar.gz`; placing the extracted files
beside the aggregate CSV/JSON files makes the raw-data reviewer inputs
portable. The uncompressed local working copies remain under the ignored
path `results/tilespec_ex_minimal_runs/tilespec_ex_minimal_20260717/`.

## Claim Boundary

The quality path reconstructs full-length visual embeddings and is not speed
evidence. The aligned latency path includes compact packing, thumbnail/crop/
text payloads, the language decoder, and first-token logits. It does not
measure the visual encoder, native multimodal position construction,
end-to-end TTFT, a fused kernel, or model-wide memory improvement.

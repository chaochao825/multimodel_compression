# Review Remediation

## Review Scope

One Review Agent audited the implementation, raw and aggregate result
accounting, scientific gates, latency claim boundary, and regression tests.
The same agent performed the final re-review after remediation.

## Initial Findings And Fixes

1. Structured latency budgets did not match the method contract. The
   benchmark now uses exact 75% base plus 25% exception splits: 96+32 and
   192+64 crop tokens.
2. Fixed-slot timing did not consume real per-tile selector output. It now
   uses the selected block indices and includes raster-to-block layout work.
3. The decoder proxy omitted thumbnail/text context and was described too
   broadly. The aligned diagnostic now includes 256 thumbnail tokens, 64 text
   tokens, compact crop tokens, the full decoder, and first-token logits; the
   structured scientific gate is explicitly `INCONCLUSIVE` because the visual
   encoder, native multimodal positions, and end-to-end TTFT are absent.
4. The reviewer trusted aggregate booleans. It now recomputes answer fields,
   quality/oracle gate inputs, latency reductions, and gate status from raw
   JSONL/CSV records.
5. Rescoring did not refresh all derived answer fields. It now updates score,
   normalized prediction, and agreement with the full-token result.
6. Regression coverage was missing. Tests now cover exact budgets,
   fixed-slot and dynamic mappings, signed metrics, and rescore behavior.

## Final Re-review

**PASS.** All six findings are closed and no new major or blocking issue was
found. The verified scientific status is:

- Tile-local better than global: **FAIL** overall.
- Risk exceptions better than energy exceptions: **FAIL** at both rates.
- Structured blocks have a real latency benefit: **INCONCLUSIVE**.
- All-three fused-kernel investment gate: **FAIL**.

The compressed raw audit inputs are Git-tracked in `raw_records.tar.gz`
(SHA-256
`dc7a006531746d1be715626d3a588c6e988ff626f5eaee10bdd77fc7dbe518de`),
closing the reviewer's checkout-level reproducibility concern.

## Residual Risks

- A conclusive structured-latency result requires native compact multimodal
  execution, real mRoPE/position construction, the visual encoder, and
  end-to-end TTFT.
- Current latency numbers are diagnostic measurements from one A800 in eager
  mode with random embeddings and a fixed 64-token text payload.
- The Review Agent performed a read-only source/result consistency review;
  remote tests and the raw-data reviewer provide the executable verification.

# TileSpec-Ex Minimal Feasibility Report

## Executive Summary

- Tile-local > global: **FAIL**
- Risk exception > energy exception: **FAIL**
- Structured block real latency benefit: **INCONCLUSIVE**
- All-three investment gate: **FAIL**
- Decision: Do not invest in a fused kernel: tile-local and risk gates failed, and the structured gate remains inconclusive.

The task-quality path reconstructs the original visual-token count. It tests representation fidelity, not speed. The aligned latency path uses the exact 75% base plus 25% exception budget, includes the thumbnail and text payload in language-decoder prefill, and emits first-token logits. It still omits the visual encoder, native multimodal position construction, and end-to-end TTFT, so the structured gate remains inconclusive.

## Experiment Contract

- Model: Qwen2.5-VL-3B-Instruct with deterministic five-image multi-tile adapter
- Samples: 200 each from GQA, TextVQA, and ChartQA
- Crop-token retention: 12.5% and 25%; the global thumbnail is always retained
- Main methods: none, average pooling, global low-pass, tile low-pass, tile+energy exception, tile+risk exception
- Structural ablation: arbitrary risk tokens, dynamic 2x2 risk blocks, fixed per-tile block slots

## Tile-Local Versus Global

- 0.125: accuracy delta -0.0058, loss-event reduction 4.8%, boundary-MSE reduction 12.5%, GQA delta -0.0200, gate FAIL.
- 0.250: accuracy delta -0.0042, loss-event reduction -7.4%, boundary-MSE reduction 22.3%, GQA delta +0.0050, gate PASS.

## Risk Versus Energy

- 0.125: TextVQA+ChartQA delta +0.0025, oracle Spearman delta +0.0008 (95% CI +0.0000 to +0.0017), scorer FLOP ratio 0.126%, gate FAIL.
- 0.250: TextVQA+ChartQA delta +0.0025, oracle Spearman delta +0.0009 (95% CI +0.0000 to +0.0018), scorer FLOP ratio 0.146%, gate FAIL.

## Structured Execution

- risk_block_dynamic @ 0.125: accuracy loss -0.0006, gather P50 reduction -278.8%, selector+gather+q_proj P50 reduction -10.1%, compact-prefill-plus-logits reduction -2.7%, diagnostic FAIL, gate INCONCLUSIVE.
- risk_block_dynamic @ 0.250: accuracy loss -0.0033, gather P50 reduction -217.4%, selector+gather+q_proj P50 reduction -6.8%, compact-prefill-plus-logits reduction 0.9%, diagnostic FAIL, gate INCONCLUSIVE.
- risk_block_fixed_slots @ 0.125: accuracy loss +0.0017, gather P50 reduction -277.9%, selector+gather+q_proj P50 reduction -15.4%, compact-prefill-plus-logits reduction -2.7%, diagnostic FAIL, gate INCONCLUSIVE.
- risk_block_fixed_slots @ 0.250: accuracy loss -0.0100, gather P50 reduction -219.3%, selector+gather+q_proj P50 reduction -13.3%, compact-prefill-plus-logits reduction -5.8%, diagnostic FAIL, gate INCONCLUSIVE.

## Claim Boundary

No fused kernel, end-to-end multimodal prefill/TTFT speedup, or model-wide memory reduction is claimed. The aligned latency rows are diagnostics only; an end-to-end structured gate decision requires a native compact multimodal execution path.

# TileSpec-Ex Related-Work Positioning

This comparison is current as of 2026-07-17 and uses primary paper pages.
Numbers are not ranked directly because models, datasets, insertion points,
and latency boundaries differ.

| Work | Existing coverage | Consequence for TileSpec-Ex |
|---|---|---|
| [Fourier Compressor](https://arxiv.org/abs/2508.06038) | Parameter-free frequency-domain visual-token compression; reports over 96% retained accuracy, up to 83.8% inference-FLOP reduction, and 31.2% generation speedup. | A generic DCT/FFT low-pass method is already covered. TileSpec-Ex must establish a multi-tile topology advantage rather than claim frequency compression itself. |
| [ApET](https://openaccess.thecvf.com/content/CVPR2026/html/Ma_ApET_Approximation-Error_Guided_Token_Compression_for_Efficient_VLMs_CVPR_2026_paper.html) | Reconstructs visual tokens from basis tokens and prunes using approximation error; reports 88.9% image-token compression with 95.2% retained performance. | Low-frequency base plus residual energy overlaps the broader basis-plus-error abstraction. The risk scorer must add task evidence beyond reconstruction fidelity. |
| [QuietPrune](https://openaccess.thecvf.com/content/CVPR2026/html/Gao_QuietPrune_Query-Guided_Early_Token_Pruning_for_Vision-Language_Models_CVPR_2026_paper.html) | Query-guided early ViT pruning with adjacent 2x2 semi-structured groups; reports up to 19.0% prefill-latency reduction. | Query awareness and 2x2 blocks are not standalone contributions. TileSpec-Ex must show that query relevance specifically improves residual-exception selection and that its layout is measurably faster. |
| [TokenPacker](https://arxiv.org/abs/2407.02392) | Learned coarse-to-fine visual projector using low-resolution queries and high-resolution local cues; reports 75% to 89% visual-token compression. | Global structure plus local detail is an established functional pattern. TileSpec-Ex is differentiated only by fixed orthogonal decomposition, training-free risk scoring, and a verified structured execution contract. |

## Competitive Bar

The minimal experiment is a feasibility screen, not a claim of state-of-the-art
compression. A competitive follow-up would still need:

1. task quality on the same established VLM benchmarks;
2. native model token-count reduction rather than full-length reconstruction;
3. end-to-end prefill, TTFT, peak-memory, and tail-latency evidence;
4. comparison against at least Fourier Compressor, an approximation-error
   baseline such as ApET, and a query-aware structured method such as
   QuietPrune;
5. more than one native multi-tile or AnyRes VLM.

Until those are present, a positive minimal gate only justifies the next
engineering stage. It does not establish superiority over the cited systems.

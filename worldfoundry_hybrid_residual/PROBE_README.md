# RobuQ Structured Residual Probe

This isolated experiment compares the RobuQ high-precision SVD branch with
training-free residual corrections for Wan DiT weights:

- RobuQ initialization: ternary main weight plus an SVD branch extracted from
  the full Hadamard-rotated weight.
- Quantization-error low rank: ternary/FP8 main weight plus SVD factors fitted
  directly to the quantization error.
- Structured residual: a full-grid block-circulant projection (BCM).
- Hybrid residual: BCM plus low rank, or low rank plus coarse row/tile sparse
  correction under the same FP16 parameter budget as rank-16 SVD.

The decomposition probe measures weight and activation error. The H200 probe
separately measures WorldFoundry-style dynamic FP8, a pre-quantized activation
lower bound, and the real overhead of each high-precision branch. Ternary
results are algorithmic only because the current H200 environment has no
native ternary GEMM kernel.

All generated artifacts live under an explicit output directory. The scripts
do not modify the source checkpoint or the existing World Foundry run.

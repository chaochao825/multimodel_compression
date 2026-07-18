from __future__ import annotations

import torch

from tilespec_ex.rate import (
    RateLedger,
    base_scaled_vq_bits,
    base_scalar_bits,
    residual_incremental_costs,
    residual_stream_bits,
)
from tilespec_ex.tilelogic_codec import (
    encode_residual_modes,
    extract_tile_coefficients,
    reconstruct_tile_coefficients,
)
from tilespec_ex.vq import (
    ResidualVQCodebook,
    fit_residual_vq_codebook,
    fit_scaled_codebook,
    nearest_code_indices,
    symmetric_vector_quantize,
)


def _vectors(count: int = 96, dimension: int = 8) -> torch.Tensor:
    generator = torch.Generator().manual_seed(20260718)
    centers = torch.tensor(
        [
            [1.0] * dimension,
            [-1.0] * dimension,
            [1.0, -1.0] * (dimension // 2),
        ]
    )
    labels = torch.arange(count) % len(centers)
    return centers[labels] + 0.05 * torch.randn(
        count, dimension, generator=generator
    )


def test_weighted_nearest_code_uses_declared_metric() -> None:
    vectors = torch.tensor([[0.4, 0.9]])
    codebook = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    ordinary = nearest_code_indices(vectors, codebook)
    weighted = nearest_code_indices(
        vectors, codebook, metric_weights=torch.tensor([100.0, 1.0])
    )
    assert ordinary.item() == 1
    assert weighted.item() == 0


def test_scaled_codebook_round_trip_and_state_are_deterministic() -> None:
    vectors = _vectors()
    first = fit_scaled_codebook(
        vectors, num_codes=8, num_scale_levels=4, iterations=8, seed=7
    )
    second = fit_scaled_codebook(
        vectors, num_codes=8, num_scale_levels=4, iterations=8, seed=7
    )
    torch.testing.assert_close(first.codewords, second.codewords)
    reconstructed, indices, scales = first.reconstruct(vectors)
    assert reconstructed.shape == vectors.shape
    assert indices.shape == (vectors.shape[0],)
    assert scales.shape == indices.shape
    state = first.state_dict()
    assert state["codewords"].dtype == torch.float16
    assert state["scale_levels"].dtype == torch.float16
    assert state["metric_weights"].dtype == torch.float32
    restored = type(first).from_state_dict(state)
    torch.testing.assert_close(
        restored.decode(indices, scales), reconstructed, atol=2e-3, rtol=2e-3
    )


def test_residual_vq_depth_never_worsens_per_vector() -> None:
    vectors = _vectors(count=128)
    codebook = fit_residual_vq_codebook(
        vectors,
        stages=2,
        num_codes=8,
        num_scale_levels=4,
        iterations=8,
        seed=11,
    )
    reconstructions, indices, scales = codebook.reconstructions_by_depth(vectors)
    error_0 = (vectors - reconstructions[0]).square().sum(dim=1)
    error_1 = (vectors - reconstructions[1]).square().sum(dim=1)
    error_2 = (vectors - reconstructions[2]).square().sum(dim=1)
    assert bool((error_1 <= error_0 + 1e-5).all())
    assert bool((error_2 <= error_1 + 1e-5).all())
    state = codebook.state_dict()
    assert state["codewords"].dtype == torch.float16
    assert state["scale_levels"].dtype == torch.float16
    assert state["metric_weights"].dtype == torch.float32
    restored = ResidualVQCodebook.from_state_dict(state)
    torch.testing.assert_close(
        restored.decode(indices, scales), reconstructions[2], atol=2e-3, rtol=2e-3
    )


def test_int4_scalar_quantization_accounts_codes_and_scales() -> None:
    vectors = _vectors(count=5, dimension=8)
    result = symmetric_vector_quantize(vectors, bits=4, scale_storage_bits=16)
    assert result.codes.dtype == torch.int8
    assert result.scales.dtype == torch.float16
    assert result.reconstructed.shape == vectors.shape
    assert result.stream_bits == 5 * 8 * 4 + 5 * 16


def test_default_int4_scale_executes_and_is_charged_as_float32() -> None:
    vectors = _vectors(count=5, dimension=8)
    result = symmetric_vector_quantize(vectors, bits=4)
    assert result.scales.dtype == torch.float32
    assert result.scale_storage_bits == 32
    assert result.stream_bits == 5 * 8 * 4 + 5 * 32


def _tiny_residual_codebook() -> ResidualVQCodebook:
    words = torch.zeros(2, 4, 8)
    words[:, 1:] = torch.randn(
        2, 3, 8, generator=torch.Generator().manual_seed(3)
    )
    return ResidualVQCodebook(words, torch.ones(4), torch.ones(8))


def test_rate_ledger_separates_stream_and_shared_overhead() -> None:
    base = fit_scaled_codebook(
        _vectors(), num_codes=8, num_scale_levels=4, iterations=4, seed=5
    )
    ledger = RateLedger()
    base_scaled_vq_bits(ledger, base, 12)
    assert ledger.stream_bits == 12 * (base.index_bits + base.scale_bits)
    assert ledger.shared_bits > 0
    assert ledger.shared_bits == (
        base.codewords.numel() * 16
        + base.scale_levels.numel() * 16
        + base.metric_weights.numel() * 32
    )
    metrics = ledger.metrics(
        original_vectors=64,
        vector_dimension=8,
        amortization_count=100,
    )
    assert metrics["effective_bits"] > metrics["stream_bits"]


def test_residual_rate_matches_incremental_costs_for_dynamic_stream() -> None:
    codebook = _tiny_residual_codebook()
    modes = torch.tensor([0, 1, 2, 3])
    costs = residual_incremental_costs(
        codebook,
        block_count=4,
        dynamic_positions=True,
        total_positions=16,
    )
    ledger = RateLedger()
    residual_stream_bits(
        ledger,
        codebook,
        modes,
        dynamic_positions=True,
        total_positions=16,
        include_shared=False,
    )
    expected = 5  # count header for 16 possible positions
    expected += 3 * (4 + 2)  # active positions plus mode bits
    expected += 2 * codebook.scale_bits  # two RVQ-coded blocks
    expected += 3 * codebook.index_bits  # depth one plus depth two indices
    expected += codebook.dimension * 16  # one exact block
    assert ledger.stream_bits == expected
    assert int(costs[0].sum()) > int(costs[0, 0])


def test_scalar_ledger_matches_quantizer_payload() -> None:
    ledger = RateLedger()
    base_scalar_bits(ledger, 5, 8, value_bits=4, per_vector_scale_bits=16)
    assert ledger.stream_bits == 5 * 8 * 4 + 5 * 16


def test_tile_coefficient_extract_and_reconstruct_match_lowpass_contract() -> None:
    generator = torch.Generator().manual_seed(13)
    crops = torch.randn(4, 16, 16, 8, generator=generator)
    coefficients = extract_tile_coefficients(crops, total_retained=64)
    reconstructed = reconstruct_tile_coefficients(
        coefficients, tiles=4, height=16, width=16
    )
    assert coefficients.shape == (64, 8)
    assert reconstructed.shape == crops.shape
    round_trip = extract_tile_coefficients(reconstructed, total_retained=64)
    torch.testing.assert_close(round_trip, coefficients, atol=2e-5, rtol=2e-5)


def test_exact_residual_mode_recovers_target_blocks() -> None:
    generator = torch.Generator().manual_seed(17)
    target = torch.randn(4, 16, 16, 2, generator=generator)
    base = target * 0.5
    codewords = torch.zeros(2, 4, 8)
    codewords[:, 1:] = torch.randn(2, 3, 8, generator=generator)
    codebook = ResidualVQCodebook(codewords, torch.ones(4), torch.ones(8))
    modes = torch.zeros(256, dtype=torch.long)
    modes[3] = codebook.stages + 1
    encoded = encode_residual_modes(base, target, codebook, modes)
    row = (3 // 8) * 2
    col = (3 % 8) * 2
    torch.testing.assert_close(
        encoded.reconstructed[0, row : row + 2, col : col + 2],
        target[0, row : row + 2, col : col + 2],
    )

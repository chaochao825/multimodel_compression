from __future__ import annotations

import math
import unittest

import torch

from ar_video_butterfly_lifting_core import (
    apply_cyclic_shift,
    apply_window_cyclic_shift,
    build_lifting_tree,
    canonicalize_rope_keys,
    choose_prediction_shifts,
    detail_selection_from_indices,
    estimate_storage,
    middle_frame_indices,
    reconstruct_lifting_tree,
    restore_rope_keys,
    select_detail_tiles,
)


class ButterflyLiftingCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.height = 2
        self.width = 4
        self.frames = 6
        self.heads = 3
        self.dim = 8
        self.key = torch.randn(
            self.frames, self.height * self.width, self.heads, self.dim
        )
        self.value = torch.randn_like(self.key)
        self.candidates = [(0, 0), (1, 0), (0, 1), (0, -1)]

    def test_frame_partition_is_disjoint_and_complete(self) -> None:
        exact, middle = middle_frame_indices(12, 3, 3)
        self.assertEqual(exact, (0, 1, 2, 9, 10, 11))
        self.assertEqual(middle, (3, 4, 5, 6, 7, 8))
        self.assertEqual(tuple(sorted(exact + middle)), tuple(range(12)))

    def test_known_cyclic_prediction_is_selected(self) -> None:
        shift = torch.tensor([[0, 1]])
        right_key = apply_cyclic_shift(
            self.key[0], shift, self.height, self.width
        )
        right_value = apply_cyclic_shift(
            self.value[0], shift, self.height, self.width
        )
        selected = choose_prediction_shifts(
            self.key[0],
            self.value[0],
            right_key,
            right_value,
            self.candidates,
            self.height,
            self.width,
            "shared",
        )
        self.assertEqual(selected.tolist(), [[0, 1]])

    def test_full_details_reconstruct_exactly(self) -> None:
        tree = build_lifting_tree(
            self.key,
            self.value,
            tuple(range(self.frames)),
            self.candidates,
            self.height,
            self.width,
            "shared",
        )
        selection = select_detail_tiles(tree, tile_size=3, fraction=1.0)
        keys, values = reconstruct_lifting_tree(
            tree, selection, self.height, self.width
        )
        reconstructed_key = torch.stack([keys[index] for index in range(self.frames)])
        reconstructed_value = torch.stack(
            [values[index] for index in range(self.frames)]
        )
        torch.testing.assert_close(reconstructed_key, self.key, atol=2e-6, rtol=2e-6)
        torch.testing.assert_close(
            reconstructed_value, self.value, atol=2e-6, rtol=2e-6
        )

    def test_window_shift_is_invertible(self) -> None:
        shifts = torch.tensor([[1, -1], [0, 1]])
        shifted = apply_window_cyclic_shift(
            self.key[0],
            shifts,
            self.height,
            self.width,
            (2, 2),
            (1, 1),
        )
        recovered = apply_window_cyclic_shift(
            shifted,
            -shifts,
            self.height,
            self.width,
            (2, 2),
            (1, 1),
        )
        torch.testing.assert_close(recovered, self.key[0])

    def test_windowed_full_details_reconstruct_exactly(self) -> None:
        tree = build_lifting_tree(
            self.key,
            self.value,
            tuple(range(self.frames)),
            self.candidates,
            self.height,
            self.width,
            "window_shared",
            window_shape=(2, 2),
            window_offsets=((0, 0), (1, 1)),
        )
        selection = select_detail_tiles(tree, tile_size=3, fraction=1.0)
        keys, values = reconstruct_lifting_tree(
            tree, selection, self.height, self.width
        )
        torch.testing.assert_close(
            torch.stack([keys[index] for index in range(self.frames)]),
            self.key,
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            torch.stack([values[index] for index in range(self.frames)]),
            self.value,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_exact_shift_sequence_needs_no_details(self) -> None:
        shift = torch.tensor([[0, 1]])
        key = [self.key[0]]
        value = [self.value[0]]
        for _ in range(3):
            key.append(apply_cyclic_shift(key[-1], shift, self.height, self.width))
            value.append(
                apply_cyclic_shift(value[-1], shift, self.height, self.width)
            )
        key_tensor = torch.stack(key)
        value_tensor = torch.stack(value)
        tree = build_lifting_tree(
            key_tensor,
            value_tensor,
            tuple(range(4)),
            [(0, 0), (0, 1), (0, 2)],
            self.height,
            self.width,
            "shared",
        )
        selection = select_detail_tiles(tree, tile_size=4, fraction=0.0)
        keys, values = reconstruct_lifting_tree(
            tree, selection, self.height, self.width
        )
        torch.testing.assert_close(
            torch.stack([keys[index] for index in range(4)]),
            key_tensor,
            atol=2e-6,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            torch.stack([values[index] for index in range(4)]),
            value_tensor,
            atol=2e-6,
            rtol=2e-6,
        )

    def test_rope_round_trip(self) -> None:
        positions = 32
        complex_dim = self.dim // 2
        angle = torch.arange(positions).float()[:, None] * torch.linspace(
            0.01, 0.04, complex_dim
        )[None, :]
        rope = torch.polar(torch.ones_like(angle), angle)
        canonical = self.key[:3]
        frame_ids = [2, 3, 7]
        post = restore_rope_keys(
            canonical,
            frame_ids,
            self.height,
            self.width,
            rope,
            torch.float32,
        )
        recovered = canonicalize_rope_keys(
            post, frame_ids, self.height, self.width, rope
        )
        torch.testing.assert_close(recovered, canonical, atol=2e-6, rtol=2e-6)

    def test_storage_accounting_decreases_with_pruning(self) -> None:
        tree = build_lifting_tree(
            self.key,
            self.value,
            tuple(range(self.frames)),
            self.candidates,
            self.height,
            self.width,
            "identity",
        )
        low = select_detail_tiles(tree, tile_size=4, fraction=0.2)
        high = select_detail_tiles(tree, tile_size=4, fraction=0.8)
        low_storage = estimate_storage(
            12, 6, 8, self.heads, self.dim, self.dim, tree, low
        )
        high_storage = estimate_storage(
            12, 6, 8, self.heads, self.dim, self.dim, tree, high
        )
        self.assertGreater(low_storage.compression_ratio, high_storage.compression_ratio)
        self.assertGreater(low_storage.compressed_bytes, 0)
        self.assertTrue(math.isfinite(low_storage.compression_ratio))

    def test_padded_storage_equalizes_partial_tiles(self) -> None:
        tree = build_lifting_tree(
            self.key,
            self.value,
            tuple(range(self.frames)),
            self.candidates,
            self.height,
            self.width,
            "identity",
        )
        full_tile = detail_selection_from_indices(
            tree, tile_size=6, selected_indices=[0]
        )
        partial_tile = detail_selection_from_indices(
            tree, tile_size=6, selected_indices=[1]
        )
        self.assertNotEqual(full_tile.retained_tokens, partial_tile.retained_tokens)
        full_storage = estimate_storage(
            12,
            6,
            8,
            self.heads,
            self.dim,
            self.dim,
            tree,
            full_tile,
            padded_detail_tile_size=6,
        )
        partial_storage = estimate_storage(
            12,
            6,
            8,
            self.heads,
            self.dim,
            self.dim,
            tree,
            partial_tile,
            padded_detail_tile_size=6,
        )
        self.assertEqual(full_storage.compressed_bytes, partial_storage.compressed_bytes)

    def test_explicit_detail_indices_are_deterministic(self) -> None:
        tree = build_lifting_tree(
            self.key,
            self.value,
            tuple(range(self.frames)),
            self.candidates,
            self.height,
            self.width,
            "identity",
        )
        selection = detail_selection_from_indices(tree, tile_size=4, selected_indices=[0, 2])
        self.assertEqual(selection.retained_blocks, 2)
        self.assertEqual(selection.retained_tokens, 8)
        with self.assertRaises(ValueError):
            detail_selection_from_indices(tree, tile_size=4, selected_indices=[0, 0])


if __name__ == "__main__":
    unittest.main()

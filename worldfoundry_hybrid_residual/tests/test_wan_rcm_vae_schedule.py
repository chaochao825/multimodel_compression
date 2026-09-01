from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import run_wan_rcm_vae_schedule as vae_schedule  # noqa: E402


class FakeTensor:
    dtype = "float32"

    def __init__(self, frames: tuple[float, ...]) -> None:
        self.frames = frames

    @property
    def shape(self) -> tuple[int, int, int, int, int]:
        return (1, 1, len(self.frames), 1, 1)

    def __getitem__(self, key: tuple[object, ...]) -> "FakeTensor":
        temporal = key[2]
        if not isinstance(temporal, slice):
            raise TypeError("fake tensor only supports temporal slices")
        return FakeTensor(self.frames[temporal])

    def __truediv__(self, value: float) -> "FakeTensor":
        return FakeTensor(tuple(frame / value for frame in self.frames))

    def __add__(self, value: float) -> "FakeTensor":
        return FakeTensor(tuple(frame + value for frame in self.frames))

    def __mul__(self, value: float) -> "FakeTensor":
        return FakeTensor(tuple(frame * value for frame in self.frames))

    def to(self, _dtype: object) -> "FakeTensor":
        return self


class FakeTorch:
    Tensor = FakeTensor

    @staticmethod
    def cat(values: list[FakeTensor], dim: int) -> FakeTensor:
        if dim != 2:
            raise ValueError("fake tensor only supports temporal concatenation")
        frames = tuple(frame for value in values for frame in value.frames)
        return FakeTensor(frames)


class FakeInnerModel:
    z_dim = 1

    def __init__(self) -> None:
        self.decode_widths: list[int] = []
        self.clear_count = 0
        self._feat_map: list[object] = []
        self._conv_idx = [0]

    def clear_cache(self) -> None:
        self.clear_count += 1
        self._feat_map = []
        self._conv_idx = [0]

    def conv2(self, value: FakeTensor) -> FakeTensor:
        return value

    def decoder(
        self,
        value: FakeTensor,
        *,
        feat_cache: list[object],
        feat_idx: list[int],
    ) -> FakeTensor:
        self.assert_cache_arguments(feat_cache, feat_idx)
        self.decode_widths.append(int(value.shape[2]))
        return value * 2

    def assert_cache_arguments(
        self, feat_cache: list[object], feat_idx: list[int]
    ) -> None:
        if feat_cache is not self._feat_map or feat_idx is not self._conv_idx:
            raise AssertionError("candidate did not forward the model cache objects")


class TemporalRangeTest(unittest.TestCase):
    def test_first_frame_remains_a_sentinel_call(self) -> None:
        self.assertEqual(
            vae_schedule.temporal_decode_ranges(21, 4),
            ((0, 1), (1, 5), (5, 9), (9, 13), (13, 17), (17, 21)),
        )

    def test_single_frame_and_invalid_inputs(self) -> None:
        self.assertEqual(vae_schedule.temporal_decode_ranges(1, 8), ((0, 1),))
        with self.assertRaises(ValueError):
            vae_schedule.temporal_decode_ranges(0, 4)
        with self.assertRaises(ValueError):
            vae_schedule.temporal_decode_ranges(4, 0)


class TemporalDecodeTest(unittest.TestCase):
    def test_chunked_schedule_preserves_order_and_cache_lifetime(self) -> None:
        model = FakeInnerModel()
        latent = FakeTensor(tuple(float(index) for index in range(21)))
        result = vae_schedule.decode_inner_temporal_chunks(
            model,
            latent,
            [0.0, 1.0],
            4,
            FakeTorch,
        )
        self.assertEqual(result.frames, (latent * 2).frames)
        self.assertEqual(model.decode_widths, [1, 4, 4, 4, 4, 4])
        self.assertEqual(model.clear_count, 2)

    def test_chunk_one_matches_framewise_dispatch(self) -> None:
        model = FakeInnerModel()
        latent = FakeTensor(tuple(float(index) for index in range(5)))
        result = vae_schedule.decode_inner_temporal_chunks(
            model,
            latent,
            [0.0, 1.0],
            1,
            FakeTorch,
        )
        self.assertEqual(result.frames, (latent * 2).frames)
        self.assertEqual(model.decode_widths, [1, 1, 1, 1, 1])


if __name__ == "__main__":
    unittest.main()

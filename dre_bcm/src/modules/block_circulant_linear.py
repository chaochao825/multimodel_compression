from typing import Dict, Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.circulant import circulant_from_generator, circulant_matmul_direct, circulant_matmul_fft


class BlockCirculantLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        bias: bool = False,
        use_fft: bool = False,
        use_generator_delta: bool = False,
        delta_mode: str = "horizontal",
        l1_lambda: float = 0.0,
    ) -> None:
        super().__init__()
        if in_features % block_size != 0 or out_features % block_size != 0:
            raise ValueError("in_features and out_features must be divisible by block_size")
        if delta_mode not in {"horizontal", "vertical", "2d"}:
            raise ValueError(f"unsupported delta_mode: {delta_mode}")

        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        self.num_in_blocks = in_features // block_size
        self.num_out_blocks = out_features // block_size
        self.use_fft = use_fft
        self.use_generator_delta = use_generator_delta
        self.delta_mode = delta_mode
        self.l1_lambda = l1_lambda

        shape = (self.num_out_blocks, self.num_in_blocks, self.block_size)
        init_scale = 1.0 / max(block_size, 1)
        self.generator = None
        self.base_generator = None
        self.delta_generator = None

        if use_generator_delta:
            self.base_generator = nn.Parameter(torch.randn(shape) * init_scale)
            self.delta_generator = nn.Parameter(torch.zeros(shape))
        else:
            self.generator = nn.Parameter(torch.randn(shape) * init_scale)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def reconstruct_generators(self) -> torch.Tensor:
        if not self.use_generator_delta:
            return self.generator

        base = self.base_generator
        delta = self.delta_generator
        generators = torch.zeros_like(base)

        if self.delta_mode == "horizontal":
            generators[:, 0, :] = base[:, 0, :] + delta[:, 0, :]
            for col in range(1, self.num_in_blocks):
                generators[:, col, :] = generators[:, col - 1, :] + delta[:, col, :]
        elif self.delta_mode == "vertical":
            generators[0, :, :] = base[0, :, :] + delta[0, :, :]
            for row in range(1, self.num_out_blocks):
                generators[row, :, :] = generators[row - 1, :, :] + delta[row, :, :]
        else:
            generators[0, 0, :] = base[0, 0, :] + delta[0, 0, :]
            for col in range(1, self.num_in_blocks):
                generators[0, col, :] = generators[0, col - 1, :] + delta[0, col, :]
            for row in range(1, self.num_out_blocks):
                generators[row, 0, :] = generators[row - 1, 0, :] + delta[row, 0, :]
            for row in range(1, self.num_out_blocks):
                for col in range(1, self.num_in_blocks):
                    generators[row, col, :] = (
                        generators[row - 1, col, :]
                        + generators[row, col - 1, :]
                        - generators[row - 1, col - 1, :]
                        + delta[row, col, :]
                    )

        return generators

    def dense_weight(self) -> torch.Tensor:
        generators = self.reconstruct_generators()
        weight = torch.zeros(
            self.out_features,
            self.in_features,
            device=generators.device,
            dtype=generators.dtype,
        )
        for out_idx in range(self.num_out_blocks):
            out_slice = slice(out_idx * self.block_size, (out_idx + 1) * self.block_size)
            for in_idx in range(self.num_in_blocks):
                in_slice = slice(in_idx * self.block_size, (in_idx + 1) * self.block_size)
                weight[out_slice, in_slice] = circulant_from_generator(generators[out_idx, in_idx])
        return weight

    def regularization_loss(self) -> torch.Tensor:
        if not self.use_generator_delta or self.delta_generator is None:
            return torch.zeros((), device=self.device, dtype=self.dtype)
        return self.l1_lambda * self.delta_generator.abs().mean()

    @property
    def device(self) -> torch.device:
        return self.reconstruct_generators().device

    @property
    def dtype(self) -> torch.dtype:
        return self.reconstruct_generators().dtype

    def extra_state(self) -> Dict[str, Optional[str]]:
        return {
            "use_generator_delta": self.use_generator_delta,
            "delta_mode": self.delta_mode if self.use_generator_delta else None,
            "use_fft": self.use_fft,
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        generators = self.reconstruct_generators()
        x_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)
        x_blocks = x_flat.view(-1, self.num_in_blocks, self.block_size)
        outputs = []
        matmul = circulant_matmul_fft if self.use_fft else circulant_matmul_direct

        for out_idx in range(self.num_out_blocks):
            accum = None
            for in_idx in range(self.num_in_blocks):
                block_out = matmul(generators[out_idx, in_idx], x_blocks[:, in_idx, :])
                accum = block_out if accum is None else accum + block_out
            outputs.append(accum)

        y = torch.cat(outputs, dim=-1)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*x_shape[:-1], self.out_features)

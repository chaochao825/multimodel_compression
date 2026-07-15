from typing import Literal

import torch
from torch import nn

from src.modules.block_circulant_linear import BlockCirculantLinear
from src.modules.linear_like import infer_linear_like_features, module_weight_as_linear
from src.modules.low_rank_linear import LowRankResidual


class DREBCMLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Module,
        block_size: int = 32,
        rank: int = 8,
        alpha: float = 16.0,
        use_fft: bool = False,
        train_base: bool = False,
        use_generator_delta: bool = False,
        delta_mode: str = "horizontal",
        mode: Literal["bcm_only", "lowrank_only", "bcm_plus_lowrank"] = "bcm_plus_lowrank",
    ) -> None:
        super().__init__()
        if not hasattr(base_linear, "weight"):
            raise TypeError("base_linear must expose a weight parameter")

        self.base_linear = base_linear
        self.mode = mode
        self.in_features, self.out_features = infer_linear_like_features(base_linear)
        self.bcm_adapter = BlockCirculantLinear(
            in_features=self.in_features,
            out_features=self.out_features,
            block_size=block_size,
            bias=False,
            use_fft=use_fft,
            use_generator_delta=use_generator_delta,
            delta_mode=delta_mode,
        )
        self.lowrank_adapter = LowRankResidual(
            in_features=self.in_features,
            out_features=self.out_features,
            rank=rank,
            alpha=alpha,
        )

        if not train_base:
            for parameter in self.base_linear.parameters():
                parameter.requires_grad = False

    def dense_delta_weight(self) -> torch.Tensor:
        weight = torch.zeros_like(module_weight_as_linear(self.base_linear))
        if self.mode in {"bcm_only", "bcm_plus_lowrank"}:
            weight = weight + self.bcm_adapter.dense_weight().to(weight.device, weight.dtype)
        if self.mode in {"lowrank_only", "bcm_plus_lowrank"}:
            weight = weight + self.lowrank_adapter.dense_weight().to(weight.device, weight.dtype)
        return weight

    def regularization_loss(self) -> torch.Tensor:
        if self.mode in {"bcm_only", "bcm_plus_lowrank"}:
            return self.bcm_adapter.regularization_loss()
        return torch.zeros((), device=self.base_linear.weight.device, dtype=self.base_linear.weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_linear(x)
        if self.mode in {"bcm_only", "bcm_plus_lowrank"}:
            y = y + self.bcm_adapter(x)
        if self.mode in {"lowrank_only", "bcm_plus_lowrank"}:
            y = y + self.lowrank_adapter(x)
        return y

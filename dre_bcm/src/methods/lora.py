from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.linear_like import infer_linear_like_features, module_weight_as_linear
from src.modules.low_rank_linear import LowRankResidual


class LoRALinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Module,
        rank: int = 8,
        alpha: float = 16.0,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        if not hasattr(base_linear, "weight"):
            raise TypeError("base_linear must expose a weight parameter")

        self.base_linear = base_linear
        self.in_features, self.out_features = infer_linear_like_features(base_linear)
        self.adapter = LowRankResidual(self.in_features, self.out_features, rank=rank, alpha=alpha)

        if not train_base:
            for parameter in self.base_linear.parameters():
                parameter.requires_grad = False

    def dense_delta_weight(self) -> torch.Tensor:
        base_weight = module_weight_as_linear(self.base_linear)
        return self.adapter.dense_weight().to(base_weight.device, base_weight.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base_linear(x) + self.adapter(x)

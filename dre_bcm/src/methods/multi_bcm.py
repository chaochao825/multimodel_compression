from typing import List

import torch
from torch import nn

from src.modules.block_circulant_linear import BlockCirculantLinear


class MultiBCMLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        block_size: int,
        num_bases: int = 2,
        use_fft: bool = False,
    ) -> None:
        super().__init__()
        self.adapters = nn.ModuleList(
            [
                BlockCirculantLinear(
                    in_features=in_features,
                    out_features=out_features,
                    block_size=block_size,
                    bias=False,
                    use_fft=use_fft,
                )
                for _ in range(num_bases)
            ]
        )

    def dense_weight(self) -> torch.Tensor:
        return sum(adapter.dense_weight() for adapter in self.adapters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return sum(adapter(x) for adapter in self.adapters)

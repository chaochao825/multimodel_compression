import math

import torch
from torch import nn
from torch.nn import functional as F


class LowRankResidual(nn.Module):
    def __init__(self, in_features: int, out_features: int, rank: int, alpha: float = 1.0) -> None:
        super().__init__()
        if rank < 0:
            raise ValueError("rank must be non-negative")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank if rank > 0 else 0.0
        self.register_buffer("_zero", torch.tensor(0.0), persistent=False)

        if rank > 0:
            self.A = nn.Parameter(torch.empty(rank, in_features))
            self.B = nn.Parameter(torch.empty(out_features, rank))
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.zeros_(self.B)
        else:
            self.register_parameter("A", None)
            self.register_parameter("B", None)

    def dense_weight(self) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(
                self.out_features,
                self.in_features,
                device=self._zero.device,
                dtype=self._zero.dtype,
            )
        return self.scaling * (self.B @ self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank == 0:
            return torch.zeros(*x.shape[:-1], self.out_features, device=x.device, dtype=x.dtype)
        down = F.linear(x, self.A)
        up = F.linear(down, self.B)
        return self.scaling * up

import math

import torch
from torch import nn
from torch.nn import functional as F


class SparseDeltaLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, sparse_ratio: float = 0.05) -> None:
        super().__init__()
        if not 0.0 < sparse_ratio <= 1.0:
            raise ValueError("sparse_ratio must be in (0, 1]")
        self.in_features = in_features
        self.out_features = out_features
        self.sparse_ratio = sparse_ratio
        self.delta = nn.Parameter(torch.zeros(out_features, in_features))
        nn.init.kaiming_uniform_(self.delta, a=math.sqrt(5))

    def dense_weight(self) -> torch.Tensor:
        flat = self.delta.reshape(-1)
        keep = max(1, int(flat.numel() * self.sparse_ratio))
        _, indices = flat.abs().topk(keep, sorted=False)
        mask = torch.zeros_like(flat)
        mask[indices] = 1.0
        return (flat * mask).reshape_as(self.delta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dense_weight())

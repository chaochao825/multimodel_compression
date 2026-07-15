from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F

from src.modules.linear_like import infer_linear_like_features, module_weight_as_linear


class FourierFTLinear(nn.Module):
    def __init__(
        self,
        base_linear: Optional[nn.Module] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        freq_rows: int = 16,
        freq_cols: int = 16,
        scale: float = 1.0,
        train_base: bool = False,
    ) -> None:
        super().__init__()
        self.base_linear = base_linear
        if base_linear is not None:
            if not hasattr(base_linear, "weight"):
                raise TypeError("base_linear must expose a weight parameter")
            in_features, out_features = infer_linear_like_features(base_linear)
            if not train_base:
                for parameter in base_linear.parameters():
                    parameter.requires_grad = False
        if in_features is None or out_features is None:
            raise ValueError("either base_linear or explicit in_features/out_features must be provided")

        self.in_features = in_features
        self.out_features = out_features
        self.freq_rows = min(freq_rows, out_features)
        self.freq_cols = min(freq_cols, in_features)
        self.scale = scale
        self.freq_real = nn.Parameter(torch.zeros(self.freq_rows, self.freq_cols))
        self.freq_imag = nn.Parameter(torch.zeros(self.freq_rows, self.freq_cols))

    def dense_delta_weight(self) -> torch.Tensor:
        spectrum = torch.zeros(
            self.out_features,
            self.in_features,
            device=self.freq_real.device,
            dtype=torch.complex64 if self.freq_real.dtype == torch.float32 else torch.complex128,
        )
        spectrum[: self.freq_rows, : self.freq_cols] = torch.complex(self.freq_real, self.freq_imag)
        return self.scale * torch.fft.ifft2(spectrum).real.to(self.freq_real.dtype)

    def dense_weight(self) -> torch.Tensor:
        delta = self.dense_delta_weight()
        if self.base_linear is None:
            return delta
        return module_weight_as_linear(self.base_linear).to(delta.device, delta.dtype) + delta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = F.linear(x, self.dense_delta_weight().to(x.device, x.dtype))
        if self.base_linear is None:
            return delta
        return self.base_linear(x) + delta

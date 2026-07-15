from typing import Optional

from torch import nn

from src.methods.multi_bcm import MultiBCMLinear
from src.modules.linear_like import infer_linear_like_features, module_weight_as_linear


class C3AProxyLinear(nn.Module):
    """
    Lightweight proxy baseline for circular-convolution-style adaptation.

    This is not a paper-faithful reproduction of C3A yet. It approximates the
    higher-rank circular adaptation effect with multiple block-circulant bases.
    """

    def __init__(
        self,
        base_linear: Optional[nn.Module] = None,
        in_features: Optional[int] = None,
        out_features: Optional[int] = None,
        block_size: int = 32,
        num_bases: int = 2,
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

        self.adapter = MultiBCMLinear(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            num_bases=num_bases,
            use_fft=False,
        )

    def dense_delta_weight(self):
        return self.adapter.dense_weight()

    def dense_weight(self):
        delta = self.dense_delta_weight()
        if self.base_linear is None:
            return delta
        return module_weight_as_linear(self.base_linear).to(delta.device, delta.dtype) + delta

    def forward(self, x):
        delta = self.adapter(x)
        if self.base_linear is None:
            return delta
        return self.base_linear(x) + delta

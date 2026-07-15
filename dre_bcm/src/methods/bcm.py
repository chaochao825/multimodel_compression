from torch import nn

from src.modules.linear_like import infer_linear_like_features
from src.modules.block_circulant_linear import BlockCirculantLinear


class BCMLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Module,
        block_size: int = 32,
        use_fft: bool = False,
        train_base: bool = False,
        use_generator_delta: bool = False,
        delta_mode: str = "horizontal",
    ) -> None:
        super().__init__()
        if not hasattr(base_linear, "weight"):
            raise TypeError("base_linear must expose a weight parameter")

        self.base_linear = base_linear
        in_features, out_features = infer_linear_like_features(base_linear)
        self.adapter = BlockCirculantLinear(
            in_features=in_features,
            out_features=out_features,
            block_size=block_size,
            bias=False,
            use_fft=use_fft,
            use_generator_delta=use_generator_delta,
            delta_mode=delta_mode,
        )

        if not train_base:
            for parameter in self.base_linear.parameters():
                parameter.requires_grad = False

    def dense_delta_weight(self):
        return self.adapter.dense_weight().to(self.base_linear.weight.device, self.base_linear.weight.dtype)

    def forward(self, x):
        return self.base_linear(x) + self.adapter(x)

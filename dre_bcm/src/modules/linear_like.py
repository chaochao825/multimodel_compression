from typing import Tuple

from torch import nn

try:
    from transformers.pytorch_utils import Conv1D
except Exception:  # pragma: no cover
    Conv1D = None


def is_linear_like(module: nn.Module) -> bool:
    if isinstance(module, nn.Linear):
        return True
    if Conv1D is not None and isinstance(module, Conv1D):
        return True
    return False


def infer_linear_like_features(module: nn.Module) -> Tuple[int, int]:
    if isinstance(module, nn.Linear):
        return module.in_features, module.out_features
    if Conv1D is not None and isinstance(module, Conv1D):
        in_features, out_features = module.weight.shape
        return in_features, out_features
    raise TypeError(f"unsupported linear-like module: {type(module).__name__}")


def module_weight_as_linear(module: nn.Module):
    if isinstance(module, nn.Linear):
        return module.weight
    if Conv1D is not None and isinstance(module, Conv1D):
        return module.weight.t()
    raise TypeError(f"unsupported linear-like module: {type(module).__name__}")


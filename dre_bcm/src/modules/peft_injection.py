from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple

import torch
from torch import nn

from src.modules.linear_like import is_linear_like


@dataclass
class InjectionResult:
    module_name: str
    original_type: str
    injected_type: str


def freeze_module(module: nn.Module) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = False


def resolve_parent(model: nn.Module, module_name: str) -> Tuple[nn.Module, str]:
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def inject_by_name(
    model: nn.Module,
    target_names: Iterable[str],
    factory: Callable[[nn.Module], nn.Module],
) -> List[InjectionResult]:
    targets = list(target_names)
    results: List[InjectionResult] = []
    for module_name, module in list(model.named_modules()):
        if module is model:
            continue
        if not is_linear_like(module):
            continue
        if not any(token in module_name for token in targets):
            continue

        parent, attr = resolve_parent(model, module_name)
        replacement = factory(module)
        setattr(parent, attr, replacement)
        results.append(
            InjectionResult(
                module_name=module_name,
                original_type=type(module).__name__,
                injected_type=type(replacement).__name__,
            )
        )
    return results


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)

import torch


def estimate_activation_error(
    target_weight: torch.Tensor,
    approx_weight: torch.Tensor,
    num_samples: int = 256,
    seed: int = 0,
) -> float:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    x = torch.randn(num_samples, target_weight.shape[1], generator=generator, dtype=target_weight.dtype)
    x = x.to(target_weight.device)
    target_out = x @ target_weight.t()
    approx_out = x @ approx_weight.t()
    numerator = torch.linalg.norm(target_out - approx_out, dim=-1)
    denominator = torch.linalg.norm(target_out, dim=-1).clamp_min(1e-12)
    return (numerator / denominator).mean().item()

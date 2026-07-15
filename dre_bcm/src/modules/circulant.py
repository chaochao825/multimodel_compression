from typing import Tuple

import torch


def _flatten_last_dim_pair(g: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Size]:
    if g.shape[-1] != x.shape[-1]:
        raise ValueError(f"generator and input must share the same last dim, got {g.shape} and {x.shape}")

    block_size = g.shape[-1]
    broadcast_shape = torch.broadcast_shapes(g.shape[:-1], x.shape[:-1])
    g_view = torch.broadcast_to(g, (*broadcast_shape, block_size)).reshape(-1, block_size)
    x_view = torch.broadcast_to(x, (*broadcast_shape, block_size)).reshape(-1, block_size)
    return g_view, x_view, torch.Size(broadcast_shape)


def circulant_from_generator(g: torch.Tensor) -> torch.Tensor:
    """Build a dense circulant matrix from the first row generator."""
    block_size = g.shape[-1]
    idx = (
        torch.arange(block_size, device=g.device)[None, :]
        - torch.arange(block_size, device=g.device)[:, None]
    ) % block_size
    return g[..., idx]


def circulant_matmul_direct(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multiply a circulant matrix and vector using explicit cyclic shifts."""
    g_view, x_view, broadcast_shape = _flatten_last_dim_pair(g, x)
    shifts = [torch.roll(x_view, shifts=-shift, dims=-1) for shift in range(x_view.shape[-1])]
    shifted = torch.stack(shifts, dim=-1)
    out = torch.einsum("bk,bnk->bn", g_view, shifted)
    return out.reshape(*broadcast_shape, x_view.shape[-1])


def circulant_matmul_fft(g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Multiply a circulant matrix and vector using the FFT."""
    g_view, x_view, broadcast_shape = _flatten_last_dim_pair(g, x)
    g_fft = torch.fft.fft(g_view, dim=-1)
    x_fft = torch.fft.fft(x_view, dim=-1)
    out = torch.fft.ifft(torch.conj(g_fft) * x_fft, dim=-1).real
    return out.reshape(*broadcast_shape, x_view.shape[-1])

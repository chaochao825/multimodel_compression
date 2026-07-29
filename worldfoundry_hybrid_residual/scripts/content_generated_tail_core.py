#!/usr/bin/env python3
"""Core operators for content-generated sparse-linear attention probes.

The linear branch estimates the full positive attention kernel. Exact support
then replaces, rather than adds to, the corresponding linear interactions.
This keeps both branches under one numerator and denominator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class GroupLayout:
    name: str
    indices: torch.Tensor
    valid: torch.Tensor

    @property
    def groups(self) -> int:
        return int(self.indices.shape[0])

    @property
    def width(self) -> int:
        return int(self.indices.shape[1])


@dataclass
class LinearTailState:
    phi_q: torch.Tensor
    phi_k: torch.Tensor
    numerator: torch.Tensor
    denominator: torch.Tensor
    kernel_scale: torch.Tensor


def _padded_layout(order: torch.Tensor, width: int, name: str) -> GroupLayout:
    if order.ndim != 1 or width <= 0:
        raise ValueError("order must be rank-1 and width must be positive")
    if order.numel() == 0:
        raise ValueError("cannot partition an empty token sequence")
    groups = math.ceil(order.numel() / width)
    padded = groups * width
    valid = torch.arange(padded, device=order.device) < order.numel()
    if padded > order.numel():
        order = F.pad(order, (0, padded - order.numel()), value=0)
    return GroupLayout(
        name=name,
        indices=order.reshape(groups, width),
        valid=valid.reshape(groups, width),
    )


def contiguous_layout(tokens: int, width: int, device: torch.device | str) -> GroupLayout:
    if tokens <= 0:
        raise ValueError("tokens must be positive")
    return _padded_layout(torch.arange(tokens, device=device), width, "fixed64")


def _standardize(values: torch.Tensor) -> torch.Tensor:
    return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-6)


def semantic_layout(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    width: int,
    mode: str,
    value_weight: float = 0.25,
) -> GroupLayout:
    """Build a deterministic query-tile-conditioned semantic permutation.

    This is an SVG2-style proxy, not a paper-faithful implementation. The
    semantic variant sorts by pooled Q/K relevance. The value-aware variant
    adds a standardized V-leverage term before sorting.
    """
    if q.ndim != 2 or k.ndim != 2 or v.ndim != 2:
        raise ValueError("q, k, and v must be rank-2")
    if k.shape != v.shape or q.shape[1] != k.shape[1]:
        raise ValueError("incompatible q, k, and v shapes")
    pooled_q = F.normalize(q.mean(dim=0), dim=0)
    relevance = _standardize(k @ pooled_q)
    if mode == "semantic_qk":
        score = relevance
        name = "svg2_style_semantic64_proxy"
    elif mode == "value_aware":
        leverage = _standardize(v.square().mean(dim=1).sqrt())
        score = relevance + float(value_weight) * leverage
        name = "value_aware_semantic64_proxy"
    else:
        raise ValueError(f"unsupported semantic layout mode: {mode}")
    order = torch.argsort(score, descending=True, stable=True)
    return _padded_layout(order, width, name)


def layout_tokens(layout: GroupLayout, selected: torch.Tensor) -> torch.Tensor:
    if selected.ndim != 1:
        raise ValueError("selected groups must be rank-1")
    indices = layout.indices.index_select(0, selected)
    valid = layout.valid.index_select(0, selected)
    return indices[valid]


def layout_tokens_padded(
    layout: GroupLayout, selected: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return fixed-width token indices plus a validity mask."""
    if selected.ndim != 1:
        raise ValueError("selected groups must be rank-1")
    return (
        layout.indices.index_select(0, selected).flatten(),
        layout.valid.index_select(0, selected).flatten(),
    )


def proxy_group_selection(
    layout: GroupLayout,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    budget: int,
    value_weight: float = 0.25,
) -> torch.Tensor:
    """Select groups using pooled Q/K and optional V leverage only."""
    if not 0 < budget <= layout.groups:
        raise ValueError("group budget is outside the valid range")
    safe = layout.indices
    valid = layout.valid.to(k.dtype)
    grouped_k = k.index_select(0, safe.flatten()).reshape(layout.groups, layout.width, -1)
    pooled_k = (grouped_k * valid.unsqueeze(2)).sum(dim=1) / valid.sum(dim=1).clamp_min(1).unsqueeze(1)
    pooled_q = F.normalize(q.mean(dim=0), dim=0)
    score = _standardize(pooled_k @ pooled_q)
    if "value_aware" in layout.name:
        grouped_v = v.index_select(0, safe.flatten()).reshape(layout.groups, layout.width, -1)
        leverage = (
            grouped_v.square().mean(dim=2).sqrt() * valid
        ).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        score = score + float(value_weight) * _standardize(leverage)
    return score.topk(budget).indices.sort().values


class PositiveLinearTail(nn.Module):
    """Per-head learned positive feature map with exact sparse correction."""

    def __init__(
        self,
        heads: int,
        channels: int,
        rank: int,
        q_rms: torch.Tensor,
        k_rms: torch.Tensor,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if heads <= 0 or channels <= 0 or rank <= 0:
            raise ValueError("heads, channels, and rank must be positive")
        if q_rms.shape != (heads,) or k_rms.shape != (heads,):
            raise ValueError("q_rms and k_rms must have shape [heads]")
        generator = torch.Generator(device="cpu").manual_seed(seed)
        scale = 1.0 / math.sqrt(channels)
        q_projection = torch.randn(heads, channels, rank, generator=generator) * scale
        k_projection = torch.randn(heads, channels, rank, generator=generator) * scale
        self.q_projection = nn.Parameter(q_projection)
        self.k_projection = nn.Parameter(k_projection)
        self.q_bias = nn.Parameter(torch.zeros(heads, rank))
        self.k_bias = nn.Parameter(torch.zeros(heads, rank))
        self.log_kernel_scale = nn.Parameter(torch.zeros(heads))
        self.register_buffer("q_rms", q_rms.float().clamp_min(1e-6))
        self.register_buffer("k_rms", k_rms.float().clamp_min(1e-6))
        self.heads = heads
        self.channels = channels
        self.rank = rank

    def _features(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if q.ndim != 3 or k.ndim != 3:
            raise ValueError("q and k must have shape [heads, tokens, channels]")
        if q.shape[0] != self.heads or k.shape[0] != self.heads:
            raise ValueError("head count does not match the feature map")
        q_input = q.float() / self.q_rms[:, None, None]
        k_input = k.float() / self.k_rms[:, None, None]
        phi_q = F.softplus(
            torch.einsum("hqd,hdr->hqr", q_input, self.q_projection)
            + self.q_bias[:, None, :]
        ) + 1e-4
        phi_k = F.softplus(
            torch.einsum("hnd,hdr->hnr", k_input, self.k_projection)
            + self.k_bias[:, None, :]
        ) + 1e-4
        kernel_scale = self.log_kernel_scale.clamp(-20.0, 20.0).exp() / self.rank
        return phi_q, phi_k, kernel_scale

    def prepare(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
    ) -> LinearTailState:
        if k.shape != v.shape:
            raise ValueError("k and v must have identical shapes")
        phi_q, phi_k, kernel_scale = self._features(q, k)
        kv = torch.einsum("hnr,hnd->hrd", phi_k, v.float())
        numerator = torch.einsum("hqr,hrd->hqd", phi_q, kv)
        denominator = torch.einsum("hqr,hr->hq", phi_q, phi_k.sum(dim=1))
        numerator = numerator * kernel_scale[:, None, None]
        denominator = denominator * kernel_scale[:, None]
        return LinearTailState(phi_q, phi_k, numerator, denominator, kernel_scale)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        selected_tokens: torch.Tensor,
        softmax_scale: float,
        selected_valid: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = self.prepare(q, k, v)
        if selected_tokens.ndim != 2 or selected_tokens.shape[0] != self.heads:
            raise ValueError("selected_tokens must have shape [heads, selected]")
        gather_d = selected_tokens[:, :, None].expand(-1, -1, self.channels)
        gather_r = selected_tokens[:, :, None].expand(-1, -1, self.rank)
        k_selected = torch.gather(k, 1, gather_d).float()
        v_selected = torch.gather(v, 1, gather_d).float()
        phi_k_selected = torch.gather(state.phi_k, 1, gather_r)
        logits = torch.einsum("hqd,hsd->hqs", q.float(), k_selected) * float(softmax_scale)
        if logits.detach().amax() >= 80:
            raise FloatingPointError("selected exact logits exceed the FP32 exp safety range")
        exact = logits.exp()
        approximate = torch.einsum(
            "hqr,hsr->hqs", state.phi_q, phi_k_selected
        ) * state.kernel_scale[:, None, None]
        correction = exact - approximate
        if selected_valid is not None:
            if selected_valid.shape != selected_tokens.shape:
                raise ValueError("selected_valid must match selected_tokens")
            correction = correction * selected_valid[:, None, :].to(correction.dtype)
        numerator = state.numerator + torch.einsum(
            "hqs,hsd->hqd", correction, v_selected
        )
        denominator = state.denominator + correction.sum(dim=2)
        if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
            raise FloatingPointError("non-finite sparse-linear numerator or denominator")
        output = numerator / denominator.clamp_min(1e-12).unsqueeze(2)
        return output, denominator


def group_corrections(
    model: PositiveLinearTail,
    head: int,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    layout: GroupLayout,
    softmax_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return base state and exact-minus-linear correction for every group."""
    qh = q.unsqueeze(0)
    kh = k.unsqueeze(0)
    vh = v.unsqueeze(0)
    q_input = qh.float() / model.q_rms[head : head + 1, None, None]
    k_input = kh.float() / model.k_rms[head : head + 1, None, None]
    phi_q = F.softplus(
        torch.einsum("hqd,hdr->hqr", q_input, model.q_projection[head : head + 1])
        + model.q_bias[head : head + 1, None, :]
    ) + 1e-4
    phi_k = F.softplus(
        torch.einsum("hnd,hdr->hnr", k_input, model.k_projection[head : head + 1])
        + model.k_bias[head : head + 1, None, :]
    ) + 1e-4
    kernel_scale = model.log_kernel_scale[head].clamp(-20.0, 20.0).exp() / model.rank
    base_n = torch.einsum("hqr,hrd->hqd", phi_q, torch.einsum("hnr,hnd->hrd", phi_k, vh.float()))[0]
    base_z = torch.einsum("hqr,hr->hq", phi_q, phi_k.sum(dim=1))[0]
    base_n = base_n * kernel_scale
    base_z = base_z * kernel_scale

    safe = layout.indices
    valid = layout.valid
    grouped_k = k.index_select(0, safe.flatten()).reshape(layout.groups, layout.width, -1).float()
    grouped_v = v.index_select(0, safe.flatten()).reshape(layout.groups, layout.width, -1).float()
    grouped_phi = phi_k[0].index_select(0, safe.flatten()).reshape(
        layout.groups, layout.width, model.rank
    )
    logits = torch.einsum("qd,gwd->gqw", q.float(), grouped_k) * float(softmax_scale)
    exact = logits.exp()
    approximate = torch.einsum("qr,gwr->gqw", phi_q[0], grouped_phi) * kernel_scale
    delta = (exact - approximate) * valid[:, None, :].to(exact.dtype)
    correction_n = torch.einsum("gqw,gwd->gqd", delta, grouped_v)
    correction_z = delta.sum(dim=2)
    return base_n, base_z, correction_n, correction_z


def output_for_group_selection(
    base_n: torch.Tensor,
    base_z: torch.Tensor,
    correction_n: torch.Tensor,
    correction_z: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    numerator = base_n + correction_n.index_select(0, selected).sum(dim=0)
    denominator = base_z + correction_z.index_select(0, selected).sum(dim=0)
    return numerator / denominator.clamp_min(1e-12).unsqueeze(1)


def trajectory_width_selection(
    reference: torch.Tensor,
    base_n: torch.Tensor,
    base_z: torch.Tensor,
    correction_n: torch.Tensor,
    correction_z: torch.Tensor,
    budget: int,
    add_chunk: int = 8,
    post_rank: int = 0,
) -> torch.Tensor:
    """Dense-output oracle with fixed-width groups and bounded greedy batches.

    When ``post_rank`` is positive, candidates are scored by the residual
    energy left after their best output-channel rank-r correction. This makes
    support selection optimize the same support-manifold pregate used later.
    """
    groups = correction_n.shape[0]
    if not 0 < budget <= groups or add_chunk <= 0 or post_rank < 0:
        raise ValueError("invalid trajectory-width selection budget")
    selected_mask = torch.zeros(groups, dtype=torch.bool, device=correction_n.device)
    numerator = base_n.clone()
    denominator = base_z.clone()
    selected: list[torch.Tensor] = []
    while len(selected) < budget:
        estimates = (numerator.unsqueeze(0) + correction_n) / (
            denominator.unsqueeze(0) + correction_z
        ).clamp_min(1e-12).unsqueeze(2)
        candidate_defects = reference.unsqueeze(0) - estimates
        if post_rank > 0:
            current = reference - numerator / denominator.clamp_min(1e-12).unsqueeze(1)
            used = min(post_rank, min(current.shape))
            _, _, vh = torch.linalg.svd(current, full_matrices=False)
            basis = vh[:used].T
            projected = candidate_defects - (
                (candidate_defects @ basis) @ basis.T
            )
            errors = projected.square().sum(dim=(1, 2))
        else:
            errors = candidate_defects.square().sum(dim=(1, 2))
        errors.masked_fill_(selected_mask, float("inf"))
        count = min(add_chunk, budget - len(selected))
        chosen = errors.topk(count, largest=False).indices
        selected.extend(chosen.unbind())
        selected_mask[chosen] = True
        numerator = numerator + correction_n.index_select(0, chosen).sum(dim=0)
        denominator = denominator + correction_z.index_select(0, chosen).sum(dim=0)
    return torch.stack(selected).sort().values


def adaptive_rank_residual(defect: torch.Tensor, rank: int) -> torch.Tensor:
    if defect.ndim != 2 or rank < 0:
        raise ValueError("defect must be rank-2 and rank non-negative")
    used = min(rank, min(defect.shape))
    if used == 0:
        return defect
    _, _, vh = torch.linalg.svd(defect, full_matrices=False)
    basis = vh[:used].T
    return defect - (defect @ basis) @ basis.T

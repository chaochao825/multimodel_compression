#!/usr/bin/env python3
"""Core operators for hardware-budgeted sparse-support manifold probes."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AtomPartition:
    indices: torch.Tensor
    valid: torch.Tensor
    execution_width: int
    coordinates: torch.Tensor | None = None

    @property
    def atoms(self) -> int:
        return int(self.indices.shape[0])


@dataclass
class AtomStatistics:
    contributions: torch.Tensor
    mass: torch.Tensor
    valid_counts: torch.Tensor
    execution_width: int
    coordinates: torch.Tensor | None = None


@dataclass
class GroupProblem:
    reference: torch.Tensor
    statistics: AtomStatistics
    budget: int


@dataclass
class SearchResult:
    selections: tuple[torch.Tensor, ...]
    defect: torch.Tensor
    basis: torch.Tensor
    residual_sq: float
    initial_residual_sq: float
    accepted_alternations: int
    accepted_swaps: int
    trace: tuple[dict[str, float | int | str], ...]


def _pad_segments(segments: list[torch.Tensor], width: int) -> tuple[torch.Tensor, torch.Tensor]:
    if not segments:
        raise ValueError("partition must contain at least one segment")
    indices = torch.full((len(segments), width), -1, dtype=torch.long)
    valid = torch.zeros((len(segments), width), dtype=torch.bool)
    for row, segment in enumerate(segments):
        if segment.numel() > width:
            raise ValueError("segment exceeds execution width")
        indices[row, : segment.numel()] = segment
        valid[row, : segment.numel()] = True
    return indices, valid


def contiguous_partition(tokens: int, width: int, offset: int = 0) -> AtomPartition:
    if tokens <= 0 or width <= 0 or offset < 0 or offset >= width:
        raise ValueError("invalid contiguous partition dimensions")
    segments: list[torch.Tensor] = []
    if offset:
        segments.append(torch.arange(0, min(offset, tokens)))
    start = offset
    while start < tokens:
        segments.append(torch.arange(start, min(start + width, tokens)))
        start += width
    indices, valid = _pad_segments(segments, width)
    covered = indices[valid]
    if covered.numel() != tokens or not torch.equal(covered.sort().values, torch.arange(tokens)):
        raise RuntimeError("contiguous partition does not cover every token exactly once")
    return AtomPartition(indices, valid, width)


def thw_partition(
    shape: tuple[int, int, int], tile_height: int = 8, tile_width: int = 8
) -> AtomPartition:
    temporal, height, width = shape
    if min(*shape, tile_height, tile_width) <= 0:
        raise ValueError("THW dimensions must be positive")
    segments: list[torch.Tensor] = []
    coordinates: list[tuple[int, int, int]] = []
    for time in range(temporal):
        for row in range(0, height, tile_height):
            for column in range(0, width, tile_width):
                values = [
                    time * height * width + h * width + w
                    for h in range(row, min(row + tile_height, height))
                    for w in range(column, min(column + tile_width, width))
                ]
                segments.append(torch.tensor(values, dtype=torch.long))
                coordinates.append((time, row // tile_height, column // tile_width))
    execution_width = tile_height * tile_width
    indices, valid = _pad_segments(segments, execution_width)
    return AtomPartition(
        indices,
        valid,
        execution_width,
        torch.tensor(coordinates, dtype=torch.long),
    )


def atom_statistics(
    attention: torch.Tensor,
    value: torch.Tensor,
    partition: AtomPartition,
) -> AtomStatistics:
    if attention.ndim != 2 or value.ndim != 2:
        raise ValueError("attention and value must be rank-2")
    if attention.shape[1] != value.shape[0]:
        raise ValueError("attention key dimension must match value tokens")
    indices = partition.indices.to(attention.device)
    valid = partition.valid.to(attention.device)
    safe = indices.clamp_min(0)
    gathered_attention = attention.index_select(1, safe.reshape(-1)).reshape(
        attention.shape[0], partition.atoms, partition.execution_width
    )
    gathered_attention = gathered_attention * valid.unsqueeze(0)
    gathered_value = value.index_select(0, safe.reshape(-1)).reshape(
        partition.atoms, partition.execution_width, value.shape[1]
    )
    gathered_value = gathered_value * valid.unsqueeze(2)
    contributions = torch.einsum("qbk,bkd->bqd", gathered_attention, gathered_value)
    mass = gathered_attention.sum(dim=2).T.contiguous()
    return AtomStatistics(
        contributions=contributions,
        mass=mass,
        valid_counts=valid.sum(dim=1),
        execution_width=partition.execution_width,
        coordinates=None if partition.coordinates is None else partition.coordinates.to(attention.device),
    )


def aggregate_statistics(
    statistics: AtomStatistics,
    groups: tuple[tuple[int, ...], ...],
    execution_width: int,
) -> AtomStatistics:
    if not groups or execution_width <= 0:
        raise ValueError("aggregate groups and execution width must be positive")
    contributions = []
    mass = []
    valid_counts = []
    for group in groups:
        if not group:
            raise ValueError("aggregate atom group must not be empty")
        indices = torch.tensor(group, device=statistics.contributions.device)
        contributions.append(statistics.contributions.index_select(0, indices).sum(dim=0))
        mass.append(statistics.mass.index_select(0, indices).sum(dim=0))
        valid_counts.append(statistics.valid_counts.index_select(0, indices).sum())
    return AtomStatistics(
        contributions=torch.stack(contributions),
        mass=torch.stack(mass),
        valid_counts=torch.stack(valid_counts),
        execution_width=execution_width,
    )


def slice_statistics(statistics: AtomStatistics, start: int, stop: int) -> AtomStatistics:
    if start < 0 or stop <= start or stop > statistics.contributions.shape[1]:
        raise ValueError("invalid query slice")
    return AtomStatistics(
        contributions=statistics.contributions[:, start:stop],
        mass=statistics.mass[:, start:stop],
        valid_counts=statistics.valid_counts,
        execution_width=statistics.execution_width,
        coordinates=statistics.coordinates,
    )


def selected_output(problem: GroupProblem, selected: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    stats = problem.statistics
    numerator = stats.contributions.index_select(0, selected).sum(dim=0)
    mass = stats.mass.index_select(0, selected).sum(dim=0)
    return numerator / mass.clamp_min(1e-30).unsqueeze(1), mass


def assemble_defect(
    groups: tuple[GroupProblem, ...], selections: tuple[torch.Tensor, ...]
) -> torch.Tensor:
    if len(groups) != len(selections):
        raise ValueError("group and selection counts differ")
    defects = []
    for problem, selected in zip(groups, selections):
        output, _ = selected_output(problem, selected)
        defects.append(problem.reference - output)
    return torch.cat(defects, dim=0)


def adaptive_tail(defect: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if defect.ndim != 2 or rank < 0:
        raise ValueError("defect must be rank-2 and rank non-negative")
    used = min(rank, min(defect.shape))
    if used == 0:
        return defect.new_zeros((defect.shape[1], 0)), defect, defect.new_empty(0)
    _, singular, vh = torch.linalg.svd(defect, full_matrices=False)
    basis = vh[:used].T.contiguous()
    residual = defect - (defect @ basis) @ basis.T
    return basis, residual, singular


def batched_tail_energy(defects: torch.Tensor, rank: int) -> torch.Tensor:
    if defects.ndim != 3:
        raise ValueError("batched defects must have shape [batch, queries, channels]")
    used = min(rank, min(defects.shape[1:]))
    gram = defects @ defects.transpose(1, 2)
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0)
    if used >= eigenvalues.shape[1]:
        return eigenvalues.new_zeros(eigenvalues.shape[0])
    return eigenvalues[:, : eigenvalues.shape[1] - used].sum(dim=1)


def _score_selection(problem: GroupProblem, method: str) -> torch.Tensor:
    if method == "contribution_norm":
        return problem.statistics.contributions.square().sum(dim=(1, 2))
    if method == "mass":
        return problem.statistics.mass.sum(dim=1)
    raise ValueError(f"unsupported initializer: {method}")


def initial_selections(
    groups: tuple[GroupProblem, ...], method: str
) -> tuple[torch.Tensor, ...]:
    selections = []
    for problem in groups:
        if problem.budget <= 0 or problem.budget > problem.statistics.contributions.shape[0]:
            raise ValueError("invalid atom budget")
        selections.append(_score_selection(problem, method).topk(problem.budget).indices.sort().values)
    return tuple(selections)


def _projected_scores(problem: GroupProblem, basis: torch.Tensor) -> torch.Tensor:
    contributions = problem.statistics.contributions
    projected = contributions - (contributions @ basis) @ basis.T
    return projected.square().sum(dim=(1, 2))


def _replace_group_defects(
    current: torch.Tensor,
    group_index: int,
    groups: tuple[GroupProblem, ...],
    replacements: torch.Tensor,
) -> torch.Tensor:
    start = sum(group.reference.shape[0] for group in groups[:group_index])
    stop = start + groups[group_index].reference.shape[0]
    output = current.unsqueeze(0).expand(replacements.shape[0], -1, -1).clone()
    output[:, start:stop] = replacements
    return output


def _best_swap(
    groups: tuple[GroupProblem, ...],
    selections: tuple[torch.Tensor, ...],
    defect: torch.Tensor,
    basis: torch.Tensor,
    rank: int,
    shortlist: int,
) -> tuple[int, int, int, float] | None:
    current_sq = float((defect - (defect @ basis) @ basis.T).square().sum())
    best: tuple[int, int, int, float] | None = None
    for group_index, (problem, selected) in enumerate(zip(groups, selections)):
        scores = _projected_scores(problem, basis)
        selected_mask = torch.zeros_like(scores, dtype=torch.bool)
        selected_mask[selected] = True
        remove = selected[scores.index_select(0, selected).argsort()[:shortlist]]
        available = (~selected_mask).nonzero(as_tuple=False).flatten()
        add = available[scores.index_select(0, available).argsort(descending=True)[:shortlist]]
        if remove.numel() == 0 or add.numel() == 0:
            continue
        remove_grid = remove[:, None].expand(-1, add.numel()).reshape(-1)
        add_grid = add[None, :].expand(remove.numel(), -1).reshape(-1)
        stats = problem.statistics
        numerator = stats.contributions.index_select(0, selected).sum(dim=0)
        mass = stats.mass.index_select(0, selected).sum(dim=0)
        candidate_numerator = (
            numerator.unsqueeze(0)
            - stats.contributions.index_select(0, remove_grid)
            + stats.contributions.index_select(0, add_grid)
        )
        candidate_mass = (
            mass.unsqueeze(0)
            - stats.mass.index_select(0, remove_grid)
            + stats.mass.index_select(0, add_grid)
        )
        replacement = problem.reference.unsqueeze(0) - candidate_numerator / candidate_mass.clamp_min(
            1e-30
        ).unsqueeze(2)
        candidate_defects = _replace_group_defects(defect, group_index, groups, replacement)
        energies = batched_tail_energy(candidate_defects, rank)
        index = int(energies.argmin())
        value = float(energies[index])
        if value + max(1e-12, current_sq * 1e-7) < current_sq and (
            best is None or value < best[3]
        ):
            best = (group_index, int(remove_grid[index]), int(add_grid[index]), value)
    return best


def optimize_support(
    groups: tuple[GroupProblem, ...],
    rank: int,
    alternations: int = 2,
    swap_steps: int = 4,
    shortlist: int = 16,
    explicit_initial: tuple[torch.Tensor, ...] | None = None,
) -> SearchResult:
    candidates: list[tuple[str, tuple[torch.Tensor, ...]]] = []
    if explicit_initial is not None:
        candidates.append(("explicit", explicit_initial))
    candidates.extend((method, initial_selections(groups, method)) for method in ("contribution_norm", "mass"))
    evaluated = []
    for name, selections in candidates:
        defect = assemble_defect(groups, selections)
        basis, residual, _ = adaptive_tail(defect, rank)
        evaluated.append((float(residual.square().sum()), name, selections, defect, basis))
    residual_sq, initializer, selections, defect, basis = min(evaluated, key=lambda item: item[0])
    initial_residual_sq = residual_sq
    trace: list[dict[str, float | int | str]] = [
        {"stage": "initial", "method": initializer, "residual_sq": residual_sq}
    ]
    accepted_alternations = 0
    for iteration in range(alternations):
        proposed = tuple(
            _projected_scores(problem, basis).topk(problem.budget).indices.sort().values
            for problem in groups
        )
        proposed_defect = assemble_defect(groups, proposed)
        proposed_basis, proposed_residual, _ = adaptive_tail(proposed_defect, rank)
        proposed_sq = float(proposed_residual.square().sum())
        accepted = proposed_sq + max(1e-12, residual_sq * 1e-7) < residual_sq
        trace.append(
            {
                "stage": "alternation",
                "iteration": iteration,
                "accepted": int(accepted),
                "residual_sq": proposed_sq,
            }
        )
        if accepted:
            selections, defect, basis, residual_sq = proposed, proposed_defect, proposed_basis, proposed_sq
            accepted_alternations += 1
    accepted_swaps = 0
    for iteration in range(swap_steps):
        swap = _best_swap(groups, selections, defect, basis, rank, shortlist)
        if swap is None:
            break
        group_index, remove, add, _ = swap
        updated = list(selections)
        selected = updated[group_index]
        updated[group_index] = torch.cat((selected[selected != remove], selected.new_tensor([add]))).sort().values
        selections = tuple(updated)
        defect = assemble_defect(groups, selections)
        basis, residual, _ = adaptive_tail(defect, rank)
        residual_sq = float(residual.square().sum())
        accepted_swaps += 1
        trace.append(
            {
                "stage": "swap",
                "iteration": iteration,
                "group": group_index,
                "remove": remove,
                "add": add,
                "residual_sq": residual_sq,
            }
        )
    return SearchResult(
        selections=selections,
        defect=defect,
        basis=basis,
        residual_sq=residual_sq,
        initial_residual_sq=initial_residual_sq,
        accepted_alternations=accepted_alternations,
        accepted_swaps=accepted_swaps,
        trace=tuple(trace),
    )


def motion_path_selection(
    problem: GroupProblem,
    temporal: int,
    tile_rows: int,
    tile_columns: int,
) -> torch.Tensor:
    coordinates = problem.statistics.coordinates
    if coordinates is None or coordinates.shape[0] != temporal * tile_rows * tile_columns:
        raise ValueError("motion paths require a dense THW tile partition")
    scores = problem.statistics.contributions.square().sum(dim=(1, 2)).detach().cpu()
    selected = torch.zeros(scores.numel(), dtype=torch.bool)
    spatial = tile_rows * tile_columns
    paths = problem.budget // temporal
    for _ in range(paths):
        dp = torch.full((temporal, spatial), float("-inf"))
        parent = torch.full((temporal, spatial), -1, dtype=torch.long)
        first = scores[:spatial].clone()
        first[selected[:spatial]] = float("-inf")
        dp[0] = first
        for time in range(1, temporal):
            current = scores[time * spatial : (time + 1) * spatial].clone()
            current[selected[time * spatial : (time + 1) * spatial]] = float("-inf")
            for row in range(tile_rows):
                for column in range(tile_columns):
                    state = row * tile_columns + column
                    neighbors = [
                        r * tile_columns + c
                        for r in range(max(0, row - 1), min(tile_rows, row + 2))
                        for c in range(max(0, column - 1), min(tile_columns, column + 2))
                    ]
                    values = dp[time - 1, neighbors]
                    best = int(values.argmax())
                    dp[time, state] = current[state] + values[best]
                    parent[time, state] = neighbors[best]
        state = int(dp[-1].argmax())
        if not math.isfinite(float(dp[-1, state])):
            break
        path = []
        for time in range(temporal - 1, -1, -1):
            path.append(time * spatial + state)
            state = int(parent[time, state]) if time else state
        selected[path] = True
    remaining = problem.budget - int(selected.sum())
    if remaining > 0:
        fill_scores = scores.clone()
        fill_scores[selected] = float("-inf")
        selected[fill_scores.topk(remaining).indices] = True
    return selected.nonzero(as_tuple=False).flatten().to(problem.reference.device).sort().values


def support_cost(
    groups: tuple[GroupProblem, ...], selections: tuple[torch.Tensor, ...], key_tokens: int
) -> dict[str, float | int]:
    query_tokens = sum(problem.reference.shape[0] for problem in groups)
    execution_pairs = 0
    logical_pairs = 0
    kernel_tiles = 0
    for problem, selected in zip(groups, selections):
        queries = problem.reference.shape[0]
        execution_pairs += queries * selected.numel() * problem.statistics.execution_width
        logical_pairs += queries * int(problem.statistics.valid_counts.index_select(0, selected).sum())
        kernel_tiles += selected.numel()
    dense_pairs = query_tokens * key_tokens
    return {
        "query_tokens": query_tokens,
        "key_tokens": key_tokens,
        "logical_pairs": logical_pairs,
        "execution_pairs": execution_pairs,
        "logical_density": logical_pairs / dense_pairs,
        "execution_density": execution_pairs / dense_pairs,
        "kernel_tiles": kernel_tiles,
    }

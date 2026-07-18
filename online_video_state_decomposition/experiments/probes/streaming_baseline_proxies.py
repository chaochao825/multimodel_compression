from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Callable

import numpy as np

from task_memory import normalize_rows


PROXY_METHODS = (
    "exact_recent",
    "causalmem_feature_proxy",
    "streamingtom_feature_proxy",
    "stc_feature_proxy",
    "selectstream_feature_proxy",
    "oasis_feature_proxy",
    "statekv_feature_proxy",
)


@dataclass(frozen=True)
class MemoryAccounting:
    active_state_bytes: int
    archive_bytes: int
    detailed_decode_bytes: int
    metadata_bytes: int
    shared_parameter_bytes: int
    total_retained_bytes: int
    active_state_bounded: bool
    total_state_bounded: bool


@dataclass(frozen=True)
class ProxyResult:
    method: str
    reproduction_tier: str
    evidence_vectors: np.ndarray
    evidence_indices: tuple[int, ...]
    accounting: MemoryAccounting
    write_seconds: float
    read_seconds: float
    estimated_write_flops: int
    estimated_read_flops: int
    query_conditioned: bool
    diagnostics: dict[str, float | int | str | bool]


@dataclass
class _GraphNode:
    state: np.ndarray
    representative: int
    start: int
    end: int
    surprise: float
    writes: int = 1
    reads: int = 0
    merges: int = 0


def _validate_inputs(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
) -> tuple[np.ndarray, np.ndarray]:
    values = normalize_rows(np.asarray(vectors, dtype=np.float64))
    query = normalize_rows(np.asarray(query_vector, dtype=np.float64))[0]
    if values.ndim != 2 or not len(values):
        raise ValueError("vectors must be a non-empty matrix")
    if query.shape != (values.shape[1],):
        raise ValueError("query_vector and vectors must share hidden_dim")
    if evidence_budget <= 0 or pool_capacity <= 0:
        raise ValueError("evidence and pool budgets must be positive")
    return values, query


def _bytes_for_values(count: int, hidden_dim: int, bits: int) -> int:
    return (count * hidden_dim * bits + 7) // 8


def _top_indices(
    scores: np.ndarray,
    candidates: list[int],
    count: int,
) -> list[int]:
    return sorted(
        candidates,
        key=lambda index: (float(scores[index]), index),
        reverse=True,
    )[:count]


def _fill_unique(
    primary: list[int],
    fallback: list[int],
    budget: int,
) -> list[int]:
    selected: list[int] = []
    for index in primary + fallback:
        if index not in selected:
            selected.append(index)
        if len(selected) >= budget:
            break
    return sorted(selected)


def _symmetric_quantize(values: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2 or bits > 8:
        raise ValueError("quantization bits must be in [2, 8]")
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return values.copy()
    limit = (1 << (bits - 1)) - 1
    scales = np.max(np.abs(values), axis=1, keepdims=True) / max(limit, 1)
    scales = np.maximum(scales, 1e-12)
    quantized = np.clip(np.rint(values / scales), -limit, limit)
    return normalize_rows(quantized * scales)


def exact_recent_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    storage_bits: int = 16,
    **_: object,
) -> ProxyResult:
    values, _query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    start = max(0, len(values) - evidence_budget)
    indices = list(range(start, len(values)))
    payload = _bytes_for_values(len(indices), values.shape[1], storage_bits)
    metadata = 4 * len(indices) + 16
    return ProxyResult(
        method="exact_recent",
        reproduction_tier="project_native_control",
        evidence_vectors=values[indices],
        evidence_indices=tuple(indices),
        accounting=MemoryAccounting(
            active_state_bytes=payload,
            archive_bytes=0,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=payload + metadata,
            active_state_bounded=True,
            total_state_bounded=True,
        ),
        write_seconds=0.0,
        read_seconds=0.0,
        estimated_write_flops=0,
        estimated_read_flops=0,
        query_conditioned=False,
        diagnostics={"retained_vectors": len(indices)},
    )


def causalmem_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    storage_bits: int = 16,
    basis_rank: int = 8,
    basis_decay: float = 0.9,
    max_new_basis: int = 1,
    **_: object,
) -> ProxyResult:
    """Frame-vector proxy for CausalMem's basis and residual retention.

    The official implementation operates on many projected tokens per frame.
    With one CLIP vector per frame, its per-frame background merge degenerates
    to retaining the original vector, so this proxy uses the repository's
    mem1-only retention path and records that limitation in diagnostics.
    """

    values, _query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    rank_limit = min(max(1, basis_rank), hidden_dim)
    retained: list[int] = []
    basis = np.empty((hidden_dim, 0), dtype=np.float64)
    activity = np.empty((0,), dtype=np.float64)
    write_flops = 0
    write_start = perf_counter()

    for frame_index, value in enumerate(values):
        if basis.shape[1] == 0:
            basis = value[:, None]
            activity = np.ones(1, dtype=np.float64)
        else:
            coefficients = value @ basis
            residual = value - basis @ coefficients
            residual_norm = float(np.linalg.norm(residual))
            activity = (
                basis_decay * activity
                + (1.0 - basis_decay) * np.abs(coefficients)
            )
            write_flops += 4 * hidden_dim * basis.shape[1]
            if residual_norm > 1e-8 and max_new_basis > 0:
                direction = residual / residual_norm
                direction -= basis @ (basis.T @ direction)
                direction_norm = float(np.linalg.norm(direction))
                if direction_norm > 1e-8:
                    basis = np.concatenate(
                        (basis, (direction / direction_norm)[:, None]),
                        axis=1,
                    )
                    activity = np.concatenate((activity, np.ones(1)))
            if basis.shape[1] > rank_limit:
                keep = np.argsort(activity, kind="stable")[-rank_limit:]
                keep.sort()
                basis = basis[:, keep]
                activity = activity[keep]

        retained.append(frame_index)
        if len(retained) > evidence_budget:
            memory = values[retained]
            residual = memory - (memory @ basis) @ basis.T
            energies = np.linalg.norm(residual, axis=1)
            order = sorted(
                range(len(retained)),
                key=lambda pos: (float(energies[pos]), retained[pos]),
                reverse=True,
            )[:evidence_budget]
            retained = sorted(retained[pos] for pos in order)
            write_flops += 4 * len(memory) * hidden_dim * basis.shape[1]

    write_seconds = perf_counter() - write_start
    payload = _bytes_for_values(len(retained), hidden_dim, storage_bits)
    basis_bytes = _bytes_for_values(
        basis.shape[1], hidden_dim, storage_bits
    )
    metadata = 4 * len(retained) + 4 * basis.shape[1] + 24
    total = payload + basis_bytes + metadata
    return ProxyResult(
        method="causalmem_feature_proxy",
        reproduction_tier="official_mechanism_feature_proxy",
        evidence_vectors=values[retained],
        evidence_indices=tuple(retained),
        accounting=MemoryAccounting(
            active_state_bytes=payload + basis_bytes,
            archive_bytes=0,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=True,
        ),
        write_seconds=write_seconds,
        read_seconds=0.0,
        estimated_write_flops=write_flops,
        estimated_read_flops=0,
        query_conditioned=False,
        diagnostics={
            "basis_rank": basis.shape[1],
            "basis_decay": basis_decay,
            "background_merge_represented": False,
            "official_source_variant": "llava_mem1_only",
        },
    )


def streamingtom_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    storage_bits: int = 16,
    quantization_bits: int = 4,
    **_: object,
) -> ProxyResult:
    """Feature-group proxy for CTR plus Online Quantized Memory."""

    values, query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    write_start = perf_counter()
    mean = normalize_rows(np.mean(values, axis=0))[0]
    change = np.zeros(len(values), dtype=np.float64)
    change[1:] = 1.0 - np.sum(values[1:] * values[:-1], axis=1)
    saliency = 1.0 - values @ mean
    scores = 0.5 * change + 0.5 * saliency

    # At frame-vector resolution, a two-frame window is one CTR token group.
    archive: list[int] = []
    for start in range(0, len(values), 2):
        window = list(range(start, min(start + 2, len(values))))
        archive.append(max(window, key=lambda index: (scores[index], index)))
    archive.sort()
    quantized = _symmetric_quantize(values[archive], quantization_bits)
    write_seconds = perf_counter() - write_start

    read_start = perf_counter()
    relevance = quantized @ query
    local_order = sorted(
        range(len(archive)),
        key=lambda pos: (float(relevance[pos]), archive[pos]),
        reverse=True,
    )[:evidence_budget]
    selected_positions = sorted(local_order)
    selected = [archive[pos] for pos in selected_positions]
    evidence = quantized[selected_positions]
    read_seconds = perf_counter() - read_start

    archive_payload = _bytes_for_values(
        len(archive), hidden_dim, quantization_bits
    )
    quant_metadata = len(archive) * 8
    active_payload = _bytes_for_values(
        len(selected), hidden_dim, storage_bits
    )
    metadata = 4 * (len(archive) + len(selected)) + quant_metadata + 32
    total = archive_payload + active_payload + metadata
    return ProxyResult(
        method="streamingtom_feature_proxy",
        reproduction_tier="official_mechanism_feature_group_proxy",
        evidence_vectors=evidence,
        evidence_indices=tuple(selected),
        accounting=MemoryAccounting(
            active_state_bytes=active_payload,
            archive_bytes=archive_payload,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=False,
        ),
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        estimated_write_flops=4 * len(values) * hidden_dim,
        estimated_read_flops=2 * len(archive) * hidden_dim,
        query_conditioned=True,
        diagnostics={
            "ctr_groups": len(archive),
            "quantization_bits": quantization_bits,
            "archive_growth": "linear_in_stream_length",
            "token_level_attention_saliency_represented": False,
        },
    )


def stc_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    storage_bits: int = 16,
    cache_similarity: float = 0.97,
    **_: object,
) -> ProxyResult:
    """Feature-group proxy for STC-Cacher and STC-Pruner."""

    values, _query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    write_start = perf_counter()
    effective = values.copy()
    reused = np.zeros(len(values), dtype=bool)
    for index in range(1, len(values)):
        if float(values[index] @ effective[index - 1]) >= cache_similarity:
            effective[index] = effective[index - 1]
            reused[index] = True

    temporal = np.zeros(len(values), dtype=np.float64)
    temporal[1:] = 1.0 - np.sum(effective[1:] * effective[:-1], axis=1)
    centroid = normalize_rows(np.mean(effective, axis=0))[0]
    spatial_proxy = 1.0 - effective @ centroid
    score = temporal + spatial_proxy
    dynamic_budget = max(1, evidence_budget // 2)
    dynamic = _top_indices(score, list(range(len(values))), dynamic_budget)
    uniform = [
        min(
            int(((2 * index + 1) * len(values)) // (2 * evidence_budget)),
            len(values) - 1,
        )
        for index in range(evidence_budget)
    ]
    selected = _fill_unique(dynamic, uniform + list(range(len(values) - 1, -1, -1)), evidence_budget)
    write_seconds = perf_counter() - write_start
    payload = _bytes_for_values(len(selected), hidden_dim, storage_bits)
    cacher_bytes = _bytes_for_values(1, hidden_dim, storage_bits)
    metadata = 4 * len(selected) + len(values) + 24
    total = payload + cacher_bytes + metadata
    return ProxyResult(
        method="stc_feature_proxy",
        reproduction_tier="official_mechanism_feature_group_proxy",
        evidence_vectors=effective[selected],
        evidence_indices=tuple(selected),
        accounting=MemoryAccounting(
            active_state_bytes=payload + cacher_bytes,
            archive_bytes=0,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=True,
        ),
        write_seconds=write_seconds,
        read_seconds=0.0,
        estimated_write_flops=4 * len(values) * hidden_dim,
        estimated_read_flops=0,
        query_conditioned=False,
        diagnostics={
            "cache_similarity": cache_similarity,
            "reused_frame_groups": int(np.sum(reused)),
            "spatial_token_structure_represented": False,
        },
    )


def _segment_representative(
    values: np.ndarray,
    indices: list[int],
    state: np.ndarray,
) -> int:
    return max(
        indices,
        key=lambda index: (float(values[index] @ state), index),
    )


def _merge_graph_nodes(nodes: list[_GraphNode], capacity: int) -> None:
    while len(nodes) > capacity:
        states = normalize_rows(np.stack([node.state for node in nodes]))
        similarity = states @ states.T
        np.fill_diagonal(similarity, -np.inf)
        max_surprise = max((node.surprise for node in nodes), default=1.0)
        max_end = max((node.end for node in nodes), default=1)
        best: tuple[float, int, int] | None = None
        for left in range(len(nodes)):
            for right in range(left + 1, len(nodes)):
                priority = 0.5 * (
                    nodes[left].surprise + nodes[right].surprise
                ) / max(max_surprise, 1e-12)
                recency = 0.5 * (
                    nodes[left].end + nodes[right].end + 2
                ) / max(max_end + 1, 1)
                penalty = 1.0 - float(similarity[left, right])
                penalty += 0.25 * priority + 0.1 * recency
                candidate = (penalty, left, right)
                if best is None or candidate < best:
                    best = candidate
        assert best is not None
        _, left, right = best
        first, second = nodes[left], nodes[right]
        weight = first.writes + second.writes
        state = normalize_rows(
            (
                first.writes * first.state
                + second.writes * second.state
            )
            / weight
        )[0]
        representative = (
            first.representative
            if float(first.state @ state) >= float(second.state @ state)
            else second.representative
        )
        merged = _GraphNode(
            state=state,
            representative=representative,
            start=min(first.start, second.start),
            end=max(first.end, second.end),
            surprise=first.surprise + second.surprise,
            writes=weight,
            reads=first.reads + second.reads,
            merges=first.merges + second.merges + 1,
        )
        nodes.pop(right)
        nodes.pop(left)
        nodes.append(merged)
        nodes.sort(key=lambda node: (node.start, node.end))


def selectstream_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int = 3,
    storage_bits: int = 16,
    min_segment: int = 2,
    max_segment: int = 4,
    surprise_threshold: float = 0.25,
    surprise_energy_budget: float = 0.6,
    **_: object,
) -> ProxyResult:
    values, query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    nodes: list[_GraphNode] = []
    segment: list[int] = []
    segment_energy = 0.0
    change = np.zeros(len(values), dtype=np.float64)
    change[1:] = 1.0 - np.sum(values[1:] * values[:-1], axis=1)
    write_start = perf_counter()

    def close_segment() -> None:
        nonlocal segment, segment_energy
        if not segment:
            return
        state = normalize_rows(np.mean(values[segment], axis=0))[0]
        nodes.append(
            _GraphNode(
                state=state,
                representative=_segment_representative(values, segment, state),
                start=segment[0],
                end=segment[-1],
                surprise=segment_energy,
                writes=len(segment),
            )
        )
        _merge_graph_nodes(nodes, pool_capacity)
        segment = []
        segment_energy = 0.0

    for index in range(len(values)):
        segment.append(index)
        segment_energy += float(change[index])
        length = len(segment)
        trigger = (
            length >= min_segment
            and (
                float(change[index]) >= surprise_threshold
                or segment_energy >= surprise_energy_budget
                or length >= max_segment
            )
        )
        if trigger:
            close_segment()
    close_segment()
    write_seconds = perf_counter() - write_start

    read_start = perf_counter()
    node_states = normalize_rows(np.stack([node.state for node in nodes]))
    relevance = node_states @ query
    surprise_scale = max((node.surprise for node in nodes), default=1.0)
    span_scale = max((node.end - node.start + 1 for node in nodes), default=1)
    merge_scale = max((node.merges for node in nodes), default=1)
    coarse = np.asarray(
        [
            relevance[index]
            + 0.1 * nodes[index].surprise / max(surprise_scale, 1e-12)
            - 0.05 * (nodes[index].end - nodes[index].start + 1) / span_scale
            - 0.05 * nodes[index].merges / max(merge_scale, 1)
            for index in range(len(nodes))
        ]
    )
    seed_count = min(4, len(nodes))
    seeds = _top_indices(coarse, list(range(len(nodes))), seed_count)
    routed = set(seeds)
    similarity = node_states @ node_states.T
    for seed in seeds:
        temporal_neighbors = sorted(
            range(len(nodes)),
            key=lambda other: (
                min(
                    abs(nodes[seed].start - nodes[other].end),
                    abs(nodes[other].start - nodes[seed].end),
                ),
                other,
            ),
        )[:3]
        semantic_neighbors = _top_indices(
            similarity[seed], list(range(len(nodes))), 3
        )
        routed.update(temporal_neighbors)
        routed.update(semantic_neighbors)
    refined_scores = []
    for index in routed:
        neighbors = [other for other in routed if other != index]
        support = (
            max(float(similarity[index, other]) for other in neighbors)
            if neighbors
            else 0.0
        )
        refined_scores.append((float(coarse[index]) + 0.1 * support, index))
    history_budget = max(0, evidence_budget - min(recent_anchors, evidence_budget))
    history_nodes = [index for _, index in sorted(refined_scores, reverse=True)[:history_budget]]
    recent = list(range(max(0, len(values) - recent_anchors), len(values)))
    evidence_vectors = [values[index] for index in recent]
    evidence_indices = list(recent)
    for node_index in history_nodes:
        nodes[node_index].reads += 1
        evidence_vectors.append(nodes[node_index].state)
        evidence_indices.append(nodes[node_index].representative)
    if not evidence_vectors:
        evidence_vectors = [values[-1]]
        evidence_indices = [len(values) - 1]
    read_seconds = perf_counter() - read_start

    node_payload = _bytes_for_values(len(nodes), hidden_dim, storage_bits)
    current_payload = _bytes_for_values(len(recent), hidden_dim, storage_bits)
    graph_bytes = len(nodes) * len(nodes) * 2
    metadata = len(nodes) * 40 + 4 * len(recent) + graph_bytes + 32
    total = node_payload + current_payload + metadata
    return ProxyResult(
        method="selectstream_feature_proxy",
        reproduction_tier="paper_mechanism_feature_proxy_untrained",
        evidence_vectors=normalize_rows(np.stack(evidence_vectors)),
        evidence_indices=tuple(evidence_indices),
        accounting=MemoryAccounting(
            active_state_bytes=node_payload + current_payload,
            archive_bytes=0,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=True,
        ),
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        estimated_write_flops=4 * len(values) * hidden_dim,
        estimated_read_flops=2 * len(nodes) * hidden_dim * 3,
        query_conditioned=True,
        diagnostics={
            "memory_nodes": len(nodes),
            "routed_nodes": len(routed),
            "recent_direct_vectors": len(recent),
            "learned_segment_encoder_represented": False,
            "learned_gar_represented": False,
        },
    )


def oasis_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int = 3,
    storage_bits: int = 16,
    event_window: int = 2,
    coarse_threshold: float = 0.3,
    **_: object,
) -> ProxyResult:
    values, query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    root_capacity = max(1, pool_capacity - recent_anchors)
    roots: list[_GraphNode] = []
    archived_events: list[np.ndarray] = []
    write_start = perf_counter()
    for start in range(0, max(0, len(values) - recent_anchors), event_window):
        indices = list(
            range(start, min(start + event_window, len(values) - recent_anchors))
        )
        if not indices:
            continue
        state = normalize_rows(np.mean(values[indices], axis=0))[0]
        pairwise_similarity = np.sum(
            values[indices[1:]] * values[indices[:-1]], axis=1
        )
        local_change = (
            float(np.mean(1.0 - pairwise_similarity))
            if len(indices) > 1
            else 0.0
        )
        archived_events.append(state.copy())
        roots.append(
            _GraphNode(
                state=state,
                representative=_segment_representative(values, indices, state),
                start=indices[0],
                end=indices[-1],
                surprise=local_change,
                writes=len(indices),
            )
        )
        _merge_graph_nodes(roots, root_capacity)
    write_seconds = perf_counter() - write_start

    read_start = perf_counter()
    recent = list(range(max(0, len(values) - recent_anchors), len(values)))
    recent_relevance = values[recent] @ query if recent else np.empty(0)
    coarse_sufficient = bool(
        len(recent_relevance) and np.max(recent_relevance) >= coarse_threshold
    )
    evidence_vectors = [values[index] for index in recent]
    evidence_indices = list(recent)
    retrieved = 0
    if not coarse_sufficient and roots:
        root_states = normalize_rows(np.stack([node.state for node in roots]))
        scores = root_states @ query
        retrieve_count = max(0, evidence_budget - len(recent))
        chosen = _top_indices(scores, list(range(len(roots))), retrieve_count)
        for root_index in chosen:
            evidence_vectors.append(roots[root_index].state)
            evidence_indices.append(roots[root_index].representative)
        retrieved = len(chosen)
    if not evidence_vectors:
        evidence_vectors = [values[-1]]
        evidence_indices = [len(values) - 1]
    read_seconds = perf_counter() - read_start

    root_payload = _bytes_for_values(len(roots), hidden_dim, storage_bits)
    recent_payload = _bytes_for_values(len(recent), hidden_dim, storage_bits)
    archive_payload = _bytes_for_values(
        len(archived_events), hidden_dim, storage_bits
    )
    metadata = (
        len(roots) * 48
        + len(archived_events) * 24
        + 4 * len(recent)
        + 32
    )
    total = root_payload + recent_payload + archive_payload + metadata
    return ProxyResult(
        method="oasis_feature_proxy",
        reproduction_tier="official_structure_feature_proxy_no_mllm_summaries",
        evidence_vectors=normalize_rows(np.stack(evidence_vectors)),
        evidence_indices=tuple(evidence_indices),
        accounting=MemoryAccounting(
            active_state_bytes=root_payload + recent_payload,
            archive_bytes=archive_payload,
            detailed_decode_bytes=0,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=False,
        ),
        write_seconds=write_seconds,
        read_seconds=read_seconds,
        estimated_write_flops=2 * len(values) * hidden_dim,
        estimated_read_flops=2 * len(roots) * hidden_dim,
        query_conditioned=True,
        diagnostics={
            "event_roots": len(roots),
            "archived_events": len(archived_events),
            "coarse_sufficient": coarse_sufficient,
            "retrieved_events": retrieved,
            "mllm_event_summaries_represented": False,
            "intent_tool_call_represented": False,
            "official_descendant_archive_growth": "linear_in_event_count",
        },
    )


def statekv_feature_proxy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    storage_bits: int = 16,
    **_: object,
) -> ProxyResult:
    """Frame-vector proxy for StateKV's attention-selected cstate.

    Paper-faithful StateKV decodes from the full detailed cache. This proxy does
    the same at frame-vector resolution, so its total state is intentionally
    marked unbounded and its evidence count may exceed evidence_budget.
    """

    values, _query = _validate_inputs(
        vectors,
        query_vector,
        evidence_budget=evidence_budget,
        pool_capacity=pool_capacity,
    )
    hidden_dim = values.shape[1]
    cstate_indices: list[int] = []
    write_flops = 0
    write_start = perf_counter()
    for frame_index, value in enumerate(values):
        candidates = cstate_indices + [frame_index]
        candidate_values = values[candidates]
        importance = candidate_values @ value
        local_keep = sorted(
            range(len(candidates)),
            key=lambda pos: (float(importance[pos]), candidates[pos]),
            reverse=True,
        )[:pool_capacity]
        cstate_indices = sorted(candidates[pos] for pos in local_keep)
        write_flops += 2 * len(candidates) * hidden_dim
    write_seconds = perf_counter() - write_start

    cstate_payload = _bytes_for_values(
        len(cstate_indices), hidden_dim, storage_bits
    )
    detailed_payload = _bytes_for_values(len(values), hidden_dim, storage_bits)
    metadata = 8 * (len(cstate_indices) + len(values)) + 32
    total = cstate_payload + detailed_payload + metadata
    return ProxyResult(
        method="statekv_feature_proxy",
        reproduction_tier="paper_mechanism_feature_proxy",
        evidence_vectors=values,
        evidence_indices=tuple(range(len(values))),
        accounting=MemoryAccounting(
            active_state_bytes=cstate_payload,
            archive_bytes=0,
            detailed_decode_bytes=detailed_payload,
            metadata_bytes=metadata,
            shared_parameter_bytes=0,
            total_retained_bytes=total,
            active_state_bounded=True,
            total_state_bounded=False,
        ),
        write_seconds=write_seconds,
        read_seconds=0.0,
        estimated_write_flops=write_flops,
        estimated_read_flops=0,
        query_conditioned=False,
        diagnostics={
            "cstate_vectors": len(cstate_indices),
            "detailed_decode_vectors": len(values),
            "decode_uses_full_dstate": True,
            "layerwise_kv_attention_represented": False,
        },
    )


_RUNNERS: dict[str, Callable[..., ProxyResult]] = {
    "exact_recent": exact_recent_proxy,
    "causalmem_feature_proxy": causalmem_feature_proxy,
    "streamingtom_feature_proxy": streamingtom_feature_proxy,
    "stc_feature_proxy": stc_feature_proxy,
    "selectstream_feature_proxy": selectstream_feature_proxy,
    "oasis_feature_proxy": oasis_feature_proxy,
    "statekv_feature_proxy": statekv_feature_proxy,
}


def run_proxy(
    method: str,
    vectors: np.ndarray,
    query_vector: np.ndarray,
    **kwargs: object,
) -> ProxyResult:
    try:
        runner = _RUNNERS[method]
    except KeyError as exc:
        raise ValueError(f"unknown streaming baseline proxy: {method}") from exc
    return runner(vectors, query_vector, **kwargs)

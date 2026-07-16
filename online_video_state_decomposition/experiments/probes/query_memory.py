from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from task_memory import normalize_rows


POLICIES = (
    "exact_recent",
    "offline_uniform",
    "recent_pool_query_topk",
    "recent_pool_query_mmr",
    "reservoir_recent_query_mmr",
    "diverse_recent_query_topk",
    "diverse_recent_query_mmr",
    "calibrated_diverse_recent_query_mmr",
    "learned_recent_query_topk",
    "offline_full_query_mmr",
)

DEFAULT_POLICIES = tuple(
    policy for policy in POLICIES if policy != "learned_recent_query_topk"
)

LEARNED_FEATURE_NAMES = (
    "question_relevance",
    "option_contrast",
    "recency",
    "novelty",
)


@dataclass(frozen=True)
class QueryPolicyResult:
    policy: str
    pool_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    persistent_vectors: int
    payload_bytes: int
    metadata_bytes: int
    total_state_bytes: int
    estimated_retrieval_flops: int
    ranker_parameter_bytes: int
    online_bounded: bool
    query_conditioned: bool
    option_aware: bool
    write_policy: str
    read_policy: str


@dataclass(frozen=True)
class LearnedFeatureRanker:
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    coefficients: tuple[float, ...]
    ridge: float
    training_frames: int

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> LearnedFeatureRanker:
        names = tuple(str(value) for value in payload["feature_names"])
        if names != LEARNED_FEATURE_NAMES:
            raise ValueError(
                f"unexpected learned feature order: {names}"
            )
        ranker = cls(
            feature_mean=tuple(
                float(value) for value in payload["feature_mean"]
            ),
            feature_scale=tuple(
                float(value) for value in payload["feature_scale"]
            ),
            coefficients=tuple(
                float(value) for value in payload["coefficients"]
            ),
            ridge=float(payload["ridge"]),
            training_frames=int(payload["training_frames"]),
        )
        ranker._validate()
        return ranker

    def _validate(self) -> None:
        expected = len(LEARNED_FEATURE_NAMES)
        if not (
            len(self.feature_mean)
            == len(self.feature_scale)
            == len(self.coefficients)
            == expected
        ):
            raise ValueError("learned ranker dimensions do not match features")
        if any(value <= 0.0 for value in self.feature_scale):
            raise ValueError("learned ranker feature scales must be positive")

    def score(self, features: np.ndarray) -> np.ndarray:
        self._validate()
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(
            LEARNED_FEATURE_NAMES
        ):
            raise ValueError("feature matrix has the wrong shape")
        normalized = (
            values - np.asarray(self.feature_mean)
        ) / np.asarray(self.feature_scale)
        return normalized @ np.asarray(self.coefficients)

    def to_dict(self) -> dict[str, object]:
        return {
            "format_version": 1,
            "feature_names": list(LEARNED_FEATURE_NAMES),
            "feature_mean": list(self.feature_mean),
            "feature_scale": list(self.feature_scale),
            "coefficients": list(self.coefficients),
            "ridge": self.ridge,
            "training_frames": self.training_frames,
            "parameter_bytes_fp32": (
                3 * len(LEARNED_FEATURE_NAMES) * 4
            ),
        }


def uniform_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    retained = min(total, count)
    return [
        min(
            int(((2 * index + 1) * total) // (2 * retained)),
            total - 1,
        )
        for index in range(retained)
    ]


def recent_positions(total: int, count: int) -> list[int]:
    if total <= 0 or count <= 0:
        return []
    retained = min(total, count)
    return list(range(total - retained, total))


def frame_rank_features(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    indices: list[int],
) -> np.ndarray:
    total, hidden_dim = vectors.shape
    candidate_vectors = np.asarray(candidate_vectors, dtype=np.float64)
    if candidate_vectors.ndim != 2 or not len(candidate_vectors):
        raise ValueError("candidate_vectors must be a non-empty matrix")
    if candidate_vectors.shape[1] != hidden_dim:
        raise ValueError("candidate and image vectors must share hidden_dim")
    if any(index < 0 or index >= total for index in indices):
        raise ValueError("feature indices are outside the sampled stream")
    normalized_vectors = normalize_rows(vectors)
    normalized_query = normalize_rows(np.asarray(query_vector))[0]
    normalized_candidates = normalize_rows(candidate_vectors)
    selected_vectors = normalized_vectors[indices]
    question_relevance = selected_vectors @ normalized_query
    option_similarity = selected_vectors @ normalized_candidates.T
    option_contrast = (
        np.max(option_similarity, axis=1)
        - np.mean(option_similarity, axis=1)
    )
    time_scale = max(total - 1, 1)
    recency = np.asarray(indices, dtype=np.float64) / time_scale
    novelty = np.asarray(
        [
            0.0
            if index == 0
            else 1.0
            - float(normalized_vectors[index] @ normalized_vectors[index - 1])
            for index in indices
        ],
        dtype=np.float64,
    )
    return np.column_stack(
        (question_relevance, option_contrast, recency, novelty)
    )


def frame_utility_targets(
    vectors: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    answer_index: int,
    indices: list[int],
) -> np.ndarray:
    normalized_vectors = normalize_rows(vectors)
    normalized_candidates = normalize_rows(candidate_vectors)
    if not 0 <= answer_index < len(normalized_candidates):
        raise ValueError("answer index is outside candidate vectors")
    if len(normalized_candidates) < 2:
        raise ValueError("utility target requires at least two candidates")
    similarity = normalized_vectors[indices] @ normalized_candidates.T
    incorrect = np.delete(similarity, answer_index, axis=1)
    return similarity[:, answer_index] - np.max(incorrect, axis=1)


def fit_learned_feature_ranker(
    feature_blocks: list[np.ndarray],
    target_blocks: list[np.ndarray],
    *,
    ridge: float,
) -> LearnedFeatureRanker:
    if ridge < 0.0:
        raise ValueError("ridge must be non-negative")
    if not feature_blocks or len(feature_blocks) != len(target_blocks):
        raise ValueError("feature and target blocks must be non-empty")
    features = np.concatenate(feature_blocks, axis=0).astype(np.float64)
    targets = np.concatenate(target_blocks, axis=0).astype(np.float64)
    if features.shape != (len(targets), len(LEARNED_FEATURE_NAMES)):
        raise ValueError("training features and targets have wrong shapes")
    feature_mean = np.mean(features, axis=0)
    feature_scale = np.std(features, axis=0)
    feature_scale = np.maximum(feature_scale, 1e-6)
    normalized = (features - feature_mean) / feature_scale
    centered_targets = targets - float(np.mean(targets))
    gram = normalized.T @ normalized
    regularized = gram + ridge * np.eye(gram.shape[0])
    coefficients = np.linalg.solve(
        regularized,
        normalized.T @ centered_targets,
    )
    return LearnedFeatureRanker(
        feature_mean=tuple(float(value) for value in feature_mean),
        feature_scale=tuple(float(value) for value in feature_scale),
        coefficients=tuple(float(value) for value in coefficients),
        ridge=ridge,
        training_frames=len(targets),
    )


def reservoir_recent_pool_indices(
    total: int,
    *,
    capacity: int,
    recent_capacity: int,
    seed: int,
) -> list[int]:
    if capacity <= 0:
        return []
    capacity = min(capacity, total)
    recent_capacity = min(max(recent_capacity, 0), capacity)
    history_capacity = capacity - recent_capacity
    archive: list[int] = []
    recent: list[int] = []
    seen_history = 0
    rng = np.random.default_rng(seed)
    for index in range(total):
        recent.append(index)
        if len(recent) <= recent_capacity:
            continue
        expired = recent.pop(0)
        if history_capacity == 0:
            continue
        seen_history += 1
        if len(archive) < history_capacity:
            archive.append(expired)
            continue
        replacement = int(rng.integers(0, seen_history))
        if replacement < history_capacity:
            archive[replacement] = expired
    return sorted(set(archive + recent))


def _update_diverse_archive(
    archive: list[int],
    candidate: int,
    vectors: np.ndarray,
    capacity: int,
) -> list[int]:
    if capacity <= 0:
        return []
    if len(archive) < capacity:
        return archive + [candidate]
    combined = archive + [candidate]
    selected = normalize_rows(vectors[combined])
    similarity = selected @ selected.T
    np.fill_diagonal(similarity, -np.inf)
    redundancy = np.max(similarity, axis=1)
    remove_index = int(np.argmax(redundancy))
    return [
        value
        for position, value in enumerate(combined)
        if position != remove_index
    ]


def diverse_recent_pool_indices(
    vectors: np.ndarray,
    *,
    capacity: int,
    recent_capacity: int,
) -> list[int]:
    total = int(len(vectors))
    if capacity <= 0:
        return []
    capacity = min(capacity, total)
    recent_capacity = min(max(recent_capacity, 0), capacity)
    history_capacity = capacity - recent_capacity
    archive: list[int] = []
    recent: list[int] = []
    for index in range(total):
        recent.append(index)
        if len(recent) <= recent_capacity:
            continue
        expired = recent.pop(0)
        archive = _update_diverse_archive(
            archive,
            expired,
            vectors,
            history_capacity,
        )
    return sorted(set(archive + recent))


def query_select_indices(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    pool_indices: list[int],
    budget: int,
    recent_anchors: int,
    diversity_weight: float,
    temporal_weight: float,
) -> list[int]:
    normalized_vectors = normalize_rows(vectors)
    query = normalize_rows(np.asarray(query_vector))[0]
    relevance = normalized_vectors @ query
    return query_select_from_relevance(
        vectors,
        relevance,
        pool_indices=pool_indices,
        budget=budget,
        recent_anchors=recent_anchors,
        diversity_weight=diversity_weight,
        temporal_weight=temporal_weight,
    )


def query_select_from_relevance(
    vectors: np.ndarray,
    relevance_scores: np.ndarray,
    *,
    pool_indices: list[int],
    budget: int,
    recent_anchors: int,
    diversity_weight: float,
    temporal_weight: float,
) -> list[int]:
    pool = sorted(set(int(index) for index in pool_indices))
    if budget <= 0 or not pool:
        return []
    if len(pool) <= budget:
        return pool

    normalized_vectors = normalize_rows(vectors)
    relevance = np.asarray(relevance_scores, dtype=np.float64)
    if relevance.shape != (len(vectors),):
        raise ValueError("relevance_scores must have one value per vector")
    anchor_count = min(max(recent_anchors, 0), budget, len(pool))
    selected = pool[-anchor_count:] if anchor_count else []
    remaining = [index for index in pool if index not in selected]
    time_scale = max(len(vectors) - 1, 1)

    while len(selected) < budget and remaining:
        scores = []
        for index in remaining:
            diversity = (
                max(
                    float(
                        normalized_vectors[index]
                        @ normalized_vectors[other]
                    )
                    for other in selected
                )
                if selected
                else 0.0
            )
            temporal_coverage = (
                min(abs(index - other) for other in selected) / time_scale
                if selected
                else 0.0
            )
            score = (
                float(relevance[index])
                - diversity_weight * diversity
                + temporal_weight * temporal_coverage
            )
            scores.append((score, float(relevance[index]), -index, index))
        chosen = max(scores)[-1]
        selected.append(chosen)
        remaining.remove(chosen)
    return sorted(selected)


def _accounting(
    *,
    policy: str,
    pool_indices: list[int],
    selected_indices: list[int],
    hidden_dim: int,
    storage_bits: int,
    online_bounded: bool,
    query_conditioned: bool,
    write_policy: str,
    read_policy: str,
    ranker_parameter_bytes: int = 0,
    extra_metadata_bytes: int = 0,
    extra_retrieval_flops: int = 0,
    option_aware: bool = False,
) -> QueryPolicyResult:
    persistent_vectors = len(pool_indices)
    payload_bytes = persistent_vectors * hidden_dim * storage_bits // 8
    metadata_bytes = persistent_vectors * 4 + 16 + extra_metadata_bytes
    if not query_conditioned:
        retrieval_flops = 0
    elif read_policy.endswith("topk"):
        retrieval_flops = 2 * persistent_vectors * hidden_dim
    else:
        retrieval_flops = (
            2
            * persistent_vectors
            * hidden_dim
            * (1 + len(selected_indices))
        )
    retrieval_flops += extra_retrieval_flops
    return QueryPolicyResult(
        policy=policy,
        pool_indices=tuple(pool_indices),
        selected_indices=tuple(selected_indices),
        persistent_vectors=persistent_vectors,
        payload_bytes=payload_bytes,
        metadata_bytes=metadata_bytes,
        total_state_bytes=payload_bytes + metadata_bytes,
        estimated_retrieval_flops=retrieval_flops,
        ranker_parameter_bytes=ranker_parameter_bytes,
        online_bounded=online_bounded,
        query_conditioned=query_conditioned,
        option_aware=option_aware,
        write_policy=write_policy,
        read_policy=read_policy,
    )


def apply_query_policy(
    policy: str,
    vectors: np.ndarray,
    query_vector: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int,
    diversity_weight: float,
    temporal_weight: float,
    seed: int,
    storage_bits: int = 16,
) -> QueryPolicyResult:
    if policy not in POLICIES:
        raise ValueError(f"unknown query-memory policy: {policy}")
    if policy in {
        "calibrated_diverse_recent_query_mmr",
        "learned_recent_query_topk",
    }:
        raise ValueError(
            "feature policy requires its dedicated apply function"
        )
    total, hidden_dim = vectors.shape
    if evidence_budget <= 0 or pool_capacity <= 0:
        raise ValueError("evidence and pool budgets must be positive")
    if evidence_budget > pool_capacity and policy not in {
        "exact_recent",
        "offline_uniform",
    }:
        raise ValueError("pool capacity must cover the evidence budget")

    if policy == "exact_recent":
        pool = recent_positions(total, evidence_budget)
        selected = pool
        return _accounting(
            policy=policy,
            pool_indices=pool,
            selected_indices=selected,
            hidden_dim=hidden_dim,
            storage_bits=storage_bits,
            online_bounded=True,
            query_conditioned=False,
            write_policy="exact_recent",
            read_policy="identity",
        )
    if policy == "offline_uniform":
        pool = uniform_positions(total, evidence_budget)
        selected = pool
        return _accounting(
            policy=policy,
            pool_indices=pool,
            selected_indices=selected,
            hidden_dim=hidden_dim,
            storage_bits=storage_bits,
            online_bounded=False,
            query_conditioned=False,
            write_policy="known_horizon_uniform",
            read_policy="identity",
        )

    if policy.startswith("recent_pool"):
        pool = recent_positions(total, pool_capacity)
        write_policy = "exact_recent_pool"
    elif policy.startswith("reservoir_recent"):
        pool = reservoir_recent_pool_indices(
            total,
            capacity=pool_capacity,
            recent_capacity=recent_anchors,
            seed=seed,
        )
        write_policy = "reservoir_plus_recent"
    elif policy.startswith("diverse_recent"):
        pool = diverse_recent_pool_indices(
            vectors,
            capacity=pool_capacity,
            recent_capacity=recent_anchors,
        )
        write_policy = "diverse_archive_plus_recent"
    elif policy == "offline_full_query_mmr":
        pool = list(range(total))
        write_policy = "full_history"
    else:
        raise ValueError(f"unsupported query-memory policy: {policy}")

    topk = policy.endswith("query_topk")
    selected = query_select_indices(
        vectors,
        query_vector,
        pool_indices=pool,
        budget=evidence_budget,
        recent_anchors=recent_anchors,
        diversity_weight=0.0 if topk else diversity_weight,
        temporal_weight=0.0 if topk else temporal_weight,
    )
    return _accounting(
        policy=policy,
        pool_indices=pool,
        selected_indices=selected,
        hidden_dim=hidden_dim,
        storage_bits=storage_bits,
        online_bounded=policy != "offline_full_query_mmr",
        query_conditioned=True,
        write_policy=write_policy,
        read_policy="query_topk" if topk else "query_mmr",
    )


def apply_calibrated_feature_policy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int,
    diversity_weight: float,
    temporal_weight: float,
    option_weight: float,
    recency_weight: float,
    novelty_weight: float,
    storage_bits: int = 16,
) -> QueryPolicyResult:
    total, hidden_dim = vectors.shape
    if evidence_budget <= 0 or pool_capacity <= 0:
        raise ValueError("evidence and pool budgets must be positive")
    if evidence_budget > pool_capacity:
        raise ValueError("pool capacity must cover the evidence budget")
    pool = diverse_recent_pool_indices(
        vectors,
        capacity=pool_capacity,
        recent_capacity=recent_anchors,
    )
    features = frame_rank_features(
        vectors,
        query_vector,
        candidate_vectors,
        indices=pool,
    )
    pool_relevance = (
        features[:, 0]
        + option_weight * features[:, 1]
        + recency_weight * features[:, 2]
        + novelty_weight * features[:, 3]
    )
    relevance = np.zeros(total, dtype=np.float64)
    relevance[pool] = pool_relevance
    selected = query_select_from_relevance(
        vectors,
        relevance,
        pool_indices=pool,
        budget=evidence_budget,
        recent_anchors=recent_anchors,
        diversity_weight=diversity_weight,
        temporal_weight=temporal_weight,
    )
    return _accounting(
        policy="calibrated_diverse_recent_query_mmr",
        pool_indices=pool,
        selected_indices=selected,
        hidden_dim=hidden_dim,
        storage_bits=storage_bits,
        online_bounded=True,
        query_conditioned=True,
        ranker_parameter_bytes=3 * 4,
        extra_metadata_bytes=len(pool) * 4,
        extra_retrieval_flops=(
            2 * len(pool) * hidden_dim * len(candidate_vectors)
        ),
        option_aware=True,
        write_policy="diverse_archive_plus_recent",
        read_policy="calibrated_feature_mmr",
    )


def apply_learned_feature_policy(
    vectors: np.ndarray,
    query_vector: np.ndarray,
    candidate_vectors: np.ndarray,
    ranker: LearnedFeatureRanker,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int,
    storage_bits: int = 16,
) -> QueryPolicyResult:
    total, hidden_dim = vectors.shape
    if evidence_budget <= 0 or pool_capacity <= 0:
        raise ValueError("evidence and pool budgets must be positive")
    if evidence_budget > pool_capacity:
        raise ValueError("pool capacity must cover the evidence budget")
    pool = recent_positions(total, pool_capacity)
    features = frame_rank_features(
        vectors,
        query_vector,
        candidate_vectors,
        indices=pool,
    )
    relevance = np.zeros(total, dtype=np.float64)
    relevance[pool] = ranker.score(features)
    selected = query_select_from_relevance(
        vectors,
        relevance,
        pool_indices=pool,
        budget=evidence_budget,
        recent_anchors=recent_anchors,
        diversity_weight=0.0,
        temporal_weight=0.0,
    )
    return _accounting(
        policy="learned_recent_query_topk",
        pool_indices=pool,
        selected_indices=selected,
        hidden_dim=hidden_dim,
        storage_bits=storage_bits,
        online_bounded=True,
        query_conditioned=True,
        ranker_parameter_bytes=(
            3 * len(LEARNED_FEATURE_NAMES) * 4
        ),
        extra_metadata_bytes=len(pool) * 4,
        extra_retrieval_flops=(
            2 * len(pool) * hidden_dim * len(candidate_vectors)
            + 2 * len(pool) * len(LEARNED_FEATURE_NAMES)
        ),
        option_aware=True,
        write_policy="exact_recent_pool",
        read_policy="learned_feature_topk",
    )

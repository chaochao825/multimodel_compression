from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from aggregate_mvbench_tasks import (
    exact_binomial_two_sided,
    wilson_interval,
)
from mvbench_clip_memory import stable_seed
from mvbench_utils import parse_csv_list
from query_memory import (
    DEFAULT_POLICIES,
    POLICIES,
    LearnedFeatureRanker,
    apply_calibrated_feature_policy,
    apply_learned_feature_policy,
    apply_query_policy,
    fit_learned_feature_ranker,
    frame_rank_features,
    frame_utility_targets,
    recent_positions,
)
from task_memory import softmax_pool_scores


POLICY_LABELS = {
    "exact_recent": "exact recent",
    "offline_uniform": "offline uniform",
    "recent_pool_query_topk": "recent pool + top-k",
    "recent_pool_query_mmr": "recent pool + MMR",
    "reservoir_recent_query_mmr": "reservoir + recent + MMR",
    "diverse_recent_query_topk": "diverse + recent + top-k",
    "diverse_recent_query_mmr": "diverse + recent + MMR",
    "calibrated_diverse_recent_query_mmr": (
        "calibrated option-aware MMR"
    ),
    "learned_recent_query_topk": "learned recent-pool top-k",
    "offline_full_query_mmr": "full history + MMR",
}

POLICY_COLORS = {
    "exact_recent": "#264653",
    "offline_uniform": "#e9c46a",
    "recent_pool_query_topk": "#2a9d8f",
    "recent_pool_query_mmr": "#4c78a8",
    "reservoir_recent_query_mmr": "#59a14f",
    "diverse_recent_query_topk": "#f28e2b",
    "diverse_recent_query_mmr": "#e15759",
    "calibrated_diverse_recent_query_mmr": "#b07aa1",
    "learned_recent_query_topk": "#af7aa1",
    "offline_full_query_mmr": "#9c755f",
}

HYPERPARAMETER_KEYS = (
    "diversity_weight",
    "temporal_weight",
    "option_weight",
    "recency_weight",
    "novelty_weight",
)


@dataclass(frozen=True)
class CacheRecord:
    metadata: dict[str, object]
    image_vectors: np.ndarray
    question_vector: np.ndarray
    candidate_vectors: np.ndarray
    frame_indices: np.ndarray


def parse_float_list(value: str) -> list[float]:
    return sorted({float(item) for item in parse_csv_list(value)})


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--split-manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--evidence-budget", type=int, default=8)
    parser.add_argument("--pool-capacity", type=int, default=16)
    parser.add_argument("--recent-anchors", type=int, default=3)
    parser.add_argument("--storage-bits", type=int, default=16)
    parser.add_argument("--pool-temperature", type=float, default=10.0)
    parser.add_argument("--diversity-grid", default="0,0.1,0.25,0.5")
    parser.add_argument("--temporal-grid", default="0,0.1,0.25")
    parser.add_argument("--option-weight-grid", default="0,0.25,0.5,1")
    parser.add_argument("--recency-weight-grid", default="0,0.1,0.25")
    parser.add_argument("--novelty-weight-grid", default="0,0.1,0.25")
    parser.add_argument("--fixed-hyperparameters", type=Path, default=None)
    parser.add_argument("--fixed-ranker", type=Path, default=None)
    parser.add_argument("--learned-ridge", type=float, default=1.0)
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICIES))
    parser.add_argument(
        "--primary-policy",
        default="diverse_recent_query_mmr",
    )
    parser.add_argument("--reference-policy", default="exact_recent")
    parser.add_argument("--analysis-stage", default="preregistered_mvp")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260717)
    return parser.parse_args()


def load_records(cache_dir: Path) -> list[CacheRecord]:
    records: list[CacheRecord] = []
    fingerprints: set[str] = set()
    for metadata_path in sorted(cache_dir.glob("*.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        npz_path = metadata_path.with_suffix(".npz")
        if not npz_path.exists():
            raise FileNotFoundError(f"missing cache array: {npz_path}")
        fingerprints.add(str(metadata["configuration_fingerprint"]))
        with np.load(npz_path) as payload:
            records.append(
                CacheRecord(
                    metadata=metadata,
                    image_vectors=np.asarray(
                        payload["image_vectors"],
                        dtype=np.float64,
                    ),
                    question_vector=np.asarray(
                        payload["question_vector"],
                        dtype=np.float64,
                    ),
                    candidate_vectors=np.asarray(
                        payload["candidate_vectors"],
                        dtype=np.float64,
                    ),
                    frame_indices=np.asarray(
                        payload["frame_indices"],
                        dtype=np.int64,
                    ),
                )
            )
    if not records:
        raise FileNotFoundError(f"no cache records under {cache_dir}")
    if len(fingerprints) != 1:
        raise ValueError(
            f"cache contains multiple configuration fingerprints: "
            f"{sorted(fingerprints)}"
        )
    return records


def expected_membership(
    split_manifest: dict[str, object],
) -> dict[str, str]:
    membership: dict[str, str] = {}
    for split_name in ("calibration", "evaluation"):
        for task in split_manifest["tasks"]:
            for index in split_manifest[split_name][task]:
                sample_id = f"{task}_{int(index):04d}"
                if sample_id in membership:
                    raise ValueError(f"split overlap for {sample_id}")
                membership[sample_id] = split_name
    return membership


def validate_records(
    records: list[CacheRecord],
    split_manifest: dict[str, object],
) -> dict[str, object]:
    membership = expected_membership(split_manifest)
    observed = {
        str(record.metadata["sample_id"]): str(record.metadata["split"])
        for record in records
    }
    duplicate_count = len(records) - len(observed)
    missing = sorted(set(membership) - set(observed))
    unexpected = sorted(set(observed) - set(membership))
    wrong_split = sorted(
        sample_id
        for sample_id in set(membership) & set(observed)
        if membership[sample_id] != observed[sample_id]
    )
    checks = {
        "no_duplicate_cache_records": duplicate_count == 0,
        "all_expected_records_present": not missing,
        "no_unexpected_records": not unexpected,
        "split_labels_match_manifest": not wrong_split,
    }
    return {
        "valid": all(checks.values()),
        "checks": checks,
        "records": len(records),
        "expected_records": len(membership),
        "missing": missing,
        "unexpected": unexpected,
        "wrong_split": wrong_split,
    }


def evaluate_policy(
    record: CacheRecord,
    policy: str,
    *,
    evidence_budget: int,
    pool_capacity: int,
    recent_anchors: int,
    storage_bits: int,
    pool_temperature: float,
    diversity_weight: float,
    temporal_weight: float,
    seed: int,
    option_weight: float = 0.0,
    recency_weight: float = 0.0,
    novelty_weight: float = 0.0,
    learned_ranker: LearnedFeatureRanker | None = None,
) -> dict[str, object]:
    selection_start = time.perf_counter()
    if policy == "learned_recent_query_topk":
        if learned_ranker is None:
            raise ValueError("learned policy requires a frozen ranker")
        result = apply_learned_feature_policy(
            record.image_vectors,
            record.question_vector,
            record.candidate_vectors,
            learned_ranker,
            evidence_budget=evidence_budget,
            pool_capacity=pool_capacity,
            recent_anchors=recent_anchors,
            storage_bits=storage_bits,
        )
    elif policy == "calibrated_diverse_recent_query_mmr":
        result = apply_calibrated_feature_policy(
            record.image_vectors,
            record.question_vector,
            record.candidate_vectors,
            evidence_budget=evidence_budget,
            pool_capacity=pool_capacity,
            recent_anchors=recent_anchors,
            diversity_weight=diversity_weight,
            temporal_weight=temporal_weight,
            option_weight=option_weight,
            recency_weight=recency_weight,
            novelty_weight=novelty_weight,
            storage_bits=storage_bits,
        )
    else:
        result = apply_query_policy(
            policy,
            record.image_vectors,
            record.question_vector,
            evidence_budget=evidence_budget,
            pool_capacity=pool_capacity,
            recent_anchors=recent_anchors,
            diversity_weight=diversity_weight,
            temporal_weight=temporal_weight,
            seed=stable_seed(seed, record.metadata["sample_id"], policy),
            storage_bits=storage_bits,
        )
    selection_seconds = time.perf_counter() - selection_start
    selected_positions = list(result.selected_indices)
    scores = softmax_pool_scores(
        record.candidate_vectors,
        record.image_vectors[selected_positions],
        temperature=pool_temperature,
    )
    predicted = int(np.argmax(scores))
    answer_index = int(record.metadata["answer_index"])
    selected_frame_indices = [
        int(record.frame_indices[index])
        for index in selected_positions
    ]
    pool_frame_indices = [
        int(record.frame_indices[index])
        for index in result.pool_indices
    ]
    return {
        "sample_id": record.metadata["sample_id"],
        "split": record.metadata["split"],
        "task": record.metadata["task"],
        "sample_index": record.metadata["sample_index"],
        "video": record.metadata["video"],
        "question": record.metadata["question"],
        "candidates_json": json.dumps(
            record.metadata["candidates"],
            ensure_ascii=False,
        ),
        "answer": record.metadata["answer"],
        "answer_index": answer_index,
        "policy": policy,
        "predicted_index": predicted,
        "prediction": record.metadata["candidates"][predicted],
        "correct": int(predicted == answer_index),
        "scores_json": json.dumps(
            [float(value) for value in scores],
            separators=(",", ":"),
        ),
        "diversity_weight": diversity_weight,
        "temporal_weight": temporal_weight,
        "option_weight": option_weight,
        "recency_weight": recency_weight,
        "novelty_weight": novelty_weight,
        "evidence_budget": len(selected_positions),
        "pool_capacity_config": pool_capacity,
        "recent_anchors": recent_anchors,
        "pool_stream_positions_json": json.dumps(
            list(result.pool_indices)
        ),
        "selected_stream_positions_json": json.dumps(
            selected_positions
        ),
        "pool_frame_indices_json": json.dumps(pool_frame_indices),
        "selected_frame_indices_json": json.dumps(
            selected_frame_indices
        ),
        "sampled_frames": len(record.image_vectors),
        "total_frames": record.metadata["total_frames"],
        "persistent_vectors": result.persistent_vectors,
        "payload_bytes": result.payload_bytes,
        "metadata_bytes": result.metadata_bytes,
        "total_state_bytes": result.total_state_bytes,
        "estimated_retrieval_flops": result.estimated_retrieval_flops,
        "ranker_parameter_bytes": result.ranker_parameter_bytes,
        "selection_seconds": selection_seconds,
        "online_bounded": int(result.online_bounded),
        "query_conditioned": int(result.query_conditioned),
        "option_aware": int(result.option_aware),
        "write_policy": result.write_policy,
        "read_policy": result.read_policy,
        "visual_evidence_cache_counted": 0,
    }


def accuracy_summary(rows: list[dict[str, object]]) -> dict[str, float]:
    by_task: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_task[str(row["task"])].append(int(row["correct"]))
    return {
        "micro_accuracy": float(
            np.mean([int(row["correct"]) for row in rows])
        ),
        "macro_task_accuracy": float(
            np.mean(
                [
                    np.mean(values)
                    for _, values in sorted(by_task.items())
                ]
            )
        ),
    }


def calibrate(
    records: list[CacheRecord],
    *,
    diversity_grid: list[float],
    temporal_grid: list[float],
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    calibration_records = [
        record
        for record in records
        if record.metadata["split"] == "calibration"
    ]
    rows = []
    for diversity_weight in diversity_grid:
        for temporal_weight in temporal_grid:
            predictions = [
                evaluate_policy(
                    record,
                    "diverse_recent_query_mmr",
                    evidence_budget=args.evidence_budget,
                    pool_capacity=args.pool_capacity,
                    recent_anchors=args.recent_anchors,
                    storage_bits=args.storage_bits,
                    pool_temperature=args.pool_temperature,
                    diversity_weight=diversity_weight,
                    temporal_weight=temporal_weight,
                    seed=args.seed,
                )
                for record in calibration_records
            ]
            summary = accuracy_summary(predictions)
            rows.append(
                {
                    "diversity_weight": diversity_weight,
                    "temporal_weight": temporal_weight,
                    "samples": len(predictions),
                    **summary,
                }
            )
    best = max(
        rows,
        key=lambda row: (
            float(row["macro_task_accuracy"]),
            float(row["micro_accuracy"]),
            -float(row["diversity_weight"])
            - float(row["temporal_weight"]),
            -float(row["diversity_weight"]),
            -float(row["temporal_weight"]),
        ),
    )
    return {
        "diversity_weight": float(best["diversity_weight"]),
        "temporal_weight": float(best["temporal_weight"]),
    }, rows


def calibrate_feature_ranker(
    records: list[CacheRecord],
    *,
    option_weight_grid: list[float],
    recency_weight_grid: list[float],
    novelty_weight_grid: list[float],
    diversity_weight: float,
    temporal_weight: float,
    args: argparse.Namespace,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    calibration_records = [
        record
        for record in records
        if record.metadata["split"] == "calibration"
    ]
    rows = []
    for option_weight in option_weight_grid:
        for recency_weight in recency_weight_grid:
            for novelty_weight in novelty_weight_grid:
                predictions = [
                    evaluate_policy(
                        record,
                        "calibrated_diverse_recent_query_mmr",
                        evidence_budget=args.evidence_budget,
                        pool_capacity=args.pool_capacity,
                        recent_anchors=args.recent_anchors,
                        storage_bits=args.storage_bits,
                        pool_temperature=args.pool_temperature,
                        diversity_weight=diversity_weight,
                        temporal_weight=temporal_weight,
                        option_weight=option_weight,
                        recency_weight=recency_weight,
                        novelty_weight=novelty_weight,
                        seed=args.seed,
                    )
                    for record in calibration_records
                ]
                summary = accuracy_summary(predictions)
                rows.append(
                    {
                        "option_weight": option_weight,
                        "recency_weight": recency_weight,
                        "novelty_weight": novelty_weight,
                        "diversity_weight": diversity_weight,
                        "temporal_weight": temporal_weight,
                        "samples": len(predictions),
                        **summary,
                    }
                )
    best = max(
        rows,
        key=lambda row: (
            float(row["macro_task_accuracy"]),
            float(row["micro_accuracy"]),
            -float(row["option_weight"])
            - float(row["recency_weight"])
            - float(row["novelty_weight"]),
            -float(row["option_weight"]),
            -float(row["recency_weight"]),
            -float(row["novelty_weight"]),
        ),
    )
    return {
        "option_weight": float(best["option_weight"]),
        "recency_weight": float(best["recency_weight"]),
        "novelty_weight": float(best["novelty_weight"]),
    }, rows


def fit_supervised_feature_ranker(
    records: list[CacheRecord],
    *,
    pool_capacity: int,
    ridge: float,
) -> tuple[LearnedFeatureRanker, dict[str, object]]:
    calibration_records = [
        record
        for record in records
        if record.metadata["split"] == "calibration"
    ]
    if not calibration_records:
        raise ValueError(
            "learned ranker fitting requires calibration records"
        )
    feature_blocks = []
    target_blocks = []
    for record in calibration_records:
        pool = recent_positions(
            len(record.image_vectors),
            pool_capacity,
        )
        feature_blocks.append(
            frame_rank_features(
                record.image_vectors,
                record.question_vector,
                record.candidate_vectors,
                indices=pool,
            )
        )
        target_blocks.append(
            frame_utility_targets(
                record.image_vectors,
                record.candidate_vectors,
                answer_index=int(record.metadata["answer_index"]),
                indices=pool,
            )
        )
    ranker = fit_learned_feature_ranker(
        feature_blocks,
        target_blocks,
        ridge=ridge,
    )
    payload = {
        **ranker.to_dict(),
        "training_records": len(calibration_records),
        "pool_capacity": pool_capacity,
        "target": (
            "correct_option_similarity_minus_best_incorrect_similarity"
        ),
        "evaluation_answer_labels_used": False,
    }
    return ranker, payload


def task_summary(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = (
        defaultdict(list)
    )
    for row in rows:
        grouped[
            (
                str(row["split"]),
                str(row["task"]),
                str(row["policy"]),
            )
        ].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        correct = sum(int(row["correct"]) for row in values)
        low, high = wilson_interval(correct, len(values))
        output.append(
            {
                "split": key[0],
                "task": key[1],
                "policy": key[2],
                "samples": len(values),
                "correct": correct,
                "accuracy": correct / len(values),
                "accuracy_ci95_low": low,
                "accuracy_ci95_high": high,
                "total_state_bytes": int(values[0]["total_state_bytes"]),
                "persistent_vectors": int(values[0]["persistent_vectors"]),
                "online_bounded": int(values[0]["online_bounded"]),
                "ranker_parameter_bytes": int(
                    values[0]["ranker_parameter_bytes"]
                ),
                "estimated_retrieval_flops": int(
                    values[0]["estimated_retrieval_flops"]
                ),
                "query_conditioned": int(
                    values[0]["query_conditioned"]
                ),
                "option_aware": int(values[0]["option_aware"]),
                "mean_selection_seconds": float(
                    np.mean(
                        [
                            float(row["selection_seconds"])
                            for row in values
                        ]
                    )
                ),
            }
        )
    return output


def overall_summary(
    task_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(
        list
    )
    for row in task_rows:
        grouped[(str(row["split"]), str(row["policy"]))].append(row)
    output = []
    for key in sorted(grouped):
        values = grouped[key]
        samples = sum(int(row["samples"]) for row in values)
        correct = sum(int(row["correct"]) for row in values)
        low, high = wilson_interval(correct, samples)
        output.append(
            {
                "split": key[0],
                "policy": key[1],
                "tasks": len(values),
                "samples": samples,
                "micro_accuracy": correct / samples,
                "micro_accuracy_ci95_low": low,
                "micro_accuracy_ci95_high": high,
                "macro_task_accuracy": float(
                    np.mean([float(row["accuracy"]) for row in values])
                ),
                "total_state_bytes": int(values[0]["total_state_bytes"]),
                "persistent_vectors": int(values[0]["persistent_vectors"]),
                "online_bounded": int(values[0]["online_bounded"]),
                "ranker_parameter_bytes": int(
                    values[0]["ranker_parameter_bytes"]
                ),
                "estimated_retrieval_flops": int(
                    values[0]["estimated_retrieval_flops"]
                ),
                "query_conditioned": int(
                    values[0]["query_conditioned"]
                ),
                "option_aware": int(values[0]["option_aware"]),
                "mean_selection_seconds": (
                    sum(
                        float(row["mean_selection_seconds"])
                        * int(row["samples"])
                        for row in values
                    )
                    / samples
                ),
            }
        )
    references = {
        str(row["split"]): float(row["macro_task_accuracy"])
        for row in output
        if row["policy"] == "exact_recent"
    }
    for row in output:
        row["macro_gain_vs_exact_recent"] = (
            float(row["macro_task_accuracy"])
            - references[str(row["split"])]
        )
    return output


def paired_comparisons(
    rows: list[dict[str, object]],
    *,
    reference: str,
    seed: int,
    bootstrap_samples: int,
) -> list[dict[str, object]]:
    evaluation = [
        row for row in rows if row["split"] == "evaluation"
    ]
    lookup = {
        (str(row["sample_id"]), str(row["policy"])): int(row["correct"])
        for row in evaluation
    }
    policies = sorted(
        {
            str(row["policy"])
            for row in evaluation
            if row["policy"] != reference
        }
    )
    rng = np.random.default_rng(seed)
    output = []
    for policy in policies:
        sample_ids = sorted(
            sample_id
            for sample_id, candidate_policy in lookup
            if candidate_policy == policy
            and (sample_id, reference) in lookup
        )
        differences = np.asarray(
            [
                lookup[(sample_id, policy)]
                - lookup[(sample_id, reference)]
                for sample_id in sample_ids
            ],
            dtype=np.float64,
        )
        resampled = rng.choice(
            differences,
            size=(bootstrap_samples, len(differences)),
            replace=True,
        ).mean(axis=1)
        low, high = np.quantile(resampled, [0.025, 0.975])
        better = int(np.sum(differences > 0))
        worse = int(np.sum(differences < 0))
        output.append(
            {
                "policy": policy,
                "reference": reference,
                "paired_samples": len(differences),
                "accuracy_gain": float(np.mean(differences)),
                "bootstrap_ci95_low": float(low),
                "bootstrap_ci95_high": float(high),
                "better_samples": better,
                "worse_samples": worse,
                "tied_samples": int(np.sum(differences == 0)),
                "mcnemar_exact_p": exact_binomial_two_sided(
                    min(better, worse),
                    better + worse,
                ),
            }
        )
    return output


def calibration_transfer(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    lookup = {
        (str(row["split"]), str(row["policy"])): row
        for row in rows
    }
    policies = sorted(
        {
            policy
            for split, policy in lookup
            if split == "calibration"
            and ("evaluation", policy) in lookup
        }
    )
    return [
        {
            "policy": policy,
            "calibration_macro_accuracy": float(
                lookup[("calibration", policy)]["macro_task_accuracy"]
            ),
            "evaluation_macro_accuracy": float(
                lookup[("evaluation", policy)]["macro_task_accuracy"]
            ),
            "evaluation_minus_calibration": float(
                lookup[("evaluation", policy)]["macro_task_accuracy"]
            )
            - float(
                lookup[("calibration", policy)]["macro_task_accuracy"]
            ),
        }
        for policy in policies
    ]


def selection_overlap(
    rows: list[dict[str, object]],
    *,
    reference: str,
) -> list[dict[str, object]]:
    evaluation = [
        row for row in rows if row["split"] == "evaluation"
    ]
    lookup = {
        (str(row["sample_id"]), str(row["policy"])): row
        for row in evaluation
    }
    policies = [
        policy
        for policy in POLICIES
        if policy != reference
        and any(row["policy"] == policy for row in evaluation)
    ]
    output = []
    for policy in policies:
        sample_ids = sorted(
            sample_id
            for sample_id, candidate_policy in lookup
            if candidate_policy == policy
            and (sample_id, reference) in lookup
        )
        jaccards = []
        exact_matches = 0
        for sample_id in sample_ids:
            reference_indices = set(
                json.loads(
                    str(
                        lookup[(sample_id, reference)][
                            "selected_stream_positions_json"
                        ]
                    )
                )
            )
            candidate_indices = set(
                json.loads(
                    str(
                        lookup[(sample_id, policy)][
                            "selected_stream_positions_json"
                        ]
                    )
                )
            )
            union = reference_indices | candidate_indices
            jaccards.append(
                len(reference_indices & candidate_indices) / len(union)
                if union
                else 1.0
            )
            exact_matches += int(reference_indices == candidate_indices)
        output.append(
            {
                "policy": policy,
                "reference": reference,
                "samples": len(sample_ids),
                "mean_jaccard": float(np.mean(jaccards)),
                "exact_selection_rate": exact_matches / len(sample_ids),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot_calibration(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    diversity = sorted({float(row["diversity_weight"]) for row in rows})
    temporal = sorted({float(row["temporal_weight"]) for row in rows})
    lookup = {
        (
            float(row["diversity_weight"]),
            float(row["temporal_weight"]),
        ): float(row["macro_task_accuracy"])
        for row in rows
    }
    matrix = np.asarray(
        [
            [lookup[(left, right)] for right in temporal]
            for left in diversity
        ]
    )
    margin = max(0.01, 0.1 * float(np.ptp(matrix)))
    vmin = max(0.0, float(np.min(matrix)) - margin)
    vmax = min(1.0, float(np.max(matrix)) + margin)
    fig, axis = plt.subplots(figsize=(5.8, 4.5))
    image = axis.imshow(matrix, vmin=vmin, vmax=vmax, cmap="YlGnBu")
    axis.set_xticks(range(len(temporal)), temporal)
    axis.set_yticks(range(len(diversity)), diversity)
    axis.set_xlabel("Temporal coverage weight")
    axis.set_ylabel("Diversity weight")
    axis.set_title("Calibration macro accuracy")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.3f}",
                ha="center",
                va="center",
                fontsize=9,
            )
    fig.colorbar(image, ax=axis, label="Macro task accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_feature_calibration(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    option_weights = sorted(
        {float(row["option_weight"]) for row in rows}
    )
    recency_weights = sorted(
        {float(row["recency_weight"]) for row in rows}
    )
    novelty_weights = sorted(
        {float(row["novelty_weight"]) for row in rows}
    )
    lookup = {
        (
            float(row["option_weight"]),
            float(row["recency_weight"]),
            float(row["novelty_weight"]),
        ): float(row["macro_task_accuracy"])
        for row in rows
    }
    values = np.asarray(list(lookup.values()), dtype=np.float64)
    margin = max(0.01, 0.1 * float(np.ptp(values)))
    vmin = max(0.0, float(np.min(values)) - margin)
    vmax = min(1.0, float(np.max(values)) + margin)
    columns = min(2, len(option_weights))
    row_count = math.ceil(len(option_weights) / columns)
    fig, axes = plt.subplots(
        row_count,
        columns,
        figsize=(5.4 * columns, 4.1 * row_count),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for panel, option_weight in enumerate(option_weights):
        axis = axes.flat[panel]
        matrix = np.asarray(
            [
                [
                    lookup[(option_weight, recency, novelty)]
                    for novelty in novelty_weights
                ]
                for recency in recency_weights
            ]
        )
        image = axis.imshow(
            matrix,
            vmin=vmin,
            vmax=vmax,
            cmap="YlGnBu",
        )
        axis.set_xticks(range(len(novelty_weights)), novelty_weights)
        axis.set_yticks(range(len(recency_weights)), recency_weights)
        axis.set_xlabel("Novelty weight")
        axis.set_ylabel("Recency weight")
        axis.set_title(f"Option weight = {option_weight:g}")
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    f"{matrix[row_index, column_index]:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    for panel in range(len(option_weights), axes.size):
        axes.flat[panel].axis("off")
    if image is not None:
        fig.colorbar(
            image,
            ax=axes.ravel().tolist(),
            label="Macro task accuracy",
            shrink=0.78,
            pad=0.02,
        )
    fig.suptitle("Calibration-only option-aware feature ranker")
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_calibration_transfer(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    values = [
        float(row[key])
        for row in rows
        for key in (
            "calibration_macro_accuracy",
            "evaluation_macro_accuracy",
        )
    ]
    lower = max(0.0, min(values) - 0.03)
    upper = min(1.0, max(values) + 0.03)
    fig, axis = plt.subplots(figsize=(7.8, 5.2))
    for row in rows:
        policy = str(row["policy"])
        axis.plot(
            [0, 1],
            [
                float(row["calibration_macro_accuracy"]),
                float(row["evaluation_macro_accuracy"]),
            ],
            marker="o",
            linewidth=2,
            color=POLICY_COLORS.get(policy, "#457b9d"),
            label=POLICY_LABELS.get(policy, policy),
        )
    axis.set_xlim(-0.08, 1.08)
    axis.set_ylim(lower, upper)
    axis.set_xticks(
        [0, 1],
        ["Calibration (100 samples)", "Evaluation (200 samples)"],
    )
    axis.set_ylabel("Macro task accuracy")
    axis.set_title("Calibration-to-evaluation method transfer")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_selection_overlap(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    x = np.arange(len(rows))
    width = 0.38
    jaccard = [float(row["mean_jaccard"]) for row in rows]
    exact = [float(row["exact_selection_rate"]) for row in rows]
    labels = [
        POLICY_LABELS.get(str(row["policy"]), str(row["policy"]))
        for row in rows
    ]
    fig, axis = plt.subplots(figsize=(1.05 * len(rows) + 3.6, 4.8))
    axis.bar(
        x - width / 2,
        jaccard,
        width,
        label="Mean Jaccard",
        color="#2a9d8f",
    )
    axis.bar(
        x + width / 2,
        exact,
        width,
        label="Exact selection rate",
        color="#e9c46a",
    )
    axis.set_xticks(x, labels, rotation=28, ha="right")
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Overlap with exact recent")
    axis.set_title("How much each policy changes the selected evidence")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_accuracy_state(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["split"] == "evaluation"]
    accuracies = [
        float(row["macro_task_accuracy"])
        for row in selected
    ]
    grouped: dict[int, list[dict[str, object]]] = defaultdict(list)
    for row in selected:
        grouped[int(row["total_state_bytes"])].append(row)
    plot_x: dict[str, float] = {}
    for state_bytes, values in grouped.items():
        ordered = sorted(values, key=lambda row: str(row["policy"]))
        for index, row in enumerate(ordered):
            log_offset = (index - (len(ordered) - 1) / 2) * 0.035
            policy = str(row["policy"])
            plot_x[policy] = state_bytes / 1024 * (2**log_offset)
    lower = max(0.0, min(accuracies) - 0.05)
    upper = min(1.0, max(accuracies) + 0.05)
    if upper - lower < 0.12:
        center = 0.5 * (lower + upper)
        lower = max(0.0, center - 0.06)
        upper = min(1.0, center + 0.06)
    fig, axis = plt.subplots(figsize=(7.4, 5.0))
    for row in selected:
        policy = str(row["policy"])
        x = plot_x[policy]
        y = float(row["macro_task_accuracy"])
        marker = (
            "D"
            if int(row["option_aware"])
            else "X"
            if not int(row["online_bounded"])
            else "o"
        )
        axis.scatter(
            x,
            y,
            s=90,
            color=POLICY_COLORS.get(policy, "#457b9d"),
            marker=marker,
            label=POLICY_LABELS.get(policy, policy),
        )
    axis.set_xscale("log", base=2)
    axis.set_ylim(lower, upper)
    axis.set_xlabel("Persistent selection state (KiB, log2)")
    axis.set_ylabel("Macro task accuracy")
    axis.set_title("Query-memory accuracy vs persistent state")
    axis.grid(alpha=0.25)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.17),
        ncol=3,
        frameon=False,
        fontsize=7.2,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(
        path.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def plot_task_heatmap(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row["split"] == "evaluation"]
    tasks = sorted({str(row["task"]) for row in selected})
    policies = [
        policy
        for policy in POLICIES
        if any(row["policy"] == policy for row in selected)
    ]
    lookup = {
        (str(row["task"]), str(row["policy"])): float(row["accuracy"])
        for row in selected
    }
    matrix = np.asarray(
        [
            [lookup[(task, policy)] for policy in policies]
            for task in tasks
        ]
    )
    fig, axis = plt.subplots(
        figsize=(1.35 * len(policies) + 2.8, 0.65 * len(tasks) + 2.2)
    )
    image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    axis.set_xticks(
        range(len(policies)),
        [POLICY_LABELS.get(policy, policy) for policy in policies],
        rotation=30,
        ha="right",
    )
    axis.set_yticks(range(len(tasks)), tasks)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.colorbar(image, ax=axis, label="Accuracy")
    axis.set_title("Evaluation task accuracy by selection policy")
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_paired(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = [
        POLICY_LABELS.get(str(row["policy"]), str(row["policy"]))
        for row in rows
    ]
    values = np.asarray([float(row["accuracy_gain"]) for row in rows])
    low = np.asarray([float(row["bootstrap_ci95_low"]) for row in rows])
    high = np.asarray([float(row["bootstrap_ci95_high"]) for row in rows])
    x = np.arange(len(rows))
    colors = ["#2a9d8f" if value >= 0 else "#e76f51" for value in values]
    fig, axis = plt.subplots(figsize=(1.15 * len(rows) + 3.0, 4.6))
    axis.bar(x, values, color=colors)
    axis.errorbar(
        x,
        values,
        yerr=np.vstack((values - low, high - values)),
        fmt="none",
        ecolor="#343a40",
        capsize=4,
    )
    axis.axhline(0.0, color="#343a40", linewidth=1)
    axis.set_xticks(x, labels, rotation=30, ha="right")
    axis.set_ylabel("Paired accuracy gain vs exact recent")
    axis.set_title("Evaluation paired comparison")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def plot_temporal_selection(
    rows: list[dict[str, object]],
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    selected_policies = (
        "exact_recent",
        "reservoir_recent_query_mmr",
        "diverse_recent_query_mmr",
        "calibrated_diverse_recent_query_mmr",
        "learned_recent_query_topk",
        "offline_full_query_mmr",
    )
    fig, axis = plt.subplots(figsize=(7.0, 4.7))
    bins = np.linspace(0.0, 1.0, 12)
    for policy in selected_policies:
        positions = []
        for row in rows:
            if row["split"] != "evaluation" or row["policy"] != policy:
                continue
            total = max(int(row["sampled_frames"]) - 1, 1)
            positions.extend(
                value / total
                for value in json.loads(
                    str(row["selected_stream_positions_json"])
                )
            )
        if positions:
            axis.hist(
                positions,
                bins=bins,
                density=True,
                histtype="step",
                linewidth=2,
                label=POLICY_LABELS.get(policy, policy),
            )
    axis.set_xlabel("Normalized position in sampled stream")
    axis.set_ylabel("Selection density")
    axis.set_title("Temporal distribution of selected evidence")
    axis.grid(alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=220)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def selection_manifest(
    rows: list[dict[str, object]],
    *,
    hyperparameters: dict[str, float],
) -> dict[str, object]:
    samples: dict[str, dict[str, object]] = {}
    policy_accounting: dict[str, dict[str, object]] = {}
    for row in rows:
        if row["split"] != "evaluation":
            continue
        policy_accounting.setdefault(
            str(row["policy"]),
            {
                "persistent_vectors": int(row["persistent_vectors"]),
                "total_state_bytes": int(row["total_state_bytes"]),
                "estimated_retrieval_flops": int(
                    row["estimated_retrieval_flops"]
                ),
                "online_bounded": int(row["online_bounded"]),
                "query_conditioned": int(row["query_conditioned"]),
                "option_aware": int(row["option_aware"]),
                "ranker_parameter_bytes": int(
                    row["ranker_parameter_bytes"]
                ),
                "write_policy": row["write_policy"],
                "read_policy": row["read_policy"],
                "visual_evidence_cache_counted": 0,
            },
        )
        sample_id = str(row["sample_id"])
        sample = samples.setdefault(
            sample_id,
            {
                "task": row["task"],
                "sample_index": row["sample_index"],
                "video": row["video"],
                "policies": {},
            },
        )
        sample["policies"][str(row["policy"])] = json.loads(
            str(row["selected_frame_indices_json"])
        )
    return {
        "format_version": 2,
        "split": "evaluation",
        "hyperparameters": hyperparameters,
        "policy_accounting": policy_accounting,
        "samples": samples,
    }


def promotion_decision(
    task_rows: list[dict[str, object]],
    paired_rows: list[dict[str, object]],
    validation: dict[str, object],
    *,
    primary: str,
    reference: str,
) -> dict[str, object]:
    paired = next(row for row in paired_rows if row["policy"] == primary)
    evaluation_tasks = [
        row
        for row in task_rows
        if row["split"] == "evaluation"
    ]
    reference_accuracy = {
        str(row["task"]): float(row["accuracy"])
        for row in evaluation_tasks
        if row["policy"] == reference
    }
    candidate = {
        str(row["task"]): float(row["accuracy"])
        for row in evaluation_tasks
        if row["policy"] == primary
    }
    task_gains = {
        task: candidate[task] - reference_accuracy[task]
        for task in sorted(reference_accuracy)
    }
    non_worse_tasks = sum(gain >= 0.0 for gain in task_gains.values())
    strong_gain = float(paired["accuracy_gain"]) >= 0.03
    trend_gate = (
        float(paired["accuracy_gain"]) > 0.0
        and non_worse_tasks >= 3
        and float(paired["bootstrap_ci95_low"]) > -0.02
    )
    leakage_checks = bool(validation["valid"])
    decision = {
        "primary_policy": primary,
        "reference_policy": reference,
        "paired": paired,
        "task_gains": task_gains,
        "non_worse_tasks": non_worse_tasks,
        "strong_gain_gate": strong_gain,
        "positive_trend_gate": trend_gate,
        "split_and_cache_validation": leakage_checks,
        "advance_to_llava_anchor": (
            leakage_checks and (strong_gain or trend_gate)
        ),
        "claim_boundary": (
            "The visual evidence cache is not counted in this CLIP proxy. "
            "A positive result promotes only a selection-policy anchor."
        ),
    }
    exploratory = "calibrated_diverse_recent_query_mmr"
    exploratory_paired = next(
        (
            row
            for row in paired_rows
            if row["policy"] == exploratory
        ),
        None,
    )
    if exploratory_paired is not None:
        exploratory_candidate = {
            str(row["task"]): float(row["accuracy"])
            for row in evaluation_tasks
            if row["policy"] == exploratory
        }
        exploratory_gains = {
            task: exploratory_candidate[task] - reference_accuracy[task]
            for task in sorted(reference_accuracy)
        }
        decision["exploratory_option_aware"] = {
            "policy": exploratory,
            "paired": exploratory_paired,
            "task_gains": exploratory_gains,
            "non_worse_tasks": sum(
                gain >= 0.0 for gain in exploratory_gains.values()
            ),
            "claim_boundary": (
                "This secondary policy uses all answer options "
                "symmetrically and is not the preregistered primary."
            ),
        }
    learned = "learned_recent_query_topk"
    learned_paired = next(
        (row for row in paired_rows if row["policy"] == learned),
        None,
    )
    if learned_paired is not None:
        learned_candidate = {
            str(row["task"]): float(row["accuracy"])
            for row in evaluation_tasks
            if row["policy"] == learned
        }
        learned_gains = {
            task: learned_candidate[task] - reference_accuracy[task]
            for task in sorted(reference_accuracy)
        }
        decision["exploratory_learned_readout"] = {
            "policy": learned,
            "paired": learned_paired,
            "task_gains": learned_gains,
            "non_worse_tasks": sum(
                gain >= 0.0 for gain in learned_gains.values()
            ),
            "claim_boundary": (
                "The four-feature ridge readout is trained only on "
                "calibration labels and remains an exploratory baseline."
            ),
        }
    return decision


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = json.loads(
        args.split_manifest.read_text(encoding="utf-8")
    )
    records = load_records(args.cache_dir)
    validation = validate_records(records, split_manifest)
    (args.out_dir / "cache_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not validation["valid"]:
        print(json.dumps(validation, indent=2, sort_keys=True))
        return 1

    policies = parse_csv_list(args.policies)
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise ValueError(f"unknown policies: {unknown}")
    if args.reference_policy not in policies:
        raise ValueError("reference policy must be included in policies")
    if args.primary_policy not in policies:
        raise ValueError("primary policy must be included in policies")
    if args.primary_policy == args.reference_policy:
        raise ValueError("primary and reference policies must differ")

    learned_ranker: LearnedFeatureRanker | None = None
    learned_ranker_source: dict[str, object] | None = None
    if "learned_recent_query_topk" in policies:
        if args.fixed_ranker is not None:
            learned_payload = json.loads(
                args.fixed_ranker.read_text(encoding="utf-8")
            )
            learned_ranker = LearnedFeatureRanker.from_dict(
                learned_payload
            )
            learned_ranker_source = {
                "mode": "fixed_before_evaluation",
                "path": str(args.fixed_ranker),
                "sha256": file_sha256(args.fixed_ranker),
            }
        else:
            learned_ranker, learned_payload = (
                fit_supervised_feature_ranker(
                    records,
                    pool_capacity=args.pool_capacity,
                    ridge=args.learned_ridge,
                )
            )
            learned_ranker_source = {
                "mode": "fit_on_calibration",
                "calibration_records": learned_payload[
                    "training_records"
                ],
            }
        ranker_artifact = {
            **learned_ranker.to_dict(),
            "source": learned_ranker_source,
        }
        if args.fixed_ranker is None:
            ranker_artifact.update(
                {
                    "training_records": learned_payload[
                        "training_records"
                    ],
                    "pool_capacity": learned_payload["pool_capacity"],
                    "target": learned_payload["target"],
                    "evaluation_answer_labels_used": False,
                }
            )
        (args.out_dir / "learned_feature_ranker.json").write_text(
            json.dumps(ranker_artifact, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    calibration_rows: list[dict[str, object]] = []
    feature_calibration_rows: list[dict[str, object]] = []
    if args.fixed_hyperparameters is not None:
        fixed_payload = json.loads(
            args.fixed_hyperparameters.read_text(encoding="utf-8")
        )
        missing = [
            key for key in HYPERPARAMETER_KEYS if key not in fixed_payload
        ]
        if missing:
            raise ValueError(
                f"fixed hyperparameters are missing keys: {missing}"
            )
        hyperparameters = {
            key: float(fixed_payload[key])
            for key in HYPERPARAMETER_KEYS
        }
        hyperparameter_source = {
            "mode": "fixed_before_evaluation",
            "path": str(args.fixed_hyperparameters),
            "sha256": file_sha256(args.fixed_hyperparameters),
        }
    else:
        primary_hyperparameters, calibration_rows = calibrate(
            records,
            diversity_grid=parse_float_list(args.diversity_grid),
            temporal_grid=parse_float_list(args.temporal_grid),
            args=args,
        )
        feature_hyperparameters = {
            "option_weight": 0.0,
            "recency_weight": 0.0,
            "novelty_weight": 0.0,
        }
        if "calibrated_diverse_recent_query_mmr" in policies:
            feature_hyperparameters, feature_calibration_rows = (
                calibrate_feature_ranker(
                    records,
                    option_weight_grid=parse_float_list(
                        args.option_weight_grid
                    ),
                    recency_weight_grid=parse_float_list(
                        args.recency_weight_grid
                    ),
                    novelty_weight_grid=parse_float_list(
                        args.novelty_weight_grid
                    ),
                    diversity_weight=primary_hyperparameters[
                        "diversity_weight"
                    ],
                    temporal_weight=primary_hyperparameters[
                        "temporal_weight"
                    ],
                    args=args,
                )
            )
        hyperparameters = {
            **primary_hyperparameters,
            **feature_hyperparameters,
        }
        hyperparameter_source = {
            "mode": "calibrated_in_run",
            "calibration_records": sum(
                record.metadata["split"] == "calibration"
                for record in records
            ),
        }
    (args.out_dir / "selected_hyperparameters.json").write_text(
        json.dumps(hyperparameters, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    rows = []
    for record in records:
        for policy in policies:
            rows.append(
                evaluate_policy(
                    record,
                    policy,
                    evidence_budget=args.evidence_budget,
                    pool_capacity=args.pool_capacity,
                    recent_anchors=args.recent_anchors,
                    storage_bits=args.storage_bits,
                    pool_temperature=args.pool_temperature,
                    diversity_weight=hyperparameters["diversity_weight"],
                    temporal_weight=hyperparameters["temporal_weight"],
                    option_weight=hyperparameters["option_weight"],
                    recency_weight=hyperparameters["recency_weight"],
                    novelty_weight=hyperparameters["novelty_weight"],
                    learned_ranker=learned_ranker,
                    seed=args.seed,
                )
            )
    task_rows = task_summary(rows)
    overall_rows = overall_summary(task_rows)
    paired_rows = paired_comparisons(
        rows,
        reference=args.reference_policy,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
    )
    transfer_rows = calibration_transfer(overall_rows)
    overlap_rows = selection_overlap(
        rows,
        reference=args.reference_policy,
    )
    write_csv(args.out_dir / "calibration_grid.csv", calibration_rows)
    write_csv(
        args.out_dir / "feature_calibration_grid.csv",
        feature_calibration_rows,
    )
    write_csv(args.out_dir / "predictions.csv", rows)
    write_csv(args.out_dir / "task_accuracy.csv", task_rows)
    write_csv(args.out_dir / "overall_accuracy.csv", overall_rows)
    write_csv(args.out_dir / "paired_vs_exact_recent.csv", paired_rows)
    write_csv(
        args.out_dir / "calibration_to_evaluation.csv",
        transfer_rows,
    )
    write_csv(
        args.out_dir / "selection_overlap_vs_reference.csv",
        overlap_rows,
    )
    if calibration_rows:
        plot_calibration(
            calibration_rows,
            args.out_dir / "calibration_surface.png",
        )
    if feature_calibration_rows:
        plot_feature_calibration(
            feature_calibration_rows,
            args.out_dir / "feature_calibration_surface.png",
        )
    if transfer_rows:
        plot_calibration_transfer(
            transfer_rows,
            args.out_dir / "calibration_to_evaluation.png",
        )
    plot_selection_overlap(
        overlap_rows,
        args.out_dir / "selection_overlap_vs_reference.png",
    )
    plot_accuracy_state(
        overall_rows,
        args.out_dir / "accuracy_vs_persistent_state.png",
    )
    plot_task_heatmap(
        task_rows,
        args.out_dir / "task_accuracy_heatmap.png",
    )
    plot_paired(
        paired_rows,
        args.out_dir / "paired_gain_vs_exact_recent.png",
    )
    plot_temporal_selection(
        rows,
        args.out_dir / "selected_frame_temporal_distribution.png",
    )
    selections = selection_manifest(
        rows,
        hyperparameters=hyperparameters,
    )
    (args.out_dir / "llava_selection_manifest.json").write_text(
        json.dumps(selections, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    decision = promotion_decision(
        task_rows,
        paired_rows,
        validation,
        primary=args.primary_policy,
        reference=args.reference_policy,
    )
    (args.out_dir / "promotion_decision.json").write_text(
        json.dumps(decision, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    summary = {
        "validation": validation,
        "hyperparameters": hyperparameters,
        "hyperparameter_source": hyperparameter_source,
        "learned_ranker_source": learned_ranker_source,
        "analysis_stage": args.analysis_stage,
        "records": len(records),
        "prediction_rows": len(rows),
        "policies": policies,
        "promotion": decision,
        "protocol_limits": {
            "primary_question_only_retrieval": (
                args.primary_policy
                != "calibrated_diverse_recent_query_mmr"
            ),
            "secondary_option_aware_policies": [
                policy
                for policy in (
                    "calibrated_diverse_recent_query_mmr",
                    "learned_recent_query_topk",
                )
                if policy in policies
            ],
            "candidate_options_used_symmetrically": True,
            "evaluation_labels_used_for_selection": False,
            "visual_evidence_cache_counted": False,
        },
    }
    (args.out_dir / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

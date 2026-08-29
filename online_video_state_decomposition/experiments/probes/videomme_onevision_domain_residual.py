from __future__ import annotations

from collections import Counter
from pathlib import Path

import torch

from feature_memory_codec import LowRankFeatureCodec, fit_pca_codec
from videomme_onevision_pca_protocol import option_text, record_is_eligible
from videomme_onevision_pca_replication import VideoMMESample
from videomme_onevision_domain_residual_protocol import (
    CANDIDATES,
    PROTOCOL_ID,
    RANK,
    SWAP_RANKS,
)


def validate_manifest(manifest: dict[str, object]) -> None:
    if manifest["protocol_id"] != PROTOCOL_ID:
        raise ValueError("domain-residual protocol mismatch")
    if int(manifest["rank"]) != RANK:
        raise ValueError("domain-residual rank mismatch")
    if tuple(int(value) for value in manifest["swap_ranks"]) != SWAP_RANKS:
        raise ValueError("domain-residual swap ranks mismatch")
    if tuple(str(value) for value in manifest["candidates"]) != CANDIDATES:
        raise ValueError("domain-residual candidates mismatch")
    if manifest["frame_policy"] != "uniform16_pool_uniform8_reader":
        raise ValueError("domain-residual frame policy mismatch")


def load_role_samples(
    *,
    records: list[dict[str, object]],
    video_root: Path,
    manifest: dict[str, object],
    role: str,
) -> list[VideoMMESample]:
    validate_manifest(manifest)
    if role not in manifest["roles"]:
        raise KeyError(f"unknown role {role}")
    by_question = {str(record["question_id"]): record for record in records}
    samples = []
    for entry in manifest["roles"][role]:
        question_id = str(entry["question_id"])
        record = by_question[question_id]
        if not record_is_eligible(record):
            raise ValueError(f"question {question_id} is not eligible")
        identity = {
            "video_id": str(record["videoID"]),
            "duration": str(record["duration"]),
            "domain": str(record["domain"]),
            "task_type": str(record["task_type"]),
        }
        for field, value in identity.items():
            if value != str(entry[field]):
                raise ValueError(f"role metadata mismatch for {question_id}: {field}")
        options = tuple(
            option_text(option, chr(ord("A") + index))
            for index, option in enumerate(record["options"])
        )
        correct_index = ord(str(record["answer"]).strip().upper()) - ord("A")
        video_path = video_root / f"{identity['video_id']}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(video_path)
        samples.append(
            VideoMMESample(
                question_id=question_id,
                video_id=identity["video_id"],
                video_path=video_path,
                duration=identity["duration"],
                domain=identity["domain"],
                task_type=identity["task_type"],
                question=str(record["question"]),
                candidates=options,
                correct_index=correct_index,
            )
        )
    expected = int(manifest["role_counts"][role])
    if len(samples) != expected:
        raise ValueError(f"role {role} expected {expected} samples, found {len(samples)}")
    if len({sample.video_id for sample in samples}) != expected:
        raise ValueError(f"role {role} contains duplicate videos")
    return samples


def _relative_error(codec: LowRankFeatureCodec, features: torch.Tensor) -> float:
    reconstruction = codec.decode(codec.encode(features))
    numerator = torch.linalg.vector_norm(features.float() - reconstruction)
    denominator = torch.linalg.vector_norm(features.float())
    return float((numerator / denominator).item())


def _subspace_overlap(first: torch.Tensor, second: torch.Tensor) -> float:
    first_q = torch.linalg.qr(first.float(), mode="reduced").Q
    second_q = torch.linalg.qr(second.float(), mode="reduced").Q
    overlap = torch.sum((first_q.transpose(0, 1) @ second_q).square())
    return float((overlap / min(first_q.shape[1], second_q.shape[1])).item())


def fit_residual_swap_basis(
    *,
    centered: torch.Tensor,
    source_basis: torch.Tensor,
    swap_rank: int,
    seed: int,
    niter: int,
) -> tuple[torch.Tensor, dict[str, float | int]]:
    if centered.ndim != 2 or source_basis.ndim != 2:
        raise ValueError("centered features and source basis must be matrices")
    if centered.shape[1] != source_basis.shape[0]:
        raise ValueError("centered features and source basis hidden sizes differ")
    source_rank = source_basis.shape[1]
    if not 0 < swap_rank < source_rank:
        raise ValueError("swap rank must be positive and smaller than source rank")
    kept_rank = source_rank - swap_rank
    kept = torch.linalg.qr(
        source_basis.float()[:, :kept_rank],
        mode="reduced",
    ).Q
    residual = centered.float() - (centered.float() @ kept) @ kept.transpose(0, 1)
    torch.manual_seed(seed)
    _, singular_values, innovation = torch.pca_lowrank(
        residual,
        q=swap_rank,
        center=False,
        niter=niter,
    )
    innovation = innovation - kept @ (kept.transpose(0, 1) @ innovation)
    innovation = torch.linalg.qr(innovation, mode="reduced").Q[:, :swap_rank]
    basis = torch.cat((kept, innovation), dim=1)
    gram = basis.transpose(0, 1) @ basis
    orthogonality_error = float(
        torch.max(
            torch.abs(
                gram - torch.eye(source_rank, device=basis.device, dtype=basis.dtype)
            )
        ).item()
    )
    total_residual_energy = torch.sum(residual.square())
    retained_residual_energy = torch.sum(singular_values.square())
    metadata = {
        "adaptation_rank": swap_rank,
        "kept_source_rank": kept_rank,
        "residual_energy_ratio": float(
            (retained_residual_energy / total_residual_energy).item()
        ),
        "orthogonality_error": orthogonality_error,
    }
    return basis, metadata


def fit_domain_residual_codecs(
    *,
    features: torch.Tensor,
    source_codec: LowRankFeatureCodec,
    seed: int,
    niter: int,
    storage_dtype: torch.dtype = torch.float16,
) -> tuple[dict[str, LowRankFeatureCodec], dict[str, object]]:
    if features.ndim != 3:
        raise ValueError("calibration features must have [frames,tokens,hidden] shape")
    if source_codec.rank != RANK:
        raise ValueError(f"source codec rank must be {RANK}")
    if source_codec.hidden_size != features.shape[-1]:
        raise ValueError("source codec hidden size differs from calibration features")

    device = features.device
    source = source_codec.to(device=device, dtype=storage_dtype)
    flattened = features.reshape(-1, features.shape[-1]).float()
    target_mean = flattened.mean(dim=0)
    centered = flattened - target_mean
    target_features = features.float()

    candidates: dict[str, LowRankFeatureCodec] = {
        "source_r456": source,
        "target_mean_source_r456": LowRankFeatureCodec(
            mean=target_mean.to(storage_dtype),
            basis=source.basis,
        ),
    }
    torch.manual_seed(seed)
    target_codec, target_fit = fit_pca_codec(
        target_features,
        rank=RANK,
        seed=seed,
        niter=niter,
        storage_dtype=storage_dtype,
    )
    candidates["target_pca_r456"] = target_codec

    swap_metadata = {}
    source_basis = source.basis.float()
    for swap_rank in SWAP_RANKS:
        basis, metadata = fit_residual_swap_basis(
            centered=centered,
            source_basis=source_basis,
            swap_rank=swap_rank,
            seed=seed + swap_rank,
            niter=niter,
        )
        if float(metadata["orthogonality_error"]) > 1e-3:
            raise ValueError(
                f"residual swap {swap_rank} is not orthonormal: "
                f"{metadata['orthogonality_error']}"
            )
        candidate_id = f"residual_swap_r{swap_rank}"
        candidates[candidate_id] = LowRankFeatureCodec(
            mean=target_mean.to(storage_dtype),
            basis=basis.to(storage_dtype),
        )
        swap_metadata[candidate_id] = metadata

    if tuple(candidates) != CANDIDATES:
        candidates = {candidate_id: candidates[candidate_id] for candidate_id in CANDIDATES}
    candidate_metadata = {}
    for candidate_id, codec in candidates.items():
        metadata = {
            "rank": codec.rank,
            "model_parameter_bytes": codec.parameter_bytes,
            "calibration_feature_relative_l2": _relative_error(codec, features),
            "source_subspace_overlap": _subspace_overlap(
                source.basis,
                codec.basis,
            ),
        }
        if candidate_id in swap_metadata:
            metadata.update(swap_metadata[candidate_id])
        candidate_metadata[candidate_id] = metadata
    metadata = {
        "rank": RANK,
        "seed": seed,
        "niter": niter,
        "training_tokens": int(flattened.shape[0]),
        "hidden_size": int(flattened.shape[1]),
        "target_source_subspace_overlap": _subspace_overlap(
            source.basis,
            target_codec.basis,
        ),
        "target_fit": target_fit,
        "candidates": candidate_metadata,
    }
    return candidates, metadata


def role_duration_counts(samples: list[VideoMMESample]) -> dict[str, int]:
    return dict(Counter(sample.duration for sample in samples))

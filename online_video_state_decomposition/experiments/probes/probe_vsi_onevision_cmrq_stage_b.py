from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_features,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import (
    FeatureStatistics,
    centered_covariance,
    descending_eigenspace,
)
from probe_vsi_onevision_reader_risk_stage_a import (
    VSIReaderSample,
    candidate_margins,
    reconstruct,
    select_calibration_questions,
)
from reader_quotient_cmrq_stage_b import (
    DomainMoments,
    boundary_mixed_basis,
    equally_weighted_moments,
    fixed_rank_hybrid,
    orthogonality_error,
    projected_top_atoms,
    random_complement_atoms,
    summarize_exact_rows,
    trace_capture,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


@dataclass(frozen=True)
class CodecCandidate:
    name: str
    mean: torch.Tensor
    basis: torch.Tensor
    atom_family: str
    atom_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--spectral-artifact", type=Path, required=True)
    parser.add_argument("--reader-risk-artifact", type=Path, required=True)
    parser.add_argument("--reader-risk-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--risk-fit-ranges", default="")
    parser.add_argument("--risk-fit-offset", type=int, default=0)
    parser.add_argument("--risk-fit-count", type=int, default=24)
    parser.add_argument("--evaluation-offset", type=int, default=24)
    parser.add_argument("--evaluation-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--rank", type=int, default=456)
    parser.add_argument("--atom-counts", default="16,32,64,96")
    parser.add_argument("--null-atom-counts", default="16,32,64,96")
    parser.add_argument("--mix-atom-count", type=int, default=32)
    parser.add_argument("--mix-weights", default="0.03,0.1,0.3,1,3,10")
    parser.add_argument("--method-names", default="")
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def parse_atom_counts(value: str) -> tuple[int, ...]:
    counts = tuple(sorted({int(item.strip()) for item in value.split(",")}))
    if not counts or counts[0] <= 0:
        raise ValueError("atom counts must be positive")
    return counts


def parse_mix_weights(value: str) -> tuple[float, ...]:
    weights = tuple(sorted({float(item.strip()) for item in value.split(",")}))
    if not weights or weights[0] <= 0.0:
        raise ValueError("mix weights must be positive")
    return weights


def parse_index_ranges(value: str) -> tuple[int, ...]:
    if not value:
        return ()
    indices = []
    for item in value.split(","):
        start_text, separator, stop_text = item.strip().partition(":")
        if not separator:
            raise ValueError("risk-fit ranges must use START:STOP syntax")
        start = int(start_text)
        stop = int(stop_text)
        if start < 0 or stop <= start:
            raise ValueError("risk-fit ranges must be non-empty and non-negative")
        indices.extend(range(start, stop))
    if len(indices) != len(set(indices)):
        raise ValueError("risk-fit ranges overlap")
    return tuple(indices)


def feature_path_for_sample(feature_dir: Path, sample: VSIReaderSample) -> Path:
    candidates = sorted(feature_dir.glob(f"*_{sample.video_path.stem}.pt"))
    if len(candidates) != 1:
        raise ValueError(
            f"expected one feature payload for scene {sample.video_path.stem}"
        )
    return candidates[0]


def training_vsi_moments(
    feature_dir: Path,
    *,
    excluded_paths: set[Path],
    device: torch.device,
) -> tuple[DomainMoments, tuple[str, ...]]:
    paths = sorted(feature_dir.glob("*.pt"))
    training_paths = [path for path in paths if path not in excluded_paths]
    if not training_paths:
        raise ValueError("VSI feature training partition is empty")
    rows = 0
    feature_sum = None
    gram = None
    sample_ids = []
    for path in training_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        features = payload["features"].to(device=device, dtype=torch.float32)
        matrix = features.reshape(-1, features.shape[-1])
        if feature_sum is None:
            feature_sum = torch.zeros(
                matrix.shape[-1], device=device, dtype=torch.float32
            )
            gram = torch.zeros(
                matrix.shape[-1],
                matrix.shape[-1],
                device=device,
                dtype=torch.float32,
            )
        rows += matrix.shape[0]
        feature_sum.add_(matrix.sum(dim=0))
        gram.add_(matrix.transpose(0, 1) @ matrix)
        sample_ids.append(str(payload["sample_id"]))
    if feature_sum is None or gram is None:
        raise RuntimeError("failed to accumulate VSI feature statistics")
    statistics = FeatureStatistics(rows=rows, feature_sum=feature_sum, gram=gram)
    return (
        DomainMoments(
            mean=feature_sum / rows,
            covariance=centered_covariance(statistics),
        ),
        tuple(sample_ids),
    )


def build_candidates(
    spectral: dict[str, dict[str, object]],
    risk_artifact: dict[str, torch.Tensor],
    vsi_train: DomainMoments,
    *,
    rank: int,
    atom_counts: tuple[int, ...],
    null_atom_counts: tuple[int, ...],
    mix_atom_count: int,
    mix_weights: tuple[float, ...],
    seed: int,
    device: torch.device,
) -> tuple[list[CodecCandidate], DomainMoments, torch.Tensor]:
    domains = []
    for name in ("source", "target"):
        domains.append(
            DomainMoments(
                mean=spectral[name]["full_mean"].to(device=device, dtype=torch.float32),
                covariance=spectral[name]["full_covariance"].to(
                    device=device, dtype=torch.float32
                ),
            )
        )
    domains.append(vsi_train)
    pooled = equally_weighted_moments(domains)
    _, pooled_basis = descending_eigenspace(pooled.covariance, rank=rank)
    _, vsi_basis = descending_eigenspace(vsi_train.covariance, rank=rank)
    risk = risk_artifact["risk_matrix"].to(device=device, dtype=torch.float32)

    candidates = [
        CodecCandidate(
            name="source_pca_r456",
            mean=spectral["source"]["full_mean"].to(
                device=device, dtype=torch.float32
            ),
            basis=spectral["source"]["full_basis"].to(
                device=device, dtype=torch.float32
            ),
            atom_family="none",
            atom_count=0,
        ),
        CodecCandidate(
            name="target_pca_r456",
            mean=spectral["target"]["full_mean"].to(
                device=device, dtype=torch.float32
            ),
            basis=spectral["target"]["full_basis"].to(
                device=device, dtype=torch.float32
            ),
            atom_family="none",
            atom_count=0,
        ),
        CodecCandidate(
            name="vsi_pca_train96_r456",
            mean=vsi_train.mean,
            basis=vsi_basis,
            atom_family="none",
            atom_count=0,
        ),
        CodecCandidate(
            name="pooled3_pca_r456",
            mean=pooled.mean,
            basis=pooled_basis,
            atom_family="none",
            atom_count=0,
        ),
    ]
    risk_atoms_by_count = {}
    for atom_count in atom_counts:
        bulk = pooled_basis[:, : rank - atom_count]
        atoms = projected_top_atoms(risk, bulk, atom_count=atom_count)
        risk_atoms_by_count[atom_count] = atoms
        candidates.append(
            CodecCandidate(
                name=f"cmrq_risk_atoms{atom_count}_r{rank}",
                mean=pooled.mean,
                basis=fixed_rank_hybrid(
                    pooled_basis,
                    atoms,
                    rank=rank,
                    atom_count=atom_count,
                ),
                atom_family="reader_risk",
                atom_count=atom_count,
            )
        )

    generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    permutation = torch.randperm(risk.shape[0], generator=generator).to(device)
    permuted_risk = risk.index_select(0, permutation).index_select(1, permutation)
    permuted_atoms_by_count = {}
    for atom_count in null_atom_counts:
        bulk = pooled_basis[:, : rank - atom_count]
        nulls = (
            (
                "feature_only_null",
                projected_top_atoms(
                    vsi_train.covariance,
                    bulk,
                    atom_count=atom_count,
                ),
            ),
            (
                "random_null",
                random_complement_atoms(
                    bulk,
                    atom_count=atom_count,
                    seed=seed + atom_count,
                ),
            ),
            (
                "permuted_risk_null",
                projected_top_atoms(
                    permuted_risk,
                    bulk,
                    atom_count=atom_count,
                ),
            ),
        )
        permuted_atoms_by_count[atom_count] = nulls[2][1]
        for family, atoms in nulls:
            candidates.append(
                CodecCandidate(
                    name=f"{family}_atoms{atom_count}_r{rank}",
                    mean=pooled.mean,
                    basis=fixed_rank_hybrid(
                        pooled_basis,
                        atoms,
                        rank=rank,
                        atom_count=atom_count,
                    ),
                    atom_family=family,
                    atom_count=atom_count,
                )
            )
    if mix_atom_count not in risk_atoms_by_count:
        raise ValueError("mix atom count must be present in atom counts")
    if mix_atom_count not in permuted_atoms_by_count:
        raise ValueError("mix atom count must be present in null atom counts")
    for risk_weight in mix_weights:
        weight_name = str(risk_weight).replace(".", "p")
        candidates.append(
            CodecCandidate(
                name=f"cmrq_mix_g{mix_atom_count}_w{weight_name}_r{rank}",
                mean=pooled.mean,
                basis=boundary_mixed_basis(
                    pooled_basis,
                    risk_atoms_by_count[mix_atom_count],
                    pooled.covariance,
                    risk,
                    rank=rank,
                    atom_count=mix_atom_count,
                    risk_weight=risk_weight,
                ),
                atom_family="reader_risk_boundary_mix",
                atom_count=mix_atom_count,
            )
        )
        candidates.append(
            CodecCandidate(
                name=f"permuted_mix_g{mix_atom_count}_w{weight_name}_r{rank}",
                mean=pooled.mean,
                basis=boundary_mixed_basis(
                    pooled_basis,
                    permuted_atoms_by_count[mix_atom_count],
                    pooled.covariance,
                    permuted_risk,
                    rank=rank,
                    atom_count=mix_atom_count,
                    risk_weight=risk_weight,
                ),
                atom_family="permuted_risk_boundary_mix_null",
                atom_count=mix_atom_count,
            )
        )
    candidates.append(
        CodecCandidate(
            name=f"risk_only_r{rank}",
            mean=pooled.mean,
            basis=risk_artifact["risk_basis"][:, :rank].to(
                device=device, dtype=torch.float32
            ),
            atom_family="risk_only_diagnostic",
            atom_count=rank,
        )
    )
    return candidates, pooled, risk


def candidate_kl(reference: torch.Tensor, approximate: torch.Tensor) -> float:
    log_reference = torch.log_softmax(reference.float(), dim=0)
    log_approximate = torch.log_softmax(approximate.float(), dim=0)
    probability = log_reference.exp()
    return float((probability * (log_reference - log_approximate)).sum().item())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    atom_counts = parse_atom_counts(args.atom_counts)
    null_atom_counts = parse_atom_counts(args.null_atom_counts)
    mix_weights = parse_mix_weights(args.mix_weights)
    risk_fit_indices = parse_index_ranges(args.risk_fit_ranges)
    if max((*atom_counts, *null_atom_counts)) >= args.rank:
        raise ValueError("atom count must be smaller than the codec rank")
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    if args.risk_fit_offset < 0 or args.evaluation_offset < 0:
        raise ValueError("sample offsets must be non-negative")
    if risk_fit_indices and len(risk_fit_indices) != args.risk_fit_count:
        raise ValueError("risk-fit ranges and count disagree")
    maximum_risk_index = (
        max(risk_fit_indices) + 1
        if risk_fit_indices
        else args.risk_fit_offset + args.risk_fit_count
    )
    selected = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=max(
            maximum_risk_index,
            args.evaluation_offset + args.evaluation_count,
        ),
    )
    risk_fit = (
        [selected[index] for index in risk_fit_indices]
        if risk_fit_indices
        else selected[
            args.risk_fit_offset : args.risk_fit_offset + args.risk_fit_count
        ]
    )
    evaluation = selected[
        args.evaluation_offset : args.evaluation_offset + args.evaluation_count
    ]
    if {sample.sample_id for sample in risk_fit} & {
        sample.sample_id for sample in evaluation
    }:
        raise ValueError("risk-fit and exact-evaluation samples overlap")
    risk_summary = json.loads(args.reader_risk_summary.read_text(encoding="utf-8"))
    if risk_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("reader-risk protocol identity mismatch")
    if risk_summary["sample_ids"] != [sample.sample_id for sample in risk_fit]:
        raise ValueError("reader-risk fit partition does not match the frozen prefix")
    if risk_summary["sample_count"] != args.risk_fit_count:
        raise ValueError("reader-risk sample count mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    evaluation_paths = {
        feature_path_for_sample(args.feature_dir, sample) for sample in evaluation
    }
    vsi_train, vsi_train_ids = training_vsi_moments(
        args.feature_dir,
        excluded_paths=evaluation_paths,
        device=device,
    )
    if len(vsi_train_ids) + len(evaluation_paths) != len(
        tuple(args.feature_dir.glob("*.pt"))
    ):
        raise ValueError("VSI feature fit/evaluation partition is not exhaustive")
    spectral = torch.load(args.spectral_artifact, map_location="cpu", weights_only=False)
    risk_artifact = torch.load(
        args.reader_risk_artifact,
        map_location="cpu",
        weights_only=True,
    )
    candidates, pooled, risk = build_candidates(
        spectral,
        risk_artifact,
        vsi_train,
        rank=args.rank,
        atom_counts=atom_counts,
        null_atom_counts=null_atom_counts,
        mix_atom_count=args.mix_atom_count,
        mix_weights=mix_weights,
        seed=args.seed,
        device=device,
    )
    if args.method_names:
        requested_methods = tuple(
            item.strip() for item in args.method_names.split(",") if item.strip()
        )
        available_methods = {candidate.name for candidate in candidates}
        missing_methods = set(requested_methods) - available_methods
        if missing_methods:
            raise ValueError(f"unknown method names: {sorted(missing_methods)}")
        candidates = [
            candidate for candidate in candidates if candidate.name in requested_methods
        ]
    candidate_metadata = {}
    for candidate in candidates:
        candidate_metadata[candidate.name] = {
            "rank": candidate.basis.shape[1],
            "atom_family": candidate.atom_family,
            "atom_count": candidate.atom_count,
            "pooled_feature_capture": trace_capture(
                pooled.covariance, candidate.basis
            ),
            "reader_risk_capture": trace_capture(risk, candidate.basis),
            "orthogonality_error": orthogonality_error(candidate.basis),
        }

    processor, model = load_onevision_model(args.model_dir, device=args.device)
    model_dtype = next(model.parameters()).dtype
    rows = []
    margin_rows = []
    started = time.perf_counter()
    for position, sample in enumerate(evaluation, start=1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        pool_features = payload["features"]
        selected_positions = uniform_frame_indices(
            pool_features.shape[0], args.frame_budget
        )
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = pool_features.index_select(0, positions).to(
            device=device,
            dtype=model_dtype,
        )
        selected_frame_indices = [
            payload["pool_indices"][index] for index in selected_positions
        ]
        frames, _, _ = decode_video_frames(sample.video_path, selected_frame_indices)
        prompt_batch = build_prompt_batch(
            processor,
            sample,
            np.stack(frames),
            device=device,
            dtype=model_dtype,
        )
        with torch.inference_mode():
            reference_logits = first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        reference_candidate_logits = reference_logits.float().index_select(
            0, token_tensor
        )
        teacher_index = int(torch.argmax(reference_candidate_logits).item())
        competitor_indices = [
            index for index in range(len(token_ids)) if index != teacher_index
        ]
        margins = candidate_margins(
            reference_logits,
            token_ids,
            teacher_index=teacher_index,
            competitor_indices=competitor_indices,
        )
        baseline_correct = teacher_index == sample.answer_index
        for candidate in candidates:
            approximate = reconstruct(
                reference,
                mean=candidate.mean,
                basis=candidate.basis,
            ).to(model_dtype)
            with torch.inference_mode():
                approximate_logits = first_token_logits_from_features(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    features=approximate,
                )
            approximate_candidate_logits = approximate_logits.float().index_select(
                0, token_tensor
            )
            approximate_index = int(torch.argmax(approximate_candidate_logits).item())
            top_two = torch.topk(approximate_candidate_logits, k=2).values
            approximate_top1_margin = float((top_two[0] - top_two[1]).item())
            approximate_margins = candidate_margins(
                approximate_logits,
                token_ids,
                teacher_index=teacher_index,
                competitor_indices=competitor_indices,
            )
            exact_shifts = approximate_margins - margins
            normalized_adverse = (-exact_shifts).clamp_min(0) / margins.clamp_min(
                args.margin_floor
            )
            approximate_correct = approximate_index == sample.answer_index
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "method": candidate.name,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "approximate_index": approximate_index,
                    "approximate_top1_margin": approximate_top1_margin,
                    "minimum_margin": float(margins.min().item()),
                    "feature_relative_l2": float(
                        (
                            torch.linalg.vector_norm(
                                approximate.float() - reference.float()
                            )
                            / torch.linalg.vector_norm(reference.float()).clamp_min(
                                torch.finfo(torch.float32).eps
                            )
                        ).item()
                    ),
                    "candidate_kl": candidate_kl(
                        reference_candidate_logits,
                        approximate_candidate_logits,
                    ),
                    "maximum_normalized_adverse_shift": float(
                        normalized_adverse.max().item()
                    ),
                    "prediction_match": int(approximate_index == teacher_index),
                    "harmful": int(baseline_correct and not approximate_correct),
                    "beneficial": int(not baseline_correct and approximate_correct),
                }
            )
            for competitor_position, competitor_index in enumerate(competitor_indices):
                margin_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "method": candidate.name,
                        "teacher_index": teacher_index,
                        "competitor_index": competitor_index,
                        "baseline_margin": float(margins[competitor_position].item()),
                        "exact_shift": float(exact_shifts[competitor_position].item()),
                        "exact_final_margin": float(
                            approximate_margins[competitor_position].item()
                        ),
                        "normalized_adverse_shift": float(
                            normalized_adverse[competitor_position].item()
                        ),
                    }
                )
        print(
            json.dumps(
                {
                    "event": "cmrq_exact_reader_ok",
                    "position": position,
                    "total": len(evaluation),
                    "sample_id": sample.sample_id,
                }
            ),
            flush=True,
        )

    method_summaries = {}
    for candidate in candidates:
        method_rows = [row for row in rows if row["method"] == candidate.name]
        method_summaries[candidate.name] = {
            **candidate_metadata[candidate.name],
            **summarize_exact_rows(method_rows, margin_floor=args.margin_floor),
        }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_only_disjoint_risk_fit_exact_eval",
        "rank": args.rank,
        "risk_fit_ranges": args.risk_fit_ranges,
        "risk_fit_offset": args.risk_fit_offset,
        "exact_evaluation_offset": args.evaluation_offset,
        "risk_fit_sample_ids": [sample.sample_id for sample in risk_fit],
        "exact_evaluation_sample_ids": [sample.sample_id for sample in evaluation],
        "vsi_feature_fit_video_count": len(vsi_train_ids),
        "vsi_feature_evaluation_video_count": len(evaluation_paths),
        "atom_counts": atom_counts,
        "null_atom_counts": null_atom_counts,
        "mix_atom_count": args.mix_atom_count,
        "mix_weights": mix_weights,
        "method_names": [candidate.name for candidate in candidates],
        "seed": args.seed,
        "method_summaries": method_summaries,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(args.out_dir / "sample_metrics.csv", rows)
    write_csv(args.out_dir / "margin_metrics.csv", margin_rows)
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

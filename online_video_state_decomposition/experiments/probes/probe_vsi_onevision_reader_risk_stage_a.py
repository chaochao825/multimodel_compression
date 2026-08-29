from __future__ import annotations

import argparse
import csv
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from feature_memory_codec import load_codec
from mvbench_llava_anchor import write_json_atomic
from mvbench_onevision_utils import (
    build_prompt_batch,
    first_token_logits_from_features,
    load_onevision_model,
)
from mvbench_reader_quotient_support_oracle import candidate_token_ids
from mvbench_utils import decode_video_frames, uniform_frame_indices
from onevision_reader_quotient_stage_a import (
    channel_reader_risk,
    commutator_ratio,
    descending_eigenspace,
    local_linearity_summary,
    tail_energy_fraction,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records, scene_key


@dataclass(frozen=True)
class VSIReaderSample:
    sample_id: str
    video_path: Path
    question: str
    candidates: tuple[str, ...]
    answer_index: int
    subtitle: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--source-codec", type=Path, required=True)
    parser.add_argument("--spectral-artifact", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=24)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--margin-floor", type=float, default=0.05)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def strip_option_label(value: object) -> str:
    return re.sub(r"^[A-Z]\s*[.)]\s*", "", str(value).strip())


def select_calibration_questions(
    *,
    split: dict[str, object],
    records: list[dict[str, object]],
    video_root: Path,
    sample_count: int,
) -> list[VSIReaderSample]:
    by_id = {int(record["id"]): record for record in records}
    selected = []
    for scene in split["roles"]["calibration"]:
        question_ids = [int(value) for value in scene["debiased_question_ids"]]
        if not question_ids:
            continue
        record = by_id[question_ids[0]]
        if scene_key(record) != (str(scene["dataset"]), str(scene["scene_name"])):
            raise ValueError("VSI question and scene split disagree")
        answer_index = ord(str(record["ground_truth"]).strip().upper()) - ord("A")
        selected.append(
            VSIReaderSample(
                sample_id=f"vsi_question_{record['id']}",
                video_path=video_root / str(scene["relative_video_path"]),
                question=str(record["question"]),
                candidates=tuple(strip_option_label(value) for value in record["options"]),
                answer_index=answer_index,
            )
        )
        if len(selected) == sample_count:
            break
    if len(selected) != sample_count:
        raise ValueError(f"requested {sample_count} calibration questions, found {len(selected)}")
    return selected


def reconstruct(
    features: torch.Tensor,
    *,
    mean: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    centered = features.float() - mean.float()
    return (
        (centered @ basis.float()) @ basis.float().transpose(0, 1)
        + mean.float()
    )


def candidate_margins(
    logits: torch.Tensor,
    token_ids: list[int],
    *,
    teacher_index: int,
    competitor_indices: list[int],
) -> torch.Tensor:
    indices = torch.tensor(token_ids, device=logits.device, dtype=torch.long)
    candidates = logits.float().index_select(0, indices)
    teacher = candidates[teacher_index]
    competitors = candidates.index_select(
        0,
        torch.tensor(competitor_indices, device=logits.device, dtype=torch.long),
    )
    return teacher - competitors


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    if args.sample_offset < 0:
        raise ValueError("sample offset must be non-negative")
    selected = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_offset + args.sample_count,
    )
    samples = selected[args.sample_offset :]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    source_codec, _ = load_codec(
        args.source_codec,
        device=device,
        dtype=torch.float32,
    )
    spectral = torch.load(args.spectral_artifact, map_location="cpu", weights_only=False)
    vsi_spectrum = spectral["vsi"]
    vsi_mean = vsi_spectrum["full_mean"].to(device)
    vsi_basis = vsi_spectrum["full_basis"].to(device)
    vsi_covariance = vsi_spectrum["full_covariance"].to(device)
    if source_codec.rank != vsi_basis.shape[1]:
        raise ValueError("source and VSI basis ranks differ")

    methods = {
        "source_pca_r456": (source_codec.mean, source_codec.basis),
        "vsi_pca_r456": (vsi_mean, vsi_basis),
    }
    risk = torch.zeros_like(vsi_covariance, dtype=torch.float32)
    rows = []
    margin_rows = []
    exact_by_method = {method: [] for method in methods}
    linear_by_method = {method: [] for method in methods}
    baseline_by_method = {method: [] for method in methods}
    started = time.perf_counter()
    for position, sample in enumerate(samples, start=1):
        scene_sample_id = sample.video_path.stem
        candidates = sorted(args.feature_dir.glob(f"*_{scene_sample_id}.pt"))
        if len(candidates) != 1:
            raise ValueError(f"expected one feature payload for scene {scene_sample_id}")
        payload = torch.load(candidates[0], map_location="cpu", weights_only=False)
        pool_features = payload["features"]
        selected_positions = uniform_frame_indices(pool_features.shape[0], args.frame_budget)
        positions = torch.tensor(selected_positions, dtype=torch.long)
        reference = pool_features.index_select(0, positions).to(
            device=device,
            dtype=model_dtype,
        )
        selected_frame_indices = [payload["pool_indices"][index] for index in selected_positions]
        frames, _, _ = decode_video_frames(sample.video_path, selected_frame_indices)
        prompt_batch = build_prompt_batch(
            processor,
            sample,
            np.stack(frames),
            device=device,
            dtype=model_dtype,
        )
        probe = reference.detach().float().requires_grad_(True)
        reference_logits = first_token_logits_from_features(
            model=model,
            input_ids=prompt_batch["input_ids"],
            attention_mask=prompt_batch["attention_mask"],
            features=probe,
        )
        token_ids = candidate_token_ids(processor.tokenizer, len(sample.candidates))
        token_tensor = torch.tensor(token_ids, device=device, dtype=torch.long)
        teacher_index = int(
            torch.argmax(reference_logits.float().index_select(0, token_tensor)).item()
        )
        competitor_indices = [
            index for index in range(len(token_ids)) if index != teacher_index
        ]
        margins = candidate_margins(
            reference_logits,
            token_ids,
            teacher_index=teacher_index,
            competitor_indices=competitor_indices,
        )
        gradients = []
        for competitor_position, margin in enumerate(margins):
            gradient = torch.autograd.grad(
                margin,
                probe,
                retain_graph=competitor_position + 1 < len(margins),
            )[0]
            gradients.append(gradient.detach().reshape(-1, gradient.shape[-1]))
        gradient_tensor = torch.stack(gradients).float()
        risk.add_(
            channel_reader_risk(
                gradient_tensor,
                margins.detach(),
                feature_norm_squared=float(reference.float().square().sum().item()),
                margin_floor=args.margin_floor,
            )
        )

        for method, (mean, basis) in methods.items():
            approximate = reconstruct(reference, mean=mean, basis=basis).to(model_dtype)
            with torch.inference_mode():
                approximate_logits = first_token_logits_from_features(
                    model=model,
                    input_ids=prompt_batch["input_ids"],
                    attention_mask=prompt_batch["attention_mask"],
                    features=approximate,
                )
            approximate_margins = candidate_margins(
                approximate_logits,
                token_ids,
                teacher_index=teacher_index,
                competitor_indices=competitor_indices,
            )
            exact_shifts = approximate_margins - margins.detach()
            delta = (approximate.float() - reference.float()).reshape(-1, reference.shape[-1])
            linear_shifts = torch.einsum("ctd,td->c", gradient_tensor, delta)
            exact_by_method[method].append(exact_shifts.cpu())
            linear_by_method[method].append(linear_shifts.cpu())
            baseline_by_method[method].append(margins.detach().cpu())
            for competitor_position, competitor_index in enumerate(competitor_indices):
                margin_rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "method": method,
                        "teacher_index": teacher_index,
                        "competitor_index": competitor_index,
                        "baseline_margin": float(margins[competitor_position].item()),
                        "exact_shift": float(exact_shifts[competitor_position].item()),
                        "linear_shift": float(linear_shifts[competitor_position].item()),
                        "exact_final_margin": float(
                            approximate_margins[competitor_position].item()
                        ),
                    }
                )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "method": method,
                    "teacher_index": teacher_index,
                    "answer_index": sample.answer_index,
                    "minimum_margin": float(margins.min().item()),
                    "feature_relative_l2": float(
                        (
                            torch.linalg.vector_norm(approximate.float() - reference.float())
                            / torch.linalg.vector_norm(reference.float()).clamp_min(
                                torch.finfo(torch.float32).eps
                            )
                        ).item()
                    ),
                    "maximum_exact_adverse_shift": float(
                        (-exact_shifts).clamp_min(0).max().item()
                    ),
                    "maximum_linear_adverse_shift": float(
                        (-linear_shifts).clamp_min(0).max().item()
                    ),
                    "prediction_match": int((approximate_margins > 0).all().item()),
                }
            )
        print(
            json.dumps(
                {
                    "event": "reader_risk_ok",
                    "sample_id": sample.sample_id,
                    "position": position,
                    "total": len(samples),
                }
            ),
            flush=True,
        )

    risk.div_(len(samples))
    risk_eigenvalues, risk_basis = descending_eigenspace(risk, rank=source_codec.rank)
    linearity = {}
    for method in methods:
        exact = torch.cat(exact_by_method[method])
        linear = torch.cat(linear_by_method[method])
        baseline = torch.cat(baseline_by_method[method])
        threshold = torch.quantile(baseline, 0.25)
        low_margin = baseline <= threshold
        linearity[method] = {
            "all": local_linearity_summary(exact, linear, baseline),
            "low_margin_quartile": local_linearity_summary(
                exact[low_margin],
                linear[low_margin],
                baseline[low_margin],
            ),
        }
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": "calibration_only",
        "sample_count": len(samples),
        "sample_offset": args.sample_offset,
        "sample_ids": [sample.sample_id for sample in samples],
        "margin_floor": args.margin_floor,
        "rank": source_codec.rank,
        "risk_tail_energy_fraction": tail_energy_fraction(
            risk_eigenvalues.cpu(),
            rank=source_codec.rank,
        ),
        "feature_risk_commutator_ratio": commutator_ratio(vsi_covariance, risk),
        "linearity": linearity,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_csv(args.out_dir / "sample_metrics.csv", rows)
    write_csv(args.out_dir / "margin_metrics.csv", margin_rows)
    torch.save(
        {
            "risk_matrix": risk.cpu(),
            "risk_eigenvalues": risk_eigenvalues.cpu(),
            "risk_basis": risk_basis.cpu(),
        },
        args.out_dir / "reader_risk_artifact.pt",
    )
    write_json_atomic(args.out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

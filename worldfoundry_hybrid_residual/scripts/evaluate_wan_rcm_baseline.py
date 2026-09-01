#!/usr/bin/env python3
"""Evaluate the frozen EXP-047 formal videos without changing the protocol."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_artifacts import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[2]
METHODS = ("teacher20", "native4", "rcm4")
DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
    "overall_consistency",
)
VIDEO_NAME = re.compile(
    r"^(teacher20|native4|rcm4)_p(?P<prompt_index>\d{2})_s(?P<seed>\d+)\.mp4$"
)


@dataclass(frozen=True)
class FormalVideo:
    method: str
    prompt_index: int
    prompt: str
    seed: int
    path: Path

    @property
    def evaluation_name(self) -> str:
        return f"{self.method}_p{self.prompt_index:02d}_s{self.seed}.mp4"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("prepare", "vbench", "diagnostics", "summarize"),
        required=True,
    )
    parser.add_argument("--vbench-full-info", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dimensions", nargs="+", choices=DIMENSIONS, default=DIMENSIONS)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-047" or config["gate_id"] != "G-026":
        raise ValueError("config is not the frozen EXP-047/G-026 configuration")
    if tuple(config["methods"]) != METHODS:
        raise ValueError(f"method order must remain frozen as {METHODS}")
    if tuple(config["metrics"]["vbench_dimensions"]) != DIMENSIONS:
        raise ValueError("VBench dimensions differ from the frozen EXP-047 protocol")
    return config


def load_prompts(config: dict[str, Any]) -> tuple[str, ...]:
    prompt_path = PROJECT_ROOT / config["generation"]["formal_prompt_file"]
    prompts = tuple(
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if len(prompts) != 4:
        raise ValueError(f"EXP-047 requires exactly four prompts, got {len(prompts)}")
    return prompts


def discover_formal_videos(
    config: dict[str, Any], formal_root: Path
) -> tuple[FormalVideo, ...]:
    prompts = load_prompts(config)
    records: list[FormalVideo] = []
    for method in METHODS:
        for prompt_index, prompt in enumerate(prompts):
            for seed in config["generation"]["formal_seeds"]:
                run_dir = formal_root / method / f"p{prompt_index:02d}_s{seed}"
                manifest_path = run_dir / "generation_manifest.json"
                success_path = run_dir / "SUCCESS.json"
                if not manifest_path.is_file() or not success_path.is_file():
                    raise FileNotFoundError(f"incomplete formal run: {run_dir}")
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest["experiment_id"] != "EXP-047":
                    raise ValueError(f"wrong experiment identity in {manifest_path}")
                if manifest["gate_id"] != "G-026" or manifest["stage"] != "formal":
                    raise ValueError(f"wrong Gate or stage in {manifest_path}")
                if manifest["method"] != method:
                    raise ValueError(f"wrong method in {manifest_path}")
                if manifest["prompt_index"] != prompt_index or manifest["prompt"] != prompt:
                    raise ValueError(f"wrong prompt identity in {manifest_path}")
                if manifest["seed"] != seed:
                    raise ValueError(f"wrong seed identity in {manifest_path}")
                if manifest["num_frames"] != config["generation"]["num_frames"]:
                    raise ValueError(f"wrong frame count in {manifest_path}")
                if len(manifest["rows"]) != 1 or manifest["rows"][0]["status"] != "ok":
                    raise ValueError(f"formal run is not a single successful row: {manifest_path}")
                video_path = run_dir / manifest["rows"][0]["video_file"]
                if not video_path.is_file() or video_path.stat().st_size == 0:
                    raise FileNotFoundError(f"missing formal video: {video_path}")
                records.append(
                    FormalVideo(method, prompt_index, prompt, int(seed), video_path.resolve())
                )
    if len(records) != 24:
        raise ValueError(f"EXP-047 requires 24 formal videos, got {len(records)}")
    return tuple(records)


def prepare_vbench_input(records: tuple[FormalVideo, ...], work_dir: Path) -> None:
    input_dir = work_dir / "vbench_input"
    if input_dir.exists():
        raise FileExistsError(f"prepared VBench input already exists: {input_dir}")
    input_dir.mkdir(parents=True)
    prompt_map: dict[str, str] = {}
    identity_rows: list[dict[str, Any]] = []
    for record in records:
        destination = input_dir / record.evaluation_name
        os.link(record.path, destination)
        prompt_map[record.evaluation_name] = record.prompt
        identity_rows.append(
            {
                "method": record.method,
                "prompt_index": record.prompt_index,
                "prompt": record.prompt,
                "seed": record.seed,
                "source": str(record.path),
                "evaluation_file": str(destination.resolve()),
            }
        )
    atomic_write_json(work_dir / "prompt_map.json", prompt_map)
    atomic_write_json(work_dir / "formal_identity.json", {"videos": identity_rows})


def run_vbench(
    work_dir: Path,
    full_info: Path,
    device: str,
    dimensions: tuple[str, ...],
    resume: bool,
) -> None:
    import torch
    from vbench import VBench

    input_dir = (work_dir / "vbench_input").resolve()
    prompt_map = json.loads((work_dir / "prompt_map.json").read_text(encoding="utf-8"))
    result_dir = work_dir / "vbench_results"
    result_dir.mkdir(parents=True, exist_ok=True)
    for dimension in dimensions:
        name = f"exp047_{dimension}"
        output_path = result_dir / f"{name}_eval_results.json"
        if output_path.exists():
            if resume:
                continue
            raise FileExistsError(f"refusing to overwrite VBench result: {output_path}")
        evaluator = VBench(torch.device(device), str(full_info.resolve()), str(result_dir))
        evaluator.evaluate(
            videos_path=str(input_dir),
            name=name,
            prompt_list=prompt_map,
            dimension_list=[dimension],
            local=True,
            read_frame=False,
            mode="custom_input",
        )


def parse_evaluation_name(path: str) -> tuple[str, int, int]:
    match = VIDEO_NAME.fullmatch(Path(path).name)
    if match is None:
        raise ValueError(f"unexpected VBench video identity: {path}")
    return match.group(1), int(match.group("prompt_index")), int(match.group("seed"))


def summarize_vbench(work_dir: Path) -> dict[str, Any]:
    result_dir = work_dir / "vbench_results"
    by_dimension: dict[str, Any] = {}
    method_ratios: dict[str, list[float]] = {method: [] for method in METHODS[1:]}
    for dimension in DIMENSIONS:
        path = result_dir / f"exp047_{dimension}_eval_results.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        details = payload[dimension][1]
        values: dict[str, list[float]] = {method: [] for method in METHODS}
        seen: set[tuple[str, int, int]] = set()
        for row in details:
            identity = parse_evaluation_name(row["video_path"])
            if identity in seen:
                raise ValueError(f"duplicate VBench identity for {dimension}: {identity}")
            seen.add(identity)
            value = float(row["video_results"])
            if not math.isfinite(value):
                raise ValueError(f"nonfinite VBench result for {dimension}: {identity}")
            values[identity[0]].append(value)
        if len(seen) != 24 or any(len(values[method]) != 8 for method in METHODS):
            raise ValueError(f"incomplete VBench dimension {dimension}: {len(seen)} videos")
        means = {method: statistics.fmean(values[method]) for method in METHODS}
        teacher = means["teacher20"]
        if teacher <= 0.0:
            raise ValueError(f"teacher score is not positive for {dimension}: {teacher}")
        ratios = {method: means[method] / teacher for method in METHODS[1:]}
        for method in METHODS[1:]:
            method_ratios[method].append(ratios[method])
        by_dimension[dimension] = {"means": means, "teacher_normalized": ratios}
    aggregate = {
        method: {
            "mean_teacher_normalized": statistics.fmean(method_ratios[method]),
            "minimum_teacher_normalized": min(method_ratios[method]),
            "passes_mean_0_90": statistics.fmean(method_ratios[method]) >= 0.90,
            "passes_minimum_0_80": min(method_ratios[method]) >= 0.80,
        }
        for method in METHODS[1:]
    }
    summary = {"dimensions": by_dimension, "aggregate": aggregate}
    atomic_write_json(work_dir / "vbench_summary.json", summary)
    return summary


def geometric_mean(values: list[float]) -> float:
    if any(value <= 0.0 or not math.isfinite(value) for value in values):
        raise ValueError(f"geometric mean requires finite positive values: {values}")
    return math.exp(statistics.fmean(math.log(value) for value in values))


def summarize_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ("video_embedding_distance", "frame_lpips", "frame_l1")
    indexed = {(row["method"], row["prompt_index"]): row for row in rows}
    per_method: dict[str, Any] = {}
    for method in METHODS[1:]:
        prompts: list[dict[str, Any]] = []
        for prompt_index in range(4):
            teacher = indexed[("teacher20", prompt_index)]
            candidate = indexed[(method, prompt_index)]
            ratios = {metric: candidate[metric] / teacher[metric] for metric in metrics}
            aggregate_ratio = geometric_mean(list(ratios.values()))
            prompts.append(
                {
                    "prompt_index": prompt_index,
                    "metric_ratios": ratios,
                    "geometric_mean_ratio": aggregate_ratio,
                }
            )
        prompt_ratios = [row["geometric_mean_ratio"] for row in prompts]
        per_method[method] = {
            "prompts": prompts,
            "prompts_at_least_0_70": sum(value >= 0.70 for value in prompt_ratios),
            "minimum_prompt_ratio": min(prompt_ratios),
            "passes_three_of_four_0_70": sum(value >= 0.70 for value in prompt_ratios) >= 3,
            "passes_no_prompt_below_0_50": min(prompt_ratios) >= 0.50,
        }
    return per_method


def run_diagnostics(records: tuple[FormalVideo, ...], work_dir: Path, device: str) -> None:
    import cv2
    import lpips
    import numpy as np
    import torch
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    from vbench.overall_consistency import get_vid_features
    from vbench.third_party.ViCLIP.simple_tokenizer import SimpleTokenizer
    from vbench.third_party.ViCLIP.viclip import ViCLIP
    from vbench.utils import CACHE_DIR, clip_transform, read_frames_decord_by_fps

    record_index = {
        (record.method, record.prompt_index, record.seed): record for record in records
    }
    seeds = sorted({record.seed for record in records})
    if len(seeds) != 2:
        raise ValueError(f"diversity evaluation requires two seeds, got {seeds}")
    torch_device = torch.device(device)
    tokenizer = SimpleTokenizer(str(Path(CACHE_DIR) / "ViCLIP/bpe_simple_vocab_16e6.txt.gz"))
    video_model = ViCLIP(
        tokenizer=tokenizer,
        pretrain=str(Path(CACHE_DIR) / "ViCLIP/ViClip-InternVid-10M-FLT.pth"),
    ).to(torch_device).eval()
    transform = clip_transform(224)
    embedding_cache: dict[Path, torch.Tensor] = {}
    for record in records:
        images = read_frames_decord_by_fps(str(record.path), num_frames=8, sample="middle")
        images = transform(images).to(torch_device)
        embedding_cache[record.path] = get_vid_features(
            video_model, images.unsqueeze(0)
        ).cpu()[0]
    del video_model
    torch.cuda.empty_cache()

    lpips_model = lpips.LPIPS(net="alex").to(torch_device).eval()

    def read_frames(path: Path) -> list[np.ndarray]:
        capture = cv2.VideoCapture(str(path))
        frames: list[np.ndarray] = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        capture.release()
        if len(frames) != 81:
            raise ValueError(f"expected 81 decoded frames, got {len(frames)} for {path}")
        return frames

    def frame_metrics(left: Path, right: Path, mode: str) -> dict[str, float]:
        if mode not in ("diversity", "paired"):
            raise ValueError(f"unsupported frame metric mode: {mode}")
        left_frames = read_frames(left)
        right_frames = read_frames(right)
        l1_values: list[float] = []
        lpips_values: list[float] = []
        ssim_values: list[float] = []
        psnr_values: list[float] = []
        for left_frame, right_frame in zip(left_frames, right_frames, strict=True):
            left_float = left_frame.astype(np.float32) / 255.0
            right_float = right_frame.astype(np.float32) / 255.0
            if mode == "diversity":
                l1_values.append(float(np.mean(np.abs(left_float - right_float))))
            else:
                ssim_values.append(
                    float(
                        structural_similarity(
                            left_float, right_float, data_range=1.0, channel_axis=2
                        )
                    )
                )
                psnr_values.append(
                    float(peak_signal_noise_ratio(left_float, right_float, data_range=1.0))
                )
        if mode == "paired":
            return {
                "ssim": statistics.fmean(ssim_values),
                "psnr": statistics.fmean(psnr_values),
            }
        for start in range(0, len(left_frames), 8):
            left_batch = np.stack(left_frames[start : start + 8]).astype(np.float32)
            right_batch = np.stack(right_frames[start : start + 8]).astype(np.float32)
            left_tensor = torch.from_numpy(left_batch).permute(0, 3, 1, 2)
            right_tensor = torch.from_numpy(right_batch).permute(0, 3, 1, 2)
            left_tensor = (left_tensor.to(torch_device) / 127.5) - 1.0
            right_tensor = (right_tensor.to(torch_device) / 127.5) - 1.0
            with torch.no_grad():
                values = lpips_model(left_tensor, right_tensor).flatten().tolist()
            lpips_values.extend(float(value) for value in values)
        return {
            "frame_l1": statistics.fmean(l1_values),
            "frame_lpips": statistics.fmean(lpips_values),
        }

    diversity_rows: list[dict[str, Any]] = []
    for method in METHODS:
        for prompt_index in range(4):
            left = record_index[(method, prompt_index, seeds[0])]
            right = record_index[(method, prompt_index, seeds[1])]
            metrics = frame_metrics(left.path, right.path, mode="diversity")
            cosine = torch.nn.functional.cosine_similarity(
                embedding_cache[left.path].unsqueeze(0),
                embedding_cache[right.path].unsqueeze(0),
            ).item()
            diversity_rows.append(
                {
                    "method": method,
                    "prompt_index": prompt_index,
                    "video_embedding_distance": 1.0 - float(cosine),
                    **metrics,
                }
            )

    paired_rows: list[dict[str, Any]] = []
    for method in METHODS[1:]:
        for prompt_index in range(4):
            for seed in seeds:
                teacher = record_index[("teacher20", prompt_index, seed)]
                candidate = record_index[(method, prompt_index, seed)]
                paired_rows.append(
                    {
                        "method": method,
                        "prompt_index": prompt_index,
                        "seed": seed,
                        **frame_metrics(teacher.path, candidate.path, mode="paired"),
                    }
                )
    payload = {
        "diversity_rows": diversity_rows,
        "diversity_gate": summarize_diversity(diversity_rows),
        "paired_diagnostics": paired_rows,
    }
    atomic_write_json(work_dir / "diagnostics.json", payload)


def summarize_all(work_dir: Path) -> None:
    vbench = summarize_vbench(work_dir)
    diagnostics = json.loads((work_dir / "diagnostics.json").read_text(encoding="utf-8"))
    methods: dict[str, Any] = {}
    for method in METHODS[1:]:
        quality = vbench["aggregate"][method]
        diversity = diagnostics["diversity_gate"][method]
        methods[method] = {
            "quality": quality,
            "diversity": diversity,
            "passes_quality_and_diversity": (
                quality["passes_mean_0_90"]
                and quality["passes_minimum_0_80"]
                and diversity["passes_three_of_four_0_70"]
                and diversity["passes_no_prompt_below_0_50"]
            ),
        }
    atomic_write_json(
        work_dir / "quality_gate_summary.json",
        {"methods": methods, "vbench": vbench, "diagnostics": diagnostics},
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config.resolve())
    records = discover_formal_videos(config, args.formal_root.resolve())
    args.work_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "prepare":
        prepare_vbench_input(records, args.work_dir.resolve())
    elif args.phase == "vbench":
        if args.vbench_full_info is None:
            raise ValueError("--vbench-full-info is required for the VBench phase")
        run_vbench(
            args.work_dir.resolve(),
            args.vbench_full_info,
            args.device,
            tuple(args.dimensions),
            args.resume,
        )
    elif args.phase == "diagnostics":
        run_diagnostics(records, args.work_dir.resolve(), args.device)
    else:
        summarize_all(args.work_dir.resolve())


if __name__ == "__main__":
    main()

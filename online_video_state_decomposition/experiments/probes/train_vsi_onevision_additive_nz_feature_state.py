from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from probe_vsi_onevision_query_fixed_positive_gaussian_measure import LAYERS
from vsi_onevision_protocol import PROTOCOL_ID


TRAIN_POSITIONS = range(1, 73)
DEVELOPMENT_POSITIONS = range(73, 97)
ROLE = "calibration_development_additive_nz_feature_state"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--feature-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--evaluation-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument("--device", default="cuda:1")
    return parser.parse_args()


class PositiveFeatureState(nn.Module):
    def __init__(self, *, head_count: int, head_dim: int, feature_width: int) -> None:
        super().__init__()
        self.query_weight = nn.Parameter(
            torch.empty(head_count, head_dim, feature_width)
        )
        self.key_weight = nn.Parameter(torch.empty(head_count, head_dim, feature_width))
        self.query_bias = nn.Parameter(torch.zeros(head_count, feature_width))
        self.key_bias = nn.Parameter(torch.zeros(head_count, feature_width))
        self.log_visual_scale = nn.Parameter(torch.zeros(head_count))
        self.register_buffer("query_rms", torch.ones(head_count, 1))
        self.register_buffer("key_rms", torch.ones(head_count, 1))
        nn.init.normal_(self.query_weight, std=1.0 / math.sqrt(head_dim))
        nn.init.normal_(self.key_weight, std=1.0 / math.sqrt(head_dim))

    @staticmethod
    def positive(projection: torch.Tensor) -> torch.Tensor:
        return F.softplus(projection) + 1e-4

    def visual_state(
        self, key: torch.Tensor, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normalized_key = key / self.key_rms[None, :, None, :]
        key_feature = self.positive(
            torch.einsum("bhnd,hdr->bhnr", normalized_key, self.key_weight)
            + self.key_bias[None, :, None, :]
        )
        state = torch.einsum("bhnr,bhnd->bhrd", key_feature, value)
        normalizer = key_feature.sum(dim=2)
        return state, normalizer

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        nonvisual_z: torch.Tensor,
        nonvisual_n: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state, normalizer = self.visual_state(key, value)
        normalized_query = query / self.query_rms[None, :, :]
        query_feature = self.positive(
            torch.einsum("bhd,hdr->bhr", normalized_query, self.query_weight)
            + self.query_bias[None, :, :]
        )
        visual_z = torch.einsum("bhr,bhr->bh", query_feature, normalizer)
        visual_n = torch.einsum("bhr,bhrd->bhd", query_feature, state)
        scale = torch.exp(self.log_visual_scale).unsqueeze(0)
        visual_z = visual_z * scale
        visual_n = visual_n * scale.unsqueeze(-1)
        visual_output = visual_n / visual_z.unsqueeze(-1).clamp_min(1e-8)
        full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(
            -1
        ).clamp_min(1e-8)
        return visual_output, full_output, visual_z


def capture_paths(capture_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in sorted(capture_dir.glob("position_*.pt")):
        position = int(path.name.split("_", 2)[1])
        if position in paths:
            raise ValueError(f"duplicate capture position: {position}")
        paths[position] = path
    expected = set(range(1, 97))
    if set(paths) != expected:
        raise ValueError("additive-N/Z trainer requires exactly positions 1--96")
    return paths


def cpu_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().to(device="cpu", copy=True)
        for key, value in model.state_dict().items()
    }


def load_examples(
    paths: dict[int, Path], *, positions: range, layer_index: int
) -> list[dict[str, torch.Tensor]]:
    examples = []
    for position in positions:
        payload = torch.load(paths[position], map_location="cpu", weights_only=False)
        if (
            payload["protocol_id"] != PROTOCOL_ID
            or payload["sample_position"] != position
        ):
            raise ValueError("additive-N/Z capture identity mismatch")
        layer = payload["layers"][layer_index]
        examples.append({key: value for key, value in layer.items()})
    return examples


def batch_examples(
    examples: list[dict[str, torch.Tensor]],
    indices: np.ndarray,
    *,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    keys = examples[0]
    return {
        key: torch.stack([examples[int(index)][key] for index in indices])
        .to(device=device, dtype=torch.float32)
        .contiguous()
        for key in keys
    }


def relative_square_error(
    prediction: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    numerator = (prediction - target).square().sum(dim=-1)
    denominator = target.square().sum(dim=-1).clamp_min(1e-8)
    return numerator / denominator


def objective(
    model: PositiveFeatureState, batch: dict[str, torch.Tensor]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    visual, full, visual_z = model(
        batch["query_scaled"],
        batch["visual_key"],
        batch["visual_value"],
        batch["nonvisual_z"],
        batch["nonvisual_n"],
    )
    visual_error = relative_square_error(visual, batch["exact_visual_output"])
    full_error = relative_square_error(full, batch["exact_full_output"])
    log_mass_error = (
        torch.log(visual_z.clamp_min(1e-8))
        - torch.log(batch["exact_visual_z"].clamp_min(1e-8))
    ).square()
    loss = visual_error.mean() + full_error.mean() + 0.05 * log_mass_error.mean()
    return loss, {
        "visual_error": visual_error,
        "full_error": full_error,
        "log_mass_error": log_mass_error,
    }


def fit_normalization(
    model: PositiveFeatureState, examples: list[dict[str, torch.Tensor]]
) -> None:
    query_square = torch.stack(
        [example["query_scaled"].float().square().mean(dim=-1) for example in examples]
    ).mean(dim=0)
    key_square = torch.stack(
        [
            example["visual_key"].float().square().mean(dim=(1, 2))
            for example in examples
        ]
    ).mean(dim=0)
    model.query_rms.copy_(query_square.sqrt().clamp_min(1e-6).unsqueeze(-1))
    model.key_rms.copy_(key_square.sqrt().clamp_min(1e-6).unsqueeze(-1))


def initialize_visual_scale(
    model: PositiveFeatureState,
    examples: list[dict[str, torch.Tensor]],
    *,
    device: torch.device,
) -> None:
    differences = []
    with torch.no_grad():
        for start in range(0, len(examples), 2):
            indices = np.arange(start, min(start + 2, len(examples)))
            batch = batch_examples(examples, indices, device=device)
            state, normalizer = model.visual_state(
                batch["visual_key"], batch["visual_value"]
            )
            del state
            query_feature = model.positive(
                torch.einsum(
                    "bhd,hdr->bhr",
                    batch["query_scaled"] / model.query_rms[None, :, :],
                    model.query_weight,
                )
                + model.query_bias[None, :, :]
            )
            raw_z = torch.einsum("bhr,bhr->bh", query_feature, normalizer)
            differences.append(
                torch.log(batch["exact_visual_z"].clamp_min(1e-8))
                - torch.log(raw_z.clamp_min(1e-8))
            )
        model.log_visual_scale.copy_(torch.cat(differences).mean(dim=0))


def evaluate(
    model: PositiveFeatureState,
    examples: list[dict[str, torch.Tensor]],
    *,
    layer_index: int,
    split: str,
    device: torch.device,
) -> list[dict[str, object]]:
    rows = []
    model.eval()
    with torch.no_grad():
        for index, example in enumerate(examples):
            batch = batch_examples(examples, np.asarray([index]), device=device)
            visual, full, visual_z = model(
                batch["query_scaled"],
                batch["visual_key"],
                batch["visual_value"],
                batch["nonvisual_z"],
                batch["nonvisual_n"],
            )
            head_visual = relative_square_error(
                visual, batch["exact_visual_output"]
            ).sqrt()
            visual_relative = torch.linalg.vector_norm(
                visual - batch["exact_visual_output"]
            ) / torch.linalg.vector_norm(batch["exact_visual_output"]).clamp_min(1e-8)
            full_relative = torch.linalg.vector_norm(
                full - batch["exact_full_output"]
            ) / torch.linalg.vector_norm(batch["exact_full_output"]).clamp_min(1e-8)
            log_mass = (
                torch.log(visual_z.clamp_min(1e-8))
                - torch.log(batch["exact_visual_z"].clamp_min(1e-8))
            ).abs()
            rows.append(
                {
                    "split": split,
                    "position": index + (1 if split == "train" else 73),
                    "layer_index": layer_index,
                    "visual_relative_l2": float(visual_relative.item()),
                    "visual_worst_head_relative_l2": float(head_visual.max().item()),
                    "full_relative_l2": float(full_relative.item()),
                    "log_visual_mass_absolute_mean": float(log_mass.mean().item()),
                }
            )
    return rows


def summarize(rows: list[dict[str, object]]) -> dict[str, float | int]:
    visual = np.asarray([float(row["visual_relative_l2"]) for row in rows])
    full = np.asarray([float(row["full_relative_l2"]) for row in rows])
    return {
        "cell_count": len(rows),
        "visual_mean": float(visual.mean()),
        "visual_p95": float(np.quantile(visual, 0.95)),
        "visual_worst": float(visual.max()),
        "visual_worst_head": max(
            float(row["visual_worst_head_relative_l2"]) for row in rows
        ),
        "full_mean": float(full.mean()),
        "full_p95": float(np.quantile(full, 0.95)),
        "full_worst": float(full.max()),
        "log_visual_mass_absolute_mean": float(
            np.mean([float(row["log_visual_mass_absolute_mean"]) for row in rows])
        ),
    }


def classify(
    *,
    baseline: dict[str, float | int],
    learned: dict[str, float | int],
    state_ratio: float,
) -> str:
    improvement = 1.0 - float(learned["visual_mean"]) / float(baseline["visual_mean"])
    strict = (
        int(learned["cell_count"]) == 72
        and float(learned["visual_mean"]) <= 0.01
        and float(learned["visual_p95"]) <= 0.02
        and float(learned["visual_worst"]) <= 0.05
        and float(learned["full_mean"]) <= 0.005
        and float(learned["full_p95"]) <= 0.01
        and improvement >= 0.5
        and state_ratio >= 32.0
    )
    if strict:
        return "ADDITIVE_NZ_DEV_GO"
    if (
        float(learned["visual_mean"]) <= 0.05
        and float(learned["visual_p95"]) <= 0.10
        and improvement >= 0.5
    ):
        return "ADDITIVE_NZ_CAPACITY_SIGNAL"
    return "NO_ADDITIVE_NZ_FEATURE_STATE"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty additive-N/Z rows")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    if (
        args.feature_width != 32
        or args.steps != 1000
        or args.batch_size != 2
        or args.learning_rate != 1e-3
        or args.evaluation_interval != 100
        or args.seed != 20260830
    ):
        raise ValueError("registered additive-N/Z training identity changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("additive-N/Z output must be empty")
    paths = capture_paths(args.capture_dir)
    capture_summary = json.loads(
        (args.capture_dir / "summary.json").read_text(encoding="utf-8")
    )
    if capture_summary["protocol_id"] != PROTOCOL_ID:
        raise ValueError("capture protocol identity mismatch")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    history_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    learned_rows: list[dict[str, object]] = []
    started = time.perf_counter()

    for layer_index in LAYERS:
        train_examples = load_examples(
            paths, positions=TRAIN_POSITIONS, layer_index=layer_index
        )
        development_examples = load_examples(
            paths, positions=DEVELOPMENT_POSITIONS, layer_index=layer_index
        )
        head_count, head_dim = train_examples[0]["query_scaled"].shape
        model = PositiveFeatureState(
            head_count=head_count,
            head_dim=head_dim,
            feature_width=args.feature_width,
        ).to(device)
        fit_normalization(model, train_examples)
        initialize_visual_scale(model, train_examples, device=device)
        initial_state = cpu_state_dict(model)
        baseline_rows.extend(
            evaluate(
                model,
                development_examples,
                layer_index=layer_index,
                split="development",
                device=device,
            )
        )
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.learning_rate, weight_decay=1e-4
        )
        best_visual = math.inf
        best_state = cpu_state_dict(model)
        for step in range(1, args.steps + 1):
            model.train()
            indices = rng.integers(0, len(train_examples), size=args.batch_size)
            batch = batch_examples(train_examples, indices, device=device)
            loss, metrics = objective(model, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            if step == 1 or step % args.evaluation_interval == 0:
                development_rows = evaluate(
                    model,
                    development_examples,
                    layer_index=layer_index,
                    split="development",
                    device=device,
                )
                development = summarize(development_rows)
                history_rows.append(
                    {
                        "layer_index": layer_index,
                        "step": step,
                        "train_loss": float(loss.item()),
                        "train_visual_error": float(
                            metrics["visual_error"].mean().sqrt().item()
                        ),
                        "development_visual_mean": development["visual_mean"],
                        "development_visual_p95": development["visual_p95"],
                        "development_full_mean": development["full_mean"],
                    }
                )
                if float(development["visual_mean"]) < best_visual:
                    best_visual = float(development["visual_mean"])
                    best_state = cpu_state_dict(model)
        model.load_state_dict(best_state)
        learned_rows.extend(
            evaluate(
                model,
                development_examples,
                layer_index=layer_index,
                split="development",
                device=device,
            )
        )
        torch.save(
            {
                "layer_index": layer_index,
                "feature_width": args.feature_width,
                "initial_state": initial_state,
                "best_state": best_state,
            },
            args.out_dir / f"layer_{layer_index:02d}_feature_state.pt",
        )

    baseline_summary = summarize(baseline_rows)
    learned_summary = summarize(learned_rows)
    token_count = 8 * 196
    head_dim = 128
    state_ratio = (2 * token_count * head_dim) / (args.feature_width * (head_dim + 1))
    decision = classify(
        baseline=baseline_summary,
        learned=learned_summary,
        state_ratio=state_ratio,
    )
    write_csv(args.out_dir / "baseline_development_rows.csv", baseline_rows)
    write_csv(args.out_dir / "learned_development_rows.csv", learned_rows)
    write_csv(args.out_dir / "training_history.csv", history_rows)
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": ROLE,
        "decision": decision,
        "feature_width": args.feature_width,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "train_positions": [1, 72],
        "development_positions": [73, 96],
        "confirmation_positions_unread": [97, 120],
        "layers": list(LAYERS),
        "baseline": baseline_summary,
        "learned": learned_summary,
        "visual_mean_relative_improvement": 1.0
        - float(learned_summary["visual_mean"])
        / float(baseline_summary["visual_mean"]),
        "analytic_state_ratio": state_ratio,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration-development additive-state capacity only. Confirmation "
            "positions 97-120, official selection/formal, reader task behavior, "
            "writer cost, kernel latency, and wall-clock remain untested."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

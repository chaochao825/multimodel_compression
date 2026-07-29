#!/usr/bin/env python3
"""Probe a calibrated Q/K/V-conditioned sparse-linear tail on Wan F81 captures.

This experiment keeps three evidence levels separate:

* proxy: support is generated from the current Q/K/V without dense attention;
* frozen-tail oracle: the feature map is calibration-frozen, but held-out dense
  AV may choose exact support to isolate tail transfer;
* transductive capacity: both fitting and support may inspect all evaluated
  captures and therefore cannot support a deployment or generalization claim.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from content_generated_tail_core import (
    GroupLayout,
    PositiveLinearTail,
    adaptive_rank_residual,
    contiguous_layout,
    group_corrections,
    layout_tokens_padded,
    output_for_group_selection,
    proxy_group_selection,
    semantic_layout,
    trajectory_width_selection,
)


@dataclass
class CaptureData:
    sample_id: str
    path: Path
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    softmax_scale: float
    metadata: dict[str, Any]


@dataclass
class TileInstance:
    sample_id: str
    tile_index: int
    query_start: int
    q: torch.Tensor
    reference: torch.Tensor
    log_partition: torch.Tensor
    capture: CaptureData

    @property
    def key(self) -> tuple[str, int]:
        return self.sample_id, self.tile_index


@dataclass
class Support:
    tokens: torch.Tensor
    valid: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-index", type=Path, required=True)
    parser.add_argument("--protocol-config", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--run-kind", choices=("smoke", "diagnostic"), default="diagnostic")
    parser.add_argument("--execution-resource-note", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**31)


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise ValueError("CSV rows do not share a stable schema")
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def load_protocol(path: Path, rank: int) -> dict[str, Any]:
    protocol = json.loads(path.read_text(encoding="utf-8"))
    if int(protocol.get("schema_version", -1)) != 1:
        raise ValueError("unsupported protocol schema")
    scope = protocol["scope"]
    if rank not in map(int, scope["ranks"]):
        raise ValueError(f"rank {rank} is not registered in the protocol")
    split = protocol["split"]
    groups = [
        *map(str, split["calibration_sample_ids"]),
        *map(str, split["validation_sample_ids"]),
        *map(str, split["test_sample_ids"]),
    ]
    if len(groups) != len(set(groups)):
        raise ValueError("calibration, validation, and test IDs must be disjoint")
    if set(groups) != set(map(str, scope["sample_ids"])):
        raise ValueError("split IDs must exactly cover scope.sample_ids")
    if float(scope["density"]) <= 0 or float(scope["density"]) > 0.25:
        raise ValueError("registered density must be in (0, 0.25]")
    return protocol


def capture_paths(index_path: Path, protocol: dict[str, Any]) -> dict[str, Path]:
    scope = protocol["scope"]
    wanted = set(map(str, scope["sample_ids"]))
    matches: dict[str, list[Path]] = defaultdict(list)
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (
                row["sample_id"] in wanted
                and int(row["layer"]) == int(scope["layer"])
                and int(row["sampling_step"]) == int(scope["sampling_step"])
                and row["branch"] == str(scope["branch"])
            ):
                matches[row["sample_id"]].append(Path(row["path"]))
    output: dict[str, Path] = {}
    for sample_id in sorted(wanted):
        paths = matches[sample_id]
        if len(paths) != 1 or not paths[0].is_file():
            raise FileNotFoundError(f"expected one capture for {sample_id}; found {paths}")
        output[sample_id] = paths[0].resolve()
    return output


def load_captures(
    paths: dict[str, Path], device: torch.device
) -> dict[str, CaptureData]:
    output = {}
    shape: tuple[int, ...] | None = None
    for sample_id, path in paths.items():
        payload = torch.load(path, map_location="cpu", weights_only=False)
        q, k, v = (payload[name][0] for name in ("q", "k", "v"))
        if q.shape != k.shape or q.shape != v.shape:
            raise ValueError(f"Q/K/V shape mismatch in {path}")
        if shape is None:
            shape = tuple(q.shape)
        elif tuple(q.shape) != shape:
            raise ValueError("captures do not share one Q/K/V shape")
        # Wan captures are [tokens, heads, channels]; the probe is head-major.
        output[sample_id] = CaptureData(
            sample_id=sample_id,
            path=path,
            q=q.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32),
            k=k.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32),
            v=v.permute(1, 0, 2).contiguous().to(device=device, dtype=torch.float32),
            softmax_scale=float(payload.get("softmax_scale", q.shape[-1] ** -0.5)),
            metadata=dict(payload.get("metadata", {})),
        )
        del payload, q, k, v
        print(f"[content-tail] loaded sample={sample_id} path={path}", flush=True)
    return output


def query_starts(tokens: int, size: int, count: int) -> tuple[int, ...]:
    full = tokens // size
    if not 0 < count <= full:
        raise ValueError("invalid query tile count")
    if count == 1:
        return (0,)
    ids = torch.linspace(0, full - 1, count, dtype=torch.float64).round().long()
    if ids.unique().numel() != count:
        raise RuntimeError("stratified query tile selection produced duplicates")
    return tuple(int(index) * size for index in ids)


@torch.no_grad()
def build_instances(
    captures: dict[str, CaptureData], protocol: dict[str, Any]
) -> dict[tuple[str, int], TileInstance]:
    scope = protocol["scope"]
    first = next(iter(captures.values()))
    starts = query_starts(first.q.shape[1], int(scope["query_tile_size"]), int(scope["query_tiles"]))
    instances = {}
    for sample_id, capture in captures.items():
        for tile_index, start in enumerate(starts):
            q = capture.q[:, start : start + int(scope["query_tile_size"])]
            scores = torch.einsum("hqd,hnd->hqn", q, capture.k) * capture.softmax_scale
            log_partition = torch.logsumexp(scores, dim=2)
            reference = torch.softmax(scores, dim=2) @ capture.v
            instance = TileInstance(
                sample_id=sample_id,
                tile_index=tile_index,
                query_start=start,
                q=q,
                reference=reference,
                log_partition=log_partition,
                capture=capture,
            )
            instances[instance.key] = instance
            del scores
    return instances


def rms_statistics(
    captures: dict[str, CaptureData], sample_ids: list[str]
) -> tuple[torch.Tensor, torch.Tensor]:
    q_energy = None
    k_energy = None
    count = 0
    for sample_id in sample_ids:
        capture = captures[sample_id]
        current_q = capture.q.square().sum(dim=(1, 2))
        current_k = capture.k.square().sum(dim=(1, 2))
        q_energy = current_q if q_energy is None else q_energy + current_q
        k_energy = current_k if k_energy is None else k_energy + current_k
        count += capture.q.shape[1] * capture.q.shape[2]
    assert q_energy is not None and k_energy is not None
    return (q_energy / count).sqrt(), (k_energy / count).sqrt()


def budget_for(instance: TileInstance, protocol: dict[str, Any]) -> int:
    scope = protocol["scope"]
    tokens = instance.capture.k.shape[1]
    return max(1, round(float(scope["density"]) * tokens / int(scope["block_size"])))


@torch.no_grad()
def proxy_support(instance: TileInstance, protocol: dict[str, Any]) -> Support:
    width = int(protocol["scope"]["block_size"])
    heads, tokens, _ = instance.capture.k.shape
    layout = contiguous_layout(tokens, width, instance.q.device)
    budget = budget_for(instance, protocol)
    token_rows = []
    valid_rows = []
    for head in range(heads):
        selected = proxy_group_selection(
            layout,
            instance.q[head],
            instance.capture.k[head],
            instance.capture.v[head],
            budget,
        )
        tokens_row, valid_row = layout_tokens_padded(layout, selected)
        token_rows.append(tokens_row)
        valid_rows.append(valid_row)
    return Support(torch.stack(token_rows), torch.stack(valid_rows))


def initial_supports(
    instances: dict[tuple[str, int], TileInstance],
    sample_ids: list[str],
    protocol: dict[str, Any],
) -> dict[tuple[str, int], Support]:
    return {
        key: proxy_support(instance, protocol)
        for key, instance in instances.items()
        if instance.sample_id in sample_ids
    }


def per_head_relative_loss(output: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    residual = (output - reference).square().sum(dim=(1, 2))
    scale = reference.square().sum(dim=(1, 2)).clamp_min(1e-12)
    return residual / scale


def post_rank_energy(
    output: torch.Tensor, reference: torch.Tensor, rank: int
) -> torch.Tensor:
    defect = reference - output
    if rank <= 0:
        return defect.square().sum()
    return adaptive_rank_residual(defect, rank).square().sum()


@torch.no_grad()
def mean_training_objective(
    model: PositiveLinearTail,
    instances: dict[tuple[str, int], TileInstance],
    supports: dict[tuple[str, int], Support],
    protocol: dict[str, Any],
    post_rank: int,
) -> float:
    values = []
    for key in sorted(supports):
        instance = instances[key]
        support = supports[key]
        output, denominator = model(
            instance.q,
            instance.capture.k,
            instance.capture.v,
            support.tokens,
            instance.capture.softmax_scale,
            support.valid,
        )
        raw = per_head_relative_loss(output, instance.reference)
        if post_rank > 0:
            defect = instance.reference - output
            singular = torch.linalg.svdvals(defect)
            used = min(post_rank, singular.shape[1])
            objective = singular[:, used:].square().sum(dim=1) / instance.reference.square().sum(
                dim=(1, 2)
            ).clamp_min(1e-12)
        else:
            objective = raw
        partition = (
            denominator.clamp_min(1e-12).log() - instance.log_partition
        ).square().mean()
        values.append(
            objective.mean() + float(protocol["training"]["partition_loss_weight"]) * partition
        )
    return float(torch.stack(values).mean())


def train_steps(
    model: PositiveLinearTail,
    instances: dict[tuple[str, int], TileInstance],
    supports: dict[tuple[str, int], Support],
    steps: int,
    protocol: dict[str, Any],
    stage: str,
    trace: list[dict[str, object]],
    post_rank: int = 0,
) -> None:
    if steps <= 0:
        return
    training = protocol["training"]
    keys = sorted(supports)
    learning_rate = float(
        training["refine_learning_rate"] if post_rank > 0 else training["learning_rate"]
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=float(training["weight_decay"]),
    )
    model.train()
    best_value = mean_training_objective(model, instances, supports, protocol, post_rank)
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    for step in range(steps):
        key = keys[step % len(keys)]
        instance = instances[key]
        support = supports[key]
        optimizer.zero_grad(set_to_none=True)
        output, denominator = model(
            instance.q,
            instance.capture.k,
            instance.capture.v,
            support.tokens,
            instance.capture.softmax_scale,
            support.valid,
        )
        raw_output_loss = per_head_relative_loss(output, instance.reference).mean()
        if post_rank > 0:
            defect = instance.reference - output
            singular = torch.linalg.svdvals(defect)
            used = min(post_rank, singular.shape[1])
            residual_energy = singular[:, used:].square().sum(dim=1)
            reference_energy = instance.reference.square().sum(dim=(1, 2)).clamp_min(1e-12)
            objective_loss = (residual_energy / reference_energy).mean()
        else:
            objective_loss = raw_output_loss
        partition_loss = (
            denominator.clamp_min(1e-12).log() - instance.log_partition
        ).square().mean()
        loss = objective_loss + float(training["partition_loss_weight"]) * partition_loss
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss in {stage} step {step}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), float(training["gradient_clip_norm"])
        )
        optimizer.step()
        if (step + 1) % len(keys) == 0 or step + 1 == steps:
            candidate_value = mean_training_objective(
                model, instances, supports, protocol, post_rank
            )
            if candidate_value < best_value:
                best_value = candidate_value
                best_state = {
                    key: value.detach().clone() for key, value in model.state_dict().items()
                }
        if step == 0 or (step + 1) % int(training["log_every_steps"]) == 0 or step + 1 == steps:
            row = {
                "stage": stage,
                "step": step + 1,
                "sample_id": instance.sample_id,
                "tile_index": instance.tile_index,
                "post_rank_objective": post_rank,
                "learning_rate": learning_rate,
                "loss": float(loss.detach()),
                "objective_loss": float(objective_loss.detach()),
                "raw_output_loss": float(raw_output_loss.detach()),
                "partition_loss": float(partition_loss.detach()),
                "gradient_norm": float(gradient_norm),
                "kernel_scale_min": float(model.log_kernel_scale.detach().exp().min()),
                "kernel_scale_max": float(model.log_kernel_scale.detach().exp().max()),
                "best_full_training_objective": best_value,
            }
            trace.append(row)
            print(
                f"[content-tail] {stage} step={step + 1}/{steps} "
                f"objective={row['objective_loss']:.6f} raw={row['raw_output_loss']:.6f} "
                f"partition={row['partition_loss']:.6f}",
                flush=True,
            )
    model.load_state_dict(best_state)


@torch.no_grad()
def trajectory_support(
    model: PositiveLinearTail,
    instance: TileInstance,
    protocol: dict[str, Any],
) -> Support:
    width = int(protocol["scope"]["block_size"])
    layout = contiguous_layout(instance.capture.k.shape[1], width, instance.q.device)
    budget = budget_for(instance, protocol)
    add_chunk = int(protocol["routers"]["trajectory_add_chunk"])
    token_rows = []
    valid_rows = []
    for head in range(instance.q.shape[0]):
        state = group_corrections(
            model,
            head,
            instance.q[head],
            instance.capture.k[head],
            instance.capture.v[head],
            layout,
            instance.capture.softmax_scale,
        )
        selected = trajectory_width_selection(
            instance.reference[head],
            *state,
            budget=budget,
            add_chunk=add_chunk,
            post_rank=int(protocol["gates"]["post_tail_adaptive_rank"]),
        )
        proxy_selected = proxy_group_selection(
            layout,
            instance.q[head],
            instance.capture.k[head],
            instance.capture.v[head],
            budget,
        )
        selected_output = output_for_group_selection(*state, selected)
        proxy_output = output_for_group_selection(*state, proxy_selected)
        if post_rank_energy(
            proxy_output,
            instance.reference[head],
            int(protocol["gates"]["post_tail_adaptive_rank"]),
        ) <= post_rank_energy(
            selected_output,
            instance.reference[head],
            int(protocol["gates"]["post_tail_adaptive_rank"]),
        ):
            selected = proxy_selected
        tokens_row, valid_row = layout_tokens_padded(layout, selected)
        token_rows.append(tokens_row)
        valid_rows.append(valid_row)
    return Support(torch.stack(token_rows), torch.stack(valid_rows))


@torch.no_grad()
def refine_supports(
    model: PositiveLinearTail,
    instances: dict[tuple[str, int], TileInstance],
    sample_ids: list[str],
    protocol: dict[str, Any],
) -> dict[tuple[str, int], Support]:
    output = {}
    for key in sorted(instances):
        instance = instances[key]
        if instance.sample_id not in sample_ids:
            continue
        output[key] = trajectory_support(model, instance, protocol)
        print(
            f"[content-tail] refined-support sample={instance.sample_id} tile={instance.tile_index}",
            flush=True,
        )
    return output


def train_variant(
    variant: str,
    rank: int,
    captures: dict[str, CaptureData],
    instances: dict[tuple[str, int], TileInstance],
    sample_ids: list[str],
    protocol: dict[str, Any],
) -> tuple[PositiveLinearTail, list[dict[str, object]], dict[tuple[str, int], Support]]:
    q_rms, k_rms = rms_statistics(captures, sample_ids)
    first = next(iter(captures.values()))
    model = PositiveLinearTail(
        first.q.shape[0],
        first.q.shape[2],
        rank,
        q_rms,
        k_rms,
        seed=stable_seed(protocol["seed"], variant, rank),
    ).to(first.q.device)
    trace: list[dict[str, object]] = []
    supports = initial_supports(instances, sample_ids, protocol)
    train_steps(
        model,
        instances,
        supports,
        int(protocol["training"]["initial_steps"]),
        protocol,
        f"{variant}_proxy_support",
        trace,
        post_rank=0,
    )
    supports = refine_supports(model, instances, sample_ids, protocol)
    train_steps(
        model,
        instances,
        supports,
        int(protocol["training"]["refine_steps"]),
        protocol,
        f"{variant}_trajectory_support",
        trace,
        post_rank=int(protocol["gates"]["post_tail_adaptive_rank"]),
    )
    model.eval()
    return model, trace, supports


def layouts_for_head(
    instance: TileInstance, head: int, protocol: dict[str, Any]
) -> tuple[GroupLayout, ...]:
    width = int(protocol["scope"]["block_size"])
    k = instance.capture.k[head]
    v = instance.capture.v[head]
    return (
        contiguous_layout(k.shape[0], width, k.device),
        semantic_layout(instance.q[head], k, v, width, "semantic_qk"),
        semantic_layout(
            instance.q[head],
            k,
            v,
            width,
            "value_aware",
            float(protocol["routers"]["value_weight"]),
        ),
    )


def route_names(layout: GroupLayout) -> tuple[str, str]:
    if layout.name == "fixed64":
        return "proxy_fixed64_qk", "oracle_trajectory_width_fixed64"
    if layout.name == "svg2_style_semantic64_proxy":
        return "proxy_svg2_style_semantic64", "oracle_trajectory_width_semantic64"
    if layout.name == "value_aware_semantic64_proxy":
        return "proxy_value_aware_semantic64", "oracle_trajectory_width_value64"
    raise ValueError(f"unknown layout name: {layout.name}")


@torch.inference_mode()
def evaluate_sample(
    model: PositiveLinearTail,
    variant: str,
    sample_id: str,
    instances: dict[tuple[str, int], TileInstance],
    protocol: dict[str, Any],
    split_name: str,
) -> list[dict[str, object]]:
    sample_instances = sorted(
        (instance for instance in instances.values() if instance.sample_id == sample_id),
        key=lambda item: item.tile_index,
    )
    if not sample_instances:
        raise ValueError(f"no instances for sample {sample_id}")
    heads = sample_instances[0].q.shape[0]
    defects: dict[tuple[str, int], list[torch.Tensor]] = defaultdict(list)
    references: dict[int, list[torch.Tensor]] = defaultdict(list)
    per_tile_adaptive: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_tile_adaptive_sq: dict[tuple[str, int], list[float]] = defaultdict(list)
    per_tile_reference_sq: dict[tuple[str, int], list[float]] = defaultdict(list)
    family_choice: dict[tuple[str, int], list[str]] = defaultdict(list)
    selection_signatures: dict[tuple[str, int], list[str]] = defaultdict(list)
    budget = budget_for(sample_instances[0], protocol)
    add_chunk = int(protocol["routers"]["trajectory_add_chunk"])

    for instance in sample_instances:
        for head in range(heads):
            reference = instance.reference[head]
            references[head].append(reference)
            oracle_outputs: list[tuple[float, str, torch.Tensor, torch.Tensor]] = []
            for layout in layouts_for_head(instance, head, protocol):
                state = group_corrections(
                    model,
                    head,
                    instance.q[head],
                    instance.capture.k[head],
                    instance.capture.v[head],
                    layout,
                    instance.capture.softmax_scale,
                )
                proxy_selected = proxy_group_selection(
                    layout,
                    instance.q[head],
                    instance.capture.k[head],
                    instance.capture.v[head],
                    budget,
                    float(protocol["routers"]["value_weight"]),
                )
                oracle_selected = trajectory_width_selection(
                    reference,
                    *state,
                    budget=budget,
                    add_chunk=add_chunk,
                    post_rank=int(protocol["gates"]["post_tail_adaptive_rank"]),
                )
                proxy_output = output_for_group_selection(*state, proxy_selected)
                oracle_output = output_for_group_selection(*state, oracle_selected)
                adaptive_rank = int(protocol["gates"]["post_tail_adaptive_rank"])
                if post_rank_energy(proxy_output, reference, adaptive_rank) <= post_rank_energy(
                    oracle_output, reference, adaptive_rank
                ):
                    oracle_selected = proxy_selected
                    oracle_output = proxy_output
                proxy_name, oracle_name = route_names(layout)
                for name, output, selected in (
                    (proxy_name, proxy_output, proxy_selected),
                    (oracle_name, oracle_output, oracle_selected),
                ):
                    defect = reference - output
                    defects[(name, head)].append(defect)
                    tile_residual = adaptive_rank_residual(
                        defect, int(protocol["gates"]["post_tail_adaptive_rank"])
                    )
                    per_tile_adaptive[(name, head)].append(
                        math.sqrt(
                            float(tile_residual.square().sum())
                            / max(float(reference.square().sum()), 1e-30)
                        )
                    )
                    per_tile_adaptive_sq[(name, head)].append(
                        float(tile_residual.square().sum())
                    )
                    per_tile_reference_sq[(name, head)].append(
                        float(reference.square().sum())
                    )
                    selection_signatures[(name, head)].append(
                        hashlib.sha256(
                            selected.detach().cpu().numpy().tobytes()
                        ).hexdigest()
                    )
                oracle_defect = reference - oracle_output
                oracle_post_rank = adaptive_rank_residual(
                    oracle_defect, int(protocol["gates"]["post_tail_adaptive_rank"])
                )
                oracle_outputs.append(
                    (
                        float(oracle_post_rank.square().sum()),
                        oracle_name,
                        oracle_output,
                        oracle_selected,
                    )
                )
            _, chosen_name, chosen_output, chosen_selected = min(
                oracle_outputs, key=lambda item: item[0]
            )
            family_name = "oracle_trajectory_width_family"
            family_defect = reference - chosen_output
            defects[(family_name, head)].append(family_defect)
            family_residual = adaptive_rank_residual(
                family_defect, int(protocol["gates"]["post_tail_adaptive_rank"])
            )
            per_tile_adaptive[(family_name, head)].append(
                math.sqrt(
                    float(family_residual.square().sum())
                    / max(float(reference.square().sum()), 1e-30)
                )
            )
            per_tile_adaptive_sq[(family_name, head)].append(
                float(family_residual.square().sum())
            )
            per_tile_reference_sq[(family_name, head)].append(
                float(reference.square().sum())
            )
            family_choice[(family_name, head)].append(chosen_name)
            selection_signatures[(family_name, head)].append(
                hashlib.sha256(chosen_selected.detach().cpu().numpy().tobytes()).hexdigest()
            )

    rows: list[dict[str, object]] = []
    for (route, head), tile_defects in sorted(defects.items()):
        defect = torch.cat(tile_defects, dim=0)
        reference = torch.cat(references[head], dim=0)
        adaptive_rank = int(protocol["gates"]["post_tail_adaptive_rank"])
        adaptive = adaptive_rank_residual(defect, adaptive_rank)
        singular = torch.linalg.svdvals(defect)
        energy = singular.square()
        cumulative = torch.cumsum(energy, dim=0)
        threshold_sq = (
            float(protocol["gates"]["oracle_worst_record_output_relative_l2"]) ** 2
            * float(reference.square().sum())
        )
        required = energy.numel()
        for candidate in range(energy.numel() + 1):
            residual_sq = float(energy.sum()) if candidate == 0 else max(
                0.0, float(energy.sum() - cumulative[candidate - 1])
            )
            if residual_sq <= threshold_sq:
                required = candidate
                break
        oracle_access = (
            "heldout_dense_AV_support_and_family"
            if route.startswith("oracle_")
            else "current_QKV_only_proxy"
        )
        rows.append(
            {
                "model_variant": variant,
                "split": split_name,
                "sample_id": sample_id,
                "rank": model.rank,
                "head": head,
                "route": route,
                "query_tiles": len(tile_defects),
                "density_target": float(protocol["scope"]["density"]),
                "reference_sq": float(reference.square().sum()),
                "content_residual_sq": float(defect.square().sum()),
                "post_adaptive_rank16_residual_sq": float(adaptive.square().sum()),
                "post_adaptive_rank16_per_tile_residual_sq": sum(
                    per_tile_adaptive_sq[(route, head)]
                ),
                "content_output_relative_l2": math.sqrt(
                    float(defect.square().sum()) / max(float(reference.square().sum()), 1e-30)
                ),
                "post_adaptive_rank16_output_relative_l2": math.sqrt(
                    float(adaptive.square().sum()) / max(float(reference.square().sum()), 1e-30)
                ),
                "post_adaptive_rank16_worst_tile_relative_l2": max(
                    per_tile_adaptive[(route, head)]
                ),
                "post_adaptive_rank16_per_tile_output_relative_l2": math.sqrt(
                    sum(per_tile_adaptive_sq[(route, head)])
                    / max(sum(per_tile_reference_sq[(route, head)]), 1e-30)
                ),
                "rank_required_for_1pct_record_gate": required,
                "family_choices": "|".join(family_choice.get((route, head), [])),
                "selection_signatures": "|".join(selection_signatures[(route, head)]),
                "oracle_access": oracle_access,
            }
        )
    return rows


def split_map(protocol: dict[str, Any]) -> dict[str, str]:
    split = protocol["split"]
    return {
        **{sample: "calibration" for sample in map(str, split["calibration_sample_ids"])},
        **{sample: "validation" for sample in map(str, split["validation_sample_ids"])},
        **{sample: "test" for sample in map(str, split["test_sample_ids"])},
    }


def aggregate(rows: list[dict[str, object]], protocol: dict[str, Any]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(
            str(row["model_variant"]),
            str(row["split"]),
            str(row["route"]),
            int(row["rank"]),
        )].append(row)
    output = []
    for (variant, split, route, rank), group in sorted(grouped.items()):
        reference_sq = sum(float(row["reference_sq"]) for row in group)
        content_sq = sum(float(row["content_residual_sq"]) for row in group)
        adaptive_sq = sum(float(row["post_adaptive_rank16_residual_sq"]) for row in group)
        per_tile_adaptive_sq = sum(
            float(row["post_adaptive_rank16_per_tile_residual_sq"]) for row in group
        )
        adaptive_values = sorted(
            float(row["post_adaptive_rank16_output_relative_l2"]) for row in group
        )
        output.append(
            {
                "model_variant": variant,
                "split": split,
                "route": route,
                "rank": rank,
                "records": len(group),
                "content_output_relative_l2": math.sqrt(content_sq / max(reference_sq, 1e-30)),
                "post_adaptive_rank16_output_relative_l2": math.sqrt(
                    adaptive_sq / max(reference_sq, 1e-30)
                ),
                "post_adaptive_rank16_per_tile_output_relative_l2": math.sqrt(
                    per_tile_adaptive_sq / max(reference_sq, 1e-30)
                ),
                "post_adaptive_rank16_worst_record_relative_l2": max(adaptive_values),
                "post_adaptive_rank16_p95_record_relative_l2": adaptive_values[
                    max(0, math.ceil(0.95 * len(adaptive_values)) - 1)
                ],
                "post_adaptive_rank16_worst_tile_relative_l2": max(
                    float(row["post_adaptive_rank16_worst_tile_relative_l2"])
                    for row in group
                ),
                "rank_required_for_1pct_record_gate_mean": sum(
                    int(row["rank_required_for_1pct_record_gate"]) for row in group
                )
                / len(group),
                "rank_required_for_1pct_record_gate_max": max(
                    int(row["rank_required_for_1pct_record_gate"]) for row in group
                ),
                "oracle_aggregate_gate": math.sqrt(adaptive_sq / max(reference_sq, 1e-30))
                <= float(protocol["gates"]["oracle_aggregate_output_relative_l2"]),
                "oracle_worst_gate": max(adaptive_values)
                <= float(protocol["gates"]["oracle_worst_record_output_relative_l2"]),
            }
        )
    return output


def summary_row(
    summary: list[dict[str, object]], variant: str, split: str, route: str
) -> dict[str, object]:
    matches = [
        row
        for row in summary
        if row["model_variant"] == variant and row["split"] == split and row["route"] == route
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one summary row for {(variant, split, route)}")
    return matches[0]


def build_decision(summary: list[dict[str, object]], protocol: dict[str, Any]) -> dict[str, object]:
    capacity = summary_row(
        summary, "transductive_capacity", "transductive_fit", "oracle_trajectory_width_family"
    )
    frozen_oracle = summary_row(
        summary, "calibration_frozen", "test", "oracle_trajectory_width_family"
    )
    proxy_routes = (
        "proxy_fixed64_qk",
        "proxy_svg2_style_semantic64",
        "proxy_value_aware_semantic64",
    )
    validation = [summary_row(summary, "calibration_frozen", "validation", route) for route in proxy_routes]
    frozen_route = min(
        validation,
        key=lambda row: (
            float(row["post_adaptive_rank16_worst_record_relative_l2"]),
            float(row["post_adaptive_rank16_output_relative_l2"]),
        ),
    )["route"]
    deployment = summary_row(summary, "calibration_frozen", "test", str(frozen_route))
    gates = protocol["gates"]
    capacity_pass = bool(capacity["oracle_aggregate_gate"] and capacity["oracle_worst_gate"])
    frozen_oracle_pass = bool(
        frozen_oracle["oracle_aggregate_gate"] and frozen_oracle["oracle_worst_gate"]
    )
    deployment_pass = (
        float(deployment["post_adaptive_rank16_output_relative_l2"])
        <= float(gates["deployment_aggregate_output_relative_l2"])
        and float(deployment["post_adaptive_rank16_worst_record_relative_l2"])
        <= float(gates["deployment_worst_record_output_relative_l2"])
    )
    if not capacity_pass:
        next_action = "FUNCTION_CLASS_FAIL_AT_THIS_RANK"
    elif not frozen_oracle_pass:
        next_action = "CAPACITY_ONLY: TEST_1K_2K_STEP_LOW_COST_ADAPTATION"
    elif not deployment_pass:
        next_action = "TAIL_TRANSFERS_BUT_PROXY_ROUTER_FAILS: TRAIN_ROUTER_AND_BRANCH_RATIO"
    else:
        next_action = "NUMERICAL_PREGATE_PASS: PROCEED_TO_FUSED_H200_KERNEL"
    return {
        "rank": int(capacity["rank"]),
        "transductive_capacity": capacity,
        "calibration_frozen_with_heldout_support_oracle": frozen_oracle,
        "validation_selected_proxy_route": frozen_route,
        "calibration_frozen_test_proxy": deployment,
        "capacity_gate_pass": capacity_pass,
        "frozen_tail_oracle_gate_pass": frozen_oracle_pass,
        "trainfree_deployment_gate_pass": deployment_pass,
        "chart_resume_gate_pass": capacity_pass and frozen_oracle_pass,
        "next_action": next_action,
        "warning": (
            "Transductive capacity and oracle routes inspect evaluated dense AV. "
            "Only calibration_frozen proxy routes are train-free deployment evidence."
        ),
    }


def checkpoint_payload(
    model: PositiveLinearTail,
    variant: str,
    protocol_hash: str,
    rank: int,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "variant": variant,
        "rank": rank,
        "protocol_sha256": protocol_hash,
        "model_class": "PositiveLinearTail",
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {args.output_dir}")
    args.output_dir.mkdir(parents=True)
    started = time.time()
    protocol = load_protocol(args.protocol_config, args.rank)
    protocol_hash = sha256_file(args.protocol_config)
    seed = stable_seed(protocol["seed"], args.rank, args.run_kind)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")

    paths = capture_paths(args.capture_index, protocol)
    captures = load_captures(paths, device)
    instances = build_instances(captures, protocol)
    split = protocol["split"]
    calibration_ids = list(map(str, split["calibration_sample_ids"]))
    all_ids = list(map(str, protocol["scope"]["sample_ids"]))

    frozen_model, frozen_trace, _ = train_variant(
        "calibration_frozen",
        args.rank,
        captures,
        instances,
        calibration_ids,
        protocol,
    )
    transductive_model, transductive_trace, _ = train_variant(
        "transductive_capacity",
        args.rank,
        captures,
        instances,
        all_ids,
        protocol,
    )
    torch.save(
        checkpoint_payload(frozen_model, "calibration_frozen", protocol_hash, args.rank),
        args.output_dir / "calibration_frozen.pt",
    )
    torch.save(
        checkpoint_payload(
            transductive_model, "transductive_capacity", protocol_hash, args.rank
        ),
        args.output_dir / "transductive_capacity.pt",
    )

    rows: list[dict[str, object]] = []
    sample_split = split_map(protocol)
    for sample_id in all_ids:
        rows.extend(
            evaluate_sample(
                frozen_model,
                "calibration_frozen",
                sample_id,
                instances,
                protocol,
                sample_split[sample_id],
            )
        )
        rows.extend(
            evaluate_sample(
                transductive_model,
                "transductive_capacity",
                sample_id,
                instances,
                protocol,
                "transductive_fit",
            )
        )
        print(f"[content-tail] evaluated sample={sample_id}", flush=True)
    summary = aggregate(rows, protocol)
    decision = build_decision(summary, protocol)

    atomic_csv(args.output_dir / "content_tail_records.csv", rows)
    atomic_csv(args.output_dir / "content_tail_summary.csv", summary)
    atomic_csv(args.output_dir / "training_trace.csv", frozen_trace + transductive_trace)
    atomic_json(args.output_dir / "decision.json", decision)
    manifest = {
        "schema_version": 1,
        "run_kind": args.run_kind,
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "capture_fingerprints": [
            {
                "sample_id": sample_id,
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtime_ns": path.stat().st_mtime_ns,
            }
            for sample_id, path in paths.items()
        ],
        "semantics": {
            "normalization": (
                "positive linear full-kernel numerator/denominator with selected interactions "
                "replaced by exact exp(QK), followed by one shared normalization"
            ),
            "fixed_budget": "all routes execute the same count of padded 64-key groups",
            "svg2_label": "SVG2-style semantic permutation proxy; not paper-faithful",
            "adaptive_rank16": "post-hoc dense-AV oracle used only as a chart-resume pregate",
            "trajectory_width_search": (
                "bounded greedy selection scored in the current rank-r basis orthogonal complement; "
                "a heuristic witness, not a global support optimum"
            ),
            "transductive_capacity": "all evaluated captures may fit feature maps and support",
            "calibration_frozen": "feature-map parameters and normalization statistics use calibration only",
        },
        "execution_resource_note": args.execution_resource_note,
        "seed": seed,
        "elapsed_seconds": time.time() - started,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
    }
    atomic_json(args.output_dir / "manifest.json", manifest)
    atomic_json(
        args.output_dir / "SUCCESS.json",
        {
            "artifact_status": "SUCCESS",
            "decision_sha256": sha256_file(args.output_dir / "decision.json"),
            "manifest_sha256": sha256_file(args.output_dir / "manifest.json"),
            "elapsed_seconds": manifest["elapsed_seconds"],
        },
    )
    print(
        f"[content-tail] completed rank={args.rank} next={decision['next_action']} "
        f"elapsed={manifest['elapsed_seconds']:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

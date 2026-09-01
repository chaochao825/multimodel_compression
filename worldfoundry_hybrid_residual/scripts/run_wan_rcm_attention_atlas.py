#!/usr/bin/env python3
"""Run EXP-054 S0/S1 for rCM-on-policy dense attention certification."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Callable

import rcm_attention_atlas_core as atlas_core
import run_wan_rcm_baseline as baseline
import run_wan_rcm_exact_runtime as exact_runtime
from experiment_artifacts import (
    JsonlEventLog,
    atomic_write_csv,
    atomic_write_json,
    require_fresh_output_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("s0-smoke", "s1-atlas")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--s0-manifest", type=Path)
    return parser.parse_args()


def _identity_ids(config: dict[str, Any], split: str) -> tuple[str, ...]:
    return tuple(
        row["identity"]
        for row in config["atlas_identities"]
        if row["split"] == split
    )


def load_configs(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if config["experiment_id"] != "EXP-054" or config["gate_id"] != "G-033":
        raise ValueError("config is not the frozen EXP-054/G-033 configuration")
    operator = config["operator"]
    expected_operator = {
        "name": "sage_sm90_smooth",
        "tensor_layout": "NHD",
        "smooth_k": True,
        "pv_accum_dtype": "fp32+fp32",
        "query_tile_size": 64,
    }
    if operator != expected_operator:
        raise ValueError("EXP-054 operator contract changed")
    materiality = config["materiality"]
    if materiality["minimum_selected_cells"] != 87:
        raise ValueError("EXP-054 requires at least 87 selected cells")
    if materiality["minimum_attention_speedup"] != 1.4:
        raise ValueError("EXP-054 attention speed guard changed")
    if materiality["minimum_projected_request_speedup"] != 1.05:
        raise ValueError("EXP-054 request materiality guard changed")

    calibration = _identity_ids(config, "calibration")
    evaluation = _identity_ids(config, "evaluation")
    if len(calibration) != 4 or len(evaluation) != 4:
        raise ValueError("EXP-054 requires four calibration and four evaluation identities")
    if set(calibration) & set(evaluation):
        raise ValueError("EXP-054 atlas identities overlap")
    all_ids = [row["identity"] for row in config["atlas_identities"]]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("EXP-054 contains duplicate atlas identities")
    if len(config["final_prompts"]) != 4 or len(config["final_seeds"]) != 2:
        raise ValueError("EXP-054 final quality split must remain four by two")

    base_path = (PROJECT_ROOT / config["base_config"]).resolve()
    base_config = baseline.load_config(base_path)
    exact_path = (PROJECT_ROOT / config["exact_runtime_config"]).resolve()
    exact_config, exact_base = exact_runtime.load_configs(exact_path)
    if exact_config["experiment_id"] != "EXP-052" or exact_base != base_config:
        raise ValueError("EXP-054 does not resolve to the exact EXP-052 baseline")
    return config, base_config


def resolve_output_dir(
    config: dict[str, Any], stage: str, override: Path | None
) -> Path:
    if override is not None:
        return override.resolve()
    return (Path(config["remote_output_root"]) / stage).resolve()


def load_sage_backend(config: dict[str, Any]) -> Callable[..., Any]:
    from sageattention import sageattn_qk_int8_pv_fp8_cuda_sm90

    operator = config["operator"]

    def backend(q: Any, k: Any, v: Any) -> Any:
        return sageattn_qk_int8_pv_fp8_cuda_sm90(
            q,
            k,
            v,
            tensor_layout=operator["tensor_layout"],
            sm_scale=1.0 / math.sqrt(q.shape[-1]),
            smooth_k=operator["smooth_k"],
            pv_accum_dtype=operator["pv_accum_dtype"],
        )

    return backend


def _event_time_ms(torch: Any, function: Callable[[], Any]) -> tuple[float, Any]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    output = function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)), output


class AttentionAtlasDispatcher:
    """Dual-evaluate self-attention while returning the exact reference path."""

    def __init__(
        self,
        torch: Any,
        sage_backend: Callable[..., Any],
        query_tile_size: int,
        mode: str,
        benchmark_warmups: int = 0,
        benchmark_repeats: int = 0,
    ) -> None:
        if mode not in ("reference", "dual", "benchmark_once"):
            raise ValueError(f"unsupported dispatcher mode: {mode}")
        self.torch = torch
        self.sage_backend = sage_backend
        self.query_tile_size = query_tile_size
        self.mode = mode
        self.benchmark_warmups = benchmark_warmups
        self.benchmark_repeats = benchmark_repeats
        self.identity = ""
        self.split = ""
        self.call_index = 0
        self.records: list[atlas_core.CellMetric] = []
        self.benchmark: dict[str, object] | None = None

    def begin_identity(self, identity: str, split: str) -> None:
        if not identity or not split:
            raise ValueError("identity and split must be non-empty")
        if self.call_index not in (0, atlas_core.CELL_COUNT):
            raise RuntimeError(
                f"previous trajectory stopped after {self.call_index} attention cells"
            )
        self.identity = identity
        self.split = split
        self.call_index = 0

    def assert_complete(self) -> None:
        if self.call_index != atlas_core.CELL_COUNT:
            raise RuntimeError(
                f"expected {atlas_core.CELL_COUNT} self-attention calls, "
                f"observed {self.call_index}"
            )

    def _validate_output(self, q: Any, reference: Any, candidate: Any) -> None:
        if candidate.shape != reference.shape:
            raise RuntimeError("Sage output shape differs from FA3")
        if candidate.dtype != reference.dtype or candidate.dtype != q.dtype:
            raise RuntimeError("Sage output dtype differs from the reference contract")
        if candidate.device != reference.device:
            raise RuntimeError("Sage output device differs from FA3")
        if not bool(self.torch.isfinite(candidate).all().item()):
            raise RuntimeError("Sage output contains non-finite values")

    def _record(self, step: int, layer: int, reference: Any, candidate: Any) -> None:
        aggregate, head, query_tile = atlas_core.output_error_metrics(
            reference, candidate, self.query_tile_size
        )
        self.records.append(
            atlas_core.CellMetric(
                identity=self.identity,
                split=self.split,
                step=step,
                layer=layer,
                aggregate=float(aggregate.item()),
                worst_head=float(head.item()),
                worst_query_tile=float(query_tile.item()),
            )
        )

    def _benchmark_pair(
        self,
        reference_function: Callable[..., Any],
        q: Any,
        k: Any,
        v: Any,
    ) -> dict[str, object]:
        if self.benchmark_repeats <= 0:
            raise ValueError("benchmark_once requires positive repeats")
        for _ in range(self.benchmark_warmups):
            reference_function(q, k, v)
            self.sage_backend(q, k, v)
        self.torch.cuda.synchronize(q.device)

        fa3_ms: list[float] = []
        sage_ms: list[float] = []
        for repeat in range(self.benchmark_repeats):
            ordered = (
                (("fa3", lambda: reference_function(q, k, v)),
                 ("sage", lambda: self.sage_backend(q, k, v)))
                if repeat % 2 == 0
                else
                (("sage", lambda: self.sage_backend(q, k, v)),
                 ("fa3", lambda: reference_function(q, k, v)))
            )
            for name, function in ordered:
                elapsed, _output = _event_time_ms(self.torch, function)
                (fa3_ms if name == "fa3" else sage_ms).append(elapsed)
        fa3_median = float(statistics.median(fa3_ms))
        sage_median = float(statistics.median(sage_ms))
        return {
            "fa3_ms": fa3_ms,
            "sage_ms": sage_ms,
            "fa3_median_ms": fa3_median,
            "sage_median_ms": sage_median,
            "attention_speedup": fa3_median / sage_median,
            "shape": list(q.shape),
            "dtype": str(q.dtype),
        }

    def __call__(
        self,
        layer: int,
        reference_function: Callable[..., Any],
        q: Any,
        k: Any,
        v: Any,
    ) -> Any:
        if not self.identity:
            raise RuntimeError("begin_identity must precede patched attention")
        step, expected_layer = divmod(self.call_index, atlas_core.CELL_LAYERS)
        if step >= atlas_core.CELL_STEPS or layer != expected_layer:
            raise RuntimeError(
                f"unexpected attention order at call {self.call_index}: "
                f"step={step}, layer={layer}, expected_layer={expected_layer}"
            )
        self.call_index += 1
        reference = reference_function(q, k, v)
        if self.mode == "reference":
            return reference

        candidate = self.sage_backend(q, k, v).contiguous().type_as(q)
        self._validate_output(q, reference, candidate)
        self._record(step, layer, reference, candidate)
        if self.mode == "benchmark_once" and self.benchmark is None:
            self.benchmark = self._benchmark_pair(reference_function, q, k, v)
        return reference


def make_spec(
    base_config: dict[str, Any],
    prompt: str,
    seed: int,
    num_frames: int,
    output_dir: Path,
) -> baseline.RunSpec:
    return exact_runtime.make_spec(
        base_config,
        "rcm4",
        prompt,
        -1,
        seed,
        num_frames,
        output_dir,
    )


def run_denoiser(
    base_config: dict[str, Any],
    spec: baseline.RunSpec,
    network: Any,
    tokenizer: Any,
    condition: dict[str, Any],
    runtime: dict[str, Any],
    device: Any,
) -> tuple[Any, int]:
    runtime["torch"].manual_seed(spec.seed)
    runtime["torch"].cuda.manual_seed_all(spec.seed)
    noise, generator = baseline.make_initial_noise(
        base_config, spec, tokenizer, runtime, device
    )
    return baseline.denoise_rcm(
        base_config, network, noise, generator, condition, runtime, device
    )


def setup(
    config: dict[str, Any], base_config: dict[str, Any], device_name: str
) -> tuple[dict[str, Any], Any, str, Any, Any, Callable[..., Any]]:
    runtime, device, source_commit = exact_runtime.setup_runtime(
        base_config, device_name
    )
    baseline.verify_checkpoint(base_config, "rcm4")
    network, tokenizer, _load_info = baseline.load_pipeline(
        base_config, "rcm4", runtime, device
    )
    sage_backend = load_sage_backend(config)
    return runtime, device, source_commit, network, tokenizer, sage_backend


def run_s0(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    runtime: dict[str, Any],
    device: Any,
    network: Any,
    tokenizer: Any,
    sage_backend: Callable[..., Any],
) -> dict[str, object]:
    s0 = config["s0"]
    text = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    try:
        text.warm(s0["prompt"], need_negative=False)
        condition, _uncondition = text.encode_condition(
            s0["prompt"], need_negative=False
        )
        f17_spec = make_spec(
            base_config, s0["prompt"], int(s0["seed"]), 17, output_dir
        )
        unpatched, unpatched_calls = run_denoiser(
            base_config,
            f17_spec,
            network,
            tokenizer,
            condition,
            runtime,
            device,
        )
        reference_dispatcher = AttentionAtlasDispatcher(
            runtime["torch"],
            sage_backend,
            int(config["operator"]["query_tile_size"]),
            "reference",
        )
        reference_dispatcher.begin_identity("s0_f17_patch", "engineering")
        with atlas_core.WanSelfAttentionPatch(network, reference_dispatcher):
            patched, patched_calls = run_denoiser(
                base_config,
                f17_spec,
                network,
                tokenizer,
                condition,
                runtime,
                device,
            )
        reference_dispatcher.assert_complete()
        patch_equal = bool(runtime["torch"].equal(unpatched, patched))

        f81_spec = make_spec(
            base_config, s0["prompt"], int(s0["seed"]), 81, output_dir
        )
        benchmark_dispatcher = AttentionAtlasDispatcher(
            runtime["torch"],
            sage_backend,
            int(config["operator"]["query_tile_size"]),
            "benchmark_once",
            benchmark_warmups=int(s0["attention_warmups"]),
            benchmark_repeats=int(s0["attention_repeats"]),
        )
        benchmark_dispatcher.begin_identity("s0_f81_backend", "engineering")
        with atlas_core.WanSelfAttentionPatch(network, benchmark_dispatcher):
            _samples, benchmark_calls = run_denoiser(
                base_config,
                f81_spec,
                network,
                tokenizer,
                condition,
                runtime,
                device,
            )
        benchmark_dispatcher.assert_complete()
    finally:
        text.close()

    if benchmark_dispatcher.benchmark is None:
        raise RuntimeError("S0 did not benchmark a self-attention cell")
    speedup = float(benchmark_dispatcher.benchmark["attention_speedup"])
    advance = (
        patch_equal
        and unpatched_calls == patched_calls == benchmark_calls == 4
        and speedup >= float(config["materiality"]["minimum_attention_speedup"])
    )
    return {
        "patch_latent_equal": patch_equal,
        "unpatched_calls": unpatched_calls,
        "patched_calls": patched_calls,
        "benchmark_calls": benchmark_calls,
        "benchmark": benchmark_dispatcher.benchmark,
        "first_cell_error": benchmark_dispatcher.records[0].as_dict(),
        "advance": advance,
    }


def _thresholds(raw: dict[str, Any]) -> atlas_core.ErrorThresholds:
    return atlas_core.ErrorThresholds(
        aggregate=float(raw["aggregate"]),
        worst_head=float(raw["worst_head"]),
        worst_query_tile=float(raw["worst_query_tile"]),
    )


def run_s1(
    config: dict[str, Any],
    base_config: dict[str, Any],
    output_dir: Path,
    s0_manifest: dict[str, Any],
    runtime: dict[str, Any],
    device: Any,
    network: Any,
    tokenizer: Any,
    sage_backend: Callable[..., Any],
) -> dict[str, object]:
    if s0_manifest["experiment_id"] != "EXP-054":
        raise ValueError("S1 requires an EXP-054 S0 manifest")
    if s0_manifest["stage"] != "s0-smoke" or not s0_manifest["result"]["advance"]:
        raise ValueError("S1 requires a passing frozen S0 manifest")
    attention_speedup = float(
        s0_manifest["result"]["benchmark"]["attention_speedup"]
    )

    dispatcher = AttentionAtlasDispatcher(
        runtime["torch"],
        sage_backend,
        int(config["operator"]["query_tile_size"]),
        "dual",
    )
    text = exact_runtime.TextEncoderPolicy(
        runtime,
        base_config["remote"]["text_encoder"],
        device,
        resident=True,
        cache_negative=False,
    )
    try:
        text.warm(config["s0"]["prompt"], need_negative=False)
        for row in config["atlas_identities"]:
            condition, _uncondition = text.encode_condition(
                row["prompt"], need_negative=False
            )
            spec = make_spec(
                base_config,
                row["prompt"],
                int(row["seed"]),
                81,
                output_dir,
            )
            dispatcher.begin_identity(row["identity"], row["split"])
            with atlas_core.WanSelfAttentionPatch(network, dispatcher):
                _samples, calls = run_denoiser(
                    base_config,
                    spec,
                    network,
                    tokenizer,
                    condition,
                    runtime,
                    device,
                )
            dispatcher.assert_complete()
            if calls != 4:
                raise RuntimeError(f"identity {row['identity']} made {calls} network calls")
            atomic_write_csv(
                output_dir / "cell_metrics.partial.csv",
                [record.as_dict() for record in dispatcher.records],
            )
            runtime["torch"].cuda.empty_cache()
    finally:
        text.close()

    calibration_ids = _identity_ids(config, "calibration")
    evaluation_ids = _identity_ids(config, "evaluation")
    calibration_records = [
        record for record in dispatcher.records if record.split == "calibration"
    ]
    evaluation_records = [
        record for record in dispatcher.records if record.split == "evaluation"
    ]
    atlas = atlas_core.freeze_and_evaluate_atlas(
        calibration_records,
        evaluation_records,
        calibration_ids,
        evaluation_ids,
        _thresholds(config["thresholds"]["calibration"]),
        _thresholds(config["thresholds"]["evaluation"]),
        int(config["materiality"]["minimum_selected_cells"]),
    )
    projected_seconds, projected_speedup = atlas_core.projected_request(
        float(config["materiality"]["baseline_request_seconds"]),
        float(config["materiality"]["baseline_denoiser_seconds"]),
        float(config["materiality"]["historical_self_attention_share"]),
        float(atlas["coverage"]),
        attention_speedup,
    )
    atlas.update(
        {
            "attention_speedup": attention_speedup,
            "projected_request_seconds": projected_seconds,
            "projected_request_speedup": projected_speedup,
            "passes_materiality": projected_speedup
            >= float(config["materiality"]["minimum_projected_request_speedup"]),
        }
    )
    atlas["advance"] = bool(
        atlas["passes_transfer_and_count"] and atlas["passes_materiality"]
    )
    atomic_write_csv(
        output_dir / "cell_metrics.csv",
        [record.as_dict() for record in dispatcher.records],
    )
    atomic_write_json(output_dir / "atlas.json", atlas)
    return {
        "record_count": len(dispatcher.records),
        "calibration_identity_count": len(calibration_ids),
        "evaluation_identity_count": len(evaluation_ids),
        "atlas": atlas,
        "advance": atlas["advance"],
    }


def main() -> None:
    args = parse_args()
    config, base_config = load_configs(args.config.resolve())
    output_dir = resolve_output_dir(config, args.stage, args.output_dir)
    require_fresh_output_dir(output_dir)
    log = JsonlEventLog(output_dir / "events.jsonl", f"EXP-054-{args.stage}")
    log.emit("run_start", stage=args.stage)

    runtime, device, source_commit, network, tokenizer, sage_backend = setup(
        config, base_config, args.device
    )
    if args.stage == "s0-smoke":
        if args.s0_manifest is not None:
            raise ValueError("s0-smoke does not accept --s0-manifest")
        result = run_s0(
            config,
            base_config,
            output_dir,
            runtime,
            device,
            network,
            tokenizer,
            sage_backend,
        )
    else:
        s0_path = (
            args.s0_manifest.resolve()
            if args.s0_manifest is not None
            else Path(config["remote_output_root"]) / "s0-smoke" / "manifest.json"
        )
        s0_manifest = json.loads(s0_path.read_text(encoding="utf-8"))
        result = run_s1(
            config,
            base_config,
            output_dir,
            s0_manifest,
            runtime,
            device,
            network,
            tokenizer,
            sage_backend,
        )

    manifest = {
        "experiment_id": config["experiment_id"],
        "gate_id": config["gate_id"],
        "stage": args.stage,
        "source_commit": source_commit,
        "result": result,
        "environment": exact_runtime.runtime_environment(runtime["torch"], device),
    }
    atomic_write_json(output_dir / "manifest.json", manifest)
    atomic_write_json(
        output_dir / "SUCCESS.json",
        {"status": "complete", "stage": args.stage, "advance": result["advance"]},
    )
    log.emit("run_complete", stage=args.stage, advance=result["advance"])
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()

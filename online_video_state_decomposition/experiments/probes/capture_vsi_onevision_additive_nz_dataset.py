from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from mvbench_onevision_utils import (
    build_prompt_batch,
    decode_video_frames,
    first_token_logits_from_features,
    load_onevision_model,
    uniform_frame_indices,
)
from probe_vsi_onevision_cmrq_stage_b import feature_path_for_sample
from probe_vsi_onevision_query_fixed_measure_remainder import (
    AttentionCapture,
    SelectedLayerCapture,
    replay_attention,
)
from probe_vsi_onevision_query_fixed_positive_gaussian_measure import (
    LAYERS,
)
from probe_vsi_onevision_reader_risk_stage_a import select_calibration_questions
from probe_vsi_onevision_same_kernel_mass_equivalence import (
    set_language_attention_eager,
)
from vsi_onevision_protocol import PROTOCOL_ID, load_vsi_mcq_records


ROLE = "calibration_additive_nz_attention_capture"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-path", type=Path, required=True)
    parser.add_argument("--jsonl-path", type=Path, required=True)
    parser.add_argument("--pruned-ids-path", type=Path, required=True)
    parser.add_argument("--video-root", type=Path, required=True)
    parser.add_argument("--feature-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--sample-count", type=int, default=96)
    parser.add_argument("--frame-budget", type=int, default=8)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def additive_nz_payload(
    capture: AttentionCapture,
    *,
    visual_start: int,
    visual_token_count: int,
) -> tuple[dict[str, torch.Tensor], float]:
    _, replay_error = replay_attention(capture)
    query = capture.query[:, -1].float()
    key = capture.key.float()
    value = capture.value.float()
    query_scaled = query * float(capture.module.scaling)
    scores = torch.einsum("hd,hsd->hs", query_scaled, key)
    if capture.attention_mask is not None:
        scores = scores + capture.attention_mask[:, -1].float()
    exponentials = torch.exp(scores - scores.max(dim=1, keepdim=True).values)

    visual_stop = visual_start + visual_token_count
    visual_key = key[:, visual_start:visual_stop]
    visual_value = value[:, visual_start:visual_stop]
    visual_exp = exponentials[:, visual_start:visual_stop]
    visual_z = visual_exp.sum(dim=1)
    visual_n = torch.einsum("hs,hsd->hd", visual_exp, visual_value)

    nonvisual_mask = torch.ones(scores.shape[1], device=scores.device, dtype=torch.bool)
    nonvisual_mask[visual_start:visual_stop] = False
    nonvisual_exp = exponentials[:, nonvisual_mask]
    nonvisual_value = value[:, nonvisual_mask]
    nonvisual_z = nonvisual_exp.sum(dim=1)
    nonvisual_n = torch.einsum("hs,hsd->hd", nonvisual_exp, nonvisual_value)
    exact_visual_output = visual_n / visual_z.unsqueeze(-1).clamp_min(1e-8)
    exact_full_output = (visual_n + nonvisual_n) / (visual_z + nonvisual_z).unsqueeze(
        -1
    ).clamp_min(1e-8)

    payload = {
        "query_scaled": query_scaled.to(device="cpu", dtype=torch.bfloat16),
        "visual_key": visual_key.to(device="cpu", dtype=torch.bfloat16),
        "visual_value": visual_value.to(device="cpu", dtype=torch.bfloat16),
        "exact_visual_output": exact_visual_output.to(
            device="cpu", dtype=torch.float32
        ),
        "exact_full_output": exact_full_output.to(device="cpu", dtype=torch.float32),
        "exact_visual_z": visual_z.to(device="cpu", dtype=torch.float32),
        "nonvisual_z": nonvisual_z.to(device="cpu", dtype=torch.float32),
        "nonvisual_n": nonvisual_n.to(device="cpu", dtype=torch.float32),
    }
    return payload, float(replay_error.item())


def main() -> int:
    args = parse_args()
    expected_count = 1 if args.smoke else 96
    if args.sample_offset != 0 or args.sample_count != expected_count:
        raise ValueError(
            "registered capture requires position 1 smoke or positions 1-96"
        )
    if args.frame_budget != 8:
        raise ValueError("registered frame budget changed")
    if args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("additive-N/Z capture output must be empty")

    split = json.loads(args.split_path.read_text(encoding="utf-8"))
    if split["protocol_id"] != PROTOCOL_ID:
        raise ValueError("VSI split protocol identity mismatch")
    records = load_vsi_mcq_records(args.jsonl_path, args.pruned_ids_path)
    samples = select_calibration_questions(
        split=split,
        records=records,
        video_root=args.video_root,
        sample_count=args.sample_count,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    processor, model = load_onevision_model(args.model_dir, device=args.device)
    set_language_attention_eager(model)
    model_dtype = next(model.parameters()).dtype
    capture = SelectedLayerCapture(model, LAYERS)
    maximum_replay_error = 0.0
    started = time.perf_counter()
    sample_ids: list[str] = []

    for sample_position, sample in enumerate(samples, start=1):
        payload = torch.load(
            feature_path_for_sample(args.feature_dir, sample),
            map_location="cpu",
            weights_only=False,
        )
        selected_positions = uniform_frame_indices(
            payload["features"].shape[0], args.frame_budget
        )
        reference = (
            payload["features"]
            .index_select(0, torch.tensor(selected_positions, dtype=torch.long))
            .to(device=device, dtype=model_dtype)
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
        capture.clear()
        with torch.inference_mode():
            first_token_logits_from_features(
                model=model,
                input_ids=prompt_batch["input_ids"],
                attention_mask=prompt_batch["attention_mask"],
                features=reference,
            )
        if set(capture.captures) != set(LAYERS):
            raise RuntimeError("not every registered attention layer was captured")

        video_mask = prompt_batch["input_ids"][0] == model.config.video_token_index
        placeholder_positions = torch.nonzero(video_mask, as_tuple=False).flatten()
        visual_start = int(placeholder_positions[0].item())
        visual_token_count = reference.shape[0] * reference.shape[1]
        layer_payloads: dict[int, dict[str, torch.Tensor]] = {}
        for layer_index in LAYERS:
            layer_payload, replay_error = additive_nz_payload(
                capture.captures[layer_index],
                visual_start=visual_start,
                visual_token_count=visual_token_count,
            )
            maximum_replay_error = max(maximum_replay_error, replay_error)
            layer_payloads[layer_index] = layer_payload
        output = {
            "protocol_id": PROTOCOL_ID,
            "role": ROLE,
            "sample_id": sample.sample_id,
            "sample_position": sample_position,
            "layers": layer_payloads,
        }
        torch.save(
            output,
            args.out_dir / f"position_{sample_position:03d}_{sample.sample_id}.pt",
        )
        sample_ids.append(sample.sample_id)
        print(
            json.dumps(
                {
                    "event": "additive_nz_capture_sample_ok",
                    "position": sample_position,
                    "sample_id": sample.sample_id,
                    "maximum_replay_error": maximum_replay_error,
                    "elapsed_seconds": time.perf_counter() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    capture.remove()
    if maximum_replay_error > 1e-4:
        raise RuntimeError("captured Q/K/V did not reconstruct attention output")
    summary = {
        "protocol_id": PROTOCOL_ID,
        "role": f"{ROLE}_smoke" if args.smoke else ROLE,
        "sample_count": len(samples),
        "sample_positions": [1, len(samples)],
        "sample_ids": sample_ids,
        "layers": list(LAYERS),
        "maximum_replay_error": maximum_replay_error,
        "elapsed_seconds": time.perf_counter() - started,
        "claim_boundary": (
            "Calibration-development post-RoPE Q/K/V capture for the additive N/Z "
            "state Gate; confirmation positions 97-120 and official selection/formal "
            "remain unread."
        ),
    }
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

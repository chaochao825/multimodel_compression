from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path


RESULT_PREFIX = "BASELINE_SMOKE_RESULT="


CAUSALMEM_PROBE = r'''
import contextlib
import io
import json
import torch
from llava.model.llava_arch_v3 import FOSSCache

torch.manual_seed(20260719)
with contextlib.redirect_stdout(io.StringIO()):
    cache = FOSSCache(
        budget=120,
        dim=16,
        k_max=4,
        decay=0.9,
        device="cpu",
        dtype=torch.float32,
        max_new_basis=2,
    )
for frame in range(20):
    cache.process_frame(torch.randn(16, 16), frame_idx=frame)
retained = cache.process_frame(
    torch.randn(16, 16),
    frame_idx=20,
    flag=1,
    total_num_frames=21,
)
assert torch.isfinite(retained).all()
assert cache.U.shape[1] <= cache.k_max
print("BASELINE_SMOKE_RESULT=" + json.dumps({
    "requested_budget": cache.budget,
    "retained_tokens": len(retained),
    "basis_rank": cache.U.shape[1],
    "mem1_tokens": len(cache.mem1_buffer),
    "mem2_frames": len(cache.mem2_bg_vectors),
    "budget_excess": len(retained) - cache.budget,
    "budget_respected": len(retained) <= cache.budget,
}, sort_keys=True))
'''


STREAMINGTOM_PROBE = r'''
import json
import torch
from streamingtom.modules.oqm import OQM

torch.manual_seed(20260719)
oqm = OQM()
video = "synthetic"
prompt_k = torch.randn(1, 2, 2, 8)
prompt_v = torch.randn(1, 2, 2, 8)
vision_k = torch.randn(1, 2, 8, 8)
vision_v = torch.randn(1, 2, 8, 8)
oqm.store_system_prompt(video, 0, prompt_k, prompt_v)
oqm.store_kv_cache(video, 0, vision_k, vision_v)
oqm.store_token_keys_as_groups(video, 0, vision_k[0, 0])
selected_k, selected_v = oqm.get_selective_kv(
    video, 0, torch.tensor([0, 1], dtype=torch.long)
)
assert selected_k.shape == (1, 2, 10, 8)
assert selected_v.shape == (1, 2, 10, 8)
torch.testing.assert_close(selected_k[:, :, :2], prompt_k)
torch.testing.assert_close(selected_v[:, :, :2], prompt_v)
packed = oqm.quantized_storage[video][0]["keys_packed"][0]
logical_fp16 = vision_k.numel() * 2
print("BASELINE_SMOKE_RESULT=" + json.dumps({
    "shape": list(selected_k.shape),
    "key_mse": torch.mean((selected_k[:, :, 2:] - vision_k) ** 2).item(),
    "value_mse": torch.mean((selected_v[:, :, 2:] - vision_v) ** 2).item(),
    "vision_fp16_bytes": logical_fp16,
    "key_packed_bytes": packed.numel(),
    "key_payload_ratio": logical_fp16 / packed.numel(),
    "group_count": len(oqm.get_group_keys(video, 0)),
}, sort_keys=True))
'''


STC_PROBE = r'''
import importlib
import json
import torch
import transformers

modules = [
    "stc.cacher.state",
    "stc.cacher.reference_forward",
    "stc.pruner.pruner",
    "stc.integrations.hf_vit",
]
loaded = [importlib.import_module(name).__name__ for name in modules]
print("BASELINE_SMOKE_RESULT=" + json.dumps({
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "modules": loaded,
}, sort_keys=True))
'''


OASIS_PROBE = r'''
import importlib
import json
import numpy as np
import sentence_transformers
import torch
import transformers

modules = [
    "src.oasis.event.forest",
    "src.oasis.event.segmenter",
]
loaded = [importlib.import_module(name).__name__ for name in modules]
from src.oasis.event.segmenter import ShortMemory
from src.oasis.types import FramePacket

memory = ShortMemory(
    buffer_frames_limit=4,
    now_window_frames_limit=2,
    buffer_fps=1.0,
)
cuts = []
for index in range(7):
    cuts.extend(
        memory.push(
            FramePacket(
                t=float(index),
                idx=index,
                frame=np.full((2, 2, 3), index, dtype=np.uint8),
            )
        )
    )
assert [len(cut) for cut in cuts] == [4]
assert len(memory.now_window) == 1
assert len(memory.buf) == 3
print("BASELINE_SMOKE_RESULT=" + json.dumps({
    "modules": loaded,
    "configured_now_window_limit": memory.now_window_frames_limit,
    "retained_now_window_frames": len(memory.now_window),
    "configured_buffer_limit": memory.buffer_frames_limit,
    "retained_buffer_frames": len(memory.buf),
    "emitted_event_sizes": [len(cut) for cut in cuts],
    "pending_event_frames": len(memory.buf_for_event),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "sentence_transformers": sentence_transformers.__version__,
}, sort_keys=True))
'''


PROBES = {
    "causalmem": ("CausalMem", CAUSALMEM_PROBE),
    "streamingtom": ("StreamingTOM", STREAMINGTOM_PROBE),
    "stc": ("STC", STC_PROBE),
    "oasis": ("OASIS", OASIS_PROBE),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--methods", default="causalmem,streamingtom,stc"
    )
    parser.add_argument("--causalmem-python", default=sys.executable)
    parser.add_argument("--streamingtom-python", default=sys.executable)
    parser.add_argument("--stc-python", default=sys.executable)
    parser.add_argument("--oasis-python", default=sys.executable)
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    return parser.parse_args()


def _streamingtom_environment() -> dict[str, str]:
    return {
        "CTR_SIMILARITY_THRESHOLD": "0.9",
        "CTR_RETAIN_TOKENS": "4",
        "CTR_K": "3",
        "CTR_BETA": "0.6",
        "OQM_RETRIEVAL_MAX_TOKENS": "8",
        "OQM_ENABLE_QUANTIZATION": "1",
        "OQM_QUANTIZATION_BITS": "4",
        "OQM_GROUP_SIZE": "4",
        "OQM_INIT_TOKEN_COUNT": "2",
        "OQM_SLIDING_WINDOW_SIZE": "8",
        "STREAMING_ENCODER_BATCH_SIZE": "2",
    }


def _output_tail(value: str | bytes | None) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return (value or "")[-4000:]


def run_probe(
    method: str,
    *,
    external_root: Path,
    python: str,
    timeout_seconds: float,
) -> dict[str, object]:
    repository, code = PROBES[method]
    checkout = external_root / repository
    if not checkout.is_dir():
        return {
            "method": method,
            "repository": repository,
            "status": "missing_checkout",
            "passed": False,
        }

    with tempfile.TemporaryDirectory(prefix=f"{method}-pycache-") as cache:
        environment = os.environ.copy()
        environment["PYTHONPYCACHEPREFIX"] = cache
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(checkout), environment.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        if method == "streamingtom":
            environment.update(_streamingtom_environment())
        try:
            completed = subprocess.run(
                [python, "-c", code],
                cwd=checkout,
                env=environment,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "method": method,
                "repository": repository,
                "python": python,
                "status": "timeout",
                "passed": False,
                "timeout_seconds": timeout_seconds,
                "stdout_tail": _output_tail(exc.stdout),
                "stderr_tail": _output_tail(exc.stderr),
            }

    metrics: dict[str, object] | None = None
    for line in completed.stdout.splitlines():
        if line.startswith(RESULT_PREFIX):
            metrics = json.loads(line[len(RESULT_PREFIX) :])
    passed = completed.returncode == 0 and metrics is not None
    return {
        "method": method,
        "repository": repository,
        "python": python,
        "status": "passed" if passed else "failed",
        "passed": passed,
        "returncode": completed.returncode,
        "metrics": metrics,
        "stdout_tail": completed.stdout[-4000:],
        "stderr_tail": completed.stderr[-4000:],
    }


def main() -> int:
    args = parse_args()
    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    unknown = sorted(set(methods) - set(PROBES))
    if unknown:
        raise ValueError(f"unknown methods: {unknown}")
    python_by_method = {
        "causalmem": args.causalmem_python,
        "streamingtom": args.streamingtom_python,
        "stc": args.stc_python,
        "oasis": args.oasis_python,
    }
    rows = [
        run_probe(
            method,
            external_root=args.external_root,
            python=python_by_method[method],
            timeout_seconds=args.timeout_seconds,
        )
        for method in methods
    ]
    payload = {
        "format_version": 1,
        "methods": methods,
        "all_requested_passed": all(row["passed"] for row in rows),
        "results": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["all_requested_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

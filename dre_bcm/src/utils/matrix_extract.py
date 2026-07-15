import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

try:
    from safetensors import safe_open
except ImportError:  # pragma: no cover
    safe_open = None


LLAMA_STYLE_SUFFIXES = {
    "q_proj": "q_proj.weight",
    "k_proj": "k_proj.weight",
    "v_proj": "v_proj.weight",
    "o_proj": "o_proj.weight",
    "gate_proj": "gate_proj.weight",
    "up_proj": "up_proj.weight",
    "down_proj": "down_proj.weight",
}

BERT_STYLE_SUFFIXES = {
    "query": "attention.self.query.weight",
    "key": "attention.self.key.weight",
    "value": "attention.self.value.weight",
    "attn_out": "attention.output.dense.weight",
    "intermediate": "intermediate.dense.weight",
    "output": "output.dense.weight",
}

GPT2_STYLE_SUFFIXES = {
    "c_attn": "attn.c_attn.weight",
    "c_proj": "attn.c_proj.weight",
    "c_fc": "mlp.c_fc.weight",
    "mlp_c_proj": "mlp.c_proj.weight",
}


class CheckpointReader:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = model_dir
        self.safetensor_weight_map: Dict[str, str] = {}
        self.bin_weight_map: Dict[str, str] = {}
        self._bin_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self._single_safetensor = None
        self._single_bin = None
        self._single_bin_state = None
        self._discover()

    def _discover(self) -> None:
        safetensor_index = self.model_dir / "model.safetensors.index.json"
        if safetensor_index.exists():
            with safetensor_index.open("r", encoding="utf-8") as handle:
                self.safetensor_weight_map = json.load(handle)["weight_map"]
            return

        single_safetensor = self.model_dir / "model.safetensors"
        if single_safetensor.exists():
            self._single_safetensor = single_safetensor
            return

        bin_index = self.model_dir / "pytorch_model.bin.index.json"
        if bin_index.exists():
            with bin_index.open("r", encoding="utf-8") as handle:
                self.bin_weight_map = json.load(handle)["weight_map"]
            return

        single_bin = self.model_dir / "pytorch_model.bin"
        if single_bin.exists():
            self._single_bin = single_bin
            return

        raise FileNotFoundError(f"no supported checkpoint file found under {self.model_dir}")

    def list_keys(self) -> List[str]:
        if self.safetensor_weight_map:
            return sorted(self.safetensor_weight_map.keys())
        if self._single_safetensor is not None:
            if safe_open is None:
                raise ImportError("safetensors is required to read .safetensors checkpoints")
            with safe_open(self._single_safetensor, framework="pt", device="cpu") as handle:
                return sorted(handle.keys())
        if self.bin_weight_map:
            return sorted(self.bin_weight_map.keys())
        if self._single_bin is not None:
            if self._single_bin_state is None:
                self._single_bin_state = torch.load(self._single_bin, map_location="cpu")
            return sorted(self._single_bin_state.keys())
        return []

    def load_tensor(self, key: str) -> torch.Tensor:
        if key in self.safetensor_weight_map:
            if safe_open is None:
                raise ImportError("safetensors is required to read .safetensors checkpoints")
            file_path = self.model_dir / self.safetensor_weight_map[key]
            with safe_open(file_path, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        if self._single_safetensor is not None:
            if safe_open is None:
                raise ImportError("safetensors is required to read .safetensors checkpoints")
            with safe_open(self._single_safetensor, framework="pt", device="cpu") as handle:
                return handle.get_tensor(key)
        if key in self.bin_weight_map:
            file_name = self.bin_weight_map[key]
            if file_name not in self._bin_cache:
                self._bin_cache[file_name] = torch.load(self.model_dir / file_name, map_location="cpu")
            return self._bin_cache[file_name][key]
        if self._single_bin is not None:
            if self._single_bin_state is None:
                self._single_bin_state = torch.load(self._single_bin, map_location="cpu")
            return self._single_bin_state[key]
        raise KeyError(key)


def load_config(model_dir: Path) -> Dict:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def candidate_suffixes(config: Dict) -> Dict[str, str]:
    model_type = str(config.get("model_type", "")).lower()
    architectures = " ".join(config.get("architectures", []))
    if any(token in model_type for token in ["llama", "mistral", "qwen", "bitnet", "opt"]):
        return LLAMA_STYLE_SUFFIXES
    if "bert" in model_type or "Bert" in architectures:
        return BERT_STYLE_SUFFIXES
    if "gpt2" in model_type:
        return GPT2_STYLE_SUFFIXES
    merged = {}
    merged.update(LLAMA_STYLE_SUFFIXES)
    merged.update(BERT_STYLE_SUFFIXES)
    merged.update(GPT2_STYLE_SUFFIXES)
    return merged


def match_layer_keys(keys: Iterable[str], requested_layers: Iterable[str], suffix_map: Dict[str, str]) -> List[Tuple[str, str]]:
    requested = list(requested_layers)
    results: List[Tuple[str, str]] = []
    for key in keys:
        if not key.endswith(".weight"):
            continue
        for layer_name in requested:
            suffix = suffix_map.get(layer_name)
            if suffix and key.endswith(suffix):
                results.append((layer_name, key))
    return results


def save_tensor(output_dir: Path, model_name: str, layer_name: str, key: str, tensor: torch.Tensor) -> None:
    layer_dir = output_dir / model_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    file_name = key.replace(".", "__") + ".pt"
    torch.save(
        {
            "model_name": model_name,
            "layer_name": layer_name,
            "key": key,
            "shape": list(tensor.shape),
            "weight": tensor.detach().cpu().float(),
        },
        layer_dir / file_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument(
        "--layers",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    parser.add_argument("--output-root", default="results/matrix_fit/raw_weights")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_root = Path(args.output_root)
    config = load_config(model_dir)
    suffix_map = candidate_suffixes(config)
    reader = CheckpointReader(model_dir)
    keys = reader.list_keys()
    matches = match_layer_keys(keys, args.layers, suffix_map)

    manifest = []
    for layer_name, key in matches:
        tensor = reader.load_tensor(key)
        save_tensor(output_root, args.model_name, layer_name, key, tensor)
        manifest.append(
            {
                "layer_name": layer_name,
                "key": key,
                "shape": list(tensor.shape),
            }
        )

    manifest_path = output_root / args.model_name / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_name": args.model_name,
                "model_dir": str(model_dir),
                "model_type": config.get("model_type"),
                "num_matches": len(manifest),
                "matches": manifest,
            },
            handle,
            indent=2,
        )

    print(f"saved {len(manifest)} weights to {manifest_path.parent}")


if __name__ == "__main__":
    main()

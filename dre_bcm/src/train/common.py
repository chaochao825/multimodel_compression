import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
)

from src.methods.bca import BCALinear
from src.methods.bcm import BCMLinear
from src.methods.c3a_wrapper import C3AProxyLinear
from src.methods.dre_bcm import DREBCMLinear
from src.methods.fourierft_wrapper import FourierFTLinear
from src.methods.lora import LoRALinear
from src.modules.peft_injection import count_trainable_parameters, freeze_module, inject_by_name
from src.utils.checkpoint import load_checkpoint, save_checkpoint


@dataclass
class TaskSpec:
    name: str
    dataset_path: str
    dataset_name: str
    text_keys: Tuple[str, Optional[str]]
    num_labels: int = 2
    label_key: str = "label"
    train_split: str = "train"
    eval_split: str = "validation"


TASK_SPECS: Dict[str, TaskSpec] = {
    "sst2": TaskSpec(
        name="sst2",
        dataset_path="glue",
        dataset_name="sst2",
        text_keys=("sentence", None),
    ),
    "rte": TaskSpec(
        name="rte",
        dataset_path="glue",
        dataset_name="rte",
        text_keys=("sentence1", "sentence2"),
    ),
    "boolq": TaskSpec(
        name="boolq",
        dataset_path="super_glue",
        dataset_name="boolq",
        text_keys=("question", "passage"),
    ),
}


LOCAL_GLUE_TASK_DIRS = {
    "sst2": "SST-2",
    "rte": "RTE",
}


RTE_LABEL_TO_ID = {
    "entailment": 1,
    "not_entailment": 0,
}


def resolve_task(task_name: str) -> TaskSpec:
    key = task_name.lower()
    if key not in TASK_SPECS:
        raise ValueError(f"unsupported task: {task_name}")
    return TASK_SPECS[key]


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def local_glue_roots(local_data_dir: Optional[str] = None) -> List[Path]:
    roots = []
    if local_data_dir:
        roots.append(Path(local_data_dir))
    roots.extend(
        [
            project_root() / "data" / "glue_data",
            Path("/home/spco/diff_bitnet/dre_bcm/data/glue_data"),
            Path("/home/wangmeiqi/chenbch/I-BERT-main/glue_data"),
        ]
    )
    unique_roots = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        unique_roots.append(root)
    return unique_roots


def parse_tsv(path: Path) -> List[Dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return []
    header = lines[0].split("\t")
    rows = []
    for line in lines[1:]:
        if not line:
            continue
        values = line.split("\t")
        if len(values) != len(header):
            continue
        rows.append(dict(zip(header, values)))
    return rows


def parse_rte_label(value: str) -> int:
    key = value.strip().lower()
    if key not in RTE_LABEL_TO_ID:
        raise ValueError(f"unsupported RTE label: {value}")
    return RTE_LABEL_TO_ID[key]


def load_local_glue_dataset(task: TaskSpec, local_data_dir: Optional[str] = None) -> Optional[DatasetDict]:
    task_dir_name = LOCAL_GLUE_TASK_DIRS.get(task.name)
    if task_dir_name is None:
        return None

    for root in local_glue_roots(local_data_dir):
        task_dir = root / task_dir_name
        train_tsv = task_dir / "train.tsv"
        dev_tsv = task_dir / "dev.tsv"
        if not train_tsv.exists() or not dev_tsv.exists():
            continue

        train_rows = parse_tsv(train_tsv)
        dev_rows = parse_tsv(dev_tsv)
        if task.name == "sst2":
            train_payload = {
                "sentence": [row["sentence"] for row in train_rows],
                "label": [int(row["label"]) for row in train_rows],
            }
            dev_payload = {
                "sentence": [row["sentence"] for row in dev_rows],
                "label": [int(row["label"]) for row in dev_rows],
            }
        elif task.name == "rte":
            train_payload = {
                "sentence1": [row["sentence1"] for row in train_rows],
                "sentence2": [row["sentence2"] for row in train_rows],
                "label": [parse_rte_label(row["label"]) for row in train_rows],
            }
            dev_payload = {
                "sentence1": [row["sentence1"] for row in dev_rows],
                "sentence2": [row["sentence2"] for row in dev_rows],
                "label": [parse_rte_label(row["label"]) for row in dev_rows],
            }
        else:
            continue

        return DatasetDict(
            {
                task.train_split: Dataset.from_dict(train_payload),
                task.eval_split: Dataset.from_dict(dev_payload),
            }
        )

    return None


def infer_default_target_modules(model_type: str) -> List[str]:
    model_type = model_type.lower()
    if model_type in {"roberta", "bert", "albert", "distilbert"}:
        return ["query", "value"]
    if model_type in {"gpt2", "gpt_neo", "gptj"}:
        return ["c_attn", "c_proj"]
    if model_type in {"opt", "llama", "mistral", "qwen2", "qwen2_moe"}:
        return ["q_proj", "v_proj"]
    return ["query", "value"]


def ensure_pad_token(tokenizer, model: nn.Module) -> None:
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            if hasattr(model, "resize_token_embeddings"):
                model.resize_token_embeddings(len(tokenizer))

    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id


def load_model_and_tokenizer(
    model_name_or_path: str,
    num_labels: int,
    cache_dir: Optional[str] = None,
):
    config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels, cache_dir=cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, cache_dir=cache_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        config=config,
        cache_dir=cache_dir,
    )
    ensure_pad_token(tokenizer, model)
    return model, tokenizer


def cache_roots(cache_dir: Optional[str] = None) -> List[Path]:
    roots = []
    if cache_dir:
        candidate = Path(cache_dir) / "datasets"
        roots.append(candidate)
    roots.extend(
        [
            Path.home() / ".cache" / "huggingface" / "datasets",
            Path("/home/wangmeiqi/.cache/huggingface/datasets"),
            Path("/home/spco/.cache/huggingface/datasets"),
        ]
    )
    unique_roots = []
    seen = set()
    for root in roots:
        root_str = str(root)
        if root_str in seen:
            continue
        seen.add(root_str)
        unique_roots.append(root)
    return unique_roots


def load_offline_arrow_dataset(task: TaskSpec, cache_dir: Optional[str] = None) -> Optional[DatasetDict]:
    prefix = task.dataset_path
    for root in cache_roots(cache_dir):
        base_dir = root / task.dataset_path / task.dataset_name / "0.0.0"
        if not base_dir.exists():
            continue
        for version_dir in sorted(base_dir.iterdir(), reverse=True):
            if not version_dir.is_dir():
                continue
            train_arrow = version_dir / f"{prefix}-{task.train_split}.arrow"
            eval_arrow = version_dir / f"{prefix}-{task.eval_split}.arrow"
            test_arrow = version_dir / f"{prefix}-test.arrow"
            if not train_arrow.exists() or not eval_arrow.exists():
                continue

            dataset = DatasetDict(
                {
                    task.train_split: Dataset.from_file(str(train_arrow)),
                    task.eval_split: Dataset.from_file(str(eval_arrow)),
                }
            )
            if test_arrow.exists():
                dataset["test"] = Dataset.from_file(str(test_arrow))
            return dataset
    return None


def load_task_dataset(
    task: TaskSpec,
    cache_dir: Optional[str] = None,
    local_data_dir: Optional[str] = None,
) -> DatasetDict:
    local_glue = load_local_glue_dataset(task, local_data_dir=local_data_dir)
    if local_glue is not None:
        return local_glue
    offline = load_offline_arrow_dataset(task, cache_dir=cache_dir)
    if offline is not None:
        return offline
    try:
        return load_dataset(task.dataset_path, task.dataset_name, cache_dir=cache_dir)
    except Exception:
        raise


def save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def subset_dataset(dataset, limit: Optional[int]):
    if limit is None or limit <= 0 or limit >= len(dataset):
        return dataset
    return dataset.select(range(limit))


def tokenize_datasets(
    tokenizer,
    task: TaskSpec,
    dataset: DatasetDict,
    max_length: int,
    max_train_samples: Optional[int] = None,
    max_eval_samples: Optional[int] = None,
) -> DatasetDict:
    text_key_a, text_key_b = task.text_keys

    def preprocess(examples):
        if text_key_b is None:
            encoded = tokenizer(examples[text_key_a], truncation=True, max_length=max_length)
        else:
            encoded = tokenizer(examples[text_key_a], examples[text_key_b], truncation=True, max_length=max_length)
        encoded["labels"] = examples[task.label_key]
        return encoded

    working = DatasetDict(
        {
            task.train_split: subset_dataset(dataset[task.train_split], max_train_samples),
            task.eval_split: subset_dataset(dataset[task.eval_split], max_eval_samples),
        }
    )
    if "test" in dataset:
        working["test"] = dataset["test"]

    processed = working.map(
        preprocess,
        batched=True,
        remove_columns=working[task.train_split].column_names,
        desc=f"tokenize_{task.name}",
    )
    return processed


def accuracy_metrics(eval_pred) -> Dict[str, float]:
    predictions = eval_pred.predictions[0] if isinstance(eval_pred.predictions, tuple) else eval_pred.predictions
    predicted = np.argmax(predictions, axis=-1)
    labels = eval_pred.label_ids
    return {"accuracy": float((predicted == labels).mean())}


def infer_classifier_keywords(model) -> List[str]:
    model_type = getattr(model.config, "model_type", "").lower()
    keywords = ["classifier", "score", "pre_classifier"]
    if model_type in {"roberta", "bert", "albert"}:
        keywords.append("pooler")
    return keywords


def unfreeze_by_keywords(model: nn.Module, keywords: Iterable[str]) -> List[str]:
    unfrozen = []
    keyword_list = list(keywords)
    for name, parameter in model.named_parameters():
        if any(keyword in name for keyword in keyword_list):
            parameter.requires_grad = True
            unfrozen.append(name)
    return unfrozen


def unfreeze_bias_terms(model: nn.Module) -> List[str]:
    unfrozen = []
    for name, parameter in model.named_parameters():
        if name.endswith(".bias") or name == "bias":
            parameter.requires_grad = True
            unfrozen.append(name)
    return unfrozen


def build_injection_factory(method: str, args):
    method = method.lower()
    if method == "lora":
        return lambda linear: LoRALinear(linear, rank=args.rank, alpha=args.alpha, train_base=args.train_base)
    if method == "bca":
        return lambda linear: BCALinear(
            linear,
            block_size=args.block_size,
            use_fft=args.use_fft,
            train_base=args.train_base,
        )
    if method == "bcm_only":
        return lambda linear: BCMLinear(
            linear,
            block_size=args.block_size,
            use_fft=args.use_fft,
            train_base=args.train_base,
            use_generator_delta=args.use_generator_delta,
            delta_mode=args.delta_mode,
        )
    if method == "dre_bcm":
        return lambda linear: DREBCMLinear(
            linear,
            block_size=args.block_size,
            rank=args.rank,
            alpha=args.alpha,
            use_fft=args.use_fft,
            train_base=args.train_base,
            use_generator_delta=args.use_generator_delta,
            delta_mode=args.delta_mode,
            mode="bcm_plus_lowrank",
        )
    if method == "dre_bcm_delta":
        return lambda linear: DREBCMLinear(
            linear,
            block_size=args.block_size,
            rank=args.rank,
            alpha=args.alpha,
            use_fft=args.use_fft,
            train_base=args.train_base,
            use_generator_delta=True,
            delta_mode=args.delta_mode,
            mode="bcm_plus_lowrank",
        )
    if method == "fourierft":
        return lambda linear: FourierFTLinear(
            base_linear=linear,
            freq_rows=args.freq_rows,
            freq_cols=args.freq_cols,
            scale=args.fourier_scale,
            train_base=args.train_base,
        )
    if method == "c3a_proxy":
        return lambda linear: C3AProxyLinear(
            base_linear=linear,
            block_size=args.block_size,
            num_bases=args.num_bases,
            train_base=args.train_base,
        )
    raise ValueError(f"unsupported injectable method: {method}")


def prepare_model_for_method(model, args):
    method = args.method.lower()
    freeze_module(model)
    injection_results = []

    if method == "full_finetune":
        for parameter in model.parameters():
            parameter.requires_grad = True
        head_parameters = [name for name, _ in model.named_parameters()]
    elif method == "linear_probe":
        head_parameters = unfreeze_by_keywords(model, infer_classifier_keywords(model))
    elif method == "bitfit":
        head_parameters = unfreeze_by_keywords(model, infer_classifier_keywords(model))
        head_parameters.extend(unfreeze_bias_terms(model))
    else:
        target_modules = list(args.target_modules) if args.target_modules else infer_default_target_modules(model.config.model_type)
        factory = build_injection_factory(method, args)
        injection_results = inject_by_name(model, target_names=target_modules, factory=factory)
        if not injection_results:
            inferred_targets = infer_default_target_modules(model.config.model_type)
            if target_modules != inferred_targets:
                target_modules = inferred_targets
                injection_results = inject_by_name(model, target_names=target_modules, factory=factory)
        if not injection_results:
            raise ValueError(
                f"no nn.Linear modules matched target tokens {target_modules} for model_type={model.config.model_type}"
            )
        head_parameters = unfreeze_by_keywords(model, infer_classifier_keywords(model))
        args.target_modules = target_modules

    return {
        "injection_results": [asdict(item) for item in injection_results],
        "head_parameter_names": sorted(set(head_parameters)),
        "trainable_params": count_trainable_parameters(model),
    }


def build_data_collator(tokenizer):
    pad_to_multiple = 8 if torch.cuda.is_available() else None
    return DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=pad_to_multiple)


def build_trainer(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    args,
    training_args_cls,
):
    training_args = training_args_cls(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available() and args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        seed=args.seed,
        remove_unused_columns=True,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=build_data_collator(tokenizer),
        compute_metrics=accuracy_metrics,
    )


def save_peft_artifacts(
    output_dir: Path,
    model: nn.Module,
    tokenizer,
    run_config: Dict,
    preparation_summary: Dict,
    train_metrics: Dict,
    eval_metrics: Dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)
    save_json(output_dir / "run_config.json", run_config)
    save_json(output_dir / "injection_summary.json", preparation_summary)
    save_json(output_dir / "train_metrics.json", train_metrics)
    save_json(output_dir / "eval_metrics.json", eval_metrics)
    save_checkpoint(
        str(output_dir / "peft_state.pt"),
        state_dict=model.state_dict(),
        run_config=run_config,
        preparation_summary=preparation_summary,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )


def load_saved_run(checkpoint_path: str) -> Dict:
    path = Path(checkpoint_path)
    if path.is_dir():
        path = path / "peft_state.pt"
    return load_checkpoint(str(path))

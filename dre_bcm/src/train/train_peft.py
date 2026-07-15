import argparse
import json
from pathlib import Path

import yaml
from transformers import TrainingArguments

from src.train.common import (
    build_trainer,
    load_model_and_tokenizer,
    load_task_dataset,
    prepare_model_for_method,
    resolve_task,
    save_peft_artifacts,
    tokenize_datasets,
)
from src.utils.logging import get_logger
from src.utils.seed import set_seed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    parser.add_argument("--model-name-or-path", required=True)
    parser.add_argument("--task-name", required=True, choices=["sst2", "rte", "boolq"])
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "full_finetune",
            "linear_probe",
            "bitfit",
            "lora",
            "bca",
            "bcm_only",
            "dre_bcm",
            "dre_bcm_delta",
            "fourierft",
            "c3a_proxy",
        ],
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-data-dir")
    parser.add_argument("--target-modules", nargs="+")
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--freq-rows", type=int, default=16)
    parser.add_argument("--freq-cols", type=int, default=16)
    parser.add_argument("--fourier-scale", type=float, default=1.0)
    parser.add_argument("--num-bases", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-eval-samples", type=int)
    parser.add_argument("--logging-steps", type=int, default=20)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-base", action="store_true")
    parser.add_argument("--use-fft", action="store_true")
    parser.add_argument("--use-generator-delta", action="store_true")
    parser.add_argument("--delta-mode", default="horizontal", choices=["horizontal", "vertical", "2d"])
    parser.add_argument("--fp16", action="store_true")
    return parser


def parse_args() -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument("--config")
    known_args, remaining = base_parser.parse_known_args()

    parser = build_parser()
    if known_args.config:
        config_path = Path(known_args.config)
        config_values = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        parser.set_defaults(**config_values)
    return parser.parse_args(remaining)


def main() -> None:
    args = parse_args()
    logger = get_logger("train_peft")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    task = resolve_task(args.task_name)
    logger.info("loading task=%s model=%s method=%s", task.name, args.model_name_or_path, args.method)
    dataset = load_task_dataset(task, cache_dir=args.cache_dir, local_data_dir=args.local_data_dir)
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=args.model_name_or_path,
        num_labels=task.num_labels,
        cache_dir=args.cache_dir,
    )
    processed = tokenize_datasets(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        max_length=args.max_length,
        max_train_samples=args.max_train_samples,
        max_eval_samples=args.max_eval_samples,
    )

    prep_summary = prepare_model_for_method(model, args)
    logger.info("trainable params=%s", prep_summary["trainable_params"])
    logger.info("target modules=%s", args.target_modules)
    logger.info("injected modules=%s", len(prep_summary["injection_results"]))

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed[task.train_split],
        eval_dataset=processed[task.eval_split],
        args=args,
        training_args_cls=TrainingArguments,
    )

    train_result = trainer.train()
    train_metrics = dict(train_result.metrics)
    eval_metrics = trainer.evaluate(eval_dataset=processed[task.eval_split])

    run_config = vars(args).copy()
    run_config["resolved_target_modules"] = args.target_modules
    run_config["task_spec"] = task.__dict__
    run_config["dataset_sizes"] = {
        "train": len(processed[task.train_split]),
        "eval": len(processed[task.eval_split]),
    }

    save_peft_artifacts(
        output_dir=output_dir,
        model=model,
        tokenizer=tokenizer,
        run_config=run_config,
        preparation_summary=prep_summary,
        train_metrics=train_metrics,
        eval_metrics=eval_metrics,
    )

    logger.info("train metrics=%s", json.dumps(train_metrics, indent=2))
    logger.info("eval metrics=%s", json.dumps(eval_metrics, indent=2))
    logger.info("saved peft artifacts to %s", output_dir)


if __name__ == "__main__":
    main()

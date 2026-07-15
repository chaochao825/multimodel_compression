import argparse
import json
from pathlib import Path

from transformers import TrainingArguments

from src.train.common import (
    build_trainer,
    load_model_and_tokenizer,
    load_task_dataset,
    load_saved_run,
    prepare_model_for_method,
    resolve_task,
    save_json,
    tokenize_datasets,
)
from src.utils.logging import get_logger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir")
    parser.add_argument("--local-data-dir")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--dataloader-num-workers", type=int, default=0)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    logger = get_logger("eval_peft")
    payload = load_saved_run(args.checkpoint)
    run_config = payload["run_config"]
    checkpoint_path = Path(args.checkpoint)
    output_dir = Path(args.output_dir) if args.output_dir else (checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent)

    task = resolve_task(run_config["task_name"])
    dataset = load_task_dataset(
        task,
        cache_dir=run_config.get("cache_dir"),
        local_data_dir=args.local_data_dir or run_config.get("local_data_dir"),
    )
    model, tokenizer = load_model_and_tokenizer(
        model_name_or_path=run_config["model_name_or_path"],
        num_labels=task.num_labels,
        cache_dir=run_config.get("cache_dir"),
    )

    namespace = argparse.Namespace(**run_config)
    namespace.eval_batch_size = args.eval_batch_size
    namespace.dataloader_num_workers = args.dataloader_num_workers
    namespace.fp16 = args.fp16
    prepare_model_for_method(model, namespace)
    model.load_state_dict(payload["state_dict"], strict=True)

    processed = tokenize_datasets(
        tokenizer=tokenizer,
        task=task,
        dataset=dataset,
        max_length=run_config["max_length"],
        max_train_samples=run_config.get("max_train_samples"),
        max_eval_samples=run_config.get("max_eval_samples"),
    )

    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed[task.train_split],
        eval_dataset=processed[task.eval_split],
        args=namespace,
        training_args_cls=TrainingArguments,
    )
    eval_metrics = trainer.evaluate(eval_dataset=processed[task.eval_split])
    save_json(output_dir / "eval_metrics_recomputed.json", eval_metrics)
    logger.info("recomputed eval metrics=%s", json.dumps(eval_metrics, indent=2))


if __name__ == "__main__":
    main()

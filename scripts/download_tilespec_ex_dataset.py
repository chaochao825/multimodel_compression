#!/usr/bin/env python3
"""Download a deterministic 200-sample GQA/TextVQA/ChartQA snapshot.

The script uses the official Hugging Face dataset viewer API so only selected
rows and images are transferred.  It records source row IDs and image hashes;
the generated data directory is intentionally ignored by git.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from PIL import Image


API_ROOT = "https://datasets-server.huggingface.co"
DATASET_SPECS = {
    "textvqa": {
        "repo": "lmms-lab/textvqa",
        "config": "default",
        "split": "validation",
    },
    "chartqa": {
        "repo": "lmms-lab/ChartQA",
        "config": "default",
        "split": "test",
    },
}


def _get_bytes(url: str, retries: int = 5) -> bytes:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={"User-Agent": "tilespec-ex/0.1"})
            with urlopen(request, timeout=120) as response:
                return response.read()
        except Exception as error:  # pragma: no cover - network dependent
            last_error = error
            time.sleep(2**attempt)
    raise RuntimeError(f"failed to download {url}") from last_error


def _get_json(endpoint: str, parameters: dict[str, Any]) -> dict[str, Any]:
    url = f"{API_ROOT}/{endpoint}?{urlencode(parameters)}"
    return json.loads(_get_bytes(url))


def _rows(
    repo: str,
    config: str,
    split: str,
    count: int,
    *,
    start: int = 0,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    while len(output) < count:
        length = min(100, count - len(output))
        payload = _get_json(
            "rows",
            {
                "dataset": repo,
                "config": config,
                "split": split,
                "offset": start + len(output),
                "length": length,
            },
        )
        page = payload.get("rows", [])
        if not page:
            break
        output.extend(page)
    if len(output) < count:
        raise RuntimeError(f"only received {len(output)} of {count} rows from {repo}")
    return output


def _download_image(url: str, target: Path) -> tuple[str, int, int]:
    raw = _get_bytes(url)
    image = Image.open(io.BytesIO(raw)).convert("RGB")
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target, format="JPEG", quality=95, optimize=True)
    payload = target.read_bytes()
    return hashlib.sha256(payload).hexdigest(), image.width, image.height


def _direct_dataset_records(dataset: str, count: int) -> list[dict[str, Any]]:
    spec = DATASET_SPECS[dataset]
    records = []
    for wrapped in _rows(spec["repo"], spec["config"], spec["split"], count):
        row = wrapped["row"]
        answers = row["answers"] if dataset == "textvqa" else [row["answer"]]
        records.append(
            {
                "dataset": dataset,
                "sample_id": str(row.get("question_id", wrapped["row_idx"])),
                "question": row["question"],
                "answers": [str(answer) for answer in answers],
                "image_url": row["image"]["src"],
                "source_row": wrapped["row_idx"],
                "source_repo": spec["repo"],
                "source_config": spec["config"],
                "source_split": spec["split"],
            }
        )
    return records


def _gqa_records(count: int) -> list[dict[str, Any]]:
    image_rows = _rows(
        "lmms-lab/GQA", "val_balanced_images", "val", count
    )
    images = {str(item["row"]["id"]): item for item in image_rows}
    questions: dict[str, dict[str, Any]] = {}
    offset = 0
    while len(questions) < count:
        page = _rows(
            "lmms-lab/GQA",
            "val_balanced_instructions",
            "val",
            100,
            start=offset,
        )
        for wrapped in page:
            image_id = str(wrapped["row"]["imageId"])
            if image_id in images and image_id not in questions:
                questions[image_id] = wrapped
        offset += len(page)
        if offset >= 20_000:
            missing = sorted(set(images) - set(questions))[:10]
            raise RuntimeError(f"could not join GQA image rows, examples: {missing}")

    records = []
    for image_id, image_wrapped in images.items():
        question_wrapped = questions[image_id]
        row = question_wrapped["row"]
        records.append(
            {
                "dataset": "gqa",
                "sample_id": str(row["id"]),
                "question": row["question"],
                "answers": [str(row["answer"])],
                "image_url": image_wrapped["row"]["image"]["src"],
                "source_row": question_wrapped["row_idx"],
                "image_source_row": image_wrapped["row_idx"],
                "source_repo": "lmms-lab/GQA",
                "source_config": "val_balanced_instructions+val_balanced_images",
                "source_split": "val",
            }
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/tilespec_ex_minimal"),
    )
    parser.add_argument("--samples-per-dataset", type=int, default=200)
    args = parser.parse_args()
    if args.samples_per_dataset <= 0:
        raise SystemExit("--samples-per-dataset must be positive")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_records: list[dict[str, Any]] = []
    for dataset in ("gqa", "textvqa", "chartqa"):
        records = (
            _gqa_records(args.samples_per_dataset)
            if dataset == "gqa"
            else _direct_dataset_records(dataset, args.samples_per_dataset)
        )
        for index, record in enumerate(records):
            image_path = output / dataset / "images" / f"{index:04d}.jpg"
            digest, width, height = _download_image(record.pop("image_url"), image_path)
            record.update(
                {
                    "dataset_index": index,
                    "image_path": image_path.relative_to(output).as_posix(),
                    "image_sha256": digest,
                    "image_width": width,
                    "image_height": height,
                }
            )
            all_records.append(record)
            print(f"{dataset}: {index + 1}/{len(records)}", flush=True)

    manifest = output / "manifest.jsonl"
    manifest.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_records),
        encoding="utf-8",
    )
    metadata = {
        "datasets": ["gqa", "textvqa", "chartqa"],
        "samples_per_dataset": args.samples_per_dataset,
        "total_samples": len(all_records),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "source": "Hugging Face dataset viewer API",
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

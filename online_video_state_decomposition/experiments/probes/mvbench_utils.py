from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


CHOICE_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@dataclass(frozen=True)
class MVBenchSample:
    task: str
    index: int
    video_path: Path
    question: str
    candidates: tuple[str, ...]
    answer: str
    subtitle: str = ""

    @property
    def sample_id(self) -> str:
        return f"{self.task}_{self.index:04d}"

    @property
    def answer_index(self) -> int:
        normalized = normalize_text(self.answer)
        for index, candidate in enumerate(self.candidates):
            if normalize_text(candidate) == normalized:
                return index
        raise ValueError(
            f"answer {self.answer!r} is not in candidates for {self.sample_id}"
        )


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower()).strip(" .,:;!?\"'")


def parse_csv_list(value: str) -> list[str]:
    output = [item.strip() for item in value.split(",") if item.strip()]
    if not output:
        raise ValueError("list arguments must contain at least one value")
    return output


def parse_int_list(value: str) -> list[int]:
    output = sorted({int(item) for item in parse_csv_list(value)})
    if output[0] <= 0:
        raise ValueError("integer lists must contain positive values")
    return output


def load_mvbench_samples(
    dataset_root: Path,
    *,
    tasks: Iterable[str],
    samples_per_task: int,
    selection_seed: int,
) -> list[MVBenchSample]:
    manifest_path = dataset_root / "test.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = manifest["root"]
    metadata = manifest["meta"]
    rng = np.random.default_rng(selection_seed)
    samples: list[MVBenchSample] = []
    for task in tasks:
        if task not in metadata or task not in roots:
            raise KeyError(f"unknown MVBench task: {task}")
        records = metadata[task]
        if samples_per_task > 0 and samples_per_task < len(records):
            selected = sorted(
                int(value)
                for value in rng.choice(
                    len(records),
                    size=samples_per_task,
                    replace=False,
                )
            )
        else:
            selected = list(range(len(records)))
        for index in selected:
            record = records[index]
            video_path = dataset_root / roots[task] / record["video"]
            samples.append(
                MVBenchSample(
                    task=task,
                    index=index,
                    video_path=video_path,
                    question=str(record["question"]),
                    candidates=tuple(str(x) for x in record["candidates"]),
                    answer=str(record["answer"]),
                    subtitle=str(record.get("subtitle", "")),
                )
            )
    return samples


def load_mvbench_samples_by_indices(
    dataset_root: Path,
    *,
    indices_by_task: dict[str, Iterable[int]],
) -> list[MVBenchSample]:
    manifest_path = dataset_root / "test.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roots = manifest["root"]
    metadata = manifest["meta"]
    samples: list[MVBenchSample] = []
    for task, requested_indices in indices_by_task.items():
        if task not in metadata or task not in roots:
            raise KeyError(f"unknown MVBench task: {task}")
        records = metadata[task]
        seen: set[int] = set()
        for raw_index in requested_indices:
            index = int(raw_index)
            if index in seen:
                raise ValueError(f"duplicate index {index} for task {task}")
            if not 0 <= index < len(records):
                raise IndexError(
                    f"index {index} is out of range for task {task}"
                )
            seen.add(index)
            record = records[index]
            samples.append(
                MVBenchSample(
                    task=task,
                    index=index,
                    video_path=dataset_root / roots[task] / record["video"],
                    question=str(record["question"]),
                    candidates=tuple(str(x) for x in record["candidates"]),
                    answer=str(record["answer"]),
                    subtitle=str(record.get("subtitle", "")),
                )
            )
    return samples


def shard_samples(
    samples: list[MVBenchSample],
    *,
    shard_index: int,
    shard_count: int,
) -> list[MVBenchSample]:
    if shard_count <= 0:
        raise ValueError("shard_count must be positive")
    if not 0 <= shard_index < shard_count:
        raise ValueError("shard_index must be in [0, shard_count)")
    return [
        sample
        for index, sample in enumerate(samples)
        if index % shard_count == shard_index
    ]


def uniform_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    retained = min(total_frames, count)
    return [
        min(
            int(((2 * index + 1) * total_frames) // (2 * retained)),
            total_frames - 1,
        )
        for index in range(retained)
    ]


def recent_frame_indices(total_frames: int, count: int) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    retained = min(total_frames, count)
    return list(range(total_frames - retained, total_frames))


def hybrid_frame_indices(
    total_frames: int,
    count: int,
    *,
    recent_count: int,
) -> list[int]:
    if total_frames <= 0 or count <= 0:
        return []
    retained = min(total_frames, count)
    recent = min(max(recent_count, 0), retained)
    recent_indices = recent_frame_indices(total_frames, recent)
    earlier_end = recent_indices[0] if recent_indices else total_frames
    long_count = retained - recent
    long_indices = uniform_frame_indices(earlier_end, long_count)
    return sorted(set(long_indices + recent_indices))


def decode_video_frames(
    video_path: Path,
    indices: Iterable[int],
) -> tuple[list[np.ndarray], float, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = capture.read()
        if not ok:
            capture.release()
            raise RuntimeError(
                f"failed to decode frame {index} from {video_path}"
            )
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    return frames, fps, total_frames


def video_metadata(video_path: Path) -> tuple[int, float]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    capture.release()
    return total_frames, fps


def choice_prompt(sample: MVBenchSample, *, include_subtitle: bool) -> str:
    prefix = ""
    if include_subtitle and sample.subtitle:
        prefix = f"Subtitle context:\n{sample.subtitle}\n\n"
    choices = "\n".join(
        f"{CHOICE_LABELS[index]}. {candidate}"
        for index, candidate in enumerate(sample.candidates)
    )
    return (
        f"{prefix}{sample.question}\n{choices}\n"
        "Answer with only the option letter."
    )


def clip_candidate_prompts(sample: MVBenchSample) -> list[str]:
    return [
        f"Question: {sample.question} Answer: {candidate}"
        for candidate in sample.candidates
    ]


def clip_question_prompt(sample: MVBenchSample) -> str:
    return f"Question: {sample.question}"


def parse_choice_output(
    output: str,
    candidates: tuple[str, ...],
) -> int | None:
    normalized = normalize_text(output)
    letter_match = re.search(
        r"(?:^|\b)(?:option\s*)?[\(\[]?([a-z])[\)\].,:]?(?:\b|$)",
        output.strip().lower(),
    )
    if letter_match:
        index = ord(letter_match.group(1)) - ord("a")
        if 0 <= index < len(candidates):
            return index
    exact_matches = [
        index
        for index, candidate in enumerate(candidates)
        if normalize_text(candidate) == normalized
    ]
    if len(exact_matches) == 1:
        return exact_matches[0]
    contained = [
        index
        for index, candidate in enumerate(candidates)
        if normalize_text(candidate)
        and normalize_text(candidate) in normalized
    ]
    return contained[0] if len(contained) == 1 else None

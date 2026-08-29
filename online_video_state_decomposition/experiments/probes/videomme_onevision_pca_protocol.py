from __future__ import annotations

import re
from collections import defaultdict

import numpy as np


DURATIONS = ("short", "medium", "long")
FORMAL_SEED = 20260829
FORMAL_SAMPLES_PER_DURATION = 200
SPLIT_PROTOCOL_ID = "videomme_onevision_pca_cross_domain_600_20260829_v1"
HISTORICAL_VIDEO_IDS = frozenset(
    {
        "0ncLRRLvdfk",
        "0w4OTD4L0GQ",
        "20gUIdXGuAs",
        "40BlVzjxu-I",
        "8np5YKYx3sU",
        "A9fQPzZ1-hg",
        "CxBbw8eT624",
        "EeTmrsVW8qE",
        "FcP0mzWFCQU",
        "Huj-zXv4DEw",
        "KlbDo9o5SC8",
        "LCfBYE97rFk",
        "Lyv_2usFQA0",
        "M69Sn3OERZo",
        "M_lwC5zPkyo",
        "NagvWwLvRik",
        "QdlP8ai8trw",
        "QhL6ICNQ_So",
        "VmNBt1tzC6k",
        "Zjl2vmy02As",
        "_tvmjsKXTu8",
        "aqTIB_q40bo",
        "dTUaWnvIOp4",
        "fJ-hp5Jlbv0",
        "iZYLeIJwe4w",
        "ooqdg9Wr-mo",
        "p9uIBCDhyr0",
        "txdfCHpxzVg",
        "uhYiRmGURwE",
        "yU5kPoc7sL4",
    }
)


def option_text(value: object, expected_label: str) -> str:
    text = str(value).strip()
    pattern = rf"^{re.escape(expected_label)}\s*[.):]\s*"
    stripped = re.sub(pattern, "", text, count=1, flags=re.IGNORECASE).strip()
    if not stripped:
        raise ValueError(f"empty option after removing label {expected_label}")
    return stripped


def record_is_eligible(record: dict[str, object]) -> bool:
    options = list(record["options"])
    answer = str(record["answer"]).strip().upper()
    return (
        str(record["duration"]) in DURATIONS
        and len(options) == 4
        and answer in tuple("ABCD")
        and bool(str(record["videoID"]).strip())
        and bool(str(record["question_id"]).strip())
    )


def build_frozen_split(
    records: list[dict[str, object]],
    *,
    available_video_ids: set[str],
    seed: int = FORMAL_SEED,
    samples_per_duration: int = FORMAL_SAMPLES_PER_DURATION,
) -> dict[str, object]:
    grouped: dict[str, dict[str, list[dict[str, object]]]] = {
        duration: defaultdict(list) for duration in DURATIONS
    }
    for record in records:
        if not record_is_eligible(record):
            continue
        video_id = str(record["videoID"])
        if video_id in HISTORICAL_VIDEO_IDS or video_id not in available_video_ids:
            continue
        grouped[str(record["duration"])][video_id].append(record)

    rng = np.random.default_rng(seed)
    selected_rows: list[dict[str, object]] = []
    for duration in DURATIONS:
        video_ids = sorted(grouped[duration])
        if len(video_ids) < samples_per_duration:
            raise ValueError(
                f"duration {duration} has only {len(video_ids)} eligible videos"
            )
        selected_positions = sorted(
            int(position)
            for position in rng.choice(
                len(video_ids),
                size=samples_per_duration,
                replace=False,
            )
        )
        for position in selected_positions:
            video_id = video_ids[position]
            questions = sorted(
                grouped[duration][video_id],
                key=lambda row: str(row["question_id"]),
            )
            selected = questions[int(rng.integers(0, len(questions)))]
            selected_rows.append(
                {
                    "question_id": str(selected["question_id"]),
                    "video_id": video_id,
                    "duration": duration,
                    "domain": str(selected["domain"]),
                    "task_type": str(selected["task_type"]),
                }
            )

    video_ids = [str(row["video_id"]) for row in selected_rows]
    question_ids = [str(row["question_id"]) for row in selected_rows]
    expected = samples_per_duration * len(DURATIONS)
    if len(selected_rows) != expected:
        raise ValueError(f"expected {expected} split rows, found {len(selected_rows)}")
    if len(set(video_ids)) != expected or len(set(question_ids)) != expected:
        raise ValueError("formal split must use unique videos and questions")
    return {
        "protocol_id": SPLIT_PROTOCOL_ID,
        "selection_seed": seed,
        "samples_per_duration": samples_per_duration,
        "historical_video_exclusions": sorted(HISTORICAL_VIDEO_IDS),
        "frame_policy": "uniform16_pool_uniform8_reader",
        "samples": selected_rows,
    }

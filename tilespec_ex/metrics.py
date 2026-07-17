"""Dataset answer metrics and small statistical helpers."""

from __future__ import annotations

from collections import Counter
import re
import string
from typing import Sequence

import numpy as np


_ARTICLES = {"a", "an", "the"}
_CONTRACTIONS = {
    "cant": "can't",
    "dont": "don't",
    "isnt": "isn't",
    "wont": "won't",
    "couldnt": "couldn't",
    "didnt": "didn't",
    "doesnt": "doesn't",
    "hadnt": "hadn't",
    "hasnt": "hasn't",
    "havent": "haven't",
    "shouldnt": "shouldn't",
    "wasnt": "wasn't",
    "werent": "weren't",
    "wouldnt": "wouldn't",
}
_PUNCTUATION = set(string.punctuation)
_NUMBER_WORDS = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}


def normalize_answer(value: object) -> str:
    text = str(value).replace("\n", " ").replace("\t", " ").strip().lower()
    text = re.sub(r"(?<=\d),(?=\d)", "", text)
    normalized_characters = []
    for index, character in enumerate(text):
        if character not in _PUNCTUATION:
            normalized_characters.append(character)
            continue
        previous_is_digit = index > 0 and text[index - 1].isdigit()
        next_is_digit = index + 1 < len(text) and text[index + 1].isdigit()
        inside_word = (
            character == "'"
            and index > 0
            and index + 1 < len(text)
            and text[index - 1].isalpha()
            and text[index + 1].isalpha()
        )
        signed_number = (
            character in {"+", "-"}
            and next_is_digit
            and (index == 0 or not text[index - 1].isalnum())
        )
        if (
            (character == "." and previous_is_digit and next_is_digit)
            or inside_word
            or signed_number
        ):
            normalized_characters.append(character)
        else:
            normalized_characters.append(" ")
    text = "".join(normalized_characters)
    words = []
    for word in text.split():
        if word in _ARTICLES:
            continue
        mapped = _NUMBER_WORDS.get(word, word)
        words.append(_CONTRACTIONS.get(mapped, mapped))
    return " ".join(words)


def relaxed_chart_match(prediction: object, answer: object) -> float:
    predicted = normalize_answer(prediction)
    target = normalize_answer(answer)
    try:
        predicted_value = float(predicted.replace(",", ""))
        target_value = float(target.replace(",", ""))
    except ValueError:
        return float(predicted == target)
    tolerance = 0.05 * abs(target_value)
    if target_value == 0:
        tolerance = 1e-6
    return float(abs(predicted_value - target_value) <= tolerance)


def textvqa_score(prediction: object, answers: Sequence[object]) -> float:
    predicted = normalize_answer(prediction)
    counts = Counter(normalize_answer(answer) for answer in answers)
    return min(1.0, counts[predicted] / 3.0)


def dataset_score(dataset: str, prediction: object, answers: Sequence[object]) -> float:
    if not answers:
        raise ValueError("answers must not be empty")
    lowered = dataset.lower()
    if lowered == "textvqa":
        return textvqa_score(prediction, answers)
    if lowered == "chartqa":
        return relaxed_chart_match(prediction, answers[0])
    if lowered == "gqa":
        return float(normalize_answer(prediction) == normalize_answer(answers[0]))
    raise ValueError(f"unsupported dataset: {dataset}")


def rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(array):
        end = start + 1
        while end < len(array) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman_correlation(lhs: Sequence[float], rhs: Sequence[float]) -> float:
    if len(lhs) != len(rhs) or len(lhs) < 2:
        raise ValueError("Spearman inputs must have equal length >= 2")
    left = rankdata(lhs)
    right = rankdata(rhs)
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def paired_bootstrap_interval(
    lhs: Sequence[float],
    rhs: Sequence[float],
    *,
    seed: int = 20260717,
    samples: int = 10_000,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    left = np.asarray(lhs, dtype=np.float64)
    right = np.asarray(rhs, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or len(left) < 2:
        raise ValueError("paired arrays must be one-dimensional and equally sized")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(left), size=(samples, len(left)))
    differences = (left[indices] - right[indices]).mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return (
        float((left - right).mean()),
        float(np.quantile(differences, alpha)),
        float(np.quantile(differences, 1.0 - alpha)),
    )

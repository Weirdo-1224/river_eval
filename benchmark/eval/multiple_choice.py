"""Multiple-choice evaluation utilities."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

@dataclass
class MCEvalMetrics:
    total: int = 0
    correct: int = 0
    accuracy: float = 0.0
    breakdown: dict[str, Any] = field(default_factory=dict)


def valid_letters(num_choices: int) -> list[str]:
    return [chr(ord("A") + idx) for idx in range(max(num_choices, 0))]


def parse_answer(raw_output: str, choices: list[str] | None = None) -> str | None:
    """Extract the first valid option letter from model output.

    Handles common API outputs such as ``E``, ``Option E``, and ``Answer: E``.
    The valid answer range is derived from the sample choices when available.
    """
    if not raw_output:
        return None
    letters = valid_letters(len(choices)) if choices is not None else list("ABCDE")
    if not letters:
        return None
    pattern = "".join(re.escape(letter) for letter in letters)
    text = raw_output.strip().upper()
    match = re.search(rf"(?:ANSWER\s*[:：]?\s*|OPTION\s+)?\b([{pattern}])\b", text)
    if match:
        return match.group(1).upper()
    return None


def evaluate_batch(predictions: list[str], ground_truths: list[str]) -> MCEvalMetrics:
    """Compute multiple-choice accuracy.

    Args:
        predictions: List of raw model outputs.
        ground_truths: List of ground-truth letters (A/B/C/D).

    Returns:
        Metrics dataclass with accuracy and per-sample breakdown.
    """
    assert len(predictions) == len(ground_truths)
    total = len(predictions)
    correct = 0
    details = []

    for pred_raw, gt in zip(predictions, ground_truths):
        pred = parse_answer(pred_raw)
        is_correct = pred == gt.upper()
        if is_correct:
            correct += 1
        details.append(
            {
                "predicted_raw": pred_raw,
                "predicted": pred,
                "ground_truth": gt.upper(),
                "is_correct": is_correct,
            }
        )

    accuracy = correct / total if total > 0 else 0.0
    return MCEvalMetrics(
        total=total,
        correct=correct,
        accuracy=accuracy,
        breakdown={"details": details},
    )

"""RIVER Retro-Memory scoring."""

from __future__ import annotations

from typing import Any

from benchmark.eval.registry import register_scorer
from benchmark.schema import Sample


@register_scorer("river_mcq_accuracy")
def score_mcq(prediction: str | None, sample: Sample) -> dict[str, Any]:
    correct = prediction == sample.answer.upper()
    return {
        "accuracy": 1.0 if correct else 0.0,
        "is_correct": correct,
        "correct_answer": sample.answer.upper(),
        "prediction": prediction,
    }

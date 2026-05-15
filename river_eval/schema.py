"""Shared dataclasses for the RIVER evaluation framework."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Sample:
    """Unified sample format across all datasets and tasks."""

    sample_id: str
    task_type: str
    video_source: str
    video_id: str
    video_path: Path  # resolved absolute path
    question: str
    choices: list[str]
    answer: str  # ground-truth letter, e.g. "A"
    question_time: float
    time_reference: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""

    sample_id: str
    predicted: str | None
    is_correct: bool
    raw_output: str
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

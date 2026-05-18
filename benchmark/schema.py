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
    answer: str  # ground-truth letter or text
    question_time: float
    time_reference: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)
    # --- Phase 2 extensions ---
    event_time: float | None = None        # Pro-Response: target event timestamp
    is_open_ended: bool = False            # True for open-ended (non-MCQ) questions
    dialog_turn: int = 0                   # Streaming: turn index (0 = single-turn)
    dialog_context: list[dict[str, Any]] | None = None  # Streaming: previous turns


@dataclass
class EvalResult:
    """Result of evaluating a single sample."""

    sample_id: str
    predicted: str | None
    is_correct: bool
    raw_output: str
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EventTrace:
    """Trace of a detected event for online evaluation (Pro-Response)."""

    trigger_time: float
    description: str
    window_start: float
    window_end: float
    raw_output: str
    latency_sec: float
    metadata: dict[str, Any] = field(default_factory=dict)

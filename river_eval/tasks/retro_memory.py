"""Retro-Memory task implementation."""

from __future__ import annotations

from typing import Any

from river_eval.eval.multiple_choice import parse_answer
from river_eval.schema import Sample
from river_eval.tasks.base import BaseTask


SYSTEM_PROMPT = (
    "You are a video understanding assistant. "
    "Answer the multiple-choice question based on the provided video frames. "
    "Respond with ONLY the letter of the correct choice: A, B, C, or D. "
    "Do not provide explanations."
)


class RetroMemoryTask(BaseTask):
    """Task adapter for RIVER Retro-Memory (multiple-choice)."""

    def __init__(self, max_frames: int = 16, frame_resolution: int = 448) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build OpenAI-format messages for a multiple-choice question."""
        choices_text = "\n".join(sample.choices)
        user_text = f"Question:\n{sample.question}\n\nChoices:\n{choices_text}\n\nAnswer:"

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def parse_output(self, raw_output: str) -> str | None:
        """Extract the predicted answer letter."""
        return parse_answer(raw_output)

    def get_visibility_policy(self, sample: Sample) -> dict[str, Any]:
        """Retro-Memory sees everything up to question_time."""
        return {
            "mode": "window",
            "window_start": 0.0,
            "window_end": sample.question_time,
            "allow_future": False,
            "use_memory": True,
            "max_frames": self.max_frames,
            "frame_resolution": self.frame_resolution,
        }

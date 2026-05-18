"""Live-Perception task implementation.

In Live-Perception, the user asks a question at the current time point.
The model can only see a recent window around question_time,
not the full history from the beginning.
"""

from __future__ import annotations

from typing import Any

from benchmark.eval.multiple_choice import parse_answer
from benchmark.schema import Sample
from benchmark.tasks.base import BaseTask
from benchmark.tasks.registry import register_task
from benchmark.video.frame_policies import FrameBundle


SYSTEM_PROMPT = (
    "You are observing a live video feed. You can only see the most recent "
    "segment of the video (the current window). Based solely on what you can "
    "see in this window, answer the multiple-choice question. Do not infer "
    "information from outside the visible window."
)


@register_task("river_live_perception")
class LivePerceptionTask(BaseTask):
    """Task adapter for RIVER Live-Perception (multiple-choice, current window only)."""

    def __init__(
        self,
        max_frames: int = 16,
        frame_resolution: int = 448,
        prompt_style: str = "live_window",
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.prompt_style = prompt_style

    def build_prompt(
        self,
        sample: Sample,
        frame_bundle: FrameBundle | None = None,
    ) -> list[dict[str, Any]]:
        """Build OpenAI-format messages for a multiple-choice question."""
        choices_text = "\n".join(sample.choices)
        letters = [chr(ord("A") + idx) for idx in range(len(sample.choices))]
        answer_range = ", ".join(letters)
        question = f"Question: {sample.question}\nOptions:\n{choices_text}".rstrip()

        if frame_bundle is not None:
            visible_start = frame_bundle.visible_range[0]
            visible_end = frame_bundle.visible_range[1]
        else:
            visible_start = max(0.0, sample.question_time - 8.0)
            visible_end = sample.question_time

        video_context = (
            f"You are viewing a live video segment from "
            f"{visible_start:.2f}s to {visible_end:.2f}s.\n"
        )

        user_text = (
            video_context
            + question
            + "\nSelect the best answer based ONLY on the visible video segment. "
            f"Respond with only the letter ({answer_range}) of the correct option."
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def parse_output(self, raw_output: str, sample: Sample | None = None) -> str | None:
        """Extract the predicted answer letter."""
        return parse_answer(raw_output, sample.choices if sample is not None else None)

    def get_visibility_policy(self, sample: Sample) -> dict[str, Any]:
        """Live-Perception sees only a recent window before question_time."""
        window_sec = 8.0  # default visible window
        return {
            "mode": "window",
            "window_start": max(0.0, sample.question_time - window_sec),
            "window_end": sample.question_time,
            "allow_future": False,
            "use_memory": False,
            "max_frames": self.max_frames,
            "frame_resolution": self.frame_resolution,
        }

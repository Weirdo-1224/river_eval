"""Retro-Memory task implementation."""

from __future__ import annotations

from typing import Any

from benchmark.eval.multiple_choice import parse_answer
from benchmark.schema import Sample
from benchmark.tasks.base import BaseTask
from benchmark.tasks.registry import register_task
from benchmark.video.frame_policies import FrameBundle


SYSTEM_PROMPT = (
    "Carefully watch the video and pay attention to the cause and sequence of "
    "events, the detail and movement of objects, and the action and pose of "
    "persons. Based on your observations, select the best option that "
    "accurately addresses the question."
)


@register_task("river_retro_memory")
class RetroMemoryTask(BaseTask):
    """Task adapter for RIVER Retro-Memory (multiple-choice)."""

    def __init__(
        self,
        max_frames: int = 16,
        frame_resolution: int = 448,
        prompt_style: str = "river_longshort",
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

        if self.prompt_style == "river_longshort":
            if frame_bundle is not None and frame_bundle.long.timestamps:
                long_end = frame_bundle.long.timestamps[-1]
            else:
                long_end = 0.0
            if frame_bundle is not None and frame_bundle.short.timestamps:
                short_start = frame_bundle.short.timestamps[0]
                short_end = frame_bundle.short.timestamps[-1]
            else:
                short_start = 0.0
                short_end = sample.question_time
            video_context = (
                f"This contains a long memory of 0.0 to {long_end:.2f} seconds. "
                f"This contains a short clip sampled in {short_start:.2f} to {short_end:.2f} seconds.\n"
            )
            user_text = (
                video_context
                + question
                + "\nSelect the best answer to the following multiple-choice question "
                f"based on the video. Respond with only the letter ({answer_range}) "
                "of the correct option."
            )
        elif self.prompt_style == "river_longshort_offline":
            user_text = (
                question
                + "\nOnly give the best option.\n"
                + "Best option:("
            )
        else:
            user_text = question + f"\nAnswer with only one option letter: {answer_range}."

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def parse_output(self, raw_output: str, sample: Sample | None = None) -> str | None:
        """Extract the predicted answer letter."""
        return parse_answer(raw_output, sample.choices if sample is not None else None)

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

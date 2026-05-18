"""Pro-Response-Streaming task implementation.

Multi-turn open-ended Q&A over a video timeline.
Each turn includes the current question plus dialog history from previous turns.
"""

from __future__ import annotations

from typing import Any

from benchmark.schema import Sample
from benchmark.tasks.base import BaseTask
from benchmark.tasks.registry import register_task
from benchmark.video.frame_policies import FrameBundle


SYSTEM_PROMPT = (
    "You are a video understanding assistant engaged in a live conversation "
    "about a video. Answer each question based on what you can see in the "
    "provided video frames. Be concise."
)


@register_task("river_pro_response_streaming")
class ProResponseStreamingTask(BaseTask):
    """Task adapter for RIVER Pro-Response-Streaming (multi-turn open-ended)."""

    def __init__(
        self,
        max_frames: int = 16,
        frame_resolution: int = 448,
        prompt_style: str = "streaming_turn",
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.prompt_style = prompt_style

    def build_prompt(
        self,
        sample: Sample,
        frame_bundle: FrameBundle | None = None,
    ) -> list[dict[str, Any]]:
        """Build multi-turn prompt with dialog history."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]

        # Append previous turns from dialog_context
        if sample.dialog_context:
            for turn in sample.dialog_context:
                messages.append({
                    "role": turn.get("role", "user"),
                    "content": turn.get("content", ""),
                })

        # Current turn: frame context + question
        if frame_bundle is not None:
            visible_start = frame_bundle.visible_range[0]
            visible_end = frame_bundle.visible_range[1]
        else:
            visible_start = max(0.0, sample.question_time - 8.0)
            visible_end = sample.question_time

        video_context = (
            f"[Video segment {visible_start:.2f}s - {visible_end:.2f}s] "
        )

        user_text = video_context + sample.question
        messages.append({"role": "user", "content": user_text})

        return messages

    def parse_output(self, raw_output: str, sample: Sample | None = None) -> str | None:
        """Return the raw output for open-ended scoring."""
        return raw_output.strip() if raw_output else None

    def get_visibility_policy(self, sample: Sample) -> dict[str, Any]:
        """Streaming sees a window around the current question_time."""
        window_sec = 8.0
        return {
            "mode": "window",
            "window_start": max(0.0, sample.question_time - window_sec),
            "window_end": sample.question_time,
            "allow_future": False,
            "use_memory": True,
            "max_frames": self.max_frames,
            "frame_resolution": self.frame_resolution,
        }

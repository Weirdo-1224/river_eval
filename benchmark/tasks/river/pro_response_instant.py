"""Pro-Response-Instant task implementation.

The model must actively monitor the video and alert when a target event occurs.
For each sliding window, the model answers whether the target event is detected.
"""

from __future__ import annotations

from typing import Any

from benchmark.schema import Sample
from benchmark.tasks.base import BaseTask
from benchmark.tasks.registry import register_task
from benchmark.video.frame_policies import FrameBundle


SYSTEM_PROMPT = (
    "You are a video monitoring assistant. Your job is to watch the video "
    "and determine whether a specific target event is occurring. "
    "Respond concisely."
)


@register_task("river_pro_response_instant")
class ProResponseInstantTask(BaseTask):
    """Task adapter for RIVER Pro-Response-Instant (event detection)."""

    def __init__(
        self,
        max_frames: int = 8,
        frame_resolution: int = 448,
        prompt_style: str = "event_detect",
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.prompt_style = prompt_style

    def build_prompt(
        self,
        sample: Sample,
        frame_bundle: FrameBundle | None = None,
    ) -> list[dict[str, Any]]:
        """Build prompt asking the model to detect a target event."""
        target_event = sample.question

        if frame_bundle is not None:
            visible_start = frame_bundle.visible_range[0]
            visible_end = frame_bundle.visible_range[1]
        else:
            visible_start = 0.0
            visible_end = sample.question_time

        window_info = (
            f"This video segment covers {visible_start:.2f}s to {visible_end:.2f}s.\n"
        )

        if sample.choices:
            # Multiple-choice variant (some Pro-Response-Instant samples have choices)
            choices_text = "\n".join(sample.choices)
            user_text = (
                window_info
                + f"Target event: {target_event}\n"
                + f"Options:\n{choices_text}\n"
                + "If the target event is occurring in this segment, select the best option. "
                + "If not, respond with NONE. "
                + "Respond with only the letter or NONE."
            )
        else:
            # Open-ended variant
            user_text = (
                window_info
                + f"Target event to monitor: '{target_event}'\n"
                + "Is this target event occurring in the video segment?\n"
                + "Respond with ONLY 'YES' if you see the event occurring, "
                + "or 'NO' if you do not. Do not explain."
            )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text},
        ]

    def parse_output(self, raw_output: str, sample: Sample | None = None) -> dict[str, Any]:
        """Parse model output into a structured detection result.

        Returns a dict with keys:
            - triggered: bool
            - description: str
            - answer: str | None (for MCQ variants)
        """
        text = (raw_output or "").strip().upper()

        result: dict[str, Any] = {
            "triggered": False,
            "description": raw_output or "",
            "answer": None,
        }

        if not text:
            return result

        # Check for YES/NO
        if "YES" in text:
            result["triggered"] = True
        elif "NO" in text:
            result["triggered"] = False
        elif "NONE" in text:
            result["triggered"] = False
        else:
            # Fallback: if output is not YES/NO/NONE, treat as triggered
            # (model may have described the event directly)
            result["triggered"] = True

        # For MCQ variants, try to extract a letter
        if sample is not None and sample.choices:
            from benchmark.eval.multiple_choice import parse_answer
            answer = parse_answer(raw_output, sample.choices)
            result["answer"] = answer
            # If a valid option is selected, consider it triggered
            if answer is not None:
                result["triggered"] = True

        return result

    def get_visibility_policy(self, sample: Sample) -> dict[str, Any]:
        """Pro-Response monitors from question_time onward."""
        return {
            "mode": "stream",
            "window_start": float(sample.question_time),
            "window_end": float(sample.metadata.get("duration_sec", sample.question_time)),
            "allow_future": True,
            "use_memory": True,
            "max_frames": self.max_frames,
            "frame_resolution": self.frame_resolution,
        }

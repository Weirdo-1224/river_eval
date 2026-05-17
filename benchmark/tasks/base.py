"""Base task interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from benchmark.schema import Sample


class BaseTask(ABC):
    """Abstract base class for all evaluation tasks."""

    @abstractmethod
    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build chat messages (OpenAI format) for the given sample.

        Returns a list of dicts with ``role`` and ``content`` keys.
        """
        ...

    @abstractmethod
    def parse_output(self, raw_output: str) -> str | None:
        """Parse raw model output into a structured answer."""
        ...

    def get_visibility_policy(self, sample: Sample) -> dict[str, Any]:
        """Return visibility policy for this task (reserved for Phase 2)."""
        return {
            "mode": "window",
            "window_start": 0.0,
            "window_end": sample.question_time,
            "allow_future": False,
            "use_memory": True,
        }

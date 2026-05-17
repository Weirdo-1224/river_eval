"""Base model interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class BaseModel(ABC):
    """Abstract base class for all model adapters."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
        memory: Any = None,
    ) -> str:
        """Generate a response from the model.

        Args:
            messages: Chat messages in OpenAI format.
            image_paths: Paths to image frames to include in the prompt.
            memory: Optional memory context (reserved for Phase 4).

        Returns:
            Raw model output string.
        """
        ...

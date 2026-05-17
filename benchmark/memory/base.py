"""Memory interface for long-form video evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseMemory(ABC):
    @abstractmethod
    def reset(self) -> None:
        """Clear all stored state."""

    @abstractmethod
    def update(self, item: Any) -> None:
        """Add an observation or intermediate state."""

    @abstractmethod
    def retrieve(self, query: Any | None = None) -> Any:
        """Return memory context for a model call."""

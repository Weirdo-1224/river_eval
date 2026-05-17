"""No-op memory implementation."""

from __future__ import annotations

from typing import Any

from benchmark.memory.base import BaseMemory


class NoMemory(BaseMemory):
    def reset(self) -> None:
        return None

    def update(self, item: Any) -> None:
        return None

    def retrieve(self, query: Any | None = None) -> None:
        return None

"""Simple sliding-window memory placeholder."""

from __future__ import annotations

from collections import deque
from typing import Any

from benchmark.memory.base import BaseMemory


class SlidingWindowMemory(BaseMemory):
    def __init__(self, max_items: int = 16) -> None:
        self.items: deque[Any] = deque(maxlen=max_items)

    def reset(self) -> None:
        self.items.clear()

    def update(self, item: Any) -> None:
        self.items.append(item)

    def retrieve(self, query: Any | None = None) -> list[Any]:
        return list(self.items)

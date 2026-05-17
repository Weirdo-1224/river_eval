"""Trace writer placeholder.

For the current Retro-Memory phase, trace fields are embedded directly in each
JSONL result record. This module reserves a stable place for richer streaming
traces in later phases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class TraceWriter:
    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def write(self, trace: dict[str, Any]) -> None:
        raise NotImplementedError("Standalone trace writing is reserved for streaming tasks.")

"""Result persistence utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.schema import EvalResult


class ResultWriter:
    """Writes evaluation results incrementally to a JSONL file."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")

    def write(self, result: EvalResult) -> None:
        record = {
            "sample_id": result.sample_id,
            "predicted": result.predicted,
            "is_correct": result.is_correct,
            "raw_output": result.raw_output,
            "latency_sec": result.latency_sec,
            **result.metadata,
        }
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def write_record(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "ResultWriter":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()


def write_summary(
    output_path: str | Path,
    total: int,
    correct: int,
    accuracy: float,
    extra: dict[str, Any] | None = None,
) -> None:
    """Write a JSON summary file."""
    summary = {
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        **(extra or {}),
    }
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")

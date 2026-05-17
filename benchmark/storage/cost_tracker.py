"""Cost tracking for API model calls."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Default pricing in USD per 1K tokens.
# Format: model_name -> (input_per_1k, output_per_1k)
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.00250, 0.01000),
    "gpt-4o-mini": (0.00015, 0.00060),
    "qwen-vl-plus": (0.00050, 0.00050),
    "qwen-vl-max": (0.00200, 0.00200),
    "qwen2-vl-72b-instruct": (0.00150, 0.00150),
}


@dataclass
class CostInfo:
    """Per-request cost breakdown."""

    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostReport:
    """Aggregated cost report for an experiment."""

    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cost_usd: float = 0.0
    avg_cost_per_sample: float = 0.0
    by_model: dict[str, dict[str, Any]] = field(default_factory=dict)


class CostTracker:
    """Tracks API usage costs across an experiment."""

    def __init__(self, pricing: dict[str, tuple[float, float]] | None = None) -> None:
        self.pricing = pricing or DEFAULT_PRICING.copy()
        self._records: list[CostInfo] = []

    def add(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Record a request and return its estimated cost in USD."""
        input_price, output_price = self.pricing.get(model, (0.0, 0.0))
        cost = (input_tokens * input_price + output_tokens * output_price) / 1000.0
        self._records.append(
            CostInfo(
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
            )
        )
        return cost

    def report(self) -> CostReport:
        """Generate aggregated cost report."""
        total_requests = len(self._records)
        total_input = sum(r.input_tokens for r in self._records)
        total_output = sum(r.output_tokens for r in self._records)
        total_cost = sum(r.cost_usd for r in self._records)

        by_model: dict[str, dict[str, Any]] = {}
        for r in self._records:
            if r.model not in by_model:
                by_model[r.model] = {
                    "requests": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cost_usd": 0.0,
                }
            by_model[r.model]["requests"] += 1
            by_model[r.model]["input_tokens"] += r.input_tokens
            by_model[r.model]["output_tokens"] += r.output_tokens
            by_model[r.model]["cost_usd"] += r.cost_usd

        avg_cost = total_cost / total_requests if total_requests > 0 else 0.0

        return CostReport(
            total_requests=total_requests,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            total_cost_usd=total_cost,
            avg_cost_per_sample=avg_cost,
            by_model=by_model,
        )

    def write_report(self, output_path: str | Path) -> None:
        """Write cost report to a JSON file."""
        report = self.report()
        data = {
            "total_requests": report.total_requests,
            "total_input_tokens": report.total_input_tokens,
            "total_output_tokens": report.total_output_tokens,
            "total_cost_usd": report.total_cost_usd,
            "avg_cost_per_sample": report.avg_cost_per_sample,
            "by_model": report.by_model,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            import json

            json.dump(data, f, indent=2, ensure_ascii=False)
            f.write("\n")

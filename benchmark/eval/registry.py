"""Scoring registry."""

from __future__ import annotations

from typing import Callable

from benchmark.schema import Sample


Scorer = Callable[[str | None, Sample], dict]

SCORERS: dict[str, Scorer] = {}


def register_scorer(name: str) -> Callable[[Scorer], Scorer]:
    def decorator(fn: Scorer) -> Scorer:
        SCORERS[name] = fn
        return fn

    return decorator


def build_scorer(name: str) -> Scorer:
    if name not in SCORERS:
        raise KeyError(f"Unknown scoring '{name}'. Available: {sorted(SCORERS)}")
    return SCORERS[name]


def import_builtin_scorers() -> None:
    import benchmark.eval.multiple_choice  # noqa: F401
    import benchmark.eval.river.retro_memory_scoring  # noqa: F401

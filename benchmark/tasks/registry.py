"""Task registry."""

from __future__ import annotations

from typing import Any, Callable

from benchmark.tasks.base import BaseTask


TASKS: dict[str, type[BaseTask]] = {}


def register_task(name: str) -> Callable[[type[BaseTask]], type[BaseTask]]:
    def decorator(cls: type[BaseTask]) -> type[BaseTask]:
        TASKS[name] = cls
        return cls

    return decorator


def build_task(name: str, **kwargs: Any) -> BaseTask:
    if name not in TASKS:
        raise KeyError(f"Unknown task '{name}'. Available: {sorted(TASKS)}")
    return TASKS[name](**kwargs)


def import_builtin_tasks() -> None:
    import benchmark.tasks.river.retro_memory  # noqa: F401

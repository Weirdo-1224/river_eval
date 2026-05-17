"""Dataset registry."""

from __future__ import annotations

from typing import Any, Callable

from benchmark.datasets.base import BaseDataset


DATASETS: dict[str, type[BaseDataset]] = {}


def register_dataset(name: str) -> Callable[[type[BaseDataset]], type[BaseDataset]]:
    def decorator(cls: type[BaseDataset]) -> type[BaseDataset]:
        DATASETS[name] = cls
        return cls

    return decorator


def build_dataset(name: str, **kwargs: Any) -> BaseDataset:
    if name not in DATASETS:
        raise KeyError(f"Unknown dataset '{name}'. Available: {sorted(DATASETS)}")
    return DATASETS[name](**kwargs)


def import_builtin_datasets() -> None:
    import benchmark.datasets.river  # noqa: F401

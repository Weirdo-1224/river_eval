"""Model registry."""

from __future__ import annotations

from typing import Any, Callable

from benchmark.models.base import BaseModel


MODELS: dict[str, type[BaseModel]] = {}


def register_model(name: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
    def decorator(cls: type[BaseModel]) -> type[BaseModel]:
        MODELS[name] = cls
        return cls

    return decorator


def build_model(name: str, **kwargs: Any) -> BaseModel:
    if name not in MODELS:
        raise KeyError(f"Unknown model '{name}'. Available: {sorted(MODELS)}")
    return MODELS[name](**kwargs)


def import_builtin_models() -> None:
    import benchmark.models.api.openai_model  # noqa: F401
    import benchmark.models.api.qwen_api_model  # noqa: F401
    import benchmark.models.local.qwen_local_model  # noqa: F401

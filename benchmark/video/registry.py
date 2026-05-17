"""Frame policy registry."""

from __future__ import annotations

from typing import Any, Callable


FRAME_POLICIES: dict[str, type[Any]] = {}


def register_frame_policy(name: str) -> Callable[[type[Any]], type[Any]]:
    def decorator(cls: type[Any]) -> type[Any]:
        FRAME_POLICIES[name] = cls
        return cls

    return decorator


def build_frame_policy(name: str, **kwargs: Any) -> Any:
    if name not in FRAME_POLICIES:
        raise KeyError(f"Unknown frame_policy '{name}'. Available: {sorted(FRAME_POLICIES)}")
    return FRAME_POLICIES[name](**kwargs)


def import_builtin_frame_policies() -> None:
    import benchmark.video.river_policies  # noqa: F401

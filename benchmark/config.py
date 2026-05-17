"""Configuration loader for RIVER evaluation experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import yaml


class Config:
    """Thin wrapper around a nested dict loaded from YAML."""

    def __init__(self, data: dict[str, Any]) -> None:
        self._data = data

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))

    def get(self, key: str, default: Any = None) -> Any:
        """Dot-notation getter, e.g. config.get('model.name')."""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def __getitem__(self, key: str) -> Any:
        return self.get(key)

    def experiment_name(self) -> str:
        return self.get("experiment.name", "unnamed")

    def output_dir(self) -> Path:
        return Path(self.get("experiment.output_dir", "results"))

    def max_samples(self) -> int | None:
        return self.get("experiment.max_samples")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RIVER evaluation.")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML config.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Override max_samples from config.",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace | None = None) -> Config:
    if args is None:
        args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.max_samples is not None:
        cfg._data.setdefault("experiment", {})
        cfg._data["experiment"]["max_samples"] = args.max_samples
    return cfg

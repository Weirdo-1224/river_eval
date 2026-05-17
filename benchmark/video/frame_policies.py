"""Generic frame policy interfaces."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmark.schema import Sample
from benchmark.storage.cache import FrameCache


@dataclass
class FrameGroup:
    name: str
    paths: list[Path]
    timestamps: list[float]


@dataclass
class FrameBundle:
    policy_name: str
    visible_range: list[float]
    long: FrameGroup
    short: FrameGroup

    @property
    def paths(self) -> list[Path]:
        return self.long.paths + self.short.paths

    @property
    def timestamps(self) -> list[float]:
        return self.long.timestamps + self.short.timestamps


class BaseFramePolicy:
    name = "base"

    def sample(self, sample: Sample, cache: FrameCache | None = None) -> FrameBundle:
        raise NotImplementedError

"""Base dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from river_eval.schema import Sample


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def load_samples(self) -> list[Sample]:
        """Load and return all samples."""
        ...

    def __len__(self) -> int:
        samples = self.load_samples()
        return len(samples)

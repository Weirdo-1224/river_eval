"""RIVER Retro-Memory dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.datasets.base import BaseDataset
from benchmark.datasets.registry import register_dataset
from benchmark.schema import Sample
from benchmark.utils.video_resolver import resolve_video_path


@register_dataset("river_retro_memory")
class RiverRetroDataset(BaseDataset):
    """Loads RIVER Retro-Memory annotations and resolves local video paths."""

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        max_samples: int | None = None,
        require_local_video: bool = True,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.video_root = Path(video_root)
        self.max_samples = max_samples
        self.require_local_video = require_local_video
        self._samples: list[Sample] | None = None

    def load_samples(self) -> list[Sample]:
        if self._samples is not None:
            return self._samples

        with self.annotation_path.open("r", encoding="utf-8") as f:
            raw_items: list[dict[str, Any]] = json.load(f)

        samples: list[Sample] = []
        for item in raw_items:
            video_path = resolve_video_path(
                self.video_root,
                item["video_source"],
                item["video_path"],
            )

            if self.require_local_video and not video_path.is_file():
                continue

            samples.append(
                Sample(
                    sample_id=item["question_id"],
                    task_type="retro_memory",
                    video_source=item["video_source"],
                    video_id=item["video_id"],
                    video_path=video_path,
                    question=item["question"],
                    choices=item.get("choices", []),
                    answer=item["correct_answer"],
                    question_time=float(item["question_time"]),
                    time_reference=[float(t) for t in item.get("time_reference", [])],
                    metadata={
                        "question_type": item.get("question_type"),
                        "duration_sec": item.get("duration_sec"),
                        "fps": item.get("fps"),
                    },
                )
            )

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        self._samples = samples
        return samples

"""RIVER dataset adapter supporting Retro-Memory, Live-Perception, and Pro-Response."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from benchmark.datasets.base import BaseDataset
from benchmark.datasets.registry import register_dataset
from benchmark.schema import Sample
from benchmark.utils.video_resolver import resolve_video_path


@register_dataset("river_retro_memory")
@register_dataset("river_live_perception")
@register_dataset("river_pro_response_instant")
@register_dataset("river_pro_response_streaming")
class RiverDataset(BaseDataset):
    """Loads RIVER annotations and resolves local video paths.

    Supports four question types:
    - retro_memory
    - live_perception
    - pro_response_instant
    - pro_response_streaming
    """

    def __init__(
        self,
        annotation_path: str | Path,
        video_root: str | Path,
        max_samples: int | None = None,
        require_local_video: bool = True,
        question_type_filter: str | list[str] | None = None,
    ) -> None:
        self.annotation_path = Path(annotation_path)
        self.video_root = Path(video_root)
        self.max_samples = max_samples
        self.require_local_video = require_local_video
        if isinstance(question_type_filter, str):
            self.question_type_filter = [question_type_filter]
        else:
            self.question_type_filter = question_type_filter
        self._samples: list[Sample] | None = None

    def load_samples(self) -> list[Sample]:
        if self._samples is not None:
            return self._samples

        with self.annotation_path.open("r", encoding="utf-8") as f:
            raw_items: list[dict[str, Any]] = json.load(f)

        samples: list[Sample] = []
        for item in raw_items:
            q_type = item.get("question_type", "")

            # Filter by question type if specified
            if self.question_type_filter is not None:
                matched = any(
                    filt in q_type or q_type in filt
                    for filt in self.question_type_filter
                )
                if not matched:
                    continue

            # Delegate to type-specific parser
            if "Streaming" in q_type:
                samples.extend(self._parse_streaming(item))
            elif "Instant" in q_type:
                samples.extend(self._parse_instant(item))
            else:
                samples.extend(self._parse_single_turn(item))

        if self.max_samples is not None:
            samples = samples[: self.max_samples]

        self._samples = samples
        return samples

    def _resolve_video(self, item: dict[str, Any]) -> Path | None:
        video_path = resolve_video_path(
            self.video_root,
            item["video_source"],
            item["video_path"],
        )
        if self.require_local_video and not video_path.is_file():
            return None
        return video_path

    def _parse_single_turn(self, item: dict[str, Any]) -> list[Sample]:
        """Parse Retro-Memory or Live-Perception (single question per entry)."""
        video_path = self._resolve_video(item)
        if video_path is None:
            return []

        q_type = item.get("question_type", "")
        task_type = "retro_memory" if "Retro" in q_type else "live_perception"

        # time_reference may be a list or a single float
        tr = item.get("time_reference", [])
        if isinstance(tr, (int, float)):
            time_reference = [float(tr)]
        else:
            time_reference = [float(t) for t in tr]

        return [
            Sample(
                sample_id=item["question_id"],
                task_type=task_type,
                video_source=item["video_source"],
                video_id=item["video_id"],
                video_path=video_path,
                question=item["question"],
                choices=item.get("choices", []),
                answer=str(item.get("correct_answer", "")),
                question_time=float(item["question_time"]),
                time_reference=time_reference,
                metadata={
                    "question_type": q_type,
                    "duration_sec": item.get("duration_sec"),
                    "fps": item.get("fps"),
                },
            )
        ]

    def _parse_instant(self, item: dict[str, Any]) -> list[Sample]:
        """Parse Pro-Response-Instant (event detection, single question)."""
        video_path = self._resolve_video(item)
        if video_path is None:
            return []

        q_type = item.get("question_type", "")
        tr = item.get("time_reference", [])
        if isinstance(tr, (int, float)):
            time_reference = [float(tr)]
            event_time = float(tr)
        else:
            time_reference = [float(t) for t in tr]
            event_time = time_reference[0] if time_reference else None

        choices = item.get("choices", [])
        correct_answer = str(item.get("correct_answer", ""))

        return [
            Sample(
                sample_id=item["question_id"],
                task_type="pro_response_instant",
                video_source=item["video_source"],
                video_id=item["video_id"],
                video_path=video_path,
                question=item["question"],
                choices=choices,
                answer=correct_answer,
                question_time=float(item.get("question_time", 0)),
                time_reference=time_reference,
                event_time=event_time,
                is_open_ended=len(choices) == 0,
                metadata={
                    "question_type": q_type,
                    "duration_sec": item.get("duration_sec"),
                    "fps": item.get("fps"),
                },
            )
        ]

    def _parse_streaming(self, item: dict[str, Any]) -> list[Sample]:
        """Parse Pro-Response-Streaming (multi-turn, flattened to samples)."""
        video_path = self._resolve_video(item)
        if video_path is None:
            return []

        q_type = item.get("question_type", "")
        q_ids = item["question_id"]  # list
        questions = item["question"]  # list
        q_times = item["question_time"]  # list
        correct_answers = item["correct_answer"]  # dict: q_id -> list[str]
        time_refs = item["time_reference"]  # dict: q_id -> list[float]

        samples: list[Sample] = []
        dialog_context: list[dict[str, Any]] = []

        for turn_idx, q_id in enumerate(q_ids):
            question = questions[turn_idx]
            q_time = float(q_times[turn_idx])
            answers = correct_answers.get(q_id, [])
            tr = time_refs.get(q_id, [])
            time_reference = [float(t) for t in tr]

            # First answer as primary; store alternatives in metadata
            answer = answers[0] if answers else ""

            samples.append(
                Sample(
                    sample_id=q_id,
                    task_type="pro_response_streaming",
                    video_source=item["video_source"],
                    video_id=item["video_id"],
                    video_path=video_path,
                    question=question,
                    choices=[],  # Streaming is open-ended
                    answer=answer,
                    question_time=q_time,
                    time_reference=time_reference,
                    is_open_ended=True,
                    dialog_turn=turn_idx,
                    dialog_context=list(dialog_context) if dialog_context else None,
                    metadata={
                        "question_type": q_type,
                        "duration_sec": item.get("duration_sec"),
                        "fps": item.get("fps"),
                        "all_correct_answers": answers,
                    },
                )
            )

            # Update context for next turn
            dialog_context.append({"role": "user", "content": question})
            dialog_context.append({"role": "assistant", "content": answer})

        return samples

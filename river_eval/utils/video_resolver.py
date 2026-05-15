"""Resolve annotation video paths to local filesystem paths."""

from __future__ import annotations

from pathlib import Path

# Source-to-directory mapping used by the existing download/check tools.
DEFAULT_SOURCE_DIRS = {
    "Ego-Ego4D-Narration-Val": "Ego-Ego4D-Narration-Val",
    "Ego4D-Narration-Val": "Ego4D-Narration-Val",
    "LVBench": "LVBench",
    "LongVideoBench": "LongVideoBench",
    "QVHighlights-Val": "QVHighlights-Val",
    "Vript-RR": "Vript-RR",
}


def resolve_video_path(video_root: Path, video_source: str, video_path: str) -> Path:
    """Return the absolute path to a local video file.

    Args:
        video_root: Root directory containing source-specific folders,
            e.g. ``data/videos/RIVER``.
        video_source: Source dataset name, e.g. ``Vript-RR``.
        video_path: Filename from the annotation, e.g. ``-n5eIlDgY5w.mp4``.
    """
    source_dir = DEFAULT_SOURCE_DIRS.get(video_source, video_source)
    return video_root / source_dir / video_path

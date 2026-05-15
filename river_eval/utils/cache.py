"""Request cache and frame cache utilities."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


class RequestCache:
    """Disk-based cache for API requests.

    Cache key is a SHA-256 hash of the request content.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def get(self, key: str) -> dict[str, Any] | None:
        path = self._cache_path(key)
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def set(self, key: str, value: dict[str, Any]) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(value, f, ensure_ascii=False)

    @staticmethod
    def make_key(
        model_name: str,
        sample_id: str,
        messages: list[dict[str, Any]],
        image_paths: list[Path],
    ) -> str:
        """Build a deterministic cache key from request components."""
        # Hash image files by content for robustness.
        image_hashes = []
        for p in sorted(image_paths):
            if p.exists():
                h = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
                image_hashes.append(h)
            else:
                image_hashes.append("missing")

        payload = json.dumps(
            {
                "model": model_name,
                "sample_id": sample_id,
                "messages": messages,
                "image_hashes": image_hashes,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class FrameCache:
    """Disk-based cache for extracted video frames."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, video_id: str, start: float, end: float, max_frames: int, resolution: int) -> Path:
        key = f"{video_id}_{start:.3f}_{end:.3f}_{max_frames}_{resolution}"
        return self.cache_dir / key

    def get(
        self,
        video_id: str,
        start: float,
        end: float,
        max_frames: int,
        resolution: int,
    ) -> list[Path] | None:
        """Return cached frame paths if they exist and are complete."""
        path = self._cache_path(video_id, start, end, max_frames, resolution)
        if not path.exists():
            return None
        frames = sorted(path.glob("frame_*.jpg"))
        if len(frames) >= max_frames:
            return frames
        return None

    def prepare_dir(
        self,
        video_id: str,
        start: float,
        end: float,
        max_frames: int,
        resolution: int,
    ) -> Path:
        """Return the cache directory path (create if needed)."""
        path = self._cache_path(video_id, start, end, max_frames, resolution)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def clear(self) -> None:
        """Remove all cached frames."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

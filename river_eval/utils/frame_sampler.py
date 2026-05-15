"""Video frame sampling using ffmpeg (batch extraction)."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from river_eval.utils.cache import FrameCache


FFMPEG = Path(__file__).parents[2] / "bin" / "ffmpeg"


def sample_frames(
    video_path: Path,
    start_time: float,
    end_time: float,
    max_frames: int,
    resolution: int = 448,
    output_format: Literal["jpg", "png"] = "jpg",
    video_id: str | None = None,
    cache: FrameCache | None = None,
) -> list[Path]:
    """Uniformly sample ``max_frames`` frames from ``[start_time, end_time]``.

    Uses a single ffmpeg command with the ``fps`` video filter for efficient
    batch extraction. Extracted frames are resized so that the shorter edge
    equals *resolution* (lanczos).

    Args:
        video_path: Path to the video file.
        start_time: Start of the sampling window (seconds).
        end_time: End of the sampling window (seconds).
        max_frames: Maximum number of frames to extract.
        resolution: Shorter-edge resolution after resize.
        output_format: Image format extension.
        video_id: Optional video identifier for cache key.
        cache: Optional frame cache instance.

    Returns:
        List of paths to the extracted frames, sorted by time.
    """
    vid = video_id or video_path.stem

    # Check frame cache first.
    if cache is not None:
        cached = cache.get(vid, start_time, end_time, max_frames, resolution)
        if cached is not None:
            return cached

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    duration = end_time - start_time
    if duration <= 0 or max_frames <= 0:
        return []

    # Determine output directory (cache if available, else temp).
    if cache is not None:
        out_dir = cache.prepare_dir(vid, start_time, end_time, max_frames, resolution)
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="river_frames_"))

    # Check if frames already exist in the output directory.
    existing = sorted(out_dir.glob("frame_*.jpg"))
    if len(existing) >= max_frames:
        return existing[:max_frames]

    ffmpeg_bin = str(FFMPEG) if FFMPEG.exists() else "ffmpeg"

    # Batch extraction: use fps filter to uniformly sample within the window.
    # fps = max_frames / duration  => one frame every (duration / max_frames) seconds.
    fps = max_frames / duration
    ext = "jpg" if output_format == "jpg" else output_format
    out_pattern = out_dir / f"frame_%04d.{ext}"

    cmd = [
        ffmpeg_bin,
        "-y",
        "-ss",
        str(start_time),
        "-to",
        str(end_time),
        "-i",
        str(video_path),
        "-vf",
        f"fps={fps},scale={resolution}:-1:flags=lanczos",
        "-q:v",
        "2",
        str(out_pattern),
    ]

    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

    frames = sorted(out_dir.glob(f"frame_*.{ext}"))
    # fps filter may produce max_frames+1 frames; cap to requested amount.
    if len(frames) > max_frames:
        # Remove excess frames from the end to keep the set uniform.
        for f in frames[max_frames:]:
            f.unlink()
        frames = frames[:max_frames]

    return frames

"""RIVER Retro-Memory frame selection policies.

The original RIVER LLaVA long-short baseline uses model-specific video
embeddings: a long historical memory and a recent short clip. API models cannot
receive those embeddings directly, so these policies reproduce the controllable
parts of the protocol: timestamp selection, visible range, and trace metadata.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from benchmark.schema import Sample
from benchmark.storage.cache import FrameCache
from benchmark.video.frame_policies import BaseFramePolicy, FrameBundle, FrameGroup, StreamFramePolicy
from benchmark.video.registry import register_frame_policy


def _linspace(start: float, end: float, count: int, include_end: bool = True) -> list[float]:
    if count <= 0:
        return []
    if count == 1:
        return [round(end if include_end else start, 3)]
    denom = count - 1 if include_end else count
    return [round(start + (end - start) * i / denom, 3) for i in range(count)]


def _middle_indices(num_frames: int, num_segments: int) -> list[int]:
    """Match RIVER longshort-off.py get_middle_index."""
    if num_segments <= 0 or num_frames <= 0:
        return []
    interval = max(num_frames // num_segments, 1)
    start = interval // 2
    return [min(start + interval * idx, max(num_frames - 1, 0)) for idx in range(num_segments)]


def _dedupe_sorted(values: list[float]) -> list[float]:
    result: list[float] = []
    for value in sorted(values):
        rounded = round(max(value, 0.0), 3)
        if not result or abs(rounded - result[-1]) > 0.001:
            result.append(rounded)
    return result


def _cache_dir(cache: FrameCache | None, sample: Sample, policy_name: str, timestamps: list[float]) -> Path:
    key_start = timestamps[0] if timestamps else 0.0
    key_end = timestamps[-1] if timestamps else 0.0
    if cache is not None:
        # Include the policy name in the video_id part to avoid collisions with
        # older uniform-prefix frame caches.
        return cache.prepare_dir(
            f"{sample.video_id}_{policy_name}",
            key_start,
            key_end,
            len(timestamps),
            0,
        )
    import tempfile

    return Path(tempfile.mkdtemp(prefix=f"river_{policy_name}_frames_"))


def _extract_frames(
    sample: Sample,
    policy_name: str,
    group_name: str,
    timestamps: list[float],
    resolution: int,
    cache: FrameCache | None,
) -> list[Path]:
    out_dir = _cache_dir(cache, sample, f"{policy_name}_{group_name}", timestamps)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    ffmpeg_bin = str(Path(__file__).parents[2] / "bin" / "ffmpeg")
    if not Path(ffmpeg_bin).exists():
        ffmpeg_bin = "ffmpeg"

    for idx, timestamp in enumerate(timestamps, 1):
        out_path = out_dir / f"frame_{idx:04d}_{timestamp:.3f}.jpg"
        if not out_path.exists():
            last_error: subprocess.CalledProcessError | None = None
            for candidate in (timestamp, max(timestamp - 0.5, 0.0), max(timestamp - 1.0, 0.0)):
                cmd = [
                    ffmpeg_bin,
                    "-y",
                    "-ss",
                    f"{candidate:.3f}",
                    "-i",
                    str(sample.video_path),
                    "-frames:v",
                    "1",
                    "-vf",
                    f"scale={resolution}:-1:flags=lanczos",
                    "-q:v",
                    "2",
                    str(out_path),
                ]
                try:
                    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
                    break
                except subprocess.CalledProcessError as exc:
                    last_error = exc
            else:
                if last_error is not None:
                    raise last_error
        frames.append(out_path)
    return frames


@register_frame_policy("api_uniform_prefix")
class APIUniformPrefixPolicy(BaseFramePolicy):
    """Legacy benchmark_new baseline: uniformly sample the visible prefix."""

    name = "api_uniform_prefix"

    def __init__(self, max_frames: int = 16, frame_resolution: int = 448, **_: object) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution

    def sample(self, sample: Sample, cache: FrameCache | None = None) -> FrameBundle:
        end = max(float(sample.question_time), 0.0)
        timestamps = _linspace(0.0, end, self.max_frames, include_end=True)
        paths = _extract_frames(sample, self.name, "all", timestamps, self.frame_resolution, cache)
        return FrameBundle(
            policy_name=self.name,
            visible_range=[0.0, end],
            long=FrameGroup("long", [], []),
            short=FrameGroup("short", paths, timestamps),
        )


@register_frame_policy("river_offline_16")
class RiverOffline16Policy(BaseFramePolicy):
    """Approximate RIVER longshort-off.py fixed 16-frame offline sampling."""

    name = "river_offline_16"

    def __init__(self, max_frames: int = 16, frame_resolution: int = 448, **_: object) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution

    def sample(self, sample: Sample, cache: FrameCache | None = None) -> FrameBundle:
        fps = float(sample.metadata.get("fps") or 1.0)
        q_frame = max(int(float(sample.question_time) * fps), 1)
        timestamps = [round(i / fps, 3) for i in _middle_indices(q_frame, self.max_frames)]
        timestamps = _dedupe_sorted(timestamps)
        paths = _extract_frames(sample, self.name, "all", timestamps, self.frame_resolution, cache)
        return FrameBundle(
            policy_name=self.name,
            visible_range=[0.0, float(sample.question_time)],
            long=FrameGroup("long", [], []),
            short=FrameGroup("short", paths, timestamps),
        )


@register_frame_policy("river_long_short")
class RiverLongShortPolicy(BaseFramePolicy):
    """API approximation of RIVER longshort.py long memory + short clip.

    Original RIVER encodes many historical frames into compressed long memory
    and separately encodes the recent window. Here we preserve timestamp
    selection and group boundaries, then pass both groups as ordered images to
    an API model.
    """

    name = "river_long_short"

    def __init__(
        self,
        max_frames: int = 16,
        frame_resolution: int = 448,
        short_frames: int = 4,
        short_window_sec: float = 16.0,
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.short_frames = min(max(short_frames, 0), max_frames)
        self.long_frames = max_frames - self.short_frames
        self.short_window_sec = short_window_sec

    def sample(self, sample: Sample, cache: FrameCache | None = None) -> FrameBundle:
        end = max(float(sample.question_time), 0.0)
        short_start = max(0.0, end - self.short_window_sec)
        long_timestamps = (
            _linspace(0.0, short_start, self.long_frames, include_end=False)
            if self.long_frames > 0 and short_start > 0
            else []
        )
        short_timestamps = _linspace(short_start, end, self.short_frames, include_end=True)
        long_timestamps = _dedupe_sorted(long_timestamps)
        short_timestamps = _dedupe_sorted(short_timestamps)
        long_paths = _extract_frames(sample, self.name, "long", long_timestamps, self.frame_resolution, cache)
        short_paths = _extract_frames(sample, self.name, "short", short_timestamps, self.frame_resolution, cache)
        return FrameBundle(
            policy_name=self.name,
            visible_range=[0.0, end],
            long=FrameGroup("long", long_paths, long_timestamps),
            short=FrameGroup("short", short_paths, short_timestamps),
        )


@register_frame_policy("live_perception_window")
class LivePerceptionWindowPolicy(BaseFramePolicy):
    """Frame policy for Live-Perception: sample a recent window before question_time.

    The model can only see the video segment [question_time - window_sec, question_time],
    simulating a live observer who only has access to the current feed.
    """

    name = "live_perception_window"

    def __init__(
        self,
        max_frames: int = 16,
        frame_resolution: int = 448,
        window_sec: float = 8.0,
        **_: object,
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.window_sec = window_sec

    def sample(self, sample: Sample, cache: FrameCache | None = None) -> FrameBundle:
        end = max(float(sample.question_time), 0.0)
        start = max(0.0, end - self.window_sec)
        timestamps = _linspace(start, end, self.max_frames, include_end=True)
        timestamps = _dedupe_sorted(timestamps)
        paths = _extract_frames(sample, self.name, "window", timestamps, self.frame_resolution, cache)
        return FrameBundle(
            policy_name=self.name,
            visible_range=[start, end],
            long=FrameGroup("long", [], []),
            short=FrameGroup("short", paths, timestamps),
        )


@register_frame_policy("sliding_window_stream")
class SlidingWindowStreamPolicy(StreamFramePolicy):
    """Sliding-window frame extraction for online / streaming evaluation.

    Divides the video timeline into overlapping windows:
    - Each window is ``window_sec`` long
    - Windows advance by ``step_sec`` each step
    - Each window extracts ``max_frames`` uniformly

    Used by OnlineRunner for Pro-Response tasks where the model must
    actively monitor the video stream.
    """

    name = "sliding_window_stream"

    def __init__(
        self,
        max_frames: int = 8,
        frame_resolution: int = 448,
        window_sec: float = 8.0,
        step_sec: float = 4.0,
        **_: object,
    ) -> None:
        self.max_frames = max_frames
        self.frame_resolution = frame_resolution
        self.window_sec = window_sec
        self.step_sec = step_sec

    def sample_stream(
        self,
        sample: Sample,
        cache: FrameCache | None = None,
    ) -> list[FrameBundle]:
        duration = float(sample.metadata.get("duration_sec", sample.question_time))
        bundles: list[FrameBundle] = []

        t = 0.0
        while t + self.step_sec <= duration:
            w_start = t
            w_end = min(t + self.window_sec, duration)
            timestamps = _linspace(w_start, w_end, self.max_frames, include_end=True)
            timestamps = _dedupe_sorted(timestamps)
            paths = _extract_frames(
                sample, self.name, f"win_{w_start:.1f}", timestamps,
                self.frame_resolution, cache,
            )
            bundles.append(
                FrameBundle(
                    policy_name=self.name,
                    visible_range=[w_start, w_end],
                    long=FrameGroup("long", [], []),
                    short=FrameGroup("short", paths, timestamps),
                )
            )
            t += self.step_sec

        # Final window if there is remaining tail
        if t < duration:
            w_start = max(0.0, duration - self.window_sec)
            w_end = duration
            timestamps = _linspace(w_start, w_end, self.max_frames, include_end=True)
            timestamps = _dedupe_sorted(timestamps)
            paths = _extract_frames(
                sample, self.name, f"win_{w_start:.1f}", timestamps,
                self.frame_resolution, cache,
            )
            bundles.append(
                FrameBundle(
                    policy_name=self.name,
                    visible_range=[w_start, w_end],
                    long=FrameGroup("long", [], []),
                    short=FrameGroup("short", paths, timestamps),
                )
            )

        return bundles

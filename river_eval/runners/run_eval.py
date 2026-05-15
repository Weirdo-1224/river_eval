"""Main evaluation runner for RIVER Phase 1."""

from __future__ import annotations

import json
import time
from pathlib import Path

from dotenv import load_dotenv

# Load local .env file if present (for API keys).
load_dotenv(Path(__file__).parents[2] / ".env")

from river_eval.config import load_config
from river_eval.datasets.river import RiverRetroDataset
from river_eval.eval.multiple_choice import evaluate_batch
from river_eval.models.openai_model import OpenAIModel
from river_eval.schema import EvalResult
from river_eval.tasks.retro_memory import RetroMemoryTask
from river_eval.utils.cache import FrameCache, RequestCache
from river_eval.utils.cost_tracker import CostTracker
from river_eval.utils.frame_sampler import sample_frames
from river_eval.utils.result_writer import ResultWriter, write_summary


def run() -> None:
    cfg = load_config()

    # --- Cache setup ---
    use_cache = cfg.get("storage.use_cache", True)
    cache_dir_raw = cfg.get("storage.cache_dir", "auto")
    if cache_dir_raw == "auto":
        cache_dir = cfg.output_dir() / "cache"
    else:
        cache_dir = Path(cache_dir_raw)

    request_cache = RequestCache(cache_dir / "requests") if use_cache else None
    frame_cache = FrameCache(cache_dir / "frames") if use_cache else None

    # --- Cost tracking ---
    track_cost = cfg.get("cost.track", True)
    pricing = cfg.get("cost.pricing")
    cost_tracker = CostTracker(pricing=pricing) if track_cost else None

    # --- Dataset ---
    dataset = RiverRetroDataset(
        annotation_path=cfg.get("dataset.annotation_path"),
        video_root=cfg.get("dataset.video_root"),
        max_samples=cfg.max_samples(),
        require_local_video=True,
    )
    samples = dataset.load_samples()
    print(f"Loaded {len(samples)} samples (after filtering for local videos).")
    if not samples:
        print("No samples to evaluate. Exiting.")
        return

    # --- Task ---
    task = RetroMemoryTask(
        max_frames=cfg.get("task.max_frames", 16),
        frame_resolution=cfg.get("task.frame_resolution", 448),
    )

    # --- Model ---
    model_name = cfg.get("model.name", "openai")
    if model_name == "openai":
        model = OpenAIModel(
            model=cfg.get("model.model", "gpt-4o"),
            temperature=cfg.get("model.temperature", 0.0),
            max_tokens=cfg.get("model.max_tokens", 16),
            cache=request_cache,
            cost_tracker=cost_tracker,
        )
    elif model_name == "qwen_api":
        from river_eval.models.qwen_api_model import QwenAPIModel

        model = QwenAPIModel(
            model=cfg.get("model.model", "qwen-vl-plus"),
            temperature=cfg.get("model.temperature", 0.0),
            max_tokens=cfg.get("model.max_tokens", 16),
            cache=request_cache,
            cost_tracker=cost_tracker,
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # --- Output ---
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    cost_path = output_dir / "cost_report.json"

    predictions: list[str] = []
    ground_truths: list[str] = []

    with ResultWriter(results_path) as writer:
        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{len(samples)}] {sample.sample_id}")

            policy = task.get_visibility_policy(sample)
            frames = sample_frames(
                video_path=sample.video_path,
                start_time=policy["window_start"],
                end_time=policy["window_end"],
                max_frames=policy["max_frames"],
                resolution=policy["frame_resolution"],
                video_id=sample.video_id,
                cache=frame_cache,
            )
            print(f"    -> sampled {len(frames)} frames")

            messages = task.build_prompt(sample)
            raw_output = model.generate(messages, frames, sample_id=sample.sample_id)
            predicted = task.parse_output(raw_output)
            latency = getattr(model, "_last_latency", 0.0)
            cached = getattr(model, "_last_cached", False)
            usage = getattr(model, "_last_usage", {})

            is_correct = predicted == sample.answer.upper()
            result = EvalResult(
                sample_id=sample.sample_id,
                predicted=predicted,
                is_correct=is_correct,
                raw_output=raw_output,
                latency_sec=latency,
                metadata={
                    "question": sample.question,
                    "correct_answer": sample.answer,
                    "frame_count": len(frames),
                    "video_source": sample.video_source,
                    "cached": cached,
                    "model_name": cfg.get("model.model"),
                    "visible_range": [policy["window_start"], policy["window_end"]],
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
            )
            writer.write(result)

            predictions.append(raw_output)
            ground_truths.append(sample.answer)
            cache_tag = "[C]" if cached else "[R]"
            print(
                f"    -> {cache_tag} pred={predicted}  gt={sample.answer}  "
                f"correct={is_correct}  lat={latency:.2f}s"
            )

    # --- Evaluation ---
    metrics = evaluate_batch(predictions, ground_truths)
    print(f"\nAccuracy: {metrics.correct}/{metrics.total} = {metrics.accuracy:.2%}")

    write_summary(
        summary_path,
        total=metrics.total,
        correct=metrics.correct,
        accuracy=metrics.accuracy,
        extra={
            "experiment": cfg.experiment_name(),
            "model": cfg.get("model.model"),
            "max_frames": cfg.get("task.max_frames"),
        },
    )
    print(f"Results saved to: {results_path}")
    print(f"Summary saved to: {summary_path}")

    # --- Cost report ---
    if cost_tracker is not None:
        cost_tracker.write_report(cost_path)
        report = cost_tracker.report()
        print(
            f"Cost: ${report.total_cost_usd:.4f}  "
            f"({report.total_input_tokens} + {report.total_output_tokens} tokens, "
            f"{report.total_requests} requests)"
        )
        print(f"Cost report saved to: {cost_path}")


if __name__ == "__main__":
    run()

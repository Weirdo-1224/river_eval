"""Configuration-driven evaluation runner."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

# Load local .env file if present (for API keys).
load_dotenv(Path(__file__).parents[2] / ".env")

from benchmark.config import load_config
from benchmark.datasets.registry import build_dataset, import_builtin_datasets
from benchmark.eval.registry import build_scorer, import_builtin_scorers
from benchmark.models.registry import build_model, import_builtin_models
from benchmark.tasks.registry import build_task, import_builtin_tasks
from benchmark.storage.cache import FrameCache, RequestCache
from benchmark.storage.cost_tracker import CostTracker
from benchmark.storage.result_writer import ResultWriter, write_summary
from benchmark.video.registry import build_frame_policy, import_builtin_frame_policies


def _run_id(exp_name: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}_{exp_name}"


def _estimate_cost(pricing: dict | None, model_name: str, usage: dict) -> float:
    if not pricing:
        return 0.0
    input_price, output_price = pricing.get(model_name, (0.0, 0.0))
    return (
        usage.get("prompt_tokens", 0) * input_price
        + usage.get("completion_tokens", 0) * output_price
    ) / 1000.0


def run() -> None:
    cfg = load_config()
    import_builtin_datasets()
    import_builtin_tasks()
    import_builtin_models()
    import_builtin_frame_policies()
    import_builtin_scorers()

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
    dataset_name = cfg.get("dataset.name", "river_retro_memory")
    dataset = build_dataset(
        dataset_name,
        annotation_path=cfg.get("dataset.annotation_path"),
        video_root=cfg.get("dataset.video_root"),
        max_samples=cfg.max_samples(),
        require_local_video=cfg.get("dataset.require_local_video", True),
    )
    samples = dataset.load_samples()
    print(f"Loaded {len(samples)} samples (after filtering for local videos).")
    if not samples:
        print("No samples to evaluate. Exiting.")
        return

    # --- Task ---
    task_name = cfg.get("task.name", "river_retro_memory")
    prompt_style = cfg.get("prompt.style", cfg.get("task.prompt_style", "river_longshort"))
    task = build_task(
        task_name,
        max_frames=cfg.get("task.max_frames", 16),
        frame_resolution=cfg.get("task.frame_resolution", 448),
        prompt_style=prompt_style,
    )

    # --- Frame policy ---
    frame_policy_name = cfg.get("video.frame_policy", cfg.get("task.frame_policy", "river_long_short"))
    frame_policy = build_frame_policy(
        frame_policy_name,
        max_frames=cfg.get("video.max_frames", cfg.get("task.max_frames", 16)),
        frame_resolution=cfg.get("video.frame_resolution", cfg.get("task.frame_resolution", 448)),
        short_frames=cfg.get("video.short_frames", 4),
        short_window_sec=cfg.get("video.short_window_sec", 16.0),
    )

    # --- Scoring ---
    scoring_name = cfg.get("evaluation.scoring", "river_mcq_accuracy")
    scorer = build_scorer(scoring_name)

    # --- Model ---
    model_name = cfg.get("model.name", "openai")
    model_id = cfg.get("model.model", "gpt-4o")
    model = build_model(
        model_name,
        model=model_id,
        temperature=cfg.get("model.temperature", 0.0),
        max_tokens=cfg.get("model.max_tokens", 16),
        api_key=cfg.get("model.api_key"),
        base_url=cfg.get("model.base_url"),
        cache=request_cache,
        cost_tracker=cost_tracker,
        cache_errors=cfg.get("storage.cache_errors", False),
        reuse_failed_cache=cfg.get("storage.reuse_failed_cache", False),
    )

    # --- Output ---
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    cost_path = output_dir / "cost_report.json"
    run_id = _run_id(cfg.experiment_name())

    total = 0
    correct = 0

    with ResultWriter(results_path) as writer:
        for idx, sample in enumerate(samples, 1):
            print(f"[{idx}/{len(samples)}] {sample.sample_id}")

            frame_bundle = frame_policy.sample(sample, cache=frame_cache)
            print(
                f"    -> sampled {len(frame_bundle.paths)} frames "
                f"({len(frame_bundle.long.paths)} long + {len(frame_bundle.short.paths)} short)"
            )

            messages = task.build_prompt(sample, frame_bundle=frame_bundle)
            raw_output = model.generate(messages, frame_bundle.paths, sample_id=sample.sample_id)
            predicted = task.parse_output(raw_output, sample=sample)
            latency = getattr(model, "_last_latency", 0.0)
            cached = getattr(model, "_last_cached", False)
            usage = getattr(model, "_last_usage", {})
            status = getattr(model, "_last_status", "success")
            error = getattr(model, "_last_error", None)

            score = scorer(predicted, sample)
            total += 1
            correct += 1 if score.get("is_correct") else 0
            record = {
                "run_id": run_id,
                "sample_id": sample.sample_id,
                "dataset_name": dataset_name,
                "task_name": task_name,
                "model_name": model_id,
                "frame_policy": frame_policy_name,
                "prompt_style": prompt_style,
                "scoring": scoring_name,
                "video_path": str(sample.video_path),
                "video_source": sample.video_source,
                "visible_range": frame_bundle.visible_range,
                "long_frame_timestamps": frame_bundle.long.timestamps,
                "short_frame_timestamps": frame_bundle.short.timestamps,
                "frame_timestamps": frame_bundle.timestamps,
                "question": sample.question,
                "choices": sample.choices,
                "answer": sample.answer,
                "raw_output": raw_output,
                "prediction": predicted,
                "score": score,
                "status": status,
                "error": error,
                "runtime": {
                    "latency_sec": latency,
                    "cached": cached,
                    "input_tokens": usage.get("prompt_tokens", 0),
                    "output_tokens": usage.get("completion_tokens", 0),
                },
                "cost": {
                    "estimated_usd": _estimate_cost(pricing, model_id, usage),
                    "usage": usage,
                },
                "metadata": sample.metadata,
            }
            writer.write_record(record)

            cache_tag = "[C]" if cached else "[R]"
            print(
                f"    -> {cache_tag} pred={predicted}  gt={sample.answer}  "
                f"correct={score.get('is_correct')}  lat={latency:.2f}s"
            )

    # --- Evaluation ---
    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")

    write_summary(
        summary_path,
        total=total,
        correct=correct,
        accuracy=accuracy,
        extra={
            "experiment": cfg.experiment_name(),
            "run_id": run_id,
            "dataset": dataset_name,
            "task": task_name,
            "model": model_id,
            "frame_policy": frame_policy_name,
            "prompt_style": prompt_style,
            "scoring": scoring_name,
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

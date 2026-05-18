"""Online evaluation runner for Pro-Response tasks.

Processes video streams with sliding windows, detects events, and scores
time-line based metrics (hit rate, precision, latency).
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Load local .env file if present (for API keys).
load_dotenv(Path(__file__).parents[2] / ".env")

from benchmark.config import load_config
from benchmark.datasets.registry import build_dataset, import_builtin_datasets
from benchmark.eval.registry import build_scorer, import_builtin_scorers
from benchmark.models.registry import build_model, import_builtin_models
from benchmark.schema import EventTrace
from benchmark.storage.cache import FrameCache, RequestCache
from benchmark.storage.cost_tracker import CostTracker
from benchmark.storage.result_writer import ResultWriter, write_summary
from benchmark.tasks.registry import build_task, import_builtin_tasks
from benchmark.video.frame_policies import StreamFramePolicy
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
    dataset_name = cfg.get("dataset.name", "river_pro_response_instant")
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

    # Group samples by video_id (Pro-Response: one video -> one monitoring task)
    video_groups: dict[str, list[Any]] = defaultdict(list)
    for s in samples:
        video_groups[s.video_id].append(s)

    # --- Task ---
    task_name = cfg.get("task.name", "river_pro_response_instant")
    task = build_task(
        task_name,
        max_frames=cfg.get("task.max_frames", 8),
        frame_resolution=cfg.get("task.frame_resolution", 448),
    )

    # --- Frame policy ---
    frame_policy_name = cfg.get("video.frame_policy", "sliding_window_stream")
    frame_policy = build_frame_policy(
        frame_policy_name,
        max_frames=cfg.get("video.max_frames", cfg.get("task.max_frames", 8)),
        frame_resolution=cfg.get("video.frame_resolution", cfg.get("task.frame_resolution", 448)),
        window_sec=cfg.get("video.window_sec", 8.0),
        step_sec=cfg.get("video.step_sec", 4.0),
    )

    # --- Scoring ---
    scoring_name = cfg.get("evaluation.scoring", "pro_response_event_detection")
    scorer = build_scorer(scoring_name)
    tolerance_sec = cfg.get("evaluation.tolerance_sec", 4.0)

    # --- Model ---
    model_name = cfg.get("model.name", "qwen_api")
    model_id = cfg.get("model.model", "qwen-vl-plus")
    model_kwargs = {
        "model": model_id,
        "temperature": cfg.get("model.temperature", 0.0),
        "max_tokens": cfg.get("model.max_tokens", 16),
        "api_key": cfg.get("model.api_key"),
        "base_url": cfg.get("model.base_url"),
        "cache": request_cache,
        "cost_tracker": cost_tracker,
        "cache_errors": cfg.get("storage.cache_errors", False),
        "reuse_failed_cache": cfg.get("storage.reuse_failed_cache", False),
    }
    for optional_key in (
        "model_path",
        "torch_dtype",
        "device_map",
        "trust_remote_code",
        "attn_implementation",
        "max_frames",
    ):
        value = cfg.get(f"model.{optional_key}")
        if value is not None:
            model_kwargs[optional_key] = value
    model = build_model(model_name, **model_kwargs)

    # --- Output ---
    output_dir = cfg.output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    cost_path = output_dir / "cost_report.json"
    run_id = _run_id(cfg.experiment_name())

    total_videos = 0
    total_hits = 0
    total_fp = 0
    total_latency = 0.0
    latency_count = 0

    with ResultWriter(results_path) as writer:
        for video_idx, (video_id, video_samples) in enumerate(video_groups.items(), 1):
            # Use the first sample as the representative for this video
            representative = video_samples[0]
            print(f"[{video_idx}/{len(video_groups)}] Video: {video_id}")

            # Extract frame bundles via streaming policy
            if isinstance(frame_policy, StreamFramePolicy):
                bundles = frame_policy.sample_stream(representative, cache=frame_cache)
            else:
                bundles = [frame_policy.sample(representative, cache=frame_cache)]

            events: list[EventTrace] = []
            window_idx = 0

            for bundle in bundles:
                window_idx += 1
                print(
                    f"  [W{window_idx}] {bundle.visible_range[0]:.1f}s - {bundle.visible_range[1]:.1f}s "
                    f"({len(bundle.paths)} frames)"
                )

                try:
                    messages = task.build_prompt(representative, frame_bundle=bundle)
                    raw_output = model.generate(
                        messages, bundle.paths, sample_id=f"{video_id}_w{window_idx}"
                    )
                    parsed = task.parse_output(raw_output, sample=representative)
                    latency = getattr(model, "_last_latency", 0.0)
                    cached = getattr(model, "_last_cached", False)
                    usage = getattr(model, "_last_usage", {})
                    status = getattr(model, "_last_status", "success")
                    error = getattr(model, "_last_error", None)

                    triggered = parsed.get("triggered", False) if isinstance(parsed, dict) else False

                    cache_tag = "[C]" if cached else "[R]"
                    if triggered:
                        print(f"    -> {cache_tag} TRIGGERED: {raw_output[:60]}...")
                        events.append(
                            EventTrace(
                                trigger_time=(bundle.visible_range[0] + bundle.visible_range[1]) / 2,
                                description=raw_output,
                                window_start=bundle.visible_range[0],
                                window_end=bundle.visible_range[1],
                                raw_output=raw_output,
                                latency_sec=latency,
                                metadata={
                                    "window_idx": window_idx,
                                    "input_tokens": usage.get("prompt_tokens", 0),
                                    "output_tokens": usage.get("completion_tokens", 0),
                                    "cached": cached,
                                    "status": status,
                                    "error": error,
                                },
                            )
                        )
                    else:
                        print(f"    -> {cache_tag} no event")

                except Exception as exc:
                    print(f"    -> [E] error: {exc}")

            # Score the detected events against ground truth
            score = scorer(
                [ev.__dict__ for ev in events],
                representative,
                tolerance_sec=tolerance_sec,
            )

            hit = score.get("hit", False)
            total_videos += 1
            if hit:
                total_hits += 1
            total_fp += score.get("false_positives", 0)
            lat = score.get("latency_sec")
            if lat is not None:
                total_latency += lat
                latency_count += 1

            record = {
                "run_id": run_id,
                "video_id": video_id,
                "sample_id": representative.sample_id,
                "dataset_name": dataset_name,
                "task_name": task_name,
                "model_name": model_id,
                "frame_policy": frame_policy_name,
                "scoring": scoring_name,
                "video_path": str(representative.video_path),
                "video_source": representative.video_source,
                "event_time": representative.event_time,
                "predicted_events": [ev.__dict__ for ev in events],
                "event_count": len(events),
                "score": score,
                "question": representative.question,
                "ground_truth_answer": representative.answer,
            }
            writer.write_record(record)

            print(
                f"  -> Hit={hit} | Events={len(events)} | FP={score.get('false_positives', 0)} | "
                f"Latency={score.get('latency_sec')} | F1={score.get('f1', 0):.2f}"
            )

    # --- Evaluation summary ---
    recall = total_hits / total_videos if total_videos else 0.0
    avg_latency = total_latency / latency_count if latency_count else 0.0

    print(f"\n{'=' * 50}")
    print(f"Online Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"Videos evaluated: {total_videos}")
    print(f"Hits: {total_hits} / {total_videos} = {recall:.2%}")
    print(f"Total false positives: {total_fp}")
    print(f"Average latency: {avg_latency:.2f}s")

    write_summary(
        summary_path,
        total=total_videos,
        correct=total_hits,
        accuracy=recall,
        extra={
            "experiment": cfg.experiment_name(),
            "run_id": run_id,
            "dataset": dataset_name,
            "task": task_name,
            "model": model_id,
            "frame_policy": frame_policy_name,
            "scoring": scoring_name,
            "tolerance_sec": tolerance_sec,
            "total_false_positives": total_fp,
            "average_latency_sec": avg_latency,
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

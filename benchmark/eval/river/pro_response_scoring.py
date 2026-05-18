"""Pro-Response scoring utilities.

Metrics for event detection tasks:
- Hit / Miss (recall)
- Precision (true positives vs all triggers)
- False positive rate
- Response latency
- F1 score
"""

from __future__ import annotations

from typing import Any

from benchmark.eval.registry import register_scorer
from benchmark.schema import Sample


@register_scorer("pro_response_event_detection")
def score_event_detection(
    predicted_events: list[dict[str, Any]],
    sample: Sample,
    tolerance_sec: float = 4.0,
) -> dict[str, Any]:
    """Score event detection performance.

    Args:
        predicted_events: List of detected events, each with at least
            {"trigger_time": float, ...}.
        sample: Ground-truth sample with ``event_time`` set.
        tolerance_sec: Time window within which a prediction is considered a hit.

    Returns:
        Dict with hit, latency, precision, recall, f1, false_positives.
    """
    gt_time = sample.event_time
    if gt_time is None:
        # No ground truth event time — treat as impossible to score
        return {
            "hit": False,
            "latency_sec": None,
            "false_positives": len(predicted_events),
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "predicted_count": len(predicted_events),
            "ground_truth_time": None,
        }

    # Match predictions to ground truth
    matched = False
    best_latency: float | None = None
    matched_event: dict[str, Any] | None = None

    for ev in predicted_events:
        pred_time = ev.get("trigger_time")
        if pred_time is None:
            continue
        latency = abs(pred_time - gt_time)
        if latency <= tolerance_sec:
            matched = True
            if best_latency is None or latency < best_latency:
                best_latency = latency
                matched_event = ev

    tp = 1 if matched else 0
    fp = len(predicted_events) - tp
    # If no predictions and no ground truth, precision is undefined; default to 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = 1.0 if matched else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "hit": matched,
        "latency_sec": best_latency,
        "false_positives": fp,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "predicted_count": len(predicted_events),
        "ground_truth_time": gt_time,
        "matched_event": matched_event,
        "is_correct": matched,  # alias for compatibility
    }


@register_scorer("pro_response_open_ended_accuracy")
def score_open_ended(
    prediction: str | None,
    sample: Sample,
) -> dict[str, Any]:
    """Score open-ended Pro-Response answers by keyword containment.

    For samples with ``all_correct_answers`` in metadata, checks whether the
    prediction contains any of the acceptable keywords.
    """
    if prediction is None:
        return {
            "accuracy": 0.0,
            "is_correct": False,
            "match_type": "none",
        }

    pred_lower = prediction.lower()
    correct_answers = sample.metadata.get("all_correct_answers", [sample.answer])

    # Try exact match first
    for ans in correct_answers:
        if ans.lower() == pred_lower:
            return {
                "accuracy": 1.0,
                "is_correct": True,
                "match_type": "exact",
            }

    # Try keyword containment
    for ans in correct_answers:
        # Extract key nouns/verbs (simple heuristic: words longer than 3 chars)
        keywords = [w.lower() for w in ans.split() if len(w) > 3]
        if keywords and all(kw in pred_lower for kw in keywords[:3]):
            return {
                "accuracy": 1.0,
                "is_correct": True,
                "match_type": "keyword",
            }

    # Try overlap ratio
    pred_words = set(pred_lower.split())
    for ans in correct_answers:
        ans_words = set(ans.lower().split())
        if ans_words:
            overlap = len(pred_words & ans_words) / len(ans_words)
            if overlap >= 0.5:
                return {
                    "accuracy": 1.0,
                    "is_correct": True,
                    "match_type": "overlap",
                }

    return {
        "accuracy": 0.0,
        "is_correct": False,
        "match_type": "none",
    }

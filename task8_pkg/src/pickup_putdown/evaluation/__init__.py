"""Shared two-pass temporal evaluator (task_8)."""
from .contracts import (
    EvaluationEvent, EvaluationPrediction, EvaluationIgnoreInterval, MatchResult,
)
from .intervals import Criterion, midpoint_distance, overlaps, tiou
from .class_aware_matching import (
    evaluate_class_aware, match_one_to_one, match_ranked, drop_ignored,
)
from .confusion_matching import evaluate_confusion
from .ap import average_precision, mean_ap
from .metrics import aggregate_metrics, slice_metrics
from .report import failure_gallery, metrics_to_json, render_html, render_markdown
from .io import (
    events_from_rows, ignores_from_rows, load_events_csv,
    load_predictions_csv, predictions_from_rows,
)

__all__ = [
    "EvaluationEvent", "EvaluationPrediction", "EvaluationIgnoreInterval", "MatchResult",
    "Criterion", "tiou", "midpoint_distance", "overlaps",
    "match_one_to_one", "match_ranked", "drop_ignored",
    "evaluate_class_aware", "evaluate_confusion", "average_precision", "mean_ap",
    "aggregate_metrics", "slice_metrics",
    "render_markdown", "render_html", "metrics_to_json", "failure_gallery",
    "events_from_rows", "predictions_from_rows", "ignores_from_rows",
    "load_events_csv", "load_predictions_csv",
]

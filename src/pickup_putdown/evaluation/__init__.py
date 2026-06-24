"""Shared two-pass temporal evaluator (task_8)."""

from .ap import average_precision, mean_ap
from .class_aware_matching import (
    drop_ignored,
    evaluate_class_aware,
    match_one_to_one,
    match_ranked,
)
from .confusion_matching import evaluate_confusion
from .contracts import (
    EvaluationEvent,
    EvaluationIgnoreInterval,
    EvaluationPrediction,
    MatchResult,
)
from .intervals import Criterion, midpoint_distance, overlaps, tiou
from .io import (
    events_from_rows,
    ignores_from_rows,
    load_events_csv,
    load_predictions_csv,
    predictions_from_rows,
)
from .metrics import aggregate_metrics, slice_metrics
from .report import failure_gallery, metrics_to_json, render_html, render_markdown

__all__ = [
    "EvaluationEvent",
    "EvaluationPrediction",
    "EvaluationIgnoreInterval",
    "MatchResult",
    "Criterion",
    "tiou",
    "midpoint_distance",
    "overlaps",
    "match_one_to_one",
    "match_ranked",
    "drop_ignored",
    "evaluate_class_aware",
    "evaluate_confusion",
    "average_precision",
    "mean_ap",
    "aggregate_metrics",
    "slice_metrics",
    "render_markdown",
    "render_html",
    "metrics_to_json",
    "failure_gallery",
    "events_from_rows",
    "predictions_from_rows",
    "ignores_from_rows",
    "load_events_csv",
    "load_predictions_csv",
]

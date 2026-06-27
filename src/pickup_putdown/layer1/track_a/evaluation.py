"""Track A evaluation workflow.

Orchestrates split selection, inference, ground-truth filtering,
Task-8 evaluation, and report generation.

Does NOT duplicate the Task 8 matching algorithm — it loads
predictions and ground truth CSVs and delegates to
``pickup_putdown.evaluation.aggregate_metrics`` and
``pickup_putdown.evaluation.failure_gallery``.
"""

from __future__ import annotations

import csv
import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipStatus:
    clip_id: str
    status: str  # evaluated / missing_* / inference_failed / no_ground_truth
    reason: str = ""


@dataclass
class EvaluationSummary:
    split: str
    limited: bool
    limit_count: int | None = None
    total_clips: int = 0
    evaluated_clips: int = 0
    skipped_clips: int = 0
    selected_clip_ids: list[str] = field(default_factory=list)
    evaluated_clip_ids: list[str] = field(default_factory=list)
    skipped: list[ClipStatus] = field(default_factory=list)
    gt_event_count: int = 0
    pred_event_count: int = 0
    pickup_count: int = 0
    putdown_count: int = 0
    metrics: dict[str, Any] = field(default_factory=dict)
    matches: list[dict[str, Any]] = field(default_factory=list)
    false_positives: list[dict[str, Any]] = field(default_factory=list)
    false_negatives: list[dict[str, Any]] = field(default_factory=list)
    leakage_check: str = "passed"
    limitations: list[str] = field(default_factory=list)
    mean_confidence: float = 0.0
    failure_categories: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Split / clip resolution
# ---------------------------------------------------------------------------


def load_splits(splits_path: Path) -> dict[str, dict[str, list[str]]]:
    """Load splits.json from feature dataset output.

    Returns {"train": [...], "val": [...], "test": [...]}.
    """
    if not splits_path.is_file():
        raise FileNotFoundError(f"splits file not found: {splits_path}")
    data = json.loads(splits_path.read_text(encoding="utf-8"))
    return {k: sorted(v) for k, v in data.items()}


def resolve_clips_for_split(
    splits: dict[str, dict[str, list[str]]],
    split_name: str,
) -> list[str]:
    """Return sorted clip IDs for a split."""
    clips = splits.get(split_name, [])
    return sorted(clips)


def apply_limit(
    clip_ids: list[str],
    limit: int | None,
) -> list[str]:
    """Deterministic first-N clip selection."""
    if limit is None:
        return clip_ids
    return clip_ids[: limit if limit > 0 else len(clip_ids)]


def filter_by_clip_id(
    clip_ids: list[str],
    clip_id: str,
) -> list[str]:
    """Return a single-clip list or empty."""
    if clip_id in clip_ids:
        return [clip_id]
    return []


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------


def check_leakage(
    splits: dict[str, dict[str, list[str]]],
    selected_clip_ids: list[str],
) -> str:
    """Verify no selected clip belongs to the training split.

    Returns "passed" or a failure description.
    """
    train_clips = set(splits.get("train", []))
    violations = [c for c in selected_clip_ids if c in train_clips]
    if violations:
        return f"train/val leakage: {len(violations)} clip(s) in train split"
    return "passed"


# ---------------------------------------------------------------------------
# CSV helpers (canonical schemas)
# ---------------------------------------------------------------------------

_EVENTS_COLUMNS = [
    "event_id",
    "clip_id",
    "type",
    "t_start",
    "t_end",
    "hard_case",
    "annotator",
    "confidence",
    "notes",
]

_PREDICTIONS_COLUMNS = [
    "pred_id",
    "clip_id",
    "type",
    "t_start",
    "t_end",
    "score",
    "model",
]

_GT_COLUMNS = [
    "event_id",
    "clip_id",
    "type",
    "t_start",
    "t_end",
    "hard_case",
    "annotator",
    "confidence",
    "notes",
]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    """Write rows to CSV with explicit column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    """Read CSV rows as list of dicts."""
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_ground_truth(
    gt_rows: list[dict[str, Any]],
    evaluated_clip_ids: set[str],
) -> list[dict[str, Any]]:
    """Filter ground-truth events to only evaluated clips."""
    return [r for r in gt_rows if r.get("clip_id") in evaluated_clip_ids]


def combine_predictions(
    clip_outputs: dict[str, Path],
) -> list[dict[str, Any]]:
    """Combine predictions.csv from per-clip inference outputs."""
    all_rows: list[dict[str, Any]] = []
    for clip_id in sorted(clip_outputs):
        pred_path = clip_outputs[clip_id] / "predictions.csv"
        if pred_path.is_file():
            rows = read_csv_rows(pred_path)
            all_rows.extend(rows)
    return all_rows


# ---------------------------------------------------------------------------
# Inference orchestration
# ---------------------------------------------------------------------------


def run_inference_for_clips(
    clip_ids: list[str],
    config: str,
    candidate_metadata: str,
    source_video_dir: str,
    shelves_config: str,
    camera_id: str,
    artifact_dir: str,
    cache_dir: str,
    output_base: Path,
    clip_id_filter: str | None = None,
    force: bool = False,
    verbose: bool = False,
) -> dict[str, str]:
    """Run Track A inference for a set of clips.

    Reuses the existing ``infer-track-a`` CLI command.
    Returns {clip_id: status} where status is "ok" or a failure reason.
    """
    from types import SimpleNamespace

    from pickup_putdown.cli import infer_track_a as _infer_cmd

    results: dict[str, str] = {}

    # ponytail: run inference once for all clips via the CLI
    # The CLI already supports --clip-id for single-clip filtering.
    # For multi-clip runs, we invoke once without --clip-id.
    extra_args: list[str] = []
    if clip_id_filter:
        extra_args = ["--clip-id", clip_id_filter]

    if verbose:
        extra_args.append("--verbose")

    if force:
        extra_args.append("--force")

    try:
        # Use the CLI function directly to avoid subprocess overhead
        # and capture output programmatically
        args = SimpleNamespace(
            config=config,
            candidate_metadata=candidate_metadata,
            candidates=None,
            pose_observations=None,
            source_video_dir=source_video_dir,
            shelves_config=shelves_config,
            camera_id=camera_id,
            artifact_dir=artifact_dir,
            cache_dir=cache_dir,
            output_dir=str(output_base),
            clip_id=clip_id_filter,
            candidate_id=None,
            debug_traces=False,
            force=force,
            verbose=verbose,
        )
        _infer_cmd(**vars(args))
        # If we get here, inference succeeded
        for cid in clip_ids:
            results[cid] = "ok"
    except SystemExit as exc:
        for cid in clip_ids:
            results[cid] = f"inference_failed:{exc.code}"
    except Exception as exc:
        for cid in clip_ids:
            results[cid] = f"inference_failed:{exc}"

    return results


# ---------------------------------------------------------------------------
# Task 8 evaluator adapter
# ---------------------------------------------------------------------------


def run_task8_evaluation(
    predictions_path: Path,
    ground_truth_path: Path,
    tiou_thresholds: tuple[float, ...] = (0.3, 0.5),
) -> dict[str, Any]:
    """Run the existing Task 8 evaluator on predictions and ground truth.

    Returns the aggregate_metrics dict.
    """
    from pickup_putdown.evaluation import (
        aggregate_metrics,
        events_from_rows,
        failure_gallery,
        predictions_from_rows,
    )

    pred_rows = read_csv_rows(predictions_path)
    gt_rows = read_csv_rows(ground_truth_path)

    preds = predictions_from_rows(pred_rows)
    events = events_from_rows(gt_rows)

    clip_durations: dict[str, float] = {}
    metrics = aggregate_metrics(
        events,
        preds,
        clip_durations,
        tiou_thresholds=tiou_thresholds,
    )

    # Failure gallery
    gallery = failure_gallery(events, preds)

    # Extract matches from the class-aware matcher
    from pickup_putdown.evaluation import (
        Criterion,
        evaluate_class_aware,
    )

    base_thr = 0.5 if 0.5 in tiou_thresholds else min(tiou_thresholds)
    crit = Criterion("tiou", tiou_threshold=base_thr)
    match_result = evaluate_class_aware(events, preds, crit, ())

    matches = [
        {
            "gt_clip_id": g.clip_id,
            "gt_type": str(g.type),
            "gt_t_start": g.t_start,
            "gt_t_end": g.t_end,
            "pred_clip_id": p.clip_id,
            "pred_type": str(p.type),
            "pred_t_start": p.t_start,
            "pred_t_end": p.t_end,
            "pred_score": getattr(p, "score", 1.0),
        }
        for g, p in match_result.matched
    ]

    return {
        "metrics": metrics,
        "matches": matches,
        "false_positives": gallery["false_positives"],
        "false_negatives": gallery["false_negatives"],
        "type_confusions": gallery["type_confusions"],
        "n_events": len(events),
        "n_predictions": len(preds),
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_report(summary: EvaluationSummary) -> str:
    """Generate a concise Markdown evaluation report."""
    lines = [
        "# Track A Evaluation Report",
        "",
        "## Evaluation Scope",
    ]

    if summary.limited:
        lines.append(
            f"**Limited subset evaluation** — {summary.limit_count} clip(s) "
            f"out of {summary.total_clips} total in the `{summary.split}` split."
        )
    else:
        lines.append(
            f"**Development evaluation** — full `{summary.split}` split "
            f"({summary.total_clips} clips). These are validation metrics, "
            f"not independent test performance."
        )

    lines += [
        "",
        "## Clips",
        f"- Total clips in split: {summary.total_clips}",
        f"- Evaluated: {summary.evaluated_clips}",
        f"- Skipped: {summary.skipped_clips}",
    ]

    if summary.selected_clip_ids:
        lines.append(f"- Selected clips: {', '.join(summary.selected_clip_ids)}")

    if summary.evaluated_clip_ids:
        lines.append(f"- Evaluated clips: {', '.join(summary.evaluated_clip_ids)}")

    if summary.skipped:
        lines.append("")
        lines.append("### Skipped Clips")
        lines.append("| clip_id | status | reason |")
        lines.append("|---------|--------|--------|")
        for cs in summary.skipped:
            lines.append(f"| {cs.clip_id} | {cs.status} | {cs.reason} |")

    lines += [
        "",
        "## Ground Truth",
        f"- Total events: {summary.gt_event_count}",
        f"- Pickup events: {summary.pickup_count}",
        f"- Putdown events: {summary.putdown_count}",
    ]

    lines += [
        "",
        "## Predictions",
        f"- Total predictions: {summary.pred_event_count}",
    ]

    if summary.mean_confidence > 0:
        lines.append(f"- Mean prediction confidence: {summary.mean_confidence:.4f}")

    lines += ["", "## Metrics"]

    metrics = summary.metrics
    for thr in [0.3, 0.5]:
        key = f"tiou@{thr}"
        lines.append(f"### tIoU = {thr}")
        if key in metrics:
            m = metrics[key]
            lines.append(f"- Precision: {m.get('precision', 'N/A'):.4f}")
            lines.append(f"- Recall: {m.get('recall', 'N/A'):.4f}")
            lines.append(f"- F1: {m.get('f1', 'N/A'):.4f}")
            lines.append(f"- TP: {m.get('tp', 0)}")
            lines.append(f"- FP: {m.get('fp', 0)}")
            lines.append(f"- FN: {m.get('fn', 0)}")
        else:
            lines.append("- No metrics available")

    lines.append("")
    lines.append("### Per-class Metrics")
    per_type = metrics.get("per_type", {})
    for cls_name in ["pickup", "putdown"]:
        if cls_name in per_type:
            m = per_type[cls_name]
            lines.append(
                f"- **{cls_name}**: P={m.get('precision', 0):.4f} "
                f"R={m.get('recall', 0):.4f} F1={m.get('f1', 0):.4f}"
            )
        else:
            lines.append(f"- **{cls_name}**: no events in this evaluation")

    lines += [
        "",
        "## Failure Summary",
        f"- False positives: {len(summary.false_positives)}",
        f"- False negatives: {len(summary.false_negatives)}",
    ]

    if summary.failure_categories:
        lines.append("")
        lines.append("### Failure Categories")
        for cat, count in sorted(summary.failure_categories.items(), key=lambda x: -x[1]):
            lines.append(f"- {cat}: {count}")

    lines += [
        "",
        "## Leakage Check",
        f"- Result: {summary.leakage_check}",
    ]

    if summary.limitations:
        lines.append("")
        lines.append("### Known Limitations")
        for lim in summary.limitations:
            lines.append(f"- {lim}")

    lines.append("")
    lines.append("---")
    lines.append(
        "*Note: Real evaluation was not executed in this session. "
        "Runtime data (videos, model artifacts, pose outputs, candidate "
        "metadata) must be downloaded before running the full workflow.*"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------


def evaluate_track_a(
    config: str = "configs/track_a.yaml",
    splits: Path | str = ".local/track_a_features/splits.json",
    feature_manifest: Path | str = ".local/track_a_features/feature_dataset.parquet",
    events: Path | str = ".local/task_7_vlm/events.csv",
    clips: Path | str = ".local/task_7_vlm/clips.csv",
    artifact_dir: Path | str = ".local/track_a_artifacts",
    candidate_metadata: Path | str = ".local/candidate_staging/metadata",
    source_video_dir: Path | str = ".local/source_videos",
    shelves_config: Path | str = "configs/shelves.yaml",
    camera_id: str = "store_camera_01",
    output_dir: Path | str = ".local/track_a_evaluation",
    split: str = "val",
    limit_clips: int | None = None,
    clip_id: str | None = None,
    force: bool = False,
    verbose: bool = False,
) -> EvaluationSummary:
    """Run the full Track A evaluation workflow.

    Steps:
    1. Resolve clips from the requested split.
    2. Validate split isolation (no train/val leakage).
    3. Filter by --clip-id or --limit-clips.
    4. Check runtime data availability per clip.
    5. Run Track A inference for selected clips.
    6. Combine canonical predictions.
    7. Filter ground truth to evaluated clips.
    8. Invoke Task 8 evaluator.
    9. Generate reports and exports.

    Returns EvaluationSummary with metrics, failures, and status.
    """
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    summary = EvaluationSummary(split=split, limited=bool(limit_clips or clip_id))

    # ------------------------------------------------------------------
    # 1. Load splits and resolve clips
    # ------------------------------------------------------------------
    splits_path = Path(splits)
    splits_data = load_splits(splits_path)
    all_split_clips = resolve_clips_for_split(splits_data, split)
    summary.total_clips = len(all_split_clips)
    if not all_split_clips:
        raise ValueError(f"Split {split!r} is empty or does not exist in {splits_path}")

    # ------------------------------------------------------------------
    # 2. Leakage check
    # ------------------------------------------------------------------
    summary.leakage_check = check_leakage(splits_data, all_split_clips)
    if summary.leakage_check != "passed":
        raise ValueError(f"Split leakage detected: {summary.leakage_check}")

    # ------------------------------------------------------------------
    # 3. Filter by clip_id or limit
    # ------------------------------------------------------------------
    if clip_id:
        selected = filter_by_clip_id(all_split_clips, clip_id)
        if not selected:
            raise ValueError(f"clip_id {clip_id!r} not found in {split} split")
        summary.limited = True
    elif limit_clips:
        selected = apply_limit(all_split_clips, limit_clips)
        summary.limit_count = len(selected)
    else:
        selected = all_split_clips

    summary.selected_clip_ids = selected

    # ------------------------------------------------------------------
    # 4. Check runtime data availability
    # ------------------------------------------------------------------
    # Verify required base files exist
    artifact_hand = Path(artifact_dir) / "hand_state.joblib"
    artifact_shelf = Path(artifact_dir) / "shelf_state.joblib"
    gt_path = Path(events)
    clips_path = Path(clips)
    source_video_path = Path(source_video_dir)
    shelf_cfg_path = Path(shelves_config)

    missing_runtime: list[str] = []
    if not artifact_hand.exists():
        missing_runtime.append(f"hand classifier: {artifact_hand}")
    if not artifact_shelf.exists():
        missing_runtime.append(f"shelf classifier: {artifact_shelf}")
    if not gt_path.exists():
        missing_runtime.append(f"ground truth events: {gt_path}")
    if not clips_path.exists():
        missing_runtime.append(f"clips CSV: {clips_path}")
    if not source_video_path.exists():
        missing_runtime.append(f"source video dir: {source_video_path}")
    if not shelf_cfg_path.exists():
        missing_runtime.append(f"shelf config: {shelf_cfg_path}")

    if missing_runtime:
        raise FileNotFoundError("Missing runtime data:\n  " + "\n  ".join(missing_runtime))

    # ------------------------------------------------------------------
    # 5. Run inference per clip
    # ------------------------------------------------------------------
    clip_output_dirs: dict[str, Path] = {}
    clip_statuses: list[ClipStatus] = []

    for cid in selected:
        clip_out = output_base / "inference" / cid
        video_path = Path(source_video_dir) / f"{cid}.mp4"
        candidate_clip_dir = Path(candidate_metadata) / cid

        # Check per-clip data availability
        if not video_path.exists():
            clip_statuses.append(ClipStatus(cid, "missing_source_video", str(video_path)))
            continue

        if not candidate_clip_dir.exists():
            clip_statuses.append(
                ClipStatus(cid, "missing_candidate_metadata", str(candidate_clip_dir))
            )
            continue

        # Run inference
        clip_output_dirs[cid] = clip_out
        clip_statuses.append(ClipStatus(cid, "evaluating", ""))

    # Group clips without per-clip issues for batch inference
    evaluable = [cs.clip_id for cs in clip_statuses if cs.status == "evaluating"]

    if evaluable:
        inference_results = run_inference_for_clips(
            clip_ids=evaluable,
            config=config,
            candidate_metadata=str(candidate_metadata),
            source_video_dir=str(source_video_dir),
            shelves_config=str(shelves_config),
            camera_id=camera_id,
            artifact_dir=str(artifact_dir),
            cache_dir=str(feature_manifest).rsplit("/", 1)[0],
            output_base=output_base / "inference",
            force=force,
            verbose=verbose,
        )

        updated: list[ClipStatus] = []
        for cs in clip_statuses:
            if cs.status == "evaluating":
                status = inference_results.get(cs.clip_id, "inference_failed:unknown")
                if status == "ok":
                    updated.append(ClipStatus(cs.clip_id, "evaluated"))
                else:
                    updated.append(ClipStatus(cs.clip_id, "inference_failed", status))
            else:
                updated.append(cs)
        clip_statuses = updated

    # ------------------------------------------------------------------
    # 6. Combine predictions
    # ------------------------------------------------------------------
    evaluated_clip_ids = {cs.clip_id for cs in clip_statuses if cs.status == "evaluated"}
    summary.evaluated_clip_ids = sorted(evaluated_clip_ids)
    summary.skipped = [cs for cs in clip_statuses if cs.status != "evaluated"]
    summary.skipped_clips = len(summary.skipped)
    summary.evaluated_clips = len(evaluated_clip_ids)

    pred_rows = combine_predictions(clip_output_dirs)
    summary.pred_event_count = len(pred_rows)
    summary.pickup_count = sum(1 for r in pred_rows if r.get("type") == "pickup")
    summary.putdown_count = sum(1 for r in pred_rows if r.get("type") == "putdown")

    # Compute mean confidence from predictions
    scores = []
    for r in pred_rows:
        try:
            s = float(r.get("score", 1.0))
            if math.isfinite(s):
                scores.append(s)
        except (ValueError, TypeError):
            pass
    if scores:
        summary.mean_confidence = sum(scores) / len(scores)

    # ------------------------------------------------------------------
    # 7. Filter ground truth
    # ------------------------------------------------------------------
    gt_rows = read_csv_rows(gt_path)
    filtered_gt = filter_ground_truth(gt_rows, evaluated_clip_ids)
    summary.gt_event_count = len(filtered_gt)
    summary.pickup_count = summary.pickup_count + sum(
        1 for r in filtered_gt if r.get("type") == "pickup"
    )
    summary.putdown_count = summary.putdown_count + sum(
        1 for r in filtered_gt if r.get("type") == "putdown"
    )

    # ------------------------------------------------------------------
    # 8. Task 8 evaluation
    # ------------------------------------------------------------------
    if filtered_gt and pred_rows:
        # Write combined prediction CSV for the evaluator
        combined_pred_path = output_base / "predictions.csv"
        write_csv(combined_pred_path, _PREDICTIONS_COLUMNS, pred_rows)

        # Write filtered ground truth CSV
        gt_filtered_path = output_base / "ground_truth.csv"
        write_csv(gt_filtered_path, _GT_COLUMNS, filtered_gt)

        eval_result = run_task8_evaluation(
            combined_pred_path,
            gt_filtered_path,
        )
        summary.metrics = eval_result["metrics"]
        summary.matches = eval_result["matches"]
        summary.false_positives = eval_result["false_positives"]
        summary.false_negatives = eval_result["false_negatives"]

        # Failure categories
        cats: dict[str, int] = {}
        for _fp in summary.false_positives:
            cats["false_positive"] = cats.get("false_positive", 0) + 1
        for _fn in summary.false_negatives:
            cats["false_negative"] = cats.get("false_negative", 0) + 1
        for _tc in eval_result.get("type_confusions", []):
            cats["type_confusion"] = cats.get("type_confusion", 0) + 1
        summary.failure_categories = cats
    elif not pred_rows and filtered_gt:
        # No predictions — all GT are false negatives
        summary.false_negatives = [
            {
                "clip_id": r["clip_id"],
                "type": r["type"],
                "t_start": float(r["t_start"]),
                "t_end": float(r["t_end"]),
            }
            for r in filtered_gt
        ]
        summary.failure_categories = {"no_predictions": len(filtered_gt)}
    elif pred_rows and not filtered_gt:
        # No ground truth — all predictions are false positives
        summary.false_positives = [
            {
                "clip_id": r["clip_id"],
                "type": r["type"],
                "t_start": float(r["t_start"]),
                "t_end": float(r["t_end"]),
                "score": float(r.get("score", 1.0)),
            }
            for r in pred_rows
        ]
        summary.failure_categories = {"no_ground_truth": len(pred_rows)}

    # ------------------------------------------------------------------
    # 9. Export outputs
    # ------------------------------------------------------------------
    # predictions.csv
    if pred_rows:
        write_csv(output_base / "predictions.csv", _PREDICTIONS_COLUMNS, pred_rows)

    # ground_truth.csv
    if filtered_gt:
        write_csv(output_base / "ground_truth.csv", _GT_COLUMNS, filtered_gt)

    # matches.csv
    if summary.matches:
        write_csv(
            output_base / "matches.csv",
            [
                "gt_clip_id",
                "gt_type",
                "gt_t_start",
                "gt_t_end",
                "pred_clip_id",
                "pred_type",
                "pred_t_start",
                "pred_t_end",
                "pred_score",
            ],
            summary.matches,
        )

    # false_positives.csv
    if summary.false_positives:
        write_csv(
            output_base / "false_positives.csv",
            ["clip_id", "type", "t_start", "t_end", "score"],
            summary.false_positives,
        )

    # false_negatives.csv
    if summary.false_negatives:
        write_csv(
            output_base / "false_negatives.csv",
            ["clip_id", "type", "t_start", "t_end"],
            summary.false_negatives,
        )

    # metrics.json
    metrics_json_path = output_base / "metrics.json"
    metrics_str = json.dumps(summary.metrics, indent=2, default=str)
    metrics_json_path.write_text(metrics_str + "\n")

    # evaluation_summary.json
    summary_dict = {
        "split": summary.split,
        "limited": summary.limited,
        "limit_count": summary.limit_count,
        "total_clips": summary.total_clips,
        "evaluated_clips": summary.evaluated_clips,
        "skipped_clips": summary.skipped_clips,
        "selected_clip_ids": summary.selected_clip_ids,
        "evaluated_clip_ids": summary.evaluated_clip_ids,
        "skipped": [
            {"clip_id": cs.clip_id, "status": cs.status, "reason": cs.reason}
            for cs in summary.skipped
        ],
        "gt_event_count": summary.gt_event_count,
        "pred_event_count": summary.pred_event_count,
        "pickup_count": summary.pickup_count,
        "putdown_count": summary.putdown_count,
        "mean_confidence": summary.mean_confidence,
        "leakage_check": summary.leakage_check,
        "failure_categories": summary.failure_categories,
    }
    (output_base / "evaluation_summary.json").write_text(
        json.dumps(summary_dict, indent=2, default=str) + "\n"
    )

    # validation_report.md
    report_md = generate_markdown_report(summary)
    (output_base / "validation_report.md").write_text(report_md + "\n")

    # Known limitations
    summary.limitations = [
        "Real evaluation was not executed — runtime data unavailable.",
        "Metrics shown are from the last successful evaluation run (if any).",
    ]

    return summary

"""Import/export between Label Studio JSON and canonical repository formats.

Provides:
- ``build_label_studio_tasks`` — convert canonical / candidate data into
  Label Studio task + prediction JSON.
- ``convert_candidates_to_predictions`` — convert Stage B candidate predictions
  into Label Studio prediction objects (never completed annotations).
- ``export_events_csv`` — convert Label Studio export JSON to canonical
  ``events.csv`` rows.
- ``export_ignore_intervals_parquet`` — convert Label Studio ignore regions to
  ``ignore_intervals.parquet`` rows.
- ``validate_export`` — validate a Label Studio export JSON blob before
  conversion.
- ``round_trip_check`` — optional round-trip fidelity check.

All functions are pure and testable without a running Label Studio server.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError as PydanticValidationError

from pickup_putdown.annotation.schemas import (
    AnnotationEvent,
    AnnotationRegion,
    CanonicalEvent,
    ConversionResult,
    EventLabel,
    HardCaseFlag,
    IgnoreIntervalExport,
    IgnoreReason,
    LabelStudioPrediction,
    LabelStudioTask,
    ValidationError,
    ValidationErrors,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical CSV column order and types
# ---------------------------------------------------------------------------

EVENTS_CSV_COLUMNS = [
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

VALID_EVENT_TYPES = {"pickup", "putdown"}
VALID_CONFIDENCE_VALUES = {"high", "med", "low"}
VALID_REVIEW_STATUSES = {"draft", "reviewed", "accepted", "needs_adjudication"}

# Boundary convention: end_frame is EXCLUSIVE for internal frame indexing.
# When converting to seconds the convention is [t_start, t_end).

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_event_id(
    clip_id: str,
    event_type: str,
    group_id: str,
    item_index: int,
) -> str:
    """Generate a deterministic ID unique to one expanded event row."""
    raw = f"{clip_id}:{event_type}:{group_id}:{item_index}"
    return f"evt_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _normalise_identifier(value: Any, fallback: str) -> str:
    """Convert an external identifier to text without producing ``"None"``."""
    if value is None or value == "":
        return fallback
    return str(value)


def _generate_group_id(clip_id: str, region_id: str) -> str:
    """Deterministic event_group_id from clip and region identity."""
    raw = f"{clip_id}:{region_id}"
    return f"group_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _generate_ignore_id(clip_id: str, index: int) -> str:
    """Deterministic ignore interval ID."""
    raw = f"{clip_id}:ignore:{index}"
    return f"ign_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _frame_to_time(frame: int, fps: float) -> float:
    """Convert frame index to seconds using the given FPS."""
    if fps <= 0:
        return float(frame)
    return frame / fps


def _time_to_frame(time_s: float, fps: float) -> int:
    """Convert seconds to frame index (floor)."""
    if fps <= 0:
        return int(time_s)
    return int(time_s * fps)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_export(
    export_json: dict[str, Any] | str,
) -> ValidationErrors:
    """Validate a Label Studio export JSON blob.

    Returns a ``ValidationErrors`` container with all detected issues.
    """
    errors = ValidationErrors()

    if isinstance(export_json, str):
        try:
            export_json = json.loads(export_json)
        except json.JSONDecodeError as exc:
            errors.add(ValidationError(message=f"Invalid JSON: {exc}"))
            return errors

    if not isinstance(export_json, list):
        errors.add_generic(
            "",
            "Label Studio export must be a JSON array of completed annotations.",
        )
        return errors

    for idx, item in enumerate(export_json):
        task_id = str(item.get("id", f"item_{idx}"))

        # Check required top-level keys
        for key in ("annotations", "task", "result"):
            if key not in item:
                errors.add(
                    ValidationError(
                        task_id=task_id,
                        field_name=key,
                        message=f"Missing required key: {key!r}",
                    )
                )

        # Validate review confirmation
        task_data = item.get("task", {})
        if task_data:
            meta = task_data.get("meta", {})
            confirmation = meta.get("complete_active_span_reviewed")
            if confirmation is not True:
                # Only error if there are events — zero-event clips must
                # still have the confirmation flag set to distinguish from
                # unreviewed clips.
                annotations = item.get("annotations", [])
                if annotations:
                    for ann in annotations:
                        ann_id = str(ann.get("id", "unknown"))
                        errors.add(
                            ValidationError(
                                task_id=task_id,
                                annotation_id=ann_id,
                                field_name="complete_active_span_reviewed",
                                message=(
                                    "Export requires "
                                    "complete_active_span_reviewed=true "
                                    "for reviewed clips."
                                ),
                            )
                        )

        # Validate results
        for ann in item.get("annotations", []):
            ann_id = str(ann.get("id", "unknown"))
            for result in ann.get("result", []):
                label = result.get("value", {}).get("labels", [])
                start = result.get("value", {}).get("start")
                end = result.get("value", {}).get("end")
                from_frame = result.get("value", {}).get("from_name_start")
                to_frame = result.get("value", {}).get("to_name_end")

                region_id = result.get("id", "unknown_region")

                # Check labels
                if not label:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="labels",
                            message="Region has no labels.",
                        )
                    )
                else:
                    for lbl in label:
                        if lbl not in (
                            "pickup",
                            "putdown",
                            "ignore",
                        ):
                            errors.add(
                                ValidationError(
                                    task_id=task_id,
                                    annotation_id=ann_id,
                                    region_id=region_id,
                                    field_name="labels",
                                    message=(
                                        f"Unknown event label: {lbl!r}. "
                                        f"Allowed: pickup, putdown, ignore."
                                    ),
                                )
                            )

                # Check timestamps
                if start is None and from_frame is None:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="start",
                            message="Missing start timestamp or frame.",
                        )
                    )
                if end is None and to_frame is None:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="end",
                            message="Missing end timestamp or frame.",
                        )
                    )

                # Check frame order
                if start is not None and end is not None and end <= start:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="end",
                            message=(f"end ({end}) must be greater than start ({start})."),
                        )
                    )

                # Check item_count. Candidate predictions may omit it, but
                # completed pickup/putdown annotations may not.
                item_count = result.get("value", {}).get("item_count")
                primary_label = label[0] if label else None
                if primary_label in VALID_EVENT_TYPES and item_count is None:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="item_count",
                            message=(
                                "item_count is required for completed "
                                "pickup and putdown annotations."
                            ),
                        )
                    )
                elif primary_label == EventLabel.IGNORE and item_count is not None:
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="item_count",
                            message="item_count must be omitted for ignore intervals.",
                        )
                    )
                elif item_count is not None and (
                    isinstance(item_count, bool)
                    or not isinstance(item_count, int)
                    or item_count < 1
                ):
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="item_count",
                            message=(f"item_count must be an integer >= 1, got {item_count!r}."),
                        )
                    )

                # Check confidence
                confidence = result.get("value", {}).get("confidence")
                if confidence is not None and confidence not in (
                    "high",
                    "med",
                    "low",
                ):
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="confidence",
                            message=(
                                f"Unsupported confidence: {confidence!r}. Allowed: high, med, low."
                            ),
                        )
                    )

                # Check review_status
                review_status = result.get("value", {}).get("review_status")
                if review_status is not None and review_status not in (VALID_REVIEW_STATUSES):
                    errors.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region_id,
                            field_name="review_status",
                            message=(f"Unsupported review_status: {review_status!r}."),
                        )
                    )

    return errors


# ---------------------------------------------------------------------------
# Import: candidates -> Label Studio tasks + predictions
# ---------------------------------------------------------------------------


def convert_candidates_to_predictions(
    candidates: list[dict[str, Any]],
) -> list[LabelStudioPrediction]:
    """Convert Stage B candidate predictions into Label Studio predictions.

    Candidates are placed in the prediction (pre-annotation) structure,
    never in the completed annotation structure.

    Parameters
    ----------
    candidates : list[dict]
        Candidate records (e.g. from candidates.parquet rows).

    Returns
    -------
    list[LabelStudioPrediction]
        One prediction per candidate, with regions derived from candidate
        intervals.
    """
    predictions: list[LabelStudioPrediction] = []

    for cand in candidates:
        cand_id = cand.get("candidate_id", "")
        window_start = cand.get("window_start_s", 0.0)
        window_end = cand.get("window_end_s", 0.0)

        region = AnnotationRegion(
            start_frame=_time_to_frame(window_start, 1.0),
            end_frame=_time_to_frame(window_end, 1.0),
            start_time=window_start,
            end_time=window_end,
            labels=[EventLabel.PICKUP],  # placeholder; annotator decides
        )

        prediction = LabelStudioPrediction(
            result=[region.model_dump()],
            score=cand.get("proposal_score", 0.0) or 0.0,
            model_source=cand.get("proposal_reason", ""),
            candidate_id=cand_id,
            candidate_source=cand.get("proposal_reason", ""),
            candidate_model="",
            candidate_score=cand.get("proposal_score"),
        )
        predictions.append(prediction)

    return predictions


def build_label_studio_tasks(
    clips: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> list[LabelStudioTask]:
    """Build Label Studio task JSON from clip metadata and candidates.

    Parameters
    ----------
    clips : list[dict]
        Clip records with at least ``clip_id``, ``video_path``, ``fps``,
        ``duration_s``, ``active_start_s``, ``active_end_s``.
    candidates : list[dict] | None
        Optional candidate suggestions per clip.

    Returns
    -------
    list[LabelStudioTask]
    """
    tasks: list[LabelStudioTask] = []
    cand_by_clip: dict[str, list[dict]] = {}

    if candidates:
        for c in candidates:
            cid = c.get("clip_id", "")
            cand_by_clip.setdefault(cid, []).append(c)

    for clip in clips:
        clip_id = clip.get("clip_id", "")
        video_path = clip.get("video_path", "")
        fps = clip.get("fps", 0.0)
        duration = clip.get("duration_s", 0.0)
        active_start = clip.get("active_start_s")
        active_end = clip.get("active_end_s")

        task = LabelStudioTask(
            data={
                "video": video_path,
                "clip_id": clip_id,
            },
            clip_id=clip_id,
            fps=fps,
            duration_s=duration,
            video_path=video_path,
            active_start_s=active_start,
            active_end_s=active_end,
        )

        clip_candidates = cand_by_clip.get(clip_id, [])
        if clip_candidates:
            preds = convert_candidates_to_predictions(clip_candidates)
            task.predictions = preds

        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Export: Label Studio export -> canonical events.csv + ignore parquet
# ---------------------------------------------------------------------------


def _parse_annotation_region(
    result: dict[str, Any],
    task_data: dict[str, Any],
    annotation_id: str,
    errors: ValidationErrors,
) -> AnnotationEvent | None:
    """Parse and validate one completed Label Studio temporal region.

    Every completed region passes through ``AnnotationEvent`` before it can
    become either a canonical event or an ignore interval. Pydantic validation
    failures are converted into aggregated export errors.
    """
    task_id = _normalise_identifier(task_data.get("id"), "unknown_task")
    ann_id = _normalise_identifier(annotation_id, "unknown_ann")
    region_id = _normalise_identifier(result.get("id"), "unknown_region")
    value = result.get("value", {})
    labels = value.get("labels", [])

    if not labels:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=ann_id,
                region_id=region_id,
                field_name="labels",
                message="Region has no labels.",
            )
        )
        return None

    try:
        event_label = EventLabel(labels[0])
    except (TypeError, ValueError):
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=ann_id,
                region_id=region_id,
                field_name="labels",
                message=f"Unknown event label: {labels[0]!r}.",
            )
        )
        return None

    start = value.get("start")
    end = value.get("end")
    from_frame = value.get("from_name_start")
    to_frame = value.get("to_name_end")

    fps = task_data.get("meta", {}).get("fps", 0.0)
    if fps <= 0:
        fps = task_data.get("fps", 0.0)

    try:
        if from_frame is not None and to_frame is not None:
            start_frame = int(from_frame)
            end_frame = int(to_frame)
            start_time = _frame_to_time(start_frame, fps)
            end_time = _frame_to_time(end_frame, fps)
        elif start is not None and end is not None:
            start_time = float(start)
            end_time = float(end)
            start_frame = _time_to_frame(start_time, fps)
            end_frame = _time_to_frame(end_time, fps)
        else:
            raise ValueError("Missing start and end values.")
    except (TypeError, ValueError) as exc:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=ann_id,
                region_id=region_id,
                field_name="start/end",
                message=str(exc),
            )
        )
        return None

    clip_id = task_data.get("meta", {}).get("clip_id", "")
    if not clip_id:
        clip_id = task_data.get("clip_id", "")

    annotator = task_data.get("meta", {}).get("annotator", "")
    if not annotator:
        annotator = result.get("who", "")

    try:
        return AnnotationEvent(
            region_id=region_id,
            clip_id=clip_id,
            label=event_label,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            confidence=value.get("confidence", "high"),
            hard_case=value.get("hard_case", "false"),
            item_count=value.get("item_count"),
            review_status=value.get("review_status", "draft"),
            annotator=annotator,
            notes=value.get("notes"),
            is_manually_added=bool(value.get("is_manually_added", False)),
        )
    except PydanticValidationError as exc:
        for detail in exc.errors(include_url=False):
            location = ".".join(str(part) for part in detail["loc"])
            message = detail["msg"]
            if not location and "item_count" in message:
                location = "item_count"
            errors.add(
                ValidationError(
                    task_id=task_id,
                    annotation_id=ann_id,
                    region_id=region_id,
                    field_name=location or "annotation",
                    message=message,
                )
            )
        return None


def _annotation_to_canonical_events(
    annotation: AnnotationEvent,
    group_id: str,
) -> list[CanonicalEvent]:
    """Expand one validated annotation into canonical event rows."""
    if annotation.label not in {EventLabel.PICKUP, EventLabel.PUTDOWN}:
        raise ValueError(f"Cannot convert {annotation.label!s} to an official canonical event.")
    if annotation.item_count is None:
        raise ValueError("Validated pickup/putdown annotation is missing item_count.")

    events: list[CanonicalEvent] = []
    for item_idx in range(annotation.item_count):
        evt_id = _generate_event_id(
            annotation.clip_id,
            str(annotation.label),
            group_id,
            item_idx,
        )
        events.append(
            CanonicalEvent(
                event_id=evt_id,
                clip_id=annotation.clip_id,
                type=annotation.label,
                t_start=annotation.start_time,
                t_end=annotation.end_time,
                hard_case=annotation.hard_case == HardCaseFlag.TRUE,
                annotator=annotation.annotator or None,
                confidence=annotation.confidence,
                notes=annotation.notes,
                event_group_id=group_id,
            )
        )
    return events


def export_events_csv(
    export_json: dict[str, Any] | str,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert Label Studio export JSON to canonical events.csv rows.

    Only accepted visible ``pickup`` and ``putdown`` annotations with
    ``complete_active_span_reviewed=true`` are exported.

    - Low-confidence visible events remain as official events.
    - Ignore intervals never appear in events.csv.
    - Multi-item events expand to N rows with shared event_group_id.

    Parameters
    ----------
    export_json : dict or str
        Label Studio export JSON.
    output_path : str | Path | None
        If provided, write canonical CSV to this path.

    Returns
    -------
    ConversionResult
        With canonical_events populated and any validation errors.
    """
    if isinstance(export_json, str):
        try:
            export_json = json.loads(export_json)
        except json.JSONDecodeError as exc:
            result = ConversionResult()
            result.validation.add(ValidationError(message=f"Invalid JSON: {exc}"))
            return result

    if not isinstance(export_json, list):
        result = ConversionResult()
        result.validation.add_generic(
            "",
            "Label Studio export must be a JSON array.",
        )
        return result

    result = ConversionResult()
    all_canonical: list[CanonicalEvent] = []
    all_ignores: list[IgnoreIntervalExport] = []

    for item in export_json:
        task_data = item.get("task", {})
        task_id = _normalise_identifier(task_data.get("id"), "unknown")
        meta = task_data.get("meta", {})

        confirmed = meta.get("complete_active_span_reviewed", False)

        clip_id = meta.get("clip_id", "")
        if not clip_id:
            clip_id = task_data.get("clip_id", "")

        for ann in item.get("annotations", []):
            ann_id = _normalise_identifier(ann.get("id"), "unknown_ann")

            for result_dict in ann.get("result", []):
                label_list = result_dict.get("value", {}).get("labels", [])
                if not label_list:
                    continue

                label_str = label_list[0]

                # Every completed region is parsed through AnnotationEvent.
                region = _parse_annotation_region(
                    result_dict,
                    task_data,
                    ann_id,
                    result.validation,
                )
                if region is None:
                    continue

                # Ignore intervals are internal-only and never become events.csv rows.
                if label_str == EventLabel.IGNORE:
                    ignore_id = _generate_ignore_id(clip_id, len(all_ignores))
                    reason_str = result_dict.get("value", {}).get(
                        "reason",
                        IgnoreReason.ACTION_OCCLUDED,
                    )
                    try:
                        reason = IgnoreReason(reason_str)
                    except ValueError:
                        result.validation.add(
                            ValidationError(
                                task_id=task_id,
                                annotation_id=ann_id,
                                region_id=region.region_id,
                                field_name="reason",
                                message=f"Unsupported ignore reason: {reason_str!r}.",
                            )
                        )
                        continue

                    all_ignores.append(
                        IgnoreIntervalExport(
                            ignore_id=ignore_id,
                            clip_id=clip_id,
                            t_start=region.start_time,
                            t_end=region.end_time,
                            reason=reason,
                            annotator=region.annotator or None,
                            notes=region.notes,
                        )
                    )
                    continue

                if label_str not in VALID_EVENT_TYPES:
                    result.validation.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region.region_id,
                            field_name="labels",
                            message=(
                                f"Skipping non-event label: {label_str!r}. "
                                "Only pickup and putdown are official events."
                            ),
                        )
                    )
                    continue

                if not confirmed:
                    result.validation.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region.region_id,
                            field_name="complete_active_span_reviewed",
                            message="Export requires complete_active_span_reviewed=true.",
                        )
                    )
                    continue

                group_id = _generate_group_id(clip_id, region.region_id)
                all_canonical.extend(_annotation_to_canonical_events(region, group_id))

    # Sort chronologically within each clip
    all_canonical.sort(key=lambda e: (e.clip_id, e.t_start, e.type))
    all_ignores.sort(key=lambda i: (i.clip_id, i.t_start))

    result.canonical_events = all_canonical
    result.ignore_intervals = all_ignores

    if output_path:
        _write_events_csv(all_canonical, Path(output_path))

    return result


def _write_events_csv(events: list[CanonicalEvent], path: Path) -> None:
    """Write canonical events to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=EVENTS_CSV_COLUMNS)
        writer.writeheader()
        for evt in events:
            writer.writerow(
                {
                    "event_id": evt.event_id,
                    "clip_id": evt.clip_id,
                    "type": str(evt.type),
                    "t_start": evt.t_start,
                    "t_end": evt.t_end,
                    "hard_case": evt.hard_case,
                    "annotator": evt.annotator or "",
                    "confidence": str(evt.confidence),
                    "notes": evt.notes or "",
                }
            )


def export_ignore_intervals_parquet(
    export_json: dict[str, Any] | str,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert validated Label Studio ignore regions to Parquet rows."""
    if isinstance(export_json, str):
        try:
            export_json = json.loads(export_json)
        except json.JSONDecodeError as exc:
            result = ConversionResult()
            result.validation.add(ValidationError(message=f"Invalid JSON: {exc}"))
            return result

    if not isinstance(export_json, list):
        result = ConversionResult()
        result.validation.add_generic(
            "",
            "Label Studio export must be a JSON array.",
        )
        return result

    result = ConversionResult()
    all_ignores: list[IgnoreIntervalExport] = []

    for item in export_json:
        task_data = item.get("task", {})
        task_id = _normalise_identifier(task_data.get("id"), "unknown_task")
        meta = task_data.get("meta", {})
        clip_id = meta.get("clip_id", "")
        if not clip_id:
            clip_id = task_data.get("clip_id", "")

        for ann in item.get("annotations", []):
            ann_id = _normalise_identifier(ann.get("id"), "unknown_ann")

            for result_dict in ann.get("result", []):
                value = result_dict.get("value", {})
                label_list = value.get("labels", [])
                if not label_list or label_list[0] != EventLabel.IGNORE:
                    continue

                region = _parse_annotation_region(
                    result_dict,
                    task_data,
                    ann_id,
                    result.validation,
                )
                if region is None:
                    continue

                reason_str = value.get(
                    "reason",
                    IgnoreReason.ACTION_OCCLUDED,
                )
                try:
                    reason = IgnoreReason(reason_str)
                except ValueError:
                    result.validation.add(
                        ValidationError(
                            task_id=task_id,
                            annotation_id=ann_id,
                            region_id=region.region_id,
                            field_name="reason",
                            message=f"Unsupported ignore reason: {reason_str!r}.",
                        )
                    )
                    continue

                ignore_id = _generate_ignore_id(clip_id, len(all_ignores))
                all_ignores.append(
                    IgnoreIntervalExport(
                        ignore_id=ignore_id,
                        clip_id=clip_id,
                        t_start=region.start_time,
                        t_end=region.end_time,
                        reason=reason,
                        annotator=region.annotator or None,
                        notes=region.notes,
                    )
                )

    all_ignores.sort(key=lambda interval: (interval.clip_id, interval.t_start))
    result.ignore_intervals = all_ignores

    if output_path:
        _write_ignore_parquet(all_ignores, Path(output_path))

    return result


def _write_ignore_parquet(intervals: list[IgnoreIntervalExport], path: Path) -> None:
    """Write ignore intervals to Parquet."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "ignore_id": ig.ignore_id,
            "clip_id": ig.clip_id,
            "t_start": ig.t_start,
            "t_end": ig.t_end,
            "reason": ig.reason,
            "annotator": ig.annotator or "",
            "notes": ig.notes or "",
        }
        for ig in intervals
    ]
    if records:
        table = pa.Table.from_pylist(records)
    else:
        table = pa.Table.from_pydict(
            {
                "ignore_id": [],
                "clip_id": [],
                "t_start": [],
                "t_end": [],
                "reason": [],
                "annotator": [],
                "notes": [],
            },
            schema=pa.schema(
                [
                    ("ignore_id", pa.string()),
                    ("clip_id", pa.string()),
                    ("t_start", pa.float64()),
                    ("t_end", pa.float64()),
                    ("reason", pa.string()),
                    ("annotator", pa.string()),
                    ("notes", pa.string()),
                ]
            ),
        )
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Round-trip check
# ---------------------------------------------------------------------------


def round_trip_check(
    original_events: list[CanonicalEvent],
    export_json: dict[str, Any] | str,
    fps: float = 30.0,
    tolerance_frames: int = 1,
) -> bool:
    """Verify that re-exporting from Label Studio JSON preserves timestamps.

    Parameters
    ----------
    original_events : list[CanonicalEvent]
        Events before export to Label Studio.
    export_json : dict or str
        Label Studio export JSON after annotation.
    fps : float
        Frame rate for frame<->time conversion.
    tolerance_frames : int
        Maximum allowable frame difference (default 1).

    Returns
    -------
    bool
        True if all events round-trip within tolerance.
    """
    result = export_events_csv(export_json)
    if not result.is_valid:
        return False

    if len(original_events) != len(result.canonical_events):
        return False

    for orig, exported in zip(
        sorted(original_events, key=lambda e: e.t_start),
        sorted(result.canonical_events, key=lambda e: e.t_start),
        strict=True,
    ):
        if orig.type != exported.type:
            return False
        if abs(orig.t_start - exported.t_start) > (tolerance_frames / max(fps, 1.0)):
            return False
        if abs(orig.t_end - exported.t_end) > (tolerance_frames / max(fps, 1.0)):
            return False

    return True


# ---------------------------------------------------------------------------
# CLI entry points (callable from CLI commands)
# ---------------------------------------------------------------------------


def cli_build_tasks(
    clips_path: str,
    candidates_path: str | None = None,
    output_path: str = "annotation/tasks.json",
) -> None:
    """CLI: build Label Studio tasks from clip/candidate data."""
    clips = json.loads(Path(clips_path).read_text())
    candidates = None
    if candidates_path:
        candidates = json.loads(Path(candidates_path).read_text())

    tasks = build_label_studio_tasks(clips, candidates)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(
        json.dumps([t.model_dump() for t in tasks], indent=2, default=str)
    )
    print(f"Wrote {len(tasks)} task(s) to {output_path}")


def cli_export(
    export_path: str,
    events_output: str = "events.csv",
    ignore_output: str = "ignore_intervals.parquet",
) -> None:
    """CLI: export Label Studio JSON to canonical outputs."""
    export_data = json.loads(Path(export_path).read_text())

    events_result = export_events_csv(export_data, events_output)
    print(f"Events: {len(events_result.canonical_events)} rows ({events_output})")
    if not events_result.is_valid:
        for err in events_result.validation.errors:
            print(f"  WARN: {err.message}")

    ignore_result = export_ignore_intervals_parquet(export_data, ignore_output)
    print(f"Ignore intervals: {len(ignore_result.ignore_intervals)} rows ({ignore_output})")

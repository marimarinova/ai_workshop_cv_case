"""Import/export between Label Studio JSON and canonical repository formats.

The converter supports both:

* real Label Studio Community exports, where tasks contain top-level ``data``
  and TimelineLabels results use ``value.ranges`` / ``value.timelinelabels``;
* the repository's earlier synthetic fixtures, where task metadata is nested
  under ``task`` and region fields are stored in a single result value.

Every completed temporal region is normalised and validated through
``AnnotationEvent`` before it can become a canonical event or ignore interval.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError as PydanticValidationError

from pickup_putdown.annotation.schemas import (
    AnnotationEvent,
    AnnotationUnit,
    CandidateValidationError,
    CanonicalEvent,
    ConversionResult,
    EventLabel,
    HardCaseFlag,
    IgnoreIntervalExport,
    IgnoreReason,
    LabelStudioPrediction,
    LabelStudioTask,
    MediaCheckReport,
    MediaCheckResult,
    ValidationError,
    ValidationErrors,
    VideoUrlMode,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical CSV column order and allowed values
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

# Official canonical columns for Task 8 evaluator compatibility.
# Must match exactly the columns expected by evaluation/io.py and
# common/schemas.Event. No provenance fields.
OFFICIAL_EVENTS_CSV_COLUMNS = [
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

# Official canonical columns for ignore_intervals.parquet.
# No provenance fields.
OFFICIAL_IGNORE_COLUMNS = [
    "ignore_id",
    "clip_id",
    "t_start",
    "t_end",
    "reason",
    "annotator",
    "notes",
]

VALID_EVENT_TYPES = {"pickup", "putdown"}
VALID_CONFIDENCE_VALUES = {"high", "med", "low"}
VALID_REVIEW_STATUSES = {"draft", "reviewed", "accepted", "needs_adjudication"}

# Tolerance in seconds for candidate-boundary checks during export.
# Allows small floating-point or frame-boundary differences.
CANDIDATE_BOUNDARY_TOLERANCE_S = 0.05

# Boundary convention: [start_frame, end_frame), where end_frame is exclusive.


@dataclass(frozen=True)
class _TaskContext:
    """Normalised metadata for either a real or legacy Label Studio task."""

    task_id: str
    clip_id: str
    fps: float
    meta: dict[str, Any]
    data: dict[str, Any]


@dataclass(frozen=True)
class _RegionBundle:
    """One temporal region plus all per-region Label Studio attributes."""

    region_id: str
    label: EventLabel
    start_frame: int | None
    end_frame: int | None
    start_time: float | None
    end_time: float | None
    attributes: dict[str, Any]
    origin: str


# ---------------------------------------------------------------------------
# Generic helpers
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
    """Convert an external identifier to text without producing ``'None'``."""
    if value is None or value == "":
        return fallback
    return str(value)


def _label_studio_region_id(value: Any, fallback_seed: str) -> str:
    """Return a Label Studio-compatible region identifier."""
    raw = _normalise_identifier(value, "")
    if raw and re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        return raw
    digest = hashlib.sha256(f"{raw}:{fallback_seed}".encode()).hexdigest()[:16]
    return f"candidate_{digest}"


def _generate_group_id(clip_id: str, region_id: str) -> str:
    """Generate a deterministic event group ID from clip and region identity."""
    raw = f"{clip_id}:{region_id}"
    return f"group_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"


def _generate_ignore_id(clip_id: str, region_id: str) -> str:
    """Generate a deterministic ignore interval ID."""
    raw = f"{clip_id}:ignore:{region_id}"
    return f"ign_{hashlib.md5(raw.encode()).hexdigest()[:12]}"


def _frame_to_time(frame: int, fps: float) -> float:
    """Convert a frame index to seconds."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero for frame-based regions")
    return frame / fps


def _time_to_frame(time_s: float, fps: float) -> int:
    """Convert seconds to a frame index using floor semantics."""
    if fps <= 0:
        raise ValueError("fps must be greater than zero for time-to-frame conversion")
    return int(time_s * fps)


def _load_export_json(
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
    errors: ValidationErrors,
) -> list[dict[str, Any]] | None:
    """Decode and type-check a Label Studio export payload."""
    if isinstance(export_json, str):
        try:
            export_json = json.loads(export_json)
        except json.JSONDecodeError as exc:
            errors.add(ValidationError(message=f"Invalid JSON: {exc}"))
            return None

    if not isinstance(export_json, list):
        errors.add_generic("", "Label Studio export must be a JSON array.")
        return None

    invalid_index = next(
        (index for index, item in enumerate(export_json) if not isinstance(item, dict)),
        None,
    )
    if invalid_index is not None:
        errors.add_generic(
            "",
            f"Label Studio export item {invalid_index} must be a JSON object.",
        )
        return None

    return export_json


def _first_value(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _task_context(item: dict[str, Any], index: int) -> _TaskContext:
    """Normalise real and legacy task layouts into one context."""
    nested_task = item.get("task")
    task = nested_task if isinstance(nested_task, dict) else item

    item_data = item.get("data") if isinstance(item.get("data"), dict) else {}
    task_data = task.get("data") if isinstance(task.get("data"), dict) else {}
    data = {**task_data, **item_data}

    item_meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    task_meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
    data_meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    meta = {**data_meta, **item_meta, **task_meta}

    task_id = _normalise_identifier(
        _first_value(task.get("id"), item.get("id")),
        f"item_{index}",
    )
    clip_id = str(
        _first_value(
            meta.get("clip_id"),
            data.get("clip_id"),
            task.get("clip_id"),
            item.get("clip_id"),
            default="",
        )
    )
    fps = _as_float(
        _first_value(
            meta.get("fps"),
            data.get("fps"),
            task.get("fps"),
            item.get("fps"),
            default=0.0,
        )
    )

    return _TaskContext(
        task_id=task_id,
        clip_id=clip_id,
        fps=fps,
        meta=meta,
        data=data,
    )


def _annotation_id(annotation: dict[str, Any]) -> str:
    return _normalise_identifier(annotation.get("id"), "unknown_ann")


def _extract_annotator(annotation: dict[str, Any], context: _TaskContext) -> str:
    """Extract annotator identity from real Label Studio annotation metadata."""
    completed_by = annotation.get("completed_by")
    if isinstance(completed_by, dict):
        completed_by = _first_value(
            completed_by.get("email"),
            completed_by.get("username"),
            completed_by.get("id"),
            default="",
        )

    return str(
        _first_value(
            completed_by,
            annotation.get("who"),
            context.meta.get("annotator"),
            context.data.get("annotator"),
            default="",
        )
    )


def _selection_values(value: dict[str, Any]) -> list[Any]:
    """Return values emitted by Choices/Checkbox-style controls."""
    for key in ("choices", "choice", "checkboxes"):
        selected = value.get(key)
        if isinstance(selected, list):
            return selected
        if selected is not None:
            return [selected]
    return []


def _is_truthy_selection(value: Any) -> bool:
    if value is True:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "confirmed"}
    if isinstance(value, list):
        return any(_is_truthy_selection(item) for item in value)
    if isinstance(value, dict):
        return any(_is_truthy_selection(item) for item in value.values())
    return False


def _annotation_review_confirmed(
    annotation: dict[str, Any],
    context: _TaskContext,
) -> bool:
    """Read complete-span confirmation from real or legacy exports."""
    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        if result.get("from_name") != "complete_active_span_reviewed":
            continue
        value = result.get("value", {})
        if isinstance(value, dict) and (
            _is_truthy_selection(_selection_values(value))
            or _is_truthy_selection(value.get("value"))
        ):
            return True

    for state in annotation.get("state", []):
        if not isinstance(state, dict):
            continue
        if state.get("name") != "complete_active_span_reviewed":
            continue
        if _is_truthy_selection(state.get("value")):
            return True

    return _is_truthy_selection(
        _first_value(
            context.meta.get("complete_active_span_reviewed"),
            context.data.get("complete_active_span_reviewed"),
            default=False,
        )
    )


def _candidate_clip_review_confirmed(
    annotation: dict[str, Any],
    context: _TaskContext,
) -> bool:
    """Read candidate-clip review confirmation from exports.

    For candidate-backed tasks, the annotator reviews the candidate clip
    window, not the full active span. This checks the candidate_clip_reviewed
    control or falls back to complete_active_span_reviewed for backward
    compatibility.
    """
    # Check dedicated candidate_clip_reviewed control
    for result in annotation.get("result", []):
        if not isinstance(result, dict):
            continue
        if result.get("from_name") != "candidate_clip_reviewed":
            continue
        value = result.get("value", {})
        if isinstance(value, dict) and (
            _is_truthy_selection(_selection_values(value))
            or _is_truthy_selection(value.get("value"))
        ):
            return True

    for state in annotation.get("state", []):
        if not isinstance(state, dict):
            continue
        if state.get("name") != "candidate_clip_reviewed":
            continue
        if _is_truthy_selection(state.get("value")):
            return True

    if _is_truthy_selection(
        _first_value(
            context.meta.get("candidate_clip_reviewed"),
            context.data.get("candidate_clip_reviewed"),
            default=False,
        )
    ):
        return True

    # Backward compatibility: if complete_active_span_reviewed is set,
    # accept it for candidate tasks too (annotator confirmed they watched
    # the clip, even if the control label was different).
    return _annotation_review_confirmed(annotation, context)


def _is_candidate_backed_task(task_data: dict[str, Any]) -> bool:
    """Check if a task is candidate-backed (has candidate_id in data)."""
    return bool(task_data.get("candidate_id"))


def _normalise_item_count(value: Any) -> Any:
    """Normalise integral Number results while leaving invalid values visible."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = float(stripped)
        except ValueError:
            return value
        if parsed.is_integer():
            return int(parsed)
    return value


def _text_value(value: dict[str, Any]) -> str | None:
    text = value.get("text")
    if isinstance(text, list):
        return "\n".join(str(item) for item in text) if text else None
    if text is None:
        return None
    return str(text)


# ---------------------------------------------------------------------------
# Label Studio result normalisation
# ---------------------------------------------------------------------------


def _extract_region_attributes(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge per-region control results that share a Label Studio region ID."""
    attributes: dict[str, Any] = {}

    for result in results:
        from_name = str(result.get("from_name", ""))
        value = result.get("value", {})
        if not isinstance(value, dict):
            continue

        if from_name == "confidence":
            selected = _selection_values(value)
            if selected:
                attributes["confidence"] = selected[0]
        elif from_name == "hard_case":
            selected = _selection_values(value)
            if selected:
                attributes["hard_case"] = selected[0]
        elif from_name == "item_count":
            number = _first_value(value.get("number"), value.get("item_count"))
            if number is not None:
                attributes["item_count"] = _normalise_item_count(number)
        elif from_name in {"ignore_reason", "reason"}:
            selected = _selection_values(value)
            reason = selected[0] if selected else value.get("reason")
            if reason is not None:
                attributes["reason"] = reason
        elif from_name == "review_status":
            selected = _selection_values(value)
            if selected:
                attributes["review_status"] = selected[0]
        elif from_name == "notes":
            notes = _text_value(value)
            if notes is not None:
                attributes["notes"] = notes

    return attributes


def _coerce_frame(
    value: Any,
    *,
    context: _TaskContext,
    annotation_id: str,
    region_id: str,
    field_name: str,
    errors: ValidationErrors,
) -> int | None:
    """Validate an integral, non-negative frame index."""
    if isinstance(value, bool):
        parsed: float | None = None
    else:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            parsed = None

    if parsed is None or not parsed.is_integer() or parsed < 0:
        errors.add(
            ValidationError(
                task_id=context.task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name=field_name,
                message=f"{field_name} must be a non-negative integer frame, got {value!r}.",
            )
        )
        return None

    return int(parsed)


def _collect_region_bundles(
    annotation: dict[str, Any],
    context: _TaskContext,
    errors: ValidationErrors,
) -> list[_RegionBundle]:
    """Collect real TimelineLabels and legacy synthetic temporal regions."""
    annotation_id = _annotation_id(annotation)
    results = [result for result in annotation.get("result", []) if isinstance(result, dict)]

    results_by_id: dict[str, list[dict[str, Any]]] = {}
    for index, result in enumerate(results):
        region_id = _normalise_identifier(result.get("id"), f"result_{index}")
        results_by_id.setdefault(region_id, []).append(result)

    bundles: list[_RegionBundle] = []

    for index, result in enumerate(results):
        value = result.get("value", {})
        if not isinstance(value, dict):
            continue

        is_real_timeline = (
            str(result.get("type", "")).lower() == "timelinelabels"
            or "timelinelabels" in value
            or "ranges" in value
        )
        is_legacy_region = "labels" in value and any(
            key in value for key in ("start", "end", "from_name_start", "to_name_end")
        )
        if not is_real_timeline and not is_legacy_region:
            continue

        region_id = _normalise_identifier(result.get("id"), f"region_{index}")
        linked_results = results_by_id.get(region_id, [result])
        attributes = _extract_region_attributes(linked_results)

        # Preserve legacy fields stored directly in the temporal result.
        for key in (
            "confidence",
            "hard_case",
            "item_count",
            "review_status",
            "notes",
            "reason",
            "is_manually_added",
        ):
            if key in value and key not in attributes:
                attributes[key] = value[key]
        if "item_count" in attributes:
            attributes["item_count"] = _normalise_item_count(attributes["item_count"])

        if is_real_timeline:
            labels = value.get("timelinelabels", [])
            ranges = value.get("ranges", [])
            if not isinstance(labels, list) or len(labels) != 1:
                errors.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        region_id=region_id,
                        field_name="timelinelabels",
                        message="TimelineLabels region must contain exactly one label.",
                    )
                )
                continue
            if not isinstance(ranges, list) or len(ranges) != 1:
                errors.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        region_id=region_id,
                        field_name="ranges",
                        message="TimelineLabels region must contain exactly one frame range.",
                    )
                )
                continue

            try:
                label = EventLabel(labels[0])
            except (TypeError, ValueError):
                errors.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        region_id=region_id,
                        field_name="timelinelabels",
                        message=f"Unknown event label: {labels[0]!r}.",
                    )
                )
                continue

            frame_range = ranges[0]
            if not isinstance(frame_range, dict):
                errors.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        region_id=region_id,
                        field_name="ranges",
                        message="TimelineLabels range must be an object.",
                    )
                )
                continue

            start_frame = _coerce_frame(
                frame_range.get("start"),
                context=context,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="start_frame",
                errors=errors,
            )
            end_frame = _coerce_frame(
                frame_range.get("end"),
                context=context,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="end_frame",
                errors=errors,
            )
            if start_frame is None or end_frame is None:
                continue

            bundles.append(
                _RegionBundle(
                    region_id=region_id,
                    label=label,
                    start_frame=start_frame,
                    end_frame=end_frame,
                    start_time=None,
                    end_time=None,
                    attributes=attributes,
                    origin=str(result.get("origin", "manual")),
                )
            )
            continue

        # Legacy synthetic fixture format.
        legacy_labels = value.get("labels", [])
        if not isinstance(legacy_labels, list) or len(legacy_labels) != 1:
            errors.add(
                ValidationError(
                    task_id=context.task_id,
                    annotation_id=annotation_id,
                    region_id=region_id,
                    field_name="labels",
                    message="Region must contain exactly one label.",
                )
            )
            continue

        try:
            label = EventLabel(legacy_labels[0])
        except (TypeError, ValueError):
            errors.add(
                ValidationError(
                    task_id=context.task_id,
                    annotation_id=annotation_id,
                    region_id=region_id,
                    field_name="labels",
                    message=f"Unknown event label: {legacy_labels[0]!r}.",
                )
            )
            continue

        from_frame = value.get("from_name_start")
        to_frame = value.get("to_name_end")
        start = value.get("start")
        end = value.get("end")

        start_frame: int | None = None
        end_frame: int | None = None
        start_time: float | None = None
        end_time: float | None = None

        if from_frame is not None and to_frame is not None:
            start_frame = _coerce_frame(
                from_frame,
                context=context,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="start_frame",
                errors=errors,
            )
            end_frame = _coerce_frame(
                to_frame,
                context=context,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="end_frame",
                errors=errors,
            )
        elif start is not None and end is not None:
            try:
                start_time = float(start)
                end_time = float(end)
            except (TypeError, ValueError):
                errors.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        region_id=region_id,
                        field_name="start/end",
                        message="start and end must be numeric timestamps.",
                    )
                )
                continue
        else:
            errors.add(
                ValidationError(
                    task_id=context.task_id,
                    annotation_id=annotation_id,
                    region_id=region_id,
                    field_name="start/end",
                    message="Missing start/end timestamps or frame range.",
                )
            )
            continue

        bundles.append(
            _RegionBundle(
                region_id=region_id,
                label=label,
                start_frame=start_frame,
                end_frame=end_frame,
                start_time=start_time,
                end_time=end_time,
                attributes=attributes,
                origin=str(result.get("origin", "manual")),
            )
        )

    return bundles


def _parse_annotation_region(
    bundle: _RegionBundle,
    context: _TaskContext,
    annotation: dict[str, Any],
    errors: ValidationErrors,
) -> AnnotationEvent | None:
    """Validate one normalised temporal region through ``AnnotationEvent``."""
    annotation_id = _annotation_id(annotation)

    try:
        if bundle.start_frame is not None and bundle.end_frame is not None:
            if context.fps <= 0:
                raise ValueError(
                    "fps is required in task data or metadata for TimelineLabels regions"
                )
            start_frame = bundle.start_frame
            end_frame = bundle.end_frame
            start_time = _frame_to_time(start_frame, context.fps)
            end_time = _frame_to_time(end_frame, context.fps)
        elif bundle.start_time is not None and bundle.end_time is not None:
            start_time = bundle.start_time
            end_time = bundle.end_time
            if context.fps > 0:
                start_frame = _time_to_frame(start_time, context.fps)
                end_frame = _time_to_frame(end_time, context.fps)
            else:
                # Legacy timestamp-only fixtures can still be validated without FPS.
                start_frame = int(start_time)
                end_frame = max(int(end_time), start_frame + 1)
        else:
            raise ValueError("Region has no usable frame range or timestamps.")
    except (TypeError, ValueError) as exc:
        errors.add(
            ValidationError(
                task_id=context.task_id,
                annotation_id=annotation_id,
                region_id=bundle.region_id,
                field_name="start/end",
                message=str(exc),
            )
        )
        return None

    attributes = bundle.attributes
    annotator = _extract_annotator(annotation, context)

    try:
        return AnnotationEvent(
            region_id=bundle.region_id,
            clip_id=context.clip_id,
            label=bundle.label,
            start_frame=start_frame,
            end_frame=end_frame,
            start_time=start_time,
            end_time=end_time,
            confidence=attributes.get("confidence", "high"),
            hard_case=attributes.get("hard_case", "false"),
            item_count=attributes.get("item_count"),
            review_status=attributes.get("review_status", "draft"),
            annotator=annotator,
            notes=attributes.get("notes"),
            is_manually_added=bool(attributes.get("is_manually_added", bundle.origin == "manual")),
        )
    except PydanticValidationError as exc:
        for detail in exc.errors(include_url=False):
            location = ".".join(str(part) for part in detail["loc"])
            message = detail["msg"]
            if not location and "item_count" in message:
                location = "item_count"
            errors.add(
                ValidationError(
                    task_id=context.task_id,
                    annotation_id=annotation_id,
                    region_id=bundle.region_id,
                    field_name=location or "annotation",
                    message=message,
                )
            )
        return None


def _ignore_reason(
    bundle: _RegionBundle,
    context: _TaskContext,
    annotation: dict[str, Any],
    errors: ValidationErrors,
) -> IgnoreReason | None:
    value = bundle.attributes.get("reason", IgnoreReason.ACTION_OCCLUDED)
    try:
        return IgnoreReason(value)
    except (TypeError, ValueError):
        errors.add(
            ValidationError(
                task_id=context.task_id,
                annotation_id=_annotation_id(annotation),
                region_id=bundle.region_id,
                field_name="ignore_reason",
                message=f"Unsupported ignore reason: {value!r}.",
            )
        )
        return None


# ---------------------------------------------------------------------------
# Validation and conversion
# ---------------------------------------------------------------------------


def _convert_export(
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
) -> ConversionResult:
    """Convert and validate one Label Studio export in a single pass."""
    conversion = ConversionResult()
    items = _load_export_json(export_json, conversion.validation)
    if items is None:
        return conversion

    events: list[CanonicalEvent] = []
    ignores: list[IgnoreIntervalExport] = []

    for index, item in enumerate(items):
        context = _task_context(item, index)
        annotations = item.get("annotations")
        if not isinstance(annotations, list):
            conversion.validation.add(
                ValidationError(
                    task_id=context.task_id,
                    field_name="annotations",
                    message="Missing or invalid annotations array.",
                )
            )
            continue

        if not context.clip_id:
            conversion.validation.add(
                ValidationError(
                    task_id=context.task_id,
                    field_name="clip_id",
                    message="clip_id is required in task data or metadata.",
                )
            )

        for annotation in annotations:
            if not isinstance(annotation, dict):
                conversion.validation.add(
                    ValidationError(
                        task_id=context.task_id,
                        field_name="annotations",
                        message="Each annotation must be a JSON object.",
                    )
                )
                continue
            if annotation.get("was_cancelled") is True:
                continue

            annotation_id = _annotation_id(annotation)
            confirmed = _annotation_review_confirmed(annotation, context)
            if not confirmed:
                conversion.validation.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        field_name="complete_active_span_reviewed",
                        message="Export requires complete_active_span_reviewed=true.",
                    )
                )

            bundles = _collect_region_bundles(
                annotation,
                context,
                conversion.validation,
            )
            for bundle in bundles:
                region = _parse_annotation_region(
                    bundle,
                    context,
                    annotation,
                    conversion.validation,
                )
                if region is None or not confirmed:
                    continue

                if region.label is EventLabel.IGNORE:
                    reason = _ignore_reason(
                        bundle,
                        context,
                        annotation,
                        conversion.validation,
                    )
                    if reason is None:
                        continue
                    ignores.append(
                        IgnoreIntervalExport(
                            ignore_id=_generate_ignore_id(
                                context.clip_id,
                                region.region_id,
                            ),
                            clip_id=context.clip_id,
                            t_start=region.start_time,
                            t_end=region.end_time,
                            reason=reason,
                            annotator=region.annotator or None,
                            notes=region.notes,
                        )
                    )
                    continue

                if region.label not in {EventLabel.PICKUP, EventLabel.PUTDOWN}:
                    conversion.validation.add(
                        ValidationError(
                            task_id=context.task_id,
                            annotation_id=annotation_id,
                            region_id=region.region_id,
                            field_name="labels",
                            message=f"Unsupported official event label: {region.label!s}.",
                        )
                    )
                    continue

                group_id = _generate_group_id(context.clip_id, region.region_id)
                events.extend(_annotation_to_canonical_events(region, group_id))

    events.sort(key=lambda event: (event.clip_id, event.t_start, str(event.type), event.event_id))
    ignores.sort(key=lambda interval: (interval.clip_id, interval.t_start, interval.ignore_id))
    conversion.canonical_events = events
    conversion.ignore_intervals = ignores
    return conversion


def validate_export(
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
) -> ValidationErrors:
    """Validate real or legacy Label Studio export JSON.

    Real Label Studio exports use top-level ``data`` and ``annotations``.
    Earlier repository fixtures used nested ``task`` and top-level ``result``;
    those legacy structural checks are retained for backward compatibility.
    """
    conversion = _convert_export(export_json)

    structural_errors = ValidationErrors()
    items = _load_export_json(export_json, structural_errors)
    if items is not None:
        for index, item in enumerate(items):
            task_id = _task_context(item, index).task_id
            if "annotations" not in item:
                structural_errors.add(
                    ValidationError(
                        task_id=task_id,
                        field_name="annotations",
                        message="Missing required key: 'annotations'",
                    )
                )
            if "data" not in item:
                for key in ("task", "result"):
                    if key not in item:
                        structural_errors.add(
                            ValidationError(
                                task_id=task_id,
                                field_name=key,
                                message=f"Missing required key: {key!r}",
                            )
                        )

    conversion.validation.errors.extend(structural_errors.errors)
    return conversion.validation


# ---------------------------------------------------------------------------
# Import: candidates -> real Label Studio tasks and predictions
# ---------------------------------------------------------------------------


def convert_candidates_to_predictions(
    candidates: list[dict[str, Any]],
    fps: float | None = None,
) -> list[LabelStudioPrediction]:
    """Convert Stage B candidates into editable TimelineLabels predictions."""
    predictions: list[LabelStudioPrediction] = []

    for index, candidate in enumerate(candidates):
        candidate_id = _normalise_identifier(
            candidate.get("candidate_id"),
            f"candidate_{index}",
        )
        region_id = _label_studio_region_id(candidate_id, str(index))

        candidate_fps = _as_float(_first_value(candidate.get("fps"), fps, default=0.0))
        start_frame_value = candidate.get("start_frame")
        end_frame_value = candidate.get("end_frame")

        if start_frame_value is not None and end_frame_value is not None:
            start_frame = int(start_frame_value)
            end_frame = int(end_frame_value)
        else:
            window_start = float(candidate.get("window_start_s", 0.0))
            window_end = float(candidate.get("window_end_s", 0.0))
            if candidate_fps <= 0:
                # Backward-compatible fallback for direct calls. build_label_studio_tasks
                # always supplies the clip FPS and should be used for real imports.
                candidate_fps = 1.0
                logger.warning(
                    "Candidate %s has no FPS; treating seconds as frame indices.",
                    candidate_id,
                )
            start_frame = _time_to_frame(window_start, candidate_fps)
            end_frame = _time_to_frame(window_end, candidate_fps)

        if end_frame <= start_frame:
            raise ValueError(
                f"Candidate {candidate_id!r} has invalid frame range [{start_frame}, {end_frame})."
            )

        proposed_label = _first_value(
            candidate.get("event_type"),
            candidate.get("type"),
            default=EventLabel.PICKUP,
        )
        if str(proposed_label) not in VALID_EVENT_TYPES:
            proposed_label = EventLabel.PICKUP

        timeline_result = {
            "id": region_id,
            "from_name": "labels",
            "to_name": "video",
            "type": "timelinelabels",
            "value": {
                "ranges": [{"start": start_frame, "end": end_frame}],
                "timelinelabels": [str(proposed_label)],
            },
        }

        predictions.append(
            LabelStudioPrediction(
                result=[timeline_result],
                score=float(candidate.get("proposal_score", 0.0) or 0.0),
                model_source=str(candidate.get("proposal_reason", "")),
                candidate_id=candidate_id,
                candidate_source=str(candidate.get("proposal_reason", "")),
                candidate_model=str(candidate.get("candidate_model", "")),
                candidate_score=candidate.get("proposal_score"),
            )
        )

    return predictions


def build_label_studio_tasks(
    clips: list[dict[str, Any]],
    candidates: list[dict[str, Any]] | None = None,
) -> list[LabelStudioTask]:
    """Build importable Label Studio tasks with optional predictions."""
    tasks: list[LabelStudioTask] = []
    candidates_by_clip: dict[str, list[dict[str, Any]]] = {}

    for candidate in candidates or []:
        clip_id = str(candidate.get("clip_id", ""))
        candidates_by_clip.setdefault(clip_id, []).append(candidate)

    for clip in clips:
        clip_id = str(clip.get("clip_id", ""))
        video_path = str(clip.get("video_path", ""))
        fps = _as_float(clip.get("fps"))
        duration = _as_float(clip.get("duration_s"))
        active_start = clip.get("active_start_s")
        active_end = clip.get("active_end_s")

        data: dict[str, Any] = {
            "video": video_path,
            "clip_id": clip_id,
            "fps": fps,
            "duration_s": duration,
        }
        if active_start is not None:
            data["active_start_s"] = active_start
        if active_end is not None:
            data["active_end_s"] = active_end
        if clip.get("annotator") is not None:
            data["annotator"] = clip["annotator"]

        task = LabelStudioTask(
            data=data,
            clip_id=clip_id,
            fps=fps,
            duration_s=duration,
            video_path=video_path,
            active_start_s=active_start,
            active_end_s=active_end,
        )

        clip_candidates = candidates_by_clip.get(clip_id, [])
        if clip_candidates:
            task.predictions = convert_candidates_to_predictions(
                clip_candidates,
                fps=fps,
            )

        tasks.append(task)

    return tasks


# ---------------------------------------------------------------------------
# Candidate metadata validation and task building (Task 6.2)
# ---------------------------------------------------------------------------


def validate_candidate_metadata(
    metadata: dict[str, Any],
) -> list[CandidateValidationError]:
    """Validate a single candidate metadata record.

    Required fields: candidate_id, clip_id, source_start_s, source_end_s,
    candidate_video (or candidate_key).

    Returns a list of validation errors. Empty list means valid.
    """
    errors: list[CandidateValidationError] = []
    cid = str(metadata.get("candidate_id", "")) or "unknown"

    required_fields: list[str] = [
        "candidate_id",
        "clip_id",
        "source_start_s",
        "source_end_s",
    ]
    for field in required_fields:
        value = metadata.get(field)
        if value is None or (isinstance(value, str) and value.strip() == ""):
            errors.append(
                CandidateValidationError(
                    candidate_id=cid,
                    field_name=field,
                    message=f"Required field {field!r} is missing or empty.",
                )
            )

    # Video location: candidate_video or candidate_key
    video = metadata.get("candidate_video") or metadata.get("candidate_key")
    if not video or (isinstance(video, str) and video.strip() == ""):
        errors.append(
            CandidateValidationError(
                candidate_id=cid,
                field_name="candidate_video",
                message="Required field candidate_video or candidate_key is missing or empty.",
            )
        )

    # Validate source interval ordering
    try:
        start_s = float(metadata.get("source_start_s", 0))
        end_s = float(metadata.get("source_end_s", 0))
        if start_s >= end_s:
            errors.append(
                CandidateValidationError(
                    candidate_id=cid,
                    field_name="source_start_s",
                    message=f"source_start_s ({start_s}) must be less than source_end_s ({end_s}).",
                )
            )
    except (TypeError, ValueError):
        errors.append(
            CandidateValidationError(
                candidate_id=cid,
                field_name="source_start_s",
                message="source_start_s and source_end_s must be numeric.",
            )
        )

    return errors


def build_candidate_tasks(
    candidate_metadata: list[dict[str, Any]],
    video_url_mode: VideoUrlMode = VideoUrlMode.S3_KEY,
    s3_bucket: str | None = None,
    s3_prefix: str | None = None,
    local_video_dir: str | None = None,
) -> tuple[list[LabelStudioTask], list[CandidateValidationError]]:
    """Build Label Studio tasks from Task 6.1 candidate metadata.

    Each candidate becomes one Label Studio task. No default event label is
    assigned — the annotator must choose the event type explicitly.

    Args:
        candidate_metadata: list of candidate metadata dicts as produced by
            Task 6.1 (see CandidateMetadata schema).
        video_url_mode: How to format the video reference for Label Studio.
        s3_bucket: S3 bucket name (required for s3_storage mode).
        s3_prefix: S3 prefix for candidate videos (default:
            anon/candidates/videos).
        local_video_dir: Local directory for candidate videos (required for
            local mode).

    Returns:
        Tuple of (tasks, errors). Tasks for valid candidates are returned even
        if some candidates produced errors.
    """
    tasks: list[LabelStudioTask] = []
    all_errors: list[CandidateValidationError] = []

    for candidate in candidate_metadata:
        errors = validate_candidate_metadata(candidate)
        if errors:
            all_errors.extend(errors)
            continue

        cid = str(candidate["candidate_id"])
        clip_id = str(candidate["clip_id"])
        source_start_s = float(candidate["source_start_s"])
        source_end_s = float(candidate["source_end_s"])
        raw_video = str(candidate.get("candidate_video") or candidate.get("candidate_key", ""))

        duration_s = candidate.get("duration_s")
        if duration_s is None:
            duration_s = source_end_s - source_start_s
        fps = candidate.get("fps")

        # Generate video URL according to configured mode
        video_url = _generate_video_url(
            raw_video,
            video_url_mode,
            s3_bucket=s3_bucket,
            s3_prefix=s3_prefix,
            local_video_dir=local_video_dir,
        )

        data: dict[str, Any] = {
            "video": video_url,
            "candidate_id": cid,
            "clip_id": clip_id,
            "source_start_s": source_start_s,
            "source_end_s": source_end_s,
            "annotation_unit": AnnotationUnit.CANDIDATE_CLIP,
        }
        # Optional metadata — included when present
        if candidate.get("actor_id"):
            data["actor_id"] = str(candidate["actor_id"])
        if candidate.get("hand_side"):
            data["hand_side"] = str(candidate["hand_side"])
        if candidate.get("region_id"):
            data["region_id"] = str(candidate["region_id"])
        if candidate.get("proposal_score") is not None:
            data["proposal_score"] = float(candidate["proposal_score"])
        if candidate.get("config_fingerprint"):
            data["config_fingerprint"] = str(candidate["config_fingerprint"])
        if fps is not None:
            data["fps"] = float(fps)
        if duration_s is not None:
            data["duration_s"] = float(duration_s)

        task = LabelStudioTask(
            data=data,
            clip_id=clip_id,
            fps=float(fps) if fps else 0.0,
            duration_s=float(duration_s) if duration_s else 0.0,
            video_path=video_url,
        )
        # No predictions — candidate is a possible interaction interval,
        # not a classified event.
        tasks.append(task)

    # Deterministic ordering by candidate_id
    tasks.sort(key=lambda t: t.data.get("candidate_id", ""))
    return tasks, all_errors


def _load_candidate_metadata_from_dir(
    metadata_dir: str | Path,
) -> list[dict[str, Any]]:
    """Load candidate metadata from JSON files in a directory.

    Supports:
    - Individual JSON files (one candidate per file or array of candidates)
    - A single metadata JSON with a "candidates" array (Task 6.1 format)

    Returns all candidates in deterministic order.
    """
    dir_path = Path(metadata_dir)
    if not dir_path.exists():
        raise FileNotFoundError(f"Metadata directory not found: {dir_path}")

    all_candidates: list[dict[str, Any]] = []

    for json_file in sorted(dir_path.glob("*.json")):
        content = json.loads(json_file.read_text())
        if isinstance(content, list):
            all_candidates.extend(content)
        elif isinstance(content, dict):
            # Task 6.1 format: {"candidates": [...], ...}
            if "candidates" in content and isinstance(content["candidates"], list):
                all_candidates.extend(content["candidates"])
            else:
                # Single candidate object
                all_candidates.append(content)

    return all_candidates


# ---------------------------------------------------------------------------
# Video URL generation (Fix 4)
# ---------------------------------------------------------------------------


def _generate_video_url(
    raw_video: str,
    mode: VideoUrlMode,
    *,
    s3_bucket: str | None = None,
    s3_prefix: str | None = "anon/candidates/videos",
    local_video_dir: str | None = None,
) -> str:
    """Generate a Label Studio-accessible video reference.

    Args:
        raw_video: Raw video path from candidate metadata (S3 key, URL, or
            local path).
        mode: Video URL mode determining output format.
        s3_bucket: S3 bucket name (for s3_storage mode).
        s3_prefix: S3 prefix for candidate videos.
        local_video_dir: Local directory for candidate videos (for local mode).

    Returns:
        Label Studio-accessible video reference string.
    """
    if mode == VideoUrlMode.LOCAL:
        if not local_video_dir:
            raise ValueError(
                "local_video_dir is required for local video URL mode. "
                "Set via --local-video-dir or ANNOTATION_VIDEO_DIR."
            )
        # Extract filename from raw path and resolve under local_video_dir
        filename = Path(raw_video).name
        return str(Path(local_video_dir) / filename)

    if mode == VideoUrlMode.S3_STORAGE:
        if not s3_bucket:
            raise ValueError(
                "s3_bucket is required for s3_storage video URL mode. "
                "Set via --s3-bucket or ANNOTATION_S3_BUCKET."
            )
        # Use s3://bucket/key format for Label Studio cloud-storage integration
        if raw_video.startswith("s3://"):
            return raw_video
        return f"s3://{s3_bucket}/{raw_video}"

    if mode == VideoUrlMode.PRESIGNED:
        # Presigned URLs must be generated externally at task-build time.
        # Pass through the raw value (assumed to be a presigned URL).
        if not raw_video.startswith(("http://", "https://")):
            raise ValueError(
                f"presigned mode requires http(s) URL, got: {raw_video!r}. "
                "Generate presigned URLs externally before task building."
            )
        return raw_video

    # Default: s3_key mode — pass through the raw S3 key
    return raw_video


# ---------------------------------------------------------------------------
# Pilot candidate selection (Fix 3)
# ---------------------------------------------------------------------------


def select_candidate_pilot(
    candidate_metadata: list[dict[str, Any]],
    *,
    limit: int | None = None,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Select a deterministic pilot subset of candidates.

    Args:
        candidate_metadata: Full list of candidate metadata dicts.
        limit: Maximum number of candidates to select. None means all valid
            candidates. Must be positive when provided.
        seed: Random seed for reproducible selection.

    Returns:
        Selected candidates in deterministic order.

    Raises:
        ValueError: If limit is non-positive.
        FileNotFoundError: If no valid candidates exist.
    """
    if limit is not None and limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")

    # Filter valid candidates
    valid: list[dict[str, Any]] = []
    for cand in candidate_metadata:
        errors = validate_candidate_metadata(cand)
        if not errors:
            valid.append(cand)

    if not valid:
        raise ValueError("No valid candidates found in metadata")

    if limit is None:
        return sorted(valid, key=lambda c: c.get("candidate_id", ""))

    rng = random.Random(seed)

    # Attempt stratified selection by available fields for representativeness
    # If metadata is incomplete, fall back to simple random sampling
    strata_fields = [
        "clip_id",
        "actor_id",
        "hand_side",
        "region_id",
    ]
    usable_strata: list[str] = []
    for field in strata_fields:
        if all(cand.get(field) for cand in valid):
            usable_strata.append(field)

    if usable_strata:
        # Group by first usable stratum
        stratum: list[str] = usable_strata[0]
        groups: dict[str, list[dict[str, Any]]] = {}
        for cand in valid:
            key = str(cand.get(stratum, ""))
            groups.setdefault(key, []).append(cand)

        # Sort groups deterministically, then sample from each
        selected: list[dict[str, Any]] = []
        sorted_keys = sorted(groups.keys())
        remaining = limit
        for key in sorted_keys:
            if remaining <= 0:
                break
            group = groups[key]
            n_from_group = max(1, len(group) * limit // len(valid))
            n_from_group = min(n_from_group, len(group), remaining)
            rng_copy = random.Random(seed)
            rng_copy.shuffle(group)
            selected.extend(group[:n_from_group])
            remaining -= n_from_group

        # Fill remaining slots from unsampled candidates
        selected_ids = {c.get("candidate_id") for c in selected}
        pool = [c for c in valid if c.get("candidate_id") not in selected_ids]
        rng.shuffle(pool)
        selected.extend(pool[:remaining])
    else:
        # Simple deterministic random sampling
        rng.shuffle(valid)
        selected = valid[:limit]

    # Sort for stable ordering
    selected.sort(key=lambda c: c.get("candidate_id", ""))
    return selected


# ---------------------------------------------------------------------------
# Media reference verification (Fix 4)
# ---------------------------------------------------------------------------


def check_media_references(
    tasks_path: str | Path,
    *,
    video_url_mode: VideoUrlMode = VideoUrlMode.S3_KEY,
    local_video_dir: str | None = None,
    s3_bucket: str | None = None,
    s3_endpoint_url: str | None = None,
    s3_region: str | None = None,
    s3_anonymous: bool = False,
) -> MediaCheckReport:
    """Verify media references in a Label Studio task file.

    Checks:
    - Each task has a video reference
    - URL/path format matches the selected mode
    - Local files exist where applicable
    - S3 objects exist where credentials and access are available

    Args:
        tasks_path: Path to Label Studio task JSON file.
        video_url_mode: Expected video URL mode.
        local_video_dir: Local video directory for local mode checks.
        s3_bucket: S3 bucket for S3 mode checks.
        s3_endpoint_url: S3 endpoint URL.
        s3_region: S3 region.
        s3_anonymous: Use anonymous S3 access.

    Returns:
        MediaCheckReport with per-task results.
    """
    tasks_data = json.loads(Path(tasks_path).read_text())
    if not isinstance(tasks_data, list):
        return MediaCheckReport(
            results=[
                MediaCheckResult(
                    task_id="",
                    ok=False,
                    message="Task file must contain a JSON array.",
                )
            ],
            total=1,
            failed=1,
        )

    results: list[MediaCheckResult] = []
    for item in tasks_data:
        task_id = str(item.get("id", item.get("task", {}).get("id", "unknown")))
        candidate_id = str(item.get("data", {}).get("candidate_id", "")) or None
        video_ref = str(item.get("data", {}).get("video", ""))

        result = MediaCheckResult(
            task_id=task_id,
            candidate_id=candidate_id,
            video_ref=video_ref,
            mode=video_url_mode,
        )

        if not video_ref:
            result.message = "Missing video reference in task data."
            results.append(result)
            continue

        if video_url_mode == VideoUrlMode.LOCAL:
            video_path = Path(video_ref)
            if not video_path.exists():
                result.message = f"Local file not found: {video_ref}"
                results.append(result)
                continue
            result.ok = True
            result.message = f"Local file exists: {video_ref}"

        elif video_url_mode == VideoUrlMode.S3_STORAGE:
            if not video_ref.startswith("s3://"):
                result.message = f"Expected s3:// URL for s3_storage mode, got: {video_ref!r}"
                results.append(result)
                continue
            # Try S3 HeadObject check
            result.ok, result.message = _check_s3_object_exists(
                video_ref,
                endpoint_url=s3_endpoint_url,
                region=s3_region,
                anonymous=s3_anonymous,
            )

        elif video_url_mode == VideoUrlMode.PRESIGNED:
            if not video_ref.startswith(("http://", "https://")):
                result.message = f"Expected http(s) URL for presigned mode, got: {video_ref!r}"
                results.append(result)
                continue
            # For presigned URLs, we can't easily check without downloading
            result.ok = True
            result.message = f"Presigned URL format OK: {video_ref[:80]}..."

        else:
            # S3_KEY mode — just check format
            result.ok = True
            result.message = f"S3 key format OK: {video_ref}"

        results.append(result)

    report = MediaCheckReport(
        results=results,
        total=len(results),
        passed=sum(1 for r in results if r.ok),
        failed=sum(1 for r in results if not r.ok),
    )
    return report


def _check_s3_object_exists(
    s3_uri: str,
    *,
    endpoint_url: str | None = None,
    region: str | None = None,
    anonymous: bool = False,
) -> tuple[bool, str]:
    """Check if an S3 object exists using HeadObject.

    Returns (ok, message) tuple.
    """
    import boto3

    try:
        match = re.match(r"^s3://([^/]+)/(.+)$", s3_uri)
        if not match:
            return False, f"Invalid S3 URI: {s3_uri}"
        bucket = match.group(1)
        key = match.group(2)

        kwargs: dict[str, Any] = {}
        if endpoint_url:
            kwargs["endpoint_url"] = endpoint_url
        if region:
            kwargs["region_name"] = region
        if anonymous:
            kwargs["aws_access_key_id"] = ""
            kwargs["aws_secret_access_key"] = ""
            kwargs["aws_session_token"] = ""

        client = boto3.client("s3", **kwargs)
        client.head_object(Bucket=bucket, Key=key)
        return True, f"S3 object exists: {s3_uri}"
    except client.exceptions.ClientError as exc:  # type: ignore[name-defined]
        error_code = exc.response["Error"]["Code"]
        if error_code == "404":
            return False, f"S3 object not found: {s3_uri}"
        return False, f"S3 check failed ({error_code}): {s3_uri}"
    except Exception as exc:
        return False, f"S3 check error: {exc}"


# ---------------------------------------------------------------------------
# Source-offset timestamp conversion (Task 6.2)
# ---------------------------------------------------------------------------


def _collect_source_offset(task_data: dict[str, Any]) -> float:
    """Extract source_start_s offset from task data.

    Returns 0.0 for legacy tasks without source offset information.
    """
    offset = task_data.get("source_start_s")
    if offset is None:
        return 0.0
    try:
        return float(offset)
    except (TypeError, ValueError):
        return 0.0


def _collect_candidate_id(task_data: dict[str, Any]) -> str | None:
    """Extract candidate_id from task data, or None for legacy tasks."""
    cid = task_data.get("candidate_id")
    if cid and str(cid).strip():
        return str(cid)
    return None


def _collect_candidate_metadata(task_data: dict[str, Any]) -> dict[str, str | None]:
    """Extract optional candidate metadata fields for traceability."""
    return {
        "actor_id": str(task_data["actor_id"]) if task_data.get("actor_id") else None,
        "hand_side": str(task_data["hand_side"]) if task_data.get("hand_side") else None,
        "region_id": str(task_data["region_id"]) if task_data.get("region_id") else None,
    }


def _apply_source_offset(
    relative_start: float,
    relative_end: float,
    source_offset: float,
    source_start_s: float,
    source_end_s: float | None,
) -> tuple[float, float]:
    """Convert candidate-relative timestamps to source-video timestamps."""
    source_start = source_offset + relative_start
    source_end = source_offset + relative_end
    return source_start, source_end


def _validate_candidate_relative_timestamps(
    relative_start: float,
    relative_end: float,
    source_offset: float,
    source_start_s: float,
    source_end_s: float,
    candidate_id: str,
    region_id: str,
    task_id: str,
    annotation_id: str,
    errors: ValidationErrors,
) -> bool:
    """Validate that relative timestamps produce valid source timestamps.

    Returns True if valid, False otherwise. Adds errors for violations.
    """
    # Relative start must be non-negative
    if relative_start < 0:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="start_time",
                message=(
                    f"Candidate-relative start {relative_start}s is negative "
                    f"(candidate={candidate_id})."
                ),
            )
        )
        return False

    # Relative end must not exceed candidate duration (with tolerance)
    candidate_duration = source_end_s - source_start_s
    if relative_end > candidate_duration + CANDIDATE_BOUNDARY_TOLERANCE_S:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="end_time",
                message=(
                    f"Candidate-relative end {relative_end}s exceeds candidate duration "
                    f"{candidate_duration}s for candidate={candidate_id}."
                ),
            )
        )
        return False

    # Compute source timestamps and validate they fall within source interval
    source_start = source_offset + relative_start
    source_end = source_offset + relative_end

    if source_start < source_start_s - CANDIDATE_BOUNDARY_TOLERANCE_S:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="start_time",
                message=(
                    f"Computed source start {source_start}s is before source interval "
                    f"({source_start_s}s) for candidate={candidate_id}."
                ),
            )
        )
        return False

    if source_end > source_end_s + CANDIDATE_BOUNDARY_TOLERANCE_S:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="end_time",
                message=(
                    f"Computed source end {source_end}s is beyond source interval "
                    f"({source_end_s}s) for candidate={candidate_id}."
                ),
            )
        )
        return False

    # Event start must be before event end
    if relative_end <= relative_start:
        errors.add(
            ValidationError(
                task_id=task_id,
                annotation_id=annotation_id,
                region_id=region_id,
                field_name="start_time/end_time",
                message=(
                    f"Event start ({relative_start}s) must be before event end "
                    f"({relative_end}s) for candidate={candidate_id}."
                ),
            )
        )
        return False

    return True


def export_candidate_annotations(
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
    events_output: str | Path | None = None,
    ignore_output: str | Path | None = None,
    provenance_output: str | Path | None = None,
) -> ConversionResult:
    """Export candidate-backed Label Studio annotations with source offset conversion.

    Converts candidate-relative timestamps to original source-video timestamps
    using the source_start_s offset embedded in each task's data.

    For legacy tasks without source_start_s, preserves existing zero-offset
    behavior (timestamps pass through unchanged).

    For candidate-backed tasks (those with candidate_id in data), uses
    candidate-clip review confirmation instead of full active-span review.
    This means candidate tasks do not fail validation when
    complete_active_span_reviewed is absent, since the annotator only
    reviews the candidate clip window.

    Args:
        export_json: Label Studio export JSON (same format as export_events_csv).
        events_output: Optional path for official canonical events.csv
            (no provenance fields).
        ignore_output: Optional path for official ignore_intervals.parquet
            (no provenance fields).
        provenance_output: Optional path for event_provenance.parquet
            containing candidate traceability metadata.

    Returns:
        ConversionResult with source-corrected canonical events and ignore intervals.
    """
    conversion = ConversionResult()
    items = _load_export_json(export_json, conversion.validation)
    if items is None:
        return conversion

    events: list[CanonicalEvent] = []
    ignores: list[IgnoreIntervalExport] = []

    for index, item in enumerate(items):
        context = _task_context(item, index)
        annotations = item.get("annotations")
        if not isinstance(annotations, list):
            conversion.validation.add(
                ValidationError(
                    task_id=context.task_id,
                    field_name="annotations",
                    message="Missing or invalid annotations array.",
                )
            )
            continue

        # Extract candidate source-mapping metadata
        task_data = context.data
        is_candidate_task = _is_candidate_backed_task(task_data)
        source_offset = _collect_source_offset(task_data)
        candidate_id = _collect_candidate_id(task_data)
        candidate_meta = _collect_candidate_metadata(task_data)

        # Get source interval for validation
        source_start_s: float | None = None
        source_end_s: float | None = None
        try:
            raw_start = task_data.get("source_start_s")
            raw_end = task_data.get("source_end_s")
            if raw_start is not None:
                source_start_s = float(raw_start)
            if raw_end is not None:
                source_end_s = float(raw_end)
        except (TypeError, ValueError):
            pass

        # Validate source interval
        if (
            source_start_s is not None
            and source_end_s is not None
            and source_start_s >= source_end_s
        ):
            conversion.validation.add(
                ValidationError(
                    task_id=context.task_id,
                    field_name="source_start_s",
                    message=(
                        f"source_start_s ({source_start_s}) >= source_end_s ({source_end_s}) "
                        f"for candidate={candidate_id}."
                    ),
                )
            )
            continue

        for annotation in annotations:
            if not isinstance(annotation, dict):
                conversion.validation.add(
                    ValidationError(
                        task_id=context.task_id,
                        field_name="annotations",
                        message="Each annotation must be a JSON object.",
                    )
                )
                continue
            if annotation.get("was_cancelled") is True:
                continue

            annotation_id = _annotation_id(annotation)

            # Fix 1: Use appropriate review confirmation based on task type
            if is_candidate_task:
                confirmed = _candidate_clip_review_confirmed(annotation, context)
                confirmation_field = "candidate_clip_reviewed"
            else:
                confirmed = _annotation_review_confirmed(annotation, context)
                confirmation_field = "complete_active_span_reviewed"

            if not confirmed:
                conversion.validation.add(
                    ValidationError(
                        task_id=context.task_id,
                        annotation_id=annotation_id,
                        field_name=confirmation_field,
                        message=(f"Export requires {confirmation_field}=true."),
                    )
                )

            bundles = _collect_region_bundles(
                annotation,
                context,
                conversion.validation,
            )
            for bundle in bundles:
                region = _parse_annotation_region(
                    bundle,
                    context,
                    annotation,
                    conversion.validation,
                )
                if region is None or not confirmed:
                    continue

                # Apply source offset conversion for candidate-backed tasks
                if region.start_time is not None and source_offset > 0:
                    if source_start_s is not None and source_end_s is not None:
                        valid = _validate_candidate_relative_timestamps(
                            region.start_time,
                            region.end_time,
                            source_offset,
                            source_start_s,
                            source_end_s,
                            candidate_id or "unknown",
                            region.region_id,
                            context.task_id,
                            annotation_id,
                            conversion.validation,
                        )
                        if not valid:
                            continue

                    region.start_time = source_offset + region.start_time
                    region.end_time = source_offset + region.end_time

                if region.label is EventLabel.IGNORE:
                    reason = _ignore_reason(
                        bundle,
                        context,
                        annotation,
                        conversion.validation,
                    )
                    if reason is None:
                        continue
                    ignores.append(
                        IgnoreIntervalExport(
                            ignore_id=_generate_ignore_id(
                                context.clip_id,
                                region.region_id,
                            ),
                            clip_id=context.clip_id,
                            t_start=region.start_time,
                            t_end=region.end_time,
                            reason=reason,
                            annotator=region.annotator or None,
                            notes=region.notes,
                            candidate_id=candidate_id,
                        )
                    )
                    continue

                if region.label not in {EventLabel.PICKUP, EventLabel.PUTDOWN}:
                    conversion.validation.add(
                        ValidationError(
                            task_id=context.task_id,
                            annotation_id=annotation_id,
                            region_id=region.region_id,
                            field_name="labels",
                            message=f"Unsupported official event label: {region.label!s}.",
                        )
                    )
                    continue

                group_id = _generate_group_id(context.clip_id, region.region_id)
                canonical_events = _annotation_to_canonical_events(region, group_id)
                # Enrich with candidate traceability metadata
                for evt in canonical_events:
                    evt.candidate_id = candidate_id
                    evt.actor_id = candidate_meta.get("actor_id")
                    evt.hand_side = candidate_meta.get("hand_side")
                    evt.region_id = candidate_meta.get("region_id")
                events.extend(canonical_events)

    events.sort(key=lambda event: (event.clip_id, event.t_start, str(event.type), event.event_id))
    ignores.sort(key=lambda interval: (interval.clip_id, interval.t_start, interval.ignore_id))
    conversion.canonical_events = events
    conversion.ignore_intervals = ignores

    if events_output is not None:
        _write_official_events_csv(conversion.canonical_events, Path(events_output))
    if ignore_output is not None:
        _write_official_ignore_parquet(conversion.ignore_intervals, Path(ignore_output))
    if provenance_output is not None:
        _write_provenance_parquet(conversion, Path(provenance_output))

    return conversion


# ---------------------------------------------------------------------------
# Canonical outputs
# ---------------------------------------------------------------------------


def _annotation_to_canonical_events(
    annotation: AnnotationEvent,
    group_id: str,
) -> list[CanonicalEvent]:
    """Expand one validated pickup/putdown into canonical event rows."""
    if annotation.label not in {EventLabel.PICKUP, EventLabel.PUTDOWN}:
        raise ValueError(f"Cannot convert {annotation.label!s} to an official event.")
    if annotation.item_count is None:
        raise ValueError("Validated pickup/putdown annotation is missing item_count.")

    events: list[CanonicalEvent] = []
    for item_index in range(annotation.item_count):
        event_id = _generate_event_id(
            annotation.clip_id,
            str(annotation.label),
            group_id,
            item_index,
        )
        events.append(
            CanonicalEvent(
                event_id=event_id,
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
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert Label Studio JSON to canonical event rows and optional CSV."""
    result = _convert_export(export_json)
    if output_path is not None:
        _write_events_csv(result.canonical_events, Path(output_path))
    return result


def _write_events_csv(events: list[CanonicalEvent], path: Path) -> None:
    """Write canonical events to CSV with exact column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=EVENTS_CSV_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "event_id": event.event_id,
                    "clip_id": event.clip_id,
                    "type": str(event.type),
                    "t_start": event.t_start,
                    "t_end": event.t_end,
                    "hard_case": event.hard_case,
                    "annotator": event.annotator or "",
                    "confidence": str(event.confidence),
                    "notes": event.notes or "",
                }
            )


def export_ignore_intervals_parquet(
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
    output_path: str | Path | None = None,
) -> ConversionResult:
    """Convert Label Studio JSON to validated internal ignore intervals."""
    result = _convert_export(export_json)
    if output_path is not None:
        _write_ignore_parquet(result.ignore_intervals, Path(output_path))
    return result


def _write_ignore_parquet(intervals: list[IgnoreIntervalExport], path: Path) -> None:
    """Write ignore intervals to Parquet using a stable schema."""
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "ignore_id": interval.ignore_id,
            "clip_id": interval.clip_id,
            "t_start": interval.t_start,
            "t_end": interval.t_end,
            "reason": str(interval.reason),
            "annotator": interval.annotator or "",
            "notes": interval.notes or "",
        }
        for interval in intervals
    ]

    schema = pa.schema(
        [
            ("ignore_id", pa.string()),
            ("clip_id", pa.string()),
            ("t_start", pa.float64()),
            ("t_end", pa.float64()),
            ("reason", pa.string()),
            ("annotator", pa.string()),
            ("notes", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Official canonical outputs (Fix 2)
# ---------------------------------------------------------------------------


def _write_official_events_csv(
    events: list[CanonicalEvent],
    path: Path,
) -> None:
    """Write official canonical events.csv with only approved columns.

    This output is compatible with Task 8 evaluator without downstream
    filtering. Provenance fields (candidate_id, actor_id, hand_side,
    region_id, event_group_id) are excluded.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=OFFICIAL_EVENTS_CSV_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow(event.canonical_dict())


def _write_official_ignore_parquet(
    intervals: list[IgnoreIntervalExport],
    path: Path,
) -> None:
    """Write official ignore_intervals.parquet with only approved schema.

    Provenance fields (candidate_id) are excluded. This output is
    compatible with Task 8 evaluator.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [interval.canonical_dict() for interval in intervals]

    schema = pa.schema(
        [
            ("ignore_id", pa.string()),
            ("clip_id", pa.string()),
            ("t_start", pa.float64()),
            ("t_end", pa.float64()),
            ("reason", pa.string()),
            ("annotator", pa.string()),
            ("notes", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, str(path))


def _write_provenance_parquet(
    conversion: ConversionResult,
    path: Path,
) -> None:
    """Write event_provenance.parquet with candidate traceability metadata.

    Contains provenance columns not present in the official canonical export:
    event_id, candidate_id, clip_id, actor_id, hand_side, region_id,
    event_group_id, source_start_s, source_end_s, proposal_score,
    config_fingerprint.

    This artifact preserves full traceability from exported events back to
    the originating candidates and generation configuration.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    provenance_records: list[dict[str, object | None]] = []
    for event in conversion.canonical_events:
        provenance_records.append(event.provenance_dict())

    schema = pa.schema(
        [
            ("event_id", pa.string()),
            ("candidate_id", pa.string()),
            ("clip_id", pa.string()),
            ("actor_id", pa.string()),
            ("hand_side", pa.string()),
            ("region_id", pa.string()),
            ("event_group_id", pa.string()),
        ]
    )
    table = pa.Table.from_pylist(provenance_records, schema=schema)
    pq.write_table(table, str(path))


# ---------------------------------------------------------------------------
# Round-trip check
# ---------------------------------------------------------------------------


def round_trip_check(
    original_events: list[CanonicalEvent],
    export_json: dict[str, Any] | list[dict[str, Any]] | str,
    fps: float = 30.0,
    tolerance_frames: int = 1,
) -> bool:
    """Verify timestamp preservation within a frame tolerance."""
    result = export_events_csv(export_json)
    if not result.is_valid:
        return False
    if len(original_events) != len(result.canonical_events):
        return False

    for original, exported in zip(
        sorted(original_events, key=lambda event: event.t_start),
        sorted(result.canonical_events, key=lambda event: event.t_start),
        strict=True,
    ):
        if original.type != exported.type:
            return False
        tolerance_s = tolerance_frames / max(fps, 1.0)
        if abs(original.t_start - exported.t_start) > tolerance_s:
            return False
        if abs(original.t_end - exported.t_end) > tolerance_s:
            return False

    return True


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def cli_build_tasks(
    clips_path: str,
    candidates_path: str | None = None,
    output_path: str = "annotation/tasks.json",
) -> None:
    """Build Label Studio tasks from clip and optional candidate JSON."""
    clips = json.loads(Path(clips_path).read_text())
    candidates = None
    if candidates_path:
        candidates = json.loads(Path(candidates_path).read_text())

    tasks = build_label_studio_tasks(clips, candidates)
    payload: list[dict[str, Any]] = []
    for task in tasks:
        predictions = []
        for prediction in task.predictions:
            model_version = prediction.candidate_model or prediction.model_source or "stage_b"
            predictions.append(
                {
                    "model_version": model_version,
                    "score": prediction.score,
                    "result": prediction.result,
                }
            )
        payload.append({"data": task.data, "predictions": predictions})

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {len(tasks)} task(s) to {output_path}")


def cli_export(
    export_path: str,
    events_output: str = "events.csv",
    ignore_output: str = "ignore_intervals.parquet",
) -> None:
    """Export Label Studio JSON to canonical event and ignore outputs."""
    export_data = json.loads(Path(export_path).read_text())
    result = _convert_export(export_data)

    _write_events_csv(result.canonical_events, Path(events_output))
    _write_ignore_parquet(result.ignore_intervals, Path(ignore_output))

    print(f"Events: {len(result.canonical_events)} rows ({events_output})")
    print(f"Ignore intervals: {len(result.ignore_intervals)} rows ({ignore_output})")
    if not result.is_valid:
        for error in result.validation.errors:
            print(
                "  WARN: "
                f"task={error.task_id or '-'} "
                f"annotation={error.annotation_id or '-'} "
                f"region={error.region_id or '-'} "
                f"field={error.field_name or '-'}: {error.message}"
            )

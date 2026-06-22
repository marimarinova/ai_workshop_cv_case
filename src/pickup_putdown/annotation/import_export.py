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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import ValidationError as PydanticValidationError

from pickup_putdown.annotation.schemas import (
    AnnotationEvent,
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

VALID_EVENT_TYPES = {"pickup", "putdown"}
VALID_CONFIDENCE_VALUES = {"high", "med", "low"}
VALID_REVIEW_STATUSES = {"draft", "reviewed", "accepted", "needs_adjudication"}

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

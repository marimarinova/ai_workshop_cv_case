"""Typed models for Label Studio task/prediction/annotation representations
and canonical conversion results.

These models sit between the canonical repository schemas
(pickup_putdown.common.schemas) and Label Studio's JSON interchange format.
They are deliberately separate so that neither side can be mutated without
going through the explicit conversion layer.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ConfidenceLevel(StrEnum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


class HardCaseFlag(StrEnum):
    TRUE = "true"
    FALSE = "false"


class ReviewStatus(StrEnum):
    DRAFT = "draft"
    REVIEWED = "reviewed"
    ACCEPTED = "accepted"
    NEEDS_ADJUDICATION = "needs_adjudication"


class EventLabel(StrEnum):
    PICKUP = "pickup"
    PUTDOWN = "putdown"
    IGNORE = "ignore"


class IgnoreReason(StrEnum):
    ACTION_OCCLUDED = "ACTION_OCCLUDED"
    ACTION_OUT_OF_FRAME = "ACTION_OUT_OF_FRAME"
    CLIP_BOUNDARY = "CLIP_BOUNDARY"
    UNLABELABLE = "UNLABELABLE"
    CORRUPT_SECTION = "CORRUPT_SECTION"


# ---------------------------------------------------------------------------
# Annotation region (per-region Label Studio data)
# ---------------------------------------------------------------------------


class AnnotationRegion(BaseModel):
    """A single temporal annotation region with metadata."""

    start_frame: int = Field(ge=0)
    end_frame: int = Field(ge=0)
    start_time: float = Field(ge=0.0)
    end_time: float = Field(ge=0.0)
    labels: list[EventLabel] = Field(min_length=1)
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    hard_case: HardCaseFlag = HardCaseFlag.FALSE
    item_count: int | None = Field(default=None, ge=1)
    notes: str | None = None

    @field_validator("end_frame")
    @classmethod
    def end_after_start_frame(cls, v: int, info) -> int:
        start = info.data.get("start_frame")
        if start is not None and v <= start:
            raise ValueError("end_frame must be greater than start_frame")
        return v

    @field_validator("end_time")
    @classmethod
    def end_after_start_time(cls, v: float, info) -> float:
        start = info.data.get("start_time")
        if start is not None and v <= start:
            raise ValueError("end_time must be greater than start_time")
        return v


# ---------------------------------------------------------------------------
# Candidate suggestion (imported from Stage B)
# ---------------------------------------------------------------------------


class CandidateSuggestion(BaseModel):
    """A candidate proposal placed in Label Studio predictions."""

    candidate_id: str
    candidate_source: str = ""
    candidate_model: str = ""
    candidate_score: float | None = None
    regions: list[AnnotationRegion] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Label Studio task
# ---------------------------------------------------------------------------


class LabelStudioTask(BaseModel):
    """A single Label Studio task with optional pre-annotated predictions."""

    data: dict[str, Any] = Field(default_factory=dict)
    predictions: list[LabelStudioPrediction] = Field(default_factory=list)
    # Task-level metadata carried through the round-trip
    clip_id: str = ""
    fps: float = 0.0
    duration_s: float = 0.0
    video_path: str = ""
    active_start_s: float | None = None
    active_end_s: float | None = None
    source_frame_index: int | None = None

    @field_validator("fps")
    @classmethod
    def non_negative_fps(cls, v: float) -> float:
        if v < 0:
            raise ValueError("fps must be non-negative")
        return v

    @field_validator("duration_s")
    @classmethod
    def non_negative_duration(cls, v: float) -> float:
        if v < 0:
            raise ValueError("duration_s must be non-negative")
        return v


# ---------------------------------------------------------------------------
# Label Studio prediction (candidate suggestions, not completed annotations)
# ---------------------------------------------------------------------------


class LabelStudioPrediction(BaseModel):
    """A prediction/pre-annotation row in a Label Studio task."""

    result: list[dict[str, Any]] = Field(default_factory=list)
    score: float = 0.0
    model_source: str = ""
    # Traceability to candidate metadata
    candidate_id: str | None = None
    candidate_source: str = ""
    candidate_model: str = ""
    candidate_score: float | None = None


# ---------------------------------------------------------------------------
# Review metadata (task-level)
# ---------------------------------------------------------------------------


class ReviewMetadata(BaseModel):
    """Per-clip review state carried in task metadata."""

    complete_active_span_reviewed: bool = False
    annotator: str = ""
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewed_at: str = ""
    notes: str | None = None


# ---------------------------------------------------------------------------
# Canonical event (export target)
# ---------------------------------------------------------------------------


class CanonicalEvent(BaseModel):
    """One row of the canonical events.csv."""

    event_id: str
    clip_id: str
    type: EventLabel
    t_start: float
    t_end: float
    hard_case: bool = False
    annotator: str | None = None
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    notes: str | None = None
    event_group_id: str = ""

    @field_validator("t_start", "t_end")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("t_end")
    @classmethod
    def t_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("t_start")
        if start is not None and v <= start:
            raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# Ignore interval export
# ---------------------------------------------------------------------------


class IgnoreIntervalExport(BaseModel):
    """One row of ignore_intervals.parquet."""

    ignore_id: str
    clip_id: str
    t_start: float
    t_end: float
    reason: IgnoreReason
    annotator: str | None = None
    notes: str | None = None

    @field_validator("t_start", "t_end")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("t_end")
    @classmethod
    def t_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("t_start")
        if start is not None and v <= start:
            raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# Annotation event (internal conversion model)
# ---------------------------------------------------------------------------


class AnnotationEvent(BaseModel):
    """Internal representation of a completed annotation region."""

    region_id: str
    clip_id: str
    label: EventLabel
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    confidence: ConfidenceLevel = ConfidenceLevel.HIGH
    hard_case: HardCaseFlag = HardCaseFlag.FALSE
    item_count: int | None = Field(default=None, ge=1)
    review_status: ReviewStatus = ReviewStatus.DRAFT
    annotator: str = ""
    notes: str | None = None
    is_manually_added: bool = False

    @field_validator("end_frame")
    @classmethod
    def end_after_start_frame(cls, v: int, info) -> int:
        start = info.data.get("start_frame")
        if start is not None and v <= start:
            raise ValueError("end_frame must be greater than start_frame")
        return v

    @field_validator("end_time")
    @classmethod
    def end_after_start_time(cls, v: float, info) -> float:
        start = info.data.get("start_time")
        if start is not None and v <= start:
            raise ValueError("end_time must be greater than start_time")
        return v

    @model_validator(mode="after")
    def validate_item_count_for_label(self) -> Self:
        """Require item counts for events, but not for ignore intervals."""
        if self.label in {EventLabel.PICKUP, EventLabel.PUTDOWN}:
            if self.item_count is None:
                raise ValueError("item_count is required for pickup and putdown annotations")
        elif self.label is EventLabel.IGNORE and self.item_count is not None:
            raise ValueError("item_count must be omitted for ignore intervals")
        return self


# ---------------------------------------------------------------------------
# Conversion result and validation errors
# ---------------------------------------------------------------------------


class ValidationError(BaseModel):
    """A single validation error attached to a field."""

    task_id: str = ""
    annotation_id: str = ""
    region_id: str = ""
    field_name: str = ""
    message: str


class ValidationErrors(BaseModel):
    """Container for all validation errors from a conversion run."""

    errors: list[ValidationError] = Field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0

    def add(self, error: ValidationError) -> None:
        self.errors.append(error)

    def add_generic(self, task_id: str, message: str) -> None:
        self.errors.append(ValidationError(task_id=task_id, message=message))


class ConversionResult(BaseModel):
    """Result of converting Label Studio export to canonical outputs."""

    canonical_events: list[CanonicalEvent] = Field(default_factory=list)
    ignore_intervals: list[IgnoreIntervalExport] = Field(default_factory=list)
    validation: ValidationErrors = Field(default_factory=ValidationErrors)

    @property
    def is_valid(self) -> bool:
        return self.validation.is_valid

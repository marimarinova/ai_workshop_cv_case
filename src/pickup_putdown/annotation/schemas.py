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


class AnnotationUnit(StrEnum):
    """Annotation unit type that determines review semantics.

    candidate_clip: Annotator reviews only the trimmed candidate window.
    active_span: Annotator must review the complete active span of the source clip.
    source_clip: Annotator reviews the full source clip.
    """

    CANDIDATE_CLIP = "candidate_clip"
    ACTIVE_SPAN = "active_span"
    SOURCE_CLIP = "source_clip"


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
    candidate_clip_reviewed: bool = False
    annotation_unit: AnnotationUnit = AnnotationUnit.ACTIVE_SPAN
    annotator: str = ""
    review_status: ReviewStatus = ReviewStatus.DRAFT
    reviewed_at: str = ""
    notes: str | None = None


# ---------------------------------------------------------------------------
# Canonical event (export target)
# ---------------------------------------------------------------------------


class CanonicalEvent(BaseModel):
    """One row of the canonical events.csv with optional provenance fields.

    Official canonical columns (Task 8 compatible):
        event_id, clip_id, type, t_start, t_end, hard_case,
        annotator, confidence, notes

    Provenance columns (internal traceability only):
        event_group_id, candidate_id, actor_id, hand_side, region_id
    """

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
    # Traceability to originating candidate (Task 6.2)
    candidate_id: str | None = None
    # Optional metadata preserved from candidate task
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None

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

    def canonical_dict(self) -> dict[str, object]:
        """Return only the official canonical columns for Task 8 compatibility."""
        return {
            "event_id": self.event_id,
            "clip_id": self.clip_id,
            "type": str(self.type),
            "t_start": self.t_start,
            "t_end": self.t_end,
            "hard_case": self.hard_case,
            "annotator": self.annotator or "",
            "confidence": str(self.confidence),
            "notes": self.notes or "",
        }

    def provenance_dict(self) -> dict[str, object | None]:
        """Return provenance traceability columns."""
        return {
            "event_id": self.event_id,
            "candidate_id": self.candidate_id,
            "clip_id": self.clip_id,
            "actor_id": self.actor_id,
            "hand_side": self.hand_side,
            "region_id": self.region_id,
            "event_group_id": self.event_group_id,
        }


# ---------------------------------------------------------------------------
# Ignore interval export
# ---------------------------------------------------------------------------


class IgnoreIntervalExport(BaseModel):
    """One row of ignore_intervals.parquet with optional provenance fields.

    Official canonical columns:
        ignore_id, clip_id, t_start, t_end, reason, annotator, notes

    Provenance columns (internal traceability only):
        candidate_id
    """

    ignore_id: str
    clip_id: str
    t_start: float
    t_end: float
    reason: IgnoreReason
    annotator: str | None = None
    notes: str | None = None
    # Traceability to originating candidate (Task 6.2)
    candidate_id: str | None = None

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

    def canonical_dict(self) -> dict[str, object]:
        """Return only the official canonical columns."""
        return {
            "ignore_id": self.ignore_id,
            "clip_id": self.clip_id,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "reason": str(self.reason),
            "annotator": self.annotator or "",
            "notes": self.notes or "",
        }

    def provenance_dict(self) -> dict[str, object | None]:
        """Return provenance traceability columns."""
        return {
            "ignore_id": self.ignore_id,
            "candidate_id": self.candidate_id,
            "clip_id": self.clip_id,
        }


# ---------------------------------------------------------------------------
# Candidate metadata (Task 6.1 → 6.2 bridge)
# ---------------------------------------------------------------------------


class CandidateMetadata(BaseModel):
    """Metadata for a single candidate clip produced by Task 6.1.

    Required fields:
        candidate_id: unique candidate identifier
        clip_id: original source video identifier
        source_start_s: candidate window start in source-video seconds
        source_end_s: candidate window end in source-video seconds
        candidate_video: S3 key, URL, or local path accessible to Label Studio

    Optional fields (preserved but not required):
        actor_id, hand_side, region_id, proposal_score, config_fingerprint,
        duration_s, fps, codec, pixel_format
    """

    candidate_id: str
    clip_id: str
    source_start_s: float
    source_end_s: float
    candidate_video: str
    actor_id: str | None = None
    hand_side: str | None = None
    region_id: str | None = None
    proposal_score: float | None = None
    config_fingerprint: str | None = None
    duration_s: float | None = None
    fps: float | None = None
    codec: str | None = None
    pixel_format: str | None = None
    candidate_key: str | None = None

    @field_validator("source_start_s", "source_end_s")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("source_end_s")
    @classmethod
    def source_end_after_start(cls, v: float, info) -> float:
        start = info.data.get("source_start_s")
        if start is not None and v <= start:
            raise ValueError("source_end_s must be greater than source_start_s")
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


class CandidateValidationError(BaseModel):
    """Validation error for candidate metadata."""

    candidate_id: str
    field_name: str
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


# ---------------------------------------------------------------------------
# Video URL mode and media verification
# ---------------------------------------------------------------------------


class VideoUrlMode(StrEnum):
    """How candidate videos are referenced in Label Studio tasks.

    local: Video path is a local filesystem path served by Label Studio
           document storage mount.
    s3_key: Video path is a raw S3 object key. Requires Label Studio
            cloud-storage integration to be configured separately.
    s3_storage: Video path uses Label Studio S3 cloud-storage integration
                format (s3://bucket/key).
    presigned: Video path is a presigned S3 URL. Note: URLs expire.
    """

    LOCAL = "local"
    S3_KEY = "s3_key"
    S3_STORAGE = "s3_storage"
    PRESIGNED = "presigned"


class MediaCheckResult(BaseModel):
    """Result of checking a single task's media reference."""

    task_id: str
    candidate_id: str | None = None
    video_ref: str = ""
    mode: VideoUrlMode | None = None
    ok: bool = False
    message: str = ""


class MediaCheckReport(BaseModel):
    """Aggregate media check results for a task file."""

    results: list[MediaCheckResult] = Field(default_factory=list)
    total: int = 0
    passed: int = 0
    failed: int = 0

    @property
    def is_all_ok(self) -> bool:
        return self.failed == 0

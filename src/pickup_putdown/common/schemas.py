"""Pydantic models for canonical data schemas."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class EventType(StrEnum):
    PICKUP = "pickup"
    PUTDOWN = "putdown"


class Confidence(StrEnum):
    HIGH = "high"
    MED = "med"
    LOW = "low"


# ---------------------------------------------------------------------------
# clips.csv
# ---------------------------------------------------------------------------


class Clip(BaseModel):
    clip_id: str
    s3_key: str
    duration_s: float
    fps: float
    width: int
    height: int
    n_person_tracks: int = 0
    usable: bool = False
    active_start_s: float | None = None
    active_end_s: float | None = None
    split: str | None = None
    session_id: str | None = None
    notes: str | None = None

    @field_validator("duration_s", "fps", "active_start_s", "active_end_s")
    @classmethod
    def non_negative(cls, v: float | None) -> float | None:
        if v is not None and v < 0:
            raise ValueError("timestamp must be non-negative")
        return v


# ---------------------------------------------------------------------------
# active_spans.parquet
# ---------------------------------------------------------------------------


class ActiveSpan(BaseModel):
    clip_id: str
    active_span_id: str
    t_start: float
    t_end: float
    n_person_tracks: int = 0

    @field_validator("t_start", "t_end")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("t_end")
    @classmethod
    def t_end_after_start(cls, v: float, info) -> float:
        if hasattr(info, "data") and isinstance(info.data, dict):
            t_start = info.data.get("t_start")
            if t_start is not None and v <= t_start:
                raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# events.csv
# ---------------------------------------------------------------------------


class Event(BaseModel):
    event_id: str
    clip_id: str
    type: EventType
    t_start: float
    t_end: float
    hard_case: bool = False
    annotator: str | None = None
    confidence: Confidence = Confidence.HIGH
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
        if hasattr(info, "data") and isinstance(info.data, dict):
            t_start = info.data.get("t_start")
            if t_start is not None and v <= t_start:
                raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# ignore_intervals.parquet
# ---------------------------------------------------------------------------


class IgnoreInterval(BaseModel):
    ignore_id: str
    clip_id: str
    t_start: float
    t_end: float
    reason: str
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
        if hasattr(info, "data") and isinstance(info.data, dict):
            t_start = info.data.get("t_start")
            if t_start is not None and v <= t_start:
                raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# predictions.csv
# ---------------------------------------------------------------------------


class Prediction(BaseModel):
    pred_id: str
    clip_id: str
    type: EventType
    t_start: float
    t_end: float
    score: float = Field(ge=0.0, le=1.0)
    model: str

    @field_validator("t_start", "t_end")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

    @field_validator("t_end")
    @classmethod
    def t_end_after_start(cls, v: float, info) -> float:
        if hasattr(info, "data") and isinstance(info.data, dict):
            t_start = info.data.get("t_start")
            if t_start is not None and v <= t_start:
                raise ValueError("t_end must be greater than t_start")
        return v


# ---------------------------------------------------------------------------
# candidates (internal)
# ---------------------------------------------------------------------------


class Candidate(BaseModel):
    candidate_id: str
    clip_id: str
    actor_id: str
    hand_side: str | None = None
    region_id: str | None = None
    raw_start_s: float
    raw_end_s: float
    window_start_s: float
    window_end_s: float
    proposal_reason: str | None = None
    proposal_score: float | None = None
    review_status: str = "pending"

    @field_validator("raw_start_s", "raw_end_s", "window_start_s", "window_end_s")
    @classmethod
    def non_negative_timestamp(cls, v: float) -> float:
        if v < 0:
            raise ValueError("timestamp must be non-negative")
        return v

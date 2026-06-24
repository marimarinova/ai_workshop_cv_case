"""Standalone evaluation records for self-contained tests.

Renamed (``EvaluationEvent`` / ``EvaluationPrediction`` / ``EvaluationIgnoreInterval``)
so they never shadow the canonical Task 1 Pydantic models. The evaluator is
duck-typed: in the repo pass the canonical `pickup_putdown.common.schemas` models
directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

VALID_TYPES: tuple[str, ...] = ("pickup", "putdown")


def type_name(t: object) -> str:
    """Normalize an event type to its string value (StrEnum, Enum, or str)."""
    return str(t.value) if isinstance(t, Enum) else str(t)


@runtime_checkable
class IntervalLike(Protocol):
    """Duck-typed interval accepted by the evaluator."""

    clip_id: str
    t_start: float
    t_end: float
    type: object


@dataclass(frozen=True)
class EvaluationEvent:
    """A ground-truth event interval (mirrors canonical `events` fields)."""

    clip_id: str
    type: str | Enum
    t_start: float
    t_end: float
    event_id: str = ""
    confidence: str = "high"
    hard_case: bool = False
    annotator: str = ""
    notes: str = ""
    n_person: int = 1
    group_id: str = ""

    def __post_init__(self) -> None:
        if type_name(self.type) not in VALID_TYPES:
            raise ValueError(f"bad event type {self.type!r}")
        if self.t_start < 0 or self.t_end < 0:
            raise ValueError("negative timestamp")
        if not self.t_start < self.t_end:
            raise ValueError("need t_start < t_end")


@dataclass(frozen=True)
class EvaluationPrediction:
    """A predicted event interval (mirrors canonical `predictions` fields)."""

    clip_id: str
    type: str | Enum
    t_start: float
    t_end: float
    pred_id: str = ""
    score: float = 1.0
    model: str = ""

    def __post_init__(self) -> None:
        if type_name(self.type) not in VALID_TYPES:
            raise ValueError(f"bad prediction type {self.type!r}")
        if self.t_start < 0 or self.t_end < 0:
            raise ValueError("negative timestamp")
        if not self.t_start < self.t_end:
            raise ValueError("need t_start < t_end")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("score out of [0,1]")


@dataclass(frozen=True)
class EvaluationIgnoreInterval:
    """A time span excluded from all official matching."""

    clip_id: str
    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        if self.t_start < 0 or self.t_end < 0:
            raise ValueError("negative timestamp")
        if not self.t_start < self.t_end:
            raise ValueError("ignore interval needs t_start < t_end")


@dataclass
class MatchResult:
    matched: list[tuple[Any, Any]] = field(default_factory=list)
    unmatched_gt: list[Any] = field(default_factory=list)
    unmatched_pred: list[Any] = field(default_factory=list)

    @property
    def tp(self) -> int:
        return len(self.matched)

    @property
    def fp(self) -> int:
        return len(self.unmatched_pred)

    @property
    def fn(self) -> int:
        return len(self.unmatched_gt)

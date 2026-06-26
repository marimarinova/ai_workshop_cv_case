"""Shared types and classifier interfaces for the Track A detector (task_10).

These decouple the state-machine and inference logic from concrete, trained
classifiers: real estimators (fit in task_7) plug in behind the Protocols
below, while tests inject deterministic fakes. The state machine and inference
layers added in later commits build on these types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

#: A frozen image embedding produced by the task_9 encoder.
Embedding = NDArray[np.float32]

HandState = Literal["empty", "holding"]
ShelfState = Literal["occupied", "vacant"]
EventType = Literal["pickup", "putdown"]

#: Canonical prediction columns consumed by the Task 8 evaluator
#: (:func:`pickup_putdown.evaluation.io.predictions_from_rows`). Track A writes
#: exactly these so its ``events.csv`` is evaluated without a column map.
CANONICAL_PREDICTION_COLUMNS: tuple[str, ...] = (
    "clip_id",
    "pred_id",
    "type",
    "t_start",
    "t_end",
    "score",
    "model",
)


@dataclass(frozen=True)
class ActorHandRegion:
    """Grouping key: one independent state machine per actor hand and region."""

    actor_id: str
    hand_side: str
    region_id: str


@dataclass(frozen=True)
class StateObservation:
    """Calibrated per-timestamp state evidence for one actor/hand/region."""

    timestamp_s: float
    sample_position: str
    p_hand_holding: float
    p_shelf_occupied: float


@dataclass(frozen=True)
class TrackAPrediction:
    """A canonical Track A event prediction (one row of ``events.csv``)."""

    clip_id: str
    pred_id: str
    type: EventType
    t_start: float
    t_end: float
    score: float
    model: str = "track_a_v1"
    #: Internal diagnostic evidence; never written to the canonical row.
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_canonical_row(self) -> dict[str, Any]:
        """Project to the canonical ``events.csv`` columns (evidence excluded)."""
        return {
            "clip_id": self.clip_id,
            "pred_id": self.pred_id,
            "type": self.type,
            "t_start": self.t_start,
            "t_end": self.t_end,
            "score": self.score,
            "model": self.model,
        }


@runtime_checkable
class HandStateClassifier(Protocol):
    """Predicts ``P(hand is holding an item)`` from a hand-crop embedding."""

    def predict_holding(self, embedding: Embedding) -> float:
        """Return the calibrated probability in ``[0, 1]``."""
        ...


@runtime_checkable
class ShelfStateClassifier(Protocol):
    """Predicts ``P(shelf slot is occupied)`` from a shelf-patch embedding."""

    def predict_occupied(self, embedding: Embedding) -> float:
        """Return the calibrated probability in ``[0, 1]``."""
        ...

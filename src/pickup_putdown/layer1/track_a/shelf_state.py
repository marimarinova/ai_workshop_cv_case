"""Shelf-transition classifier for Track A (task_10).

Inference-time interface behind :class:`ShelfStateClassifier`. A trained
estimator (fit in task_7) is loaded from ``TrackAConfig.shelf_state_checkpoint``
when present. Until then :func:`load_shelf_state_classifier` returns a
**placeholder** whose output is a deterministic heuristic — NOT a validated
prediction. :func:`is_available` reports whether a real checkpoint exists, so
callers gate accordingly and never emit placeholder output as a real result.
"""

from __future__ import annotations

import importlib
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from pickup_putdown.config import TrackAConfig
from pickup_putdown.layer1.track_a.state_types import Embedding, ShelfStateClassifier

logger = logging.getLogger(__name__)


def is_available(config: TrackAConfig) -> bool:
    """Return ``True`` only when a trained shelf-state checkpoint exists on disk."""
    return Path(config.shelf_state_checkpoint).is_file()


def _sigmoid(x: float) -> float:
    # Branch on the sign so the exponent argument is always <= 0; this avoids the
    # OverflowError that math.exp(-x) raises for large negative x.
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    exp_x = math.exp(x)
    return exp_x / (1.0 + exp_x)


class PlaceholderShelfStateClassifier:
    """PLACEHOLDER until task_7 — a deterministic heuristic, NOT validated.

    Exists so the state machine and inference layers can be wired and tested
    without trained models. ``is_placeholder`` is ``True`` and
    :func:`is_available` is ``False``, so production code must never surface its
    output as a real prediction.
    """

    is_placeholder: bool = True

    def predict_occupied(self, embedding: Embedding) -> float:
        """Heuristic ``P(occupied)`` from the embedding mean (placeholder only)."""
        return _sigmoid(float(np.mean(embedding)))


class TrainedShelfStateClassifier:
    """Wraps a task_7 estimator (sklearn-style ``predict_proba``), loaded lazily."""

    is_placeholder: bool = False

    def __init__(self, checkpoint_path: str) -> None:
        self._checkpoint_path = checkpoint_path
        self._estimator: Any | None = None

    def _load(self) -> Any:
        if self._estimator is None:
            # Lazy import: joblib/sklearn are only needed with a real checkpoint.
            joblib = importlib.import_module("joblib")
            self._estimator = joblib.load(self._checkpoint_path)
        return self._estimator

    def predict_occupied(self, embedding: Embedding) -> float:
        estimator = self._load()
        proba = estimator.predict_proba(embedding.reshape(1, -1))
        return float(proba[0][1])


def load_shelf_state_classifier(config: TrackAConfig) -> ShelfStateClassifier:
    """Return the trained classifier if a checkpoint exists, else the placeholder."""
    if is_available(config):
        return TrainedShelfStateClassifier(config.shelf_state_checkpoint)
    logger.warning(
        "No shelf-state checkpoint at %s; using PLACEHOLDER classifier — outputs "
        "are NOT validated (task_7 pending).",
        config.shelf_state_checkpoint,
    )
    return PlaceholderShelfStateClassifier()

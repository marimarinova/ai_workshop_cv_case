"""Tests for the Track A state classifiers (task_10).

These cover the inference-time contract: is_available gating on a real
checkpoint, the placeholder fallback (clearly marked, deterministic, in range),
and that loading prefers a trained estimator when a checkpoint is present.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pickup_putdown.config import TrackAConfig
from pickup_putdown.layer1.track_a import hand_state, shelf_state
from pickup_putdown.layer1.track_a.state_types import (
    Embedding,
    HandStateClassifier,
    ShelfStateClassifier,
)


def _emb(value: float, dim: int = 8) -> Embedding:
    return np.full(dim, value, dtype=np.float32)


def test_is_available_false_without_checkpoint() -> None:
    cfg = TrackAConfig()  # default checkpoint paths do not exist in the repo
    assert hand_state.is_available(cfg) is False
    assert shelf_state.is_available(cfg) is False


def test_load_returns_marked_placeholder_without_checkpoint() -> None:
    cfg = TrackAConfig()
    hand = hand_state.load_hand_state_classifier(cfg)
    shelf = shelf_state.load_shelf_state_classifier(cfg)

    assert isinstance(hand, hand_state.PlaceholderHandStateClassifier)
    assert isinstance(shelf, shelf_state.PlaceholderShelfStateClassifier)
    # Clearly flagged so callers never mistake the output for a validated result.
    assert hand.is_placeholder is True
    assert shelf.is_placeholder is True
    # Still structurally satisfy the inference protocols.
    assert isinstance(hand, HandStateClassifier)
    assert isinstance(shelf, ShelfStateClassifier)


def test_placeholder_predictions_are_deterministic_and_in_range() -> None:
    hand = hand_state.PlaceholderHandStateClassifier()
    shelf = shelf_state.PlaceholderShelfStateClassifier()

    for clf_value, predict in (
        (hand.predict_holding(_emb(2.0)), hand.predict_holding),
        (shelf.predict_occupied(_emb(2.0)), shelf.predict_occupied),
    ):
        assert 0.0 <= clf_value <= 1.0
        assert predict(_emb(2.0)) == clf_value  # deterministic

    # Responsive to input (heuristic, not random): higher mean -> higher prob.
    assert hand.predict_holding(_emb(2.0)) > hand.predict_holding(_emb(-2.0))
    assert shelf.predict_occupied(_emb(2.0)) > shelf.predict_occupied(_emb(-2.0))


def test_is_available_true_and_loads_trained_when_checkpoint_exists(tmp_path: Path) -> None:
    hand_ckpt = tmp_path / "hand_state.joblib"
    shelf_ckpt = tmp_path / "shelf_state.joblib"
    hand_ckpt.write_bytes(b"")  # presence is enough; loading is lazy
    shelf_ckpt.write_bytes(b"")
    cfg = TrackAConfig(
        hand_state_checkpoint=str(hand_ckpt),
        shelf_state_checkpoint=str(shelf_ckpt),
    )

    assert hand_state.is_available(cfg) is True
    assert shelf_state.is_available(cfg) is True
    assert isinstance(
        hand_state.load_hand_state_classifier(cfg), hand_state.TrainedHandStateClassifier
    )
    assert isinstance(
        shelf_state.load_shelf_state_classifier(cfg), shelf_state.TrainedShelfStateClassifier
    )

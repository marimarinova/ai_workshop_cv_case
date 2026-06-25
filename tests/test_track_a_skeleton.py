"""Skeleton tests for the Track A detector (task_10): config + interface types.

These lock in the public contract (TrackAConfig defaults, canonical prediction
projection aligned with the Task 8 evaluator, classifier Protocols) before the
state machine and inference layers land in later commits.
"""

from __future__ import annotations

from pathlib import Path

from pickup_putdown.config import TrackAConfig, load_config
from pickup_putdown.evaluation.io import predictions_from_rows
from pickup_putdown.layer1.track_a.state_machine import decode_events
from pickup_putdown.layer1.track_a.state_types import (
    CANONICAL_PREDICTION_COLUMNS,
    Embedding,
    HandStateClassifier,
    ShelfStateClassifier,
    StateObservation,
    TrackAPrediction,
)


def test_track_a_config_present_in_appconfig_defaults() -> None:
    cfg = load_config(None)
    assert isinstance(cfg.track_a, TrackAConfig)
    assert cfg.track_a.model_name == "track_a_v1"


def test_track_a_yaml_loads_thresholds_and_checkpoints() -> None:
    cfg = load_config(Path("configs/track_a.yaml"))
    assert cfg.track_a.hand_state_checkpoint.endswith("hand_state.joblib")
    assert cfg.track_a.shelf_state_checkpoint.endswith("shelf_state.joblib")
    assert 0.0 <= cfg.track_a.hand_holding_threshold <= 1.0
    assert cfg.track_a.min_persistence_samples >= 1


def test_prediction_projects_to_canonical_columns() -> None:
    pred = TrackAPrediction(
        clip_id="clip_x",
        pred_id="p1",
        type="pickup",
        t_start=1.0,
        t_end=2.0,
        score=0.9,
        evidence={"diagnostic": "internal-only"},
    )
    row = pred.to_canonical_row()
    assert tuple(row.keys()) == CANONICAL_PREDICTION_COLUMNS
    assert "evidence" not in row  # internal diagnostics never leak to the row


def test_prediction_row_is_consumable_by_task8_evaluator() -> None:
    pred = TrackAPrediction(
        clip_id="clip_x",
        pred_id="p1",
        type="putdown",
        t_start=3.0,
        t_end=4.5,
        score=0.0,  # 0.0 must survive (evaluator preserves it)
    )
    [evaluated] = predictions_from_rows([pred.to_canonical_row()])
    assert evaluated.clip_id == "clip_x"
    assert evaluated.type == "putdown"
    assert evaluated.pred_id == "p1"
    assert evaluated.score == 0.0
    assert evaluated.t_start == 3.0 and evaluated.t_end == 4.5


def test_boundary_fallback_flag_is_reserved_noop() -> None:
    # The flag is RESERVED (task_7 follow-up): toggling it must not change
    # decoding today. A simple pickup sequence decoded both ways must match.
    obs = [
        StateObservation(0.0, "pre", 0.1, 0.9),
        StateObservation(1.0, "pre", 0.1, 0.9),
        StateObservation(2.0, "contact", 0.9, 0.1),
        StateObservation(3.0, "post", 0.9, 0.1),
    ]
    on = decode_events(
        obs,
        clip_id="c",
        candidate_id="cand",
        config=TrackAConfig(boundary_fallback_to_wrist_region=True),
    )
    off = decode_events(
        obs,
        clip_id="c",
        candidate_id="cand",
        config=TrackAConfig(boundary_fallback_to_wrist_region=False),
    )
    assert len(on) == 1
    assert on == off


def test_fake_classifiers_satisfy_protocols() -> None:
    class FakeHand:
        def predict_holding(self, embedding: Embedding) -> float:
            return 0.8

    class FakeShelf:
        def predict_occupied(self, embedding: Embedding) -> float:
            return 0.2

    assert isinstance(FakeHand(), HandStateClassifier)
    assert isinstance(FakeShelf(), ShelfStateClassifier)

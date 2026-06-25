"""End-to-end tests for Track A inference (task_10).

A fake feature function and probe classifiers drive the real state machine, so
no video is decoded. The written events.csv is read back through the Task 8
evaluator's predictions_from_rows to prove the rows are consumable downstream.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np

from pickup_putdown.common.schemas import Candidate
from pickup_putdown.config import TrackAConfig
from pickup_putdown.evaluation.io import load_predictions_csv
from pickup_putdown.layer1.track_a.features import SampleFeatures
from pickup_putdown.layer1.track_a.inference import (
    CandidateInput,
    infer_track_a,
    write_events_csv,
)
from pickup_putdown.layer1.track_a.state_types import Embedding

if TYPE_CHECKING:
    from pickup_putdown.perception.shelf_regions import Polygon


class _ProbeHand:
    """Reads P(holding) straight from the hand embedding's first component."""

    def predict_holding(self, embedding: Embedding) -> float:
        return float(embedding[0])


class _ProbeShelf:
    """Reads P(occupied) straight from the shelf embedding's second component."""

    def predict_occupied(self, embedding: Embedding) -> float:
        return float(embedding[1])


def _feat(ts: float, p_hand: float, p_shelf: float) -> SampleFeatures:
    return SampleFeatures(
        timestamp_s=ts,
        sample_position="mid",
        hand_embedding=np.array([p_hand, 0.0, 0.0, 0.0], dtype=np.float32),
        shelf_embedding=np.array([0.0, p_shelf, 0.0, 0.0], dtype=np.float32),
    )


# Empty/occupied -> held/vacated == pickup; the mirror == putdown.
_PICKUP = [_feat(0, 0.1, 0.9), _feat(1, 0.1, 0.9), _feat(2, 0.9, 0.1), _feat(3, 0.9, 0.1)]
_PUTDOWN = [_feat(10, 0.9, 0.1), _feat(11, 0.9, 0.1), _feat(12, 0.1, 0.9), _feat(13, 0.1, 0.9)]


def _candidate(candidate_id: str) -> Candidate:
    return Candidate(
        candidate_id=candidate_id,
        clip_id="clip_x",
        actor_id=f"actor-{candidate_id}",
        hand_side="left",
        region_id="shelf1",
        raw_start_s=0.0,
        raw_end_s=20.0,
        window_start_s=0.0,
        window_end_s=20.0,
    )


def _candidate_input(candidate_id: str) -> CandidateInput:
    return CandidateInput(
        candidate=_candidate(candidate_id),
        pose_observations=(),
        shelf_region=cast("Polygon", object()),
    )


def test_infer_track_a_writes_task8_consumable_events(tmp_path: Path) -> None:
    features_by_candidate = {"candP": _PICKUP, "candQ": _PUTDOWN}

    def feature_fn(candidate_input: CandidateInput) -> list[SampleFeatures]:
        return features_by_candidate[candidate_input.candidate.candidate_id]

    predictions = infer_track_a(
        "clip_x",
        [_candidate_input("candP"), _candidate_input("candQ")],
        TrackAConfig(),
        hand_classifier=_ProbeHand(),
        shelf_classifier=_ProbeShelf(),
        feature_fn=feature_fn,
    )

    assert [p.type for p in predictions] == ["pickup", "putdown"]

    events_path = write_events_csv(tmp_path / "events.csv", predictions)

    # Contract: the rows must be consumable by the Task 8 evaluator.
    evaluated = load_predictions_csv(str(events_path))
    assert [e.type for e in evaluated] == ["pickup", "putdown"]
    assert all(e.clip_id == "clip_x" for e in evaluated)
    assert all(e.score > 0.0 for e in evaluated)
    assert {e.pred_id for e in evaluated} == {p.pred_id for p in predictions}


def test_infer_track_a_skips_candidates_without_features(tmp_path: Path) -> None:
    def feature_fn(candidate_input: CandidateInput) -> list[SampleFeatures]:
        return []  # no features -> candidate contributes nothing

    predictions = infer_track_a(
        "clip_x",
        [_candidate_input("candP")],
        TrackAConfig(),
        hand_classifier=_ProbeHand(),
        shelf_classifier=_ProbeShelf(),
        feature_fn=feature_fn,
    )
    assert predictions == []

    events_path = write_events_csv(tmp_path / "events.csv", predictions)
    assert load_predictions_csv(str(events_path)) == []  # header-only, valid

"""Tests for the DEFERRED Task 16 Track A stage adapter (task_10).

The adapter is not wired anywhere; these verify it stays usable on its own:
availability gating on checkpoints, the deferred-default guard, and that run()
delegates to the state machine when hooks are injected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

from pickup_putdown.common.schemas import Candidate
from pickup_putdown.config import AppConfig, TrackAConfig
from pickup_putdown.layer1.track_a.features import SampleFeatures
from pickup_putdown.layer1.track_a.inference import CandidateInput
from pickup_putdown.layer1.track_a.stage import StageContextLike, TrackAStage
from pickup_putdown.layer1.track_a.state_types import Embedding

if TYPE_CHECKING:
    from pickup_putdown.perception.shelf_regions import Polygon


@dataclass
class _Ctx:
    clip_id: str = "clip_x"
    cache_dir: Path = Path(".")
    config: dict[str, Any] = field(default_factory=dict)
    upstream: dict[str, Any] = field(default_factory=dict)


class _ProbeHand:
    def predict_holding(self, embedding: Embedding) -> float:
        return float(embedding[0])


class _ProbeShelf:
    def predict_occupied(self, embedding: Embedding) -> float:
        return float(embedding[1])


def _feat(ts: float, p_hand: float, p_shelf: float) -> SampleFeatures:
    return SampleFeatures(
        timestamp_s=ts,
        sample_position="mid",
        hand_embedding=np.array([p_hand, 0.0, 0.0, 0.0], dtype=np.float32),
        shelf_embedding=np.array([0.0, p_shelf, 0.0, 0.0], dtype=np.float32),
    )


def _candidate_input() -> CandidateInput:
    candidate = Candidate(
        candidate_id="cand1",
        clip_id="clip_x",
        actor_id="a1",
        hand_side="left",
        region_id="shelf1",
        raw_start_s=0.0,
        raw_end_s=4.0,
        window_start_s=0.0,
        window_end_s=4.0,
    )
    return CandidateInput(
        candidate=candidate, pose_observations=(), shelf_region=cast("Polygon", [])
    )


def test_is_available_gates_on_checkpoints(tmp_path: Path) -> None:
    assert TrackAStage(AppConfig()).is_available() is False

    hand_ckpt = tmp_path / "hand.joblib"
    shelf_ckpt = tmp_path / "shelf.joblib"
    hand_ckpt.write_bytes(b"")
    shelf_ckpt.write_bytes(b"")
    available_cfg = AppConfig(
        track_a=TrackAConfig(
            hand_state_checkpoint=str(hand_ckpt), shelf_state_checkpoint=str(shelf_ckpt)
        )
    )
    assert TrackAStage(available_cfg).is_available() is True


def test_run_raises_when_default_wiring_is_deferred() -> None:
    stage = TrackAStage(AppConfig())
    with pytest.raises(NotImplementedError, match="deferred"):
        stage.run(cast(StageContextLike, _Ctx()))


def test_run_delegates_with_injected_hooks() -> None:
    pickup = [_feat(0, 0.1, 0.9), _feat(1, 0.1, 0.9), _feat(2, 0.9, 0.1), _feat(3, 0.9, 0.1)]
    stage = TrackAStage(
        AppConfig(),
        candidate_loader=lambda ctx: [_candidate_input()],
        hand_classifier=_ProbeHand(),
        shelf_classifier=_ProbeShelf(),
        feature_fn=lambda ci: pickup,
    )

    summary = stage.run(cast(StageContextLike, _Ctx()))

    assert summary["status"] == "ok"
    assert summary["n_events"] == 1
    assert summary["predictions"][0]["type"] == "pickup"


def test_run_is_gated_when_unavailable_and_no_injected_classifiers() -> None:
    # No checkpoints (AppConfig defaults) and no injected classifiers: the stage
    # must NOT fall back to placeholders and emit them as real predictions.
    pickup = [_feat(0, 0.1, 0.9), _feat(1, 0.1, 0.9), _feat(2, 0.9, 0.1), _feat(3, 0.9, 0.1)]
    stage = TrackAStage(
        AppConfig(),
        candidate_loader=lambda ctx: [_candidate_input()],
        feature_fn=lambda ci: pickup,
        # classifiers intentionally NOT injected
    )

    summary = stage.run(cast(StageContextLike, _Ctx()))

    assert summary["status"] == "unavailable"
    assert summary["n_events"] == 0
    assert summary["predictions"] == []  # no placeholder events surfaced

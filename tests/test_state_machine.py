"""Tests for the repeating interaction state machine.

All tests use fully synthetic observations. No classifier artifacts,
videos, YOLO, MobileNet, or real feature data are loaded.
"""

from __future__ import annotations

import pytest

from pickup_putdown.layer1.track_a.state_machine import (
    DebugTrace,
    RepeatingInteractionStateMachine,
    StateMachineConfig,
    TrackAObservation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BASE_KEY = {
    "clip_id": "clip_1",
    "candidate_id": "cand_1",
    "actor_id": "actor_1",
    "hand_side": "right",
    "region_id": "region_1",
}


def _obs(
    t: float,
    inside: bool = False,
    distance: float | None = None,
    traj_conf: float = 0.8,
    hand_empty: float = 0.5,
    hand_carrying: float = 0.3,
    hand_uncertain: float = 0.2,
    shelf_removed: float = 0.3,
    shelf_placed: float = 0.3,
    shelf_no_change: float = 0.3,
    shelf_uncertain: float = 0.1,
    **extra: object,
) -> TrackAObservation:
    kwargs: dict[str, object] = {**BASE_KEY, **extra}
    return TrackAObservation(
        timestamp_s=t,
        inside_region=inside,
        wrist_to_region_distance_px=distance,
        trajectory_confidence=traj_conf,
        hand_prob_empty=hand_empty,
        hand_prob_carrying=hand_carrying,
        hand_prob_uncertain=hand_uncertain,
        shelf_prob_object_removed=shelf_removed,
        shelf_prob_object_placed=shelf_placed,
        shelf_prob_no_change=shelf_no_change,
        shelf_prob_uncertain=shelf_uncertain,
        **kwargs,
    )


def _config(**overrides: object) -> StateMachineConfig:
    defaults: dict[str, object] = {
        "region_entry_distance_px": 40.0,
        "minimum_approach_s": 0.10,
        "minimum_contact_s": 0.15,
        "minimum_transfer_s": 0.20,
        "minimum_withdrawal_s": 0.10,
        "maximum_observation_gap_s": 0.50,
        "state_timeout_s": 3.00,
        "cycle_reset_timeout_s": 2.00,
        "debounce_s": 0.10,
        "minimum_event_separation_s": 0.30,
        "hand_probability_threshold": 0.55,
        "shelf_probability_threshold": 0.50,
        "maximum_uncertainty_ratio": 0.50,
        "event_confidence_threshold": 0.50,
        "confidence_weight_hand": 0.40,
        "confidence_weight_shelf": 0.40,
        "confidence_weight_trajectory": 0.20,
    }
    defaults.update(overrides)
    return StateMachineConfig(**defaults)


def _machine(**cfg_overrides: object) -> RepeatingInteractionStateMachine:
    return RepeatingInteractionStateMachine(
        config=_config(**cfg_overrides),
        debug=True,
    )


# ---------------------------------------------------------------------------
# 1-5: Basic state transitions
# ---------------------------------------------------------------------------


class TestBasicStateTransitions:
    def test_outside_observations_remain_outside(self) -> None:
        """Outside observations stay OUTSIDE."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=90.0),
            _obs(0.2, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_region_approach_enters_approaching(self) -> None:
        """Wrist near region enters APPROACHING state."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=35.0),  # within entry distance
        ]
        events = m.process(observations)
        assert len(events) == 0
        traces = m.debug_traces
        assert len(traces) == 1
        key = list(traces.keys())[0]
        state_names = [t.new_state for t in traces[key]]
        assert "APPROACHING" in state_names

    def test_stable_proximity_enters_contact(self) -> None:
        """Stable proximity enters CONTACT state."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=35.0),
            _obs(0.2, inside=True, distance=10.0),  # past min_approach_s
        ]
        events = m.process(observations)
        assert len(events) == 0
        traces = m.debug_traces
        key = list(traces.keys())[0]
        state_names = [t.new_state for t in traces[key]]
        assert "CONTACT" in state_names

    def test_withdrawal_without_transfer_returns_outside(self) -> None:
        """Withdrawal without transfer returns to OUTSIDE."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=35.0),
            _obs(0.2, inside=True, distance=10.0),
            _obs(0.3, inside=True, distance=10.0),
            _obs(0.4, inside=False, distance=50.0),  # withdrawal
            _obs(0.5, inside=False, distance=80.0),  # past min_withdrawal_s
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_out_of_order_observations_sorted(self) -> None:
        """Out-of-order observations are sorted deterministically."""
        m = _machine()
        observations = [
            _obs(0.2, inside=True, distance=10.0),
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=35.0),
        ]
        events = m.process(observations)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# 6-10: Pickup
# ---------------------------------------------------------------------------


class TestPickup:
    def test_empty_to_carrying_plus_removed_emits_pickup(self) -> None:
        """Full pickup evidence emits one pickup event."""
        m = _machine()
        observations = [
            # Outside
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            # Approach
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            # Contact with empty hand
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Transfer: hand becomes carrying, shelf shows removed
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            # Withdrawal
            _obs(0.6, inside=False, distance=50.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.7, inside=False, distance=80.0, hand_empty=0.1, hand_carrying=0.85),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].label == "pickup"

    def test_hand_transition_without_shelf_no_pickup(self) -> None:
        """Hand transition alone does not emit pickup."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Hand changes but shelf shows no_change
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.1,
                shelf_no_change=0.8,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.1,
                shelf_no_change=0.8,
            ),
            _obs(0.6, inside=False, distance=50.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.7, inside=False, distance=80.0, hand_empty=0.1, hand_carrying=0.85),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_shelf_removed_without_hand_transition_no_pickup(self) -> None:
        """Shelf removal alone does not emit pickup."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.5, hand_carrying=0.3),
            # Shelf shows removed but hand does not transition
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.5,
                hand_carrying=0.3,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_brief_one_frame_pickup_evidence_no_event(self) -> None:
        """One-frame pickup evidence is too brief for an event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Single frame of pickup evidence
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            # Back to empty immediately
            _obs(0.4, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_persistent_pickup_evidence_one_event(self) -> None:
        """Persistent pickup evidence emits exactly one event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.6,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.7,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.8, inside=False, distance=50.0),
            _obs(0.9, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].label == "pickup"


# ---------------------------------------------------------------------------
# 11-14: Putdown
# ---------------------------------------------------------------------------


class TestPutdown:
    def test_carrying_to_empty_plus_placed_emits_putdown(self) -> None:
        """Full putdown evidence emits one putdown event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].label == "putdown"

    def test_hand_transition_without_shelf_no_putdown(self) -> None:
        """Hand transition alone does not emit putdown."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.1,
                shelf_no_change=0.8,
            ),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_shelf_placed_without_hand_transition_no_putdown(self) -> None:
        """Shelf placement alone does not emit putdown."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.5, hand_carrying=0.3),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.5,
                hand_carrying=0.3,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_persistent_putdown_evidence_one_event(self) -> None:
        """Persistent putdown evidence emits exactly one event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.6,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.7, inside=False, distance=50.0),
            _obs(0.8, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].label == "putdown"


# ---------------------------------------------------------------------------
# 15-18: Uncertainty and conflicts
# ---------------------------------------------------------------------------


class TestUncertainty:
    def test_high_uncertainty_no_event(self) -> None:
        """High uncertainty prevents event emission."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.4,
                hand_carrying=0.2,
                hand_uncertain=0.4,
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.4,
                hand_carrying=0.2,
                hand_uncertain=0.4,
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.4,
                hand_carrying=0.2,
                hand_uncertain=0.4,
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.4,
                hand_carrying=0.2,
                hand_uncertain=0.4,
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.2,
                hand_carrying=0.2,
                hand_uncertain=0.6,
                shelf_removed=0.3,
                shelf_no_change=0.1,
                shelf_uncertain=0.6,
            ),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_contradictory_evidence_no_event(self) -> None:
        """Contradictory hand and shelf evidence prevents event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Hand says pickup, shelf says putdown
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.5, inside=False, distance=50.0),
            _obs(0.6, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_confidence_below_threshold_no_event(self) -> None:
        """Events below confidence threshold are suppressed."""
        m = _machine(event_confidence_threshold=0.95)
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0

    def test_sparse_observations_gap_reset(self) -> None:
        """Observation gap beyond maximum resets the cycle safely."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Large gap
            _obs(
                1.0,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(1.1, inside=False, distance=50.0),
            _obs(1.2, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 0


# ---------------------------------------------------------------------------
# 19-22: Repeating behavior
# ---------------------------------------------------------------------------


class TestRepeating:
    def test_pickup_then_putdown_two_events(self) -> None:
        """Pickup followed by withdrawal and putdown emits two events."""
        m = _machine()
        observations = [
            # --- Pickup cycle ---
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.7, inside=False, distance=80.0, hand_empty=0.1, hand_carrying=0.85),
            # --- Putdown cycle ---
            _obs(0.8, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.9, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(1.0, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(1.3, inside=False, distance=50.0),
            _obs(1.4, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 2
        assert events[0].label == "pickup"
        assert events[1].label == "putdown"

    def test_two_pickups_two_events(self) -> None:
        """Two pickups in separate cycles emit two events."""
        m = _machine()
        observations = [
            # --- First pickup ---
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.7, inside=False, distance=80.0, hand_empty=0.1, hand_carrying=0.85),
            # --- Second pickup ---
            _obs(0.8, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.9, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(1.0, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(1.3, inside=False, distance=50.0),
            _obs(1.4, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 2
        assert events[0].label == "pickup"
        assert events[1].label == "pickup"

    def test_sustained_evidence_no_duplicates(self) -> None:
        """Sustained evidence in one cycle does not emit duplicates."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Sustained transfer evidence
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.6,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.7,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.8,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.9, inside=False, distance=50.0),
            _obs(1.0, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 1

    def test_adjacent_pickup_putdown_not_merged(self) -> None:
        """Adjacent pickup and putdown are separate events, not merged."""
        m = _machine()
        observations = [
            # --- Pickup ---
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(0.7, inside=False, distance=80.0, hand_empty=0.1, hand_carrying=0.85),
            # --- Putdown immediately ---
            _obs(0.8, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.9, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(1.0, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(1.3, inside=False, distance=50.0),
            _obs(1.4, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 2
        assert events[0].label == "pickup"
        assert events[1].label == "putdown"
        assert events[0].cycle_id != events[1].cycle_id


# ---------------------------------------------------------------------------
# 23-26: Stream isolation
# ---------------------------------------------------------------------------


class TestStreamIsolation:
    def test_different_actors_independent(self) -> None:
        """Different actors are processed independently."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                actor_id="actor_1",
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                actor_id="actor_1",
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                actor_id="actor_1",
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                actor_id="actor_1",
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                actor_id="actor_1",
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                actor_id="actor_1",
            ),
            _obs(0.6, inside=False, distance=50.0, actor_id="actor_1"),
            _obs(0.7, inside=False, distance=80.0, actor_id="actor_1"),
            # Actor 2 with no event
            _obs(0.0, inside=False, distance=100.0, actor_id="actor_2"),
            _obs(0.1, inside=False, distance=90.0, actor_id="actor_2"),
        ]
        events = m.process(observations)
        pickup_events = [e for e in events if e.actor_id == "actor_1"]
        assert len(pickup_events) == 1
        assert pickup_events[0].label == "pickup"

    def test_left_right_hands_independent(self) -> None:
        """Left and right hands are processed independently."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                hand_side="left",
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                hand_side="left",
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                hand_side="left",
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                hand_side="left",
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                hand_side="left",
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                hand_side="left",
            ),
            _obs(0.6, inside=False, distance=50.0, hand_side="left"),
            _obs(0.7, inside=False, distance=80.0, hand_side="left"),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].hand_side == "left"

    def test_different_regions_independent(self) -> None:
        """Different shelf regions are processed independently."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                region_id="region_A",
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                region_id="region_A",
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                region_id="region_A",
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                region_id="region_A",
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                region_id="region_A",
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                region_id="region_A",
            ),
            _obs(0.6, inside=False, distance=50.0, region_id="region_A"),
            _obs(0.7, inside=False, distance=80.0, region_id="region_A"),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].region_id == "region_A"

    def test_different_clips_independent(self) -> None:
        """Different clips are processed independently."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                clip_id="clip_A",
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                clip_id="clip_A",
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                clip_id="clip_A",
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                clip_id="clip_A",
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                clip_id="clip_A",
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                clip_id="clip_A",
            ),
            _obs(0.6, inside=False, distance=50.0, clip_id="clip_A"),
            _obs(0.7, inside=False, distance=80.0, clip_id="clip_A"),
        ]
        events = m.process(observations)
        assert len(events) == 1
        assert events[0].clip_id == "clip_A"


# ---------------------------------------------------------------------------
# 27-32: Confidence and output
# ---------------------------------------------------------------------------


class TestConfidenceOutput:
    def test_confidence_in_range(self) -> None:
        """Confidence remains within 0.0-1.0."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        for e in events:
            assert 0.0 <= e.confidence <= 1.0

    def test_stronger_evidence_higher_confidence(self) -> None:
        """Stronger evidence produces higher confidence than weaker evidence."""
        # Strong evidence
        m1 = _machine()
        strong_obs = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.95, hand_carrying=0.03),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.95, hand_carrying=0.03),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.92, hand_carrying=0.05),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.92, hand_carrying=0.05),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.05,
                hand_carrying=0.92,
                shelf_removed=0.9,
                shelf_no_change=0.05,
                traj_conf=0.95,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.05,
                hand_carrying=0.92,
                shelf_removed=0.9,
                shelf_no_change=0.05,
                traj_conf=0.95,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
        ]
        events1 = m1.process(strong_obs)

        # Weaker evidence
        m2 = _machine()
        weak_obs = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.6, hand_carrying=0.2),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.6, hand_carrying=0.2),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.58, hand_carrying=0.25),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.58, hand_carrying=0.25),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.2,
                hand_carrying=0.6,
                shelf_removed=0.55,
                shelf_no_change=0.2,
                traj_conf=0.5,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.2,
                hand_carrying=0.6,
                shelf_removed=0.55,
                shelf_no_change=0.2,
                traj_conf=0.5,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
        ]
        events2 = m2.process(weak_obs)

        assert len(events1) >= 1
        if len(events2) >= 1:
            assert events1[0].confidence > events2[0].confidence

    def test_event_preserves_identity_fields(self) -> None:
        """Event output preserves identity and provenance fields."""
        m = _machine()
        observations = [
            _obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.6,
                inside=False,
                distance=50.0,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
            _obs(
                0.7,
                inside=False,
                distance=80.0,
                clip_id="clip_X",
                candidate_id="cand_Y",
                actor_id="actor_Z",
                hand_side="left",
                region_id="region_R",
            ),
        ]
        events = m.process(observations)
        assert len(events) == 1
        e = events[0]
        assert e.clip_id == "clip_X"
        assert e.candidate_id == "cand_Y"
        assert e.actor_id == "actor_Z"
        assert e.hand_side == "left"
        assert e.region_id == "region_R"

    def test_cycle_ids_deterministic_unique(self) -> None:
        """Cycle identifiers are deterministic and unique within each stream."""
        m = _machine()
        observations = [
            # Cycle 1: pickup
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(0.6, inside=False, distance=50.0),
            _obs(0.7, inside=False, distance=80.0),
            # Cycle 2: putdown
            _obs(0.8, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _obs(0.9, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(1.0, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _obs(1.3, inside=False, distance=50.0),
            _obs(1.4, inside=False, distance=80.0),
        ]
        events = m.process(observations)
        assert len(events) == 2
        cycle_ids = [e.cycle_id for e in events]
        assert len(set(cycle_ids)) == len(cycle_ids)

    def test_debug_trace_records_transitions(self) -> None:
        """Debug trace records transition reasons."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0),
            _obs(0.1, inside=False, distance=35.0),
            _obs(0.2, inside=True, distance=10.0),
            _obs(0.3, inside=False, distance=50.0),
            _obs(0.4, inside=False, distance=80.0),
        ]
        m.process(observations)
        traces = m.debug_traces
        assert len(traces) == 1
        key = list(traces.keys())[0]
        trace_list = traces[key]
        assert len(trace_list) > 0
        for trace in trace_list:
            assert isinstance(trace, DebugTrace)
            assert trace.previous_state != "" or trace.new_state != ""
            assert trace.reason != ""

    def test_finalization_incomplete_cycle(self) -> None:
        """Finalization handles incomplete cycle without inventing event."""
        m = _machine()
        observations = [
            _obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            # Stream ends mid-contact, no transfer evidence
        ]
        events = m.process(observations)
        assert len(events) == 0
        late = m.finalize()
        assert len(late) == 0


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------


class TestConfigValidation:
    def test_negative_duration_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            StateMachineConfig(minimum_approach_s=-1.0)

    def test_probability_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            StateMachineConfig(hand_probability_threshold=1.5)

    def test_weights_not_summing_to_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            StateMachineConfig(
                confidence_weight_hand=0.5,
                confidence_weight_shelf=0.5,
                confidence_weight_trajectory=0.5,
            )

    def test_default_config_valid(self) -> None:
        cfg = StateMachineConfig()
        assert cfg.hand_probability_threshold == 0.55
        assert (
            abs(
                cfg.confidence_weight_hand
                + cfg.confidence_weight_shelf
                + cfg.confidence_weight_trajectory
                - 1.0
            )
            < 0.01
        )

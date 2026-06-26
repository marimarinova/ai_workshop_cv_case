"""Deterministic tests for the Track A state machine (task_10).

Synthetic state-observation sequences (known input -> known output) exercise the
acceptance behaviours: pickup+putdown -> two rows, multi-actor independence,
boundaries not copied from candidates, no different-type merge, opt-in same-type
merge, and background suppression (no shelf corroboration / below score gate).
"""

from __future__ import annotations

from pickup_putdown.config import TrackAConfig
from pickup_putdown.layer1.track_a.state_machine import (
    GroupInput,
    decode_all,
    decode_events,
    merge_same_type,
)
from pickup_putdown.layer1.track_a.state_types import (
    ActorHandRegion,
    StateObservation,
    TrackAPrediction,
)


def _o(ts: float, hand: float, shelf: float) -> StateObservation:
    return StateObservation(
        timestamp_s=ts, sample_position="mid", p_hand_holding=hand, p_shelf_occupied=shelf
    )


# Empty hand over an occupied shelf, then a held hand over a vacated shelf.
_PICKUP = [_o(0, 0.1, 0.9), _o(1, 0.1, 0.9), _o(2, 0.9, 0.1), _o(3, 0.9, 0.1)]


def test_pickup_then_putdown_emits_two_ordered_rows() -> None:
    cfg = TrackAConfig()
    obs = [
        _o(0, 0.1, 0.9),
        _o(1, 0.1, 0.9),  # empty / occupied
        _o(2, 0.9, 0.1),
        _o(3, 0.9, 0.1),  # held / vacated   -> pickup
        _o(4, 0.1, 0.9),
        _o(5, 0.1, 0.9),  # empty / occupied -> putdown
    ]
    events = decode_events(obs, clip_id="clip_x", candidate_id="cand1", config=cfg)

    assert [e.type for e in events] == ["pickup", "putdown"]
    assert events[0].pred_id != events[1].pred_id
    # Boundaries come from the transition window, not the candidate bounds.
    assert (events[0].t_start, events[0].t_end) == (1.0, 3.0)
    assert (events[1].t_start, events[1].t_end) == (3.0, 5.0)
    assert all(0.0 <= e.score <= 1.0 for e in events)


def test_touch_only_without_shelf_change_is_background() -> None:
    cfg = TrackAConfig()
    # Hand goes held then empty but the shelf never changes -> no transfer.
    obs = [_o(0, 0.1, 0.9), _o(1, 0.1, 0.9), _o(2, 0.9, 0.9), _o(3, 0.9, 0.9), _o(4, 0.1, 0.9)]
    assert decode_events(obs, clip_id="c", candidate_id="cand", config=cfg) == []


def test_short_flicker_is_debounced() -> None:
    cfg = TrackAConfig()  # min_persistence_samples = 2
    obs = [_o(0, 0.1, 0.9), _o(1, 0.1, 0.9), _o(2, 0.9, 0.1), _o(3, 0.1, 0.9), _o(4, 0.1, 0.9)]
    # The single held sample does not persist -> no confirmed transition.
    assert decode_events(obs, clip_id="c", candidate_id="cand", config=cfg) == []


def test_multiple_actors_are_decoded_independently() -> None:
    cfg = TrackAConfig()
    group_a = GroupInput(ActorHandRegion("actorA", "left", "shelf1"), "candA", tuple(_PICKUP))
    # Same shape but later in time, a different actor doing a putdown.
    later = [
        _o(t + 10, h, s)
        for (t, h, s) in [(0, 0.9, 0.1), (1, 0.9, 0.1), (2, 0.1, 0.9), (3, 0.1, 0.9)]
    ]
    group_b = GroupInput(ActorHandRegion("actorB", "right", "shelf2"), "candB", tuple(later))

    events = decode_all("clip_x", [group_a, group_b], cfg)

    assert [e.type for e in events] == ["pickup", "putdown"]
    assert events[0].pred_id.startswith("candA")
    assert events[1].pred_id.startswith("candB")
    assert events[0].t_start < events[1].t_start  # globally time-ordered


def test_below_score_gate_is_suppressed() -> None:
    # Corroborated transition but modest probabilities: a high score gate (proxy
    # for trained-classifier-scored background / restocking) suppresses it.
    cfg = TrackAConfig(min_event_score=0.95)
    obs = [_o(0, 0.4, 0.6), _o(1, 0.4, 0.6), _o(2, 0.6, 0.4), _o(3, 0.6, 0.4)]
    assert decode_events(obs, clip_id="c", candidate_id="cand", config=cfg) == []
    # The same transition passes a default gate.
    assert len(decode_events(obs, clip_id="c", candidate_id="cand", config=TrackAConfig())) == 1


def test_score_gate_uses_unrounded_value_and_output_is_4dp() -> None:
    # Raw score 0.94999 is just under a 0.95 gate -> rejected, even though it
    # would round up to exactly 0.95.
    reject_cfg = TrackAConfig(min_event_score=0.95, min_persistence_samples=1)
    obs = [_o(0, 0.1, 0.9), _o(1, 0.94999, 0.05001)]
    assert decode_events(obs, clip_id="c", candidate_id="cand", config=reject_cfg) == []

    # A passing score is rounded to 4 decimals on output.
    pass_cfg = TrackAConfig(min_event_score=0.5, min_persistence_samples=1)
    obs2 = [_o(0, 0.1, 0.9), _o(1, 0.96666, 0.03334)]
    events = decode_events(obs2, clip_id="c", candidate_id="cand", config=pass_cfg)
    assert len(events) == 1
    assert events[0].score == 0.9667


def test_merge_same_type_only_when_configured() -> None:
    p1 = TrackAPrediction("c", "c-0", "pickup", 1.0, 2.0, 0.8)
    p2 = TrackAPrediction("c", "c-1", "pickup", 2.2, 3.0, 0.7)
    p3 = TrackAPrediction("c", "c-2", "putdown", 2.2, 3.0, 0.7)

    # Disabled (gap 0) -> untouched.
    assert len(merge_same_type([p1, p2], 0.0)) == 2
    # Enabled and within gap -> one merged span.
    merged = merge_same_type([p1, p2], 0.5)
    assert len(merged) == 1
    assert (merged[0].t_start, merged[0].t_end) == (1.0, 3.0)
    # Different types are never merged, even with a huge gap.
    assert len(merge_same_type([p1, p3], 10.0)) == 2


def test_merge_same_type_accumulates_history_across_three_merges() -> None:
    p0 = TrackAPrediction("c", "c-0", "pickup", 1.0, 2.0, 0.8, evidence={"flip_index": 1})
    p1 = TrackAPrediction("c", "c-1", "pickup", 2.1, 3.0, 0.7)
    p2 = TrackAPrediction("c", "c-2", "pickup", 3.1, 4.0, 0.9)

    merged = merge_same_type([p0, p1, p2], 0.5)

    assert len(merged) == 1
    # Full, ordered merge history (not overwritten on the 3rd merge).
    assert merged[0].evidence["merged_from"] == ["c-0", "c-1", "c-2"]
    # Prior evidence from the first event is preserved.
    assert merged[0].evidence["flip_index"] == 1
    assert (merged[0].t_start, merged[0].t_end) == (1.0, 4.0)
    assert merged[0].score == 0.9

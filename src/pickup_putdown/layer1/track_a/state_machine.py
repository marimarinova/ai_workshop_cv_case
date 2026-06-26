"""Repeating temporal state machine for the Track A detector (task_10).

Pure, deterministic decoding of per-timestamp state evidence into zero, one, or
many ordered pickup/putdown events for a single ``actor_id + hand_side +
region_id`` group. No models, no I/O — trained classifiers (task_7) feed the
:class:`StateObservation` probabilities; tests feed synthetic ones.

Decoding rules (see task_10):
* pickup  = a persistent shelf->hand transfer (shelf goes occupied->vacant while
  the hand goes empty->holding);
* putdown = a persistent hand->shelf transfer (the mirror);
* touch-only / browsing / reaching = background: a hand flip without the
  corroborating shelf change, or a flip too short to be persistent, emits
  nothing;
* boundaries come from the transition window, never from candidate bounds;
* different-type events are never merged; same-type merge is opt-in via config.

Visible-restocking suppression is by design a *classifier* concern (it is a
background/negative class trained in task_7) plus the ``min_event_score`` gate;
the state machine only fires on corroborated, sufficiently-scored transitions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from pickup_putdown.config import TrackAConfig
from pickup_putdown.layer1.track_a.state_types import (
    ActorHandRegion,
    EventType,
    StateObservation,
    TrackAPrediction,
)


@dataclass(frozen=True)
class GroupInput:
    """One actor/hand/region group's ordered observations for a candidate."""

    group: ActorHandRegion
    candidate_id: str
    observations: tuple[StateObservation, ...]


def _confirmed_transitions(
    states: list[bool], min_persistence: int
) -> list[tuple[int, int, bool]]:
    """Debounced transitions in a boolean signal.

    Returns ``(flip_index, confirm_index, new_state)`` for each change that
    persists for at least ``min_persistence`` consecutive samples; shorter
    flickers are ignored. ``flip_index`` is the first sample of the new run and
    ``confirm_index`` the sample at which persistence is met.
    """
    k = max(1, min_persistence)
    transitions: list[tuple[int, int, bool]] = []
    confirmed = states[0]
    i = 1
    n = len(states)
    while i < n:
        if states[i] == confirmed:
            i += 1
            continue
        run_end = i
        while run_end < n and states[run_end] == states[i]:
            run_end += 1
        if run_end - i >= k:
            transitions.append((i, i + k - 1, states[i]))
            confirmed = states[i]
        i = run_end
    return transitions


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def decode_events(
    observations: Sequence[StateObservation],
    *,
    clip_id: str,
    candidate_id: str,
    config: TrackAConfig,
) -> list[TrackAPrediction]:
    """Decode one group's observations into ordered Track A predictions."""
    obs = sorted(observations, key=lambda o: o.timestamp_s)
    if len(obs) < 2:
        return []

    holding = [o.p_hand_holding >= config.hand_holding_threshold for o in obs]
    occupied = [o.p_shelf_occupied >= config.shelf_occupied_threshold for o in obs]

    events: list[TrackAPrediction] = []
    for flip, confirm, new_holding in _confirmed_transitions(
        holding, config.min_persistence_samples
    ):
        event_type: EventType = "pickup" if new_holding else "putdown"

        # Corroborating shelf change across the transition (dual-signal). A
        # hand flip without it is background (touch-only / browsing / reaching).
        shelf_before = occupied[flip - 1]
        shelf_after = occupied[confirm]
        if new_holding and not (shelf_before and not shelf_after):
            continue
        if not new_holding and not (not shelf_before and shelf_after):
            continue

        window = obs[flip : confirm + 1]
        if new_holding:
            hand_ev = _mean([o.p_hand_holding for o in window])
            shelf_ev = _mean([1.0 - o.p_shelf_occupied for o in window])
        else:
            hand_ev = _mean([1.0 - o.p_hand_holding for o in window])
            shelf_ev = _mean([o.p_shelf_occupied for o in window])
        # Gate on the full-precision score; round only for the emitted value so
        # a score just under the threshold cannot slip through via rounding.
        raw_score = _clamp01((hand_ev + shelf_ev) / 2.0)
        if raw_score < config.min_event_score:
            continue
        score = round(raw_score, 4)

        events.append(
            TrackAPrediction(
                clip_id=clip_id,
                pred_id=f"{candidate_id}-trackA-{len(events)}",
                type=event_type,
                # Start at the final transfer onset, end at the stabilised state.
                t_start=obs[flip - 1].timestamp_s,
                t_end=obs[confirm].timestamp_s,
                score=score,
                model=config.model_name,
                evidence={
                    "flip_index": flip,
                    "confirm_index": confirm,
                    "hand_evidence": round(hand_ev, 4),
                    "shelf_evidence": round(shelf_ev, 4),
                },
            )
        )

    if config.same_type_merge_gap_s > 0.0:
        events = merge_same_type(events, config.same_type_merge_gap_s)
    return events


def merge_same_type(events: list[TrackAPrediction], gap_s: float) -> list[TrackAPrediction]:
    """Merge adjacent *same-type* events whose gap is within ``gap_s``.

    Different-type events are never merged. Input is assumed time-ordered.
    """
    if gap_s <= 0.0 or not events:
        return list(events)
    merged: list[TrackAPrediction] = [events[0]]
    for event in events[1:]:
        prev = merged[-1]
        if event.type == prev.type and event.t_start - prev.t_end <= gap_s:
            # Preserve the prior event's evidence and accumulate the merge
            # history so chains of >2 merges keep every contributing pred_id.
            evidence = dict(prev.evidence)
            prior = evidence.get("merged_from")
            history = list(prior) if isinstance(prior, list) else [prev.pred_id]
            history.append(event.pred_id)
            evidence["merged_from"] = history
            merged[-1] = TrackAPrediction(
                clip_id=prev.clip_id,
                pred_id=prev.pred_id,
                type=prev.type,
                t_start=prev.t_start,
                t_end=max(prev.t_end, event.t_end),
                score=max(prev.score, event.score),
                model=prev.model,
                evidence=evidence,
            )
        else:
            merged.append(event)
    return merged


def decode_all(
    clip_id: str,
    groups: Sequence[GroupInput],
    config: TrackAConfig,
) -> list[TrackAPrediction]:
    """Decode several actor/hand/region groups independently, time-ordered."""
    events: list[TrackAPrediction] = []
    for group in groups:
        events.extend(
            decode_events(
                group.observations,
                clip_id=clip_id,
                candidate_id=group.candidate_id,
                config=config,
            )
        )
    events.sort(key=lambda e: (e.t_start, e.pred_id))
    return events

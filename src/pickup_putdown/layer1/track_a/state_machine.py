"""Repeating temporal state machine for Track A interaction detection.

Converts hand-state, shelf-transition, and trajectory evidence into
pickup, putdown, or background decisions. Operates per
(clip_id, actor_id, hand_side, region_id) stream and supports multiple
interaction cycles within a single stream.

Ponytail: dataclasses throughout to match contracts.py convention. No
external dependency beyond stdlib + numpy already in pyproject.toml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, auto

logger = logging.getLogger(__name__)

# Floating-point tolerance for timestamp comparisons
_EPS = 1e-9


# ---------------------------------------------------------------------------
# State enum
# ---------------------------------------------------------------------------


class InteractionState(Enum):
    """Explicit interaction states for the repeating state machine."""

    OUTSIDE = auto()
    APPROACHING = auto()
    CONTACT = auto()
    TRANSFER = auto()
    WITHDRAWING = auto()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StateMachineConfig:
    """Configurable thresholds for the interaction state machine.

    All values are validated at construction time.
    """

    # --- Spatial ---
    region_entry_distance_px: float = 40.0

    # --- Temporal durations (seconds) ---
    minimum_approach_s: float = 0.10
    minimum_contact_s: float = 0.15
    minimum_transfer_s: float = 0.20
    minimum_withdrawal_s: float = 0.10
    maximum_observation_gap_s: float = 0.50
    state_timeout_s: float = 3.00
    cycle_reset_timeout_s: float = 2.00
    debounce_s: float = 0.10
    minimum_event_separation_s: float = 0.30

    # --- Probability thresholds ---
    hand_probability_threshold: float = 0.55
    shelf_probability_threshold: float = 0.50
    maximum_uncertainty_ratio: float = 0.50
    event_confidence_threshold: float = 0.50

    # --- Confidence weights ---
    confidence_weight_hand: float = 0.40
    confidence_weight_shelf: float = 0.40
    confidence_weight_trajectory: float = 0.20

    def __post_init__(self) -> None:
        _validate_config(self)


def _validate_config(cfg: StateMachineConfig) -> None:
    """Validate all configuration values."""
    # Durations must be non-negative
    for name in (
        "region_entry_distance_px",
        "minimum_approach_s",
        "minimum_contact_s",
        "minimum_transfer_s",
        "minimum_withdrawal_s",
        "maximum_observation_gap_s",
        "state_timeout_s",
        "cycle_reset_timeout_s",
        "debounce_s",
        "minimum_event_separation_s",
    ):
        val = getattr(cfg, name)
        if val < 0:
            raise ValueError(f"{name} must be non-negative, got {val}")

    # Probabilities in [0, 1]
    for name in (
        "hand_probability_threshold",
        "shelf_probability_threshold",
        "maximum_uncertainty_ratio",
        "event_confidence_threshold",
    ):
        val = getattr(cfg, name)
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {val}")

    # Confidence weights in [0, 1] and sum to ~1.0
    w_hand = cfg.confidence_weight_hand
    w_shelf = cfg.confidence_weight_shelf
    w_traj = cfg.confidence_weight_trajectory
    for w, n in [
        (w_hand, "confidence_weight_hand"),
        (w_shelf, "confidence_weight_shelf"),
        (w_traj, "confidence_weight_trajectory"),
    ]:
        if not (0.0 <= w <= 1.0):
            raise ValueError(f"{n} must be in [0, 1], got {w}")
    weight_sum = w_hand + w_shelf + w_traj
    if abs(weight_sum - 1.0) > 0.01:
        raise ValueError(f"Confidence weights must sum to 1.0, got {weight_sum}")

    # Logical consistency
    if cfg.minimum_event_separation_s < cfg.debounce_s:
        raise ValueError(
            f"minimum_event_separation_s ({cfg.minimum_event_separation_s}) "
            f"should be >= debounce_s ({cfg.debounce_s})"
        )


# ---------------------------------------------------------------------------
# Input observation
# ---------------------------------------------------------------------------


@dataclass
class TrackAObservation:
    """Single temporal observation fed into the state machine.

    Carries all evidence needed for state transitions within one
    (clip_id, actor_id, hand_side, region_id) stream.
    """

    clip_id: str
    candidate_id: str
    actor_id: str
    hand_side: str
    region_id: str
    timestamp_s: float

    # Wrist position (frame pixels)
    wrist_x: float | None = None
    wrist_y: float | None = None

    # Region interaction
    wrist_to_region_distance_px: float | None = None
    inside_region: bool = False

    # Trajectory confidence (0-1, from tracking quality)
    trajectory_confidence: float = 0.5

    # Hand-state probabilities (sum to ~1.0)
    hand_prob_empty: float = 0.0
    hand_prob_carrying: float = 0.0
    hand_prob_uncertain: float = 0.0

    # Shelf-state probabilities (sum to ~1.0)
    shelf_prob_object_removed: float = 0.0
    shelf_prob_object_placed: float = 0.0
    shelf_prob_no_change: float = 0.0
    shelf_prob_uncertain: float = 0.0


# ---------------------------------------------------------------------------
# Evidence summary
# ---------------------------------------------------------------------------


@dataclass
class EvidenceSummary:
    """Aggregated evidence for a decision point."""

    pre_transfer_hand_empty: float = 0.0
    pre_transfer_hand_carrying: float = 0.0
    post_transfer_hand_empty: float = 0.0
    post_transfer_hand_carrying: float = 0.0
    shelf_transition_prob: float = 0.0
    trajectory_confidence: float = 0.0
    n_supporting_observations: int = 0
    evidence_duration_s: float = 0.0
    uncertainty_proportion: float = 0.0


# ---------------------------------------------------------------------------
# Output event
# ---------------------------------------------------------------------------


@dataclass
class StateMachineEvent:
    """Intermediate event emitted by the state machine.

    Suitable for Phase 4 inference pipeline consumption.
    """

    clip_id: str
    candidate_id: str
    actor_id: str
    hand_side: str
    region_id: str
    label: str  # "pickup" or "putdown"
    start_s: float
    end_s: float
    transfer_timestamp_s: float
    confidence: float
    evidence: EvidenceSummary
    cycle_id: int


# ---------------------------------------------------------------------------
# Debug trace
# ---------------------------------------------------------------------------


@dataclass
class DebugTrace:
    """Single trace entry for debugging state transitions."""

    timestamp_s: float
    previous_state: str
    new_state: str
    reason: str
    hand_evidence: dict[str, float]
    shelf_evidence: dict[str, float]
    region_evidence: dict[str, float]
    event_emitted: bool
    event_reason: str


# ---------------------------------------------------------------------------
# Evidence aggregation helpers
# ---------------------------------------------------------------------------


def _mean_probs(probs: list[float]) -> float:
    """Return mean of a list of probabilities, 0.0 if empty."""
    return sum(probs) / len(probs) if probs else 0.0


def _uncertainty_ratio(probs: list[float]) -> float:
    """Return proportion of uncertain observations."""
    return sum(1 for p in probs if p > 0.5) / len(probs) if probs else 1.0


# ---------------------------------------------------------------------------
# Per-stream state machine
# ---------------------------------------------------------------------------


class _StreamStateMachine:
    """State machine for a single (clip, actor, hand, region) stream.

    Maintains internal state and processes observations sequentially.
    """

    def __init__(self, config: StateMachineConfig, *, debug: bool = False) -> None:
        self.config = config
        self.debug = debug
        self.state = InteractionState.OUTSIDE
        self.cycle_id: int = 0
        self._debug_traces: list[DebugTrace] = []

        # Per-cycle tracking
        self._cycle_start_s: float | None = None
        self._state_enter_s: float | None = None
        self._last_observation_s: float | None = None
        self._last_event_s: float | None = None
        self._event_emitted_this_cycle: bool = False

        # Evidence accumulation buffers
        self._approach_distances: list[float] = []
        self._contact_distances: list[float] = []
        self._contact_trajectories: list[float] = []
        self._pre_transfer_hand_empty: list[float] = []
        self._pre_transfer_hand_carrying: list[float] = []
        self._pre_transfer_hand_uncertain: list[float] = []
        self._post_transfer_hand_empty: list[float] = []
        self._post_transfer_hand_carrying: list[float] = []
        self._post_transfer_hand_uncertain: list[float] = []
        self._transfer_shelf_removed: list[float] = []
        self._transfer_shelf_placed: list[float] = []
        self._transfer_shelf_uncertain: list[float] = []
        self._transfer_trajectories: list[float] = []
        self._withdrawal_distances: list[float] = []
        self._withdrawal_trajectories: list[float] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, obs: TrackAObservation) -> StateMachineEvent | None:
        """Process one observation. Returns event if emitted, else None."""
        cfg = self.config
        prev_state = self.state
        event: StateMachineEvent | None = None
        reason = ""

        # Check observation gap
        gap_too_large = False
        if self._last_observation_s is not None:
            gap = obs.timestamp_s - self._last_observation_s
            if gap > cfg.maximum_observation_gap_s:
                gap_too_large = True

        if gap_too_large:
            reason = f"observation gap {gap:.3f}s > max {cfg.maximum_observation_gap_s}s"
            self._reset_cycle(reason)
            prev_state = self.state

        # Check state timeout
        if (
            self._state_enter_s is not None
            and obs.timestamp_s - self._state_enter_s > cfg.state_timeout_s
            and self.state != InteractionState.OUTSIDE
        ):
            reason = f"state timeout in {self.state.name}"
            self._reset_cycle(reason)
            prev_state = self.state

        # State transition logic
        if self.state == InteractionState.OUTSIDE:
            event, reason = self._handle_outside(obs, reason)
        elif self.state == InteractionState.APPROACHING:
            event, reason = self._handle_approaching(obs, reason)
        elif self.state == InteractionState.CONTACT:
            event, reason = self._handle_contact(obs, reason)
        elif self.state == InteractionState.TRANSFER:
            event, reason = self._handle_transfer(obs, reason)
        elif self.state == InteractionState.WITHDRAWING:
            event, reason = self._handle_withdrawing(obs, reason)

        new_state = self.state
        self._last_observation_s = obs.timestamp_s

        if prev_state != new_state or event is not None:
            self._record_trace(
                obs.timestamp_s,
                prev_state.name,
                new_state.name,
                reason,
                obs,
                event is not None,
                reason if event else "no event",
            )

        return event

    def finalize(self) -> list[StateMachineEvent]:
        """Finalize incomplete cycle. Returns empty list (no invented events)."""
        if self.state != InteractionState.OUTSIDE:
            self._reset_cycle("finalize: incomplete cycle discarded")
        return []

    @property
    def debug_traces(self) -> list[DebugTrace]:
        return list(self._debug_traces)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_outside(
        self, obs: TrackAObservation, reason: str
    ) -> tuple[StateMachineEvent | None, str]:
        cfg = self.config
        if obs.inside_region or (
            obs.wrist_to_region_distance_px is not None
            and obs.wrist_to_region_distance_px <= cfg.region_entry_distance_px
        ):
            self._transition_to(InteractionState.APPROACHING, obs.timestamp_s)
            self._cycle_start_s = obs.timestamp_s
            self._approach_distances = (
                [obs.wrist_to_region_distance_px]
                if obs.wrist_to_region_distance_px is not None
                else []
            )
            reason = "wrist entered or near region"
        return None, reason

    def _handle_approaching(
        self, obs: TrackAObservation, reason: str
    ) -> tuple[StateMachineEvent | None, str]:
        cfg = self.config
        if self._state_enter_s is None:
            self._state_enter_s = obs.timestamp_s

        self._approach_distances.append(
            obs.wrist_to_region_distance_px if obs.wrist_to_region_distance_px is not None else 0.0
        )

        # Check if withdrawn too early
        if (
            not obs.inside_region
            and obs.wrist_to_region_distance_px is not None
            and obs.wrist_to_region_distance_px > cfg.region_entry_distance_px * 2
        ):
            self._reset_cycle("approach abandoned: wrist left region")
            return None, "approach abandoned"

        # Check minimum approach duration to enter contact
        approach_duration = obs.timestamp_s - self._state_enter_s
        if approach_duration >= cfg.minimum_approach_s - _EPS:
            self._transition_to(InteractionState.CONTACT, obs.timestamp_s)
            reason = f"approach stable for {approach_duration:.3f}s"
            self._contact_distances = list(self._approach_distances)
            self._contact_trajectories = [obs.trajectory_confidence]
            # Start collecting pre-transfer evidence
            self._pre_transfer_hand_empty = [obs.hand_prob_empty]
            self._pre_transfer_hand_carrying = [obs.hand_prob_carrying]
            self._pre_transfer_hand_uncertain = [obs.hand_prob_uncertain]
        return None, reason

    def _handle_contact(
        self, obs: TrackAObservation, reason: str
    ) -> tuple[StateMachineEvent | None, str]:
        cfg = self.config

        # Check for withdrawal from contact BEFORE accumulating
        if (
            not obs.inside_region
            and obs.wrist_to_region_distance_px is not None
            and obs.wrist_to_region_distance_px > cfg.region_entry_distance_px
        ):
            contact_duration = obs.timestamp_s - (self._state_enter_s or obs.timestamp_s)
            if contact_duration < cfg.minimum_contact_s - _EPS:
                self._reset_cycle(
                    f"contact too brief ({contact_duration:.3f}s < {cfg.minimum_contact_s}s)"
                )
                return None, "contact too brief"
            self._transition_to(InteractionState.WITHDRAWING, obs.timestamp_s)
            self._withdrawal_distances = [obs.wrist_to_region_distance_px]
            self._withdrawal_trajectories = [obs.trajectory_confidence]
            reason = "wrist leaving region after contact"
            return None, reason

        # Check for transfer evidence BEFORE accumulating this obs into pre-transfer
        transfer_type = self._check_transfer_evidence(obs)
        if transfer_type:
            self._transition_to(InteractionState.TRANSFER, obs.timestamp_s)
            self._post_transfer_hand_empty = [obs.hand_prob_empty]
            self._post_transfer_hand_carrying = [obs.hand_prob_carrying]
            self._post_transfer_hand_uncertain = [obs.hand_prob_uncertain]
            self._transfer_shelf_removed = [obs.shelf_prob_object_removed]
            self._transfer_shelf_placed = [obs.shelf_prob_object_placed]
            self._transfer_shelf_uncertain = [obs.shelf_prob_uncertain]
            self._transfer_trajectories = [obs.trajectory_confidence]
            reason = f"transfer evidence detected: {transfer_type}"
            return None, reason

        # Still in contact, accumulate pre-transfer evidence
        self._contact_distances.append(
            obs.wrist_to_region_distance_px if obs.wrist_to_region_distance_px is not None else 0.0
        )
        self._contact_trajectories.append(obs.trajectory_confidence)
        self._pre_transfer_hand_empty.append(obs.hand_prob_empty)
        self._pre_transfer_hand_carrying.append(obs.hand_prob_carrying)
        self._pre_transfer_hand_uncertain.append(obs.hand_prob_uncertain)

        return None, reason

    def _handle_transfer(
        self, obs: TrackAObservation, reason: str
    ) -> tuple[StateMachineEvent | None, str]:
        cfg = self.config

        if self._state_enter_s is None:
            self._state_enter_s = obs.timestamp_s

        transfer_duration = obs.timestamp_s - self._state_enter_s

        # Check for withdrawal BEFORE accumulating this obs into transfer buffers
        if (
            not obs.inside_region
            and obs.wrist_to_region_distance_px is not None
            and obs.wrist_to_region_distance_px > cfg.region_entry_distance_px
        ):
            if transfer_duration >= cfg.minimum_transfer_s - _EPS:
                if not self._event_emitted_this_cycle:
                    label = self._determine_transfer_label()
                    if label:
                        event = self._build_event(label, obs.timestamp_s)
                        if event:
                            self._event_emitted_this_cycle = True
                            self._last_event_s = obs.timestamp_s
                            self._transition_to(InteractionState.WITHDRAWING, obs.timestamp_s)
                            self._withdrawal_distances = [obs.wrist_to_region_distance_px]
                            self._withdrawal_trajectories = [obs.trajectory_confidence]
                            return event, f"emitted {label} event at withdrawal"
                self._transition_to(InteractionState.WITHDRAWING, obs.timestamp_s)
                self._withdrawal_distances = [obs.wrist_to_region_distance_px]
                self._withdrawal_trajectories = [obs.trajectory_confidence]
                return None, "wrist withdrawing after transfer"
            else:
                self._transition_to(InteractionState.WITHDRAWING, obs.timestamp_s)
                self._withdrawal_distances = [obs.wrist_to_region_distance_px]
                self._withdrawal_trajectories = [obs.trajectory_confidence]
                return None, "transfer too brief, withdrawing"

        # Still in region, accumulate post-transfer evidence
        self._post_transfer_hand_empty.append(obs.hand_prob_empty)
        self._post_transfer_hand_carrying.append(obs.hand_prob_carrying)
        self._post_transfer_hand_uncertain.append(obs.hand_prob_uncertain)
        self._transfer_shelf_removed.append(obs.shelf_prob_object_removed)
        self._transfer_shelf_placed.append(obs.shelf_prob_object_placed)
        self._transfer_shelf_uncertain.append(obs.shelf_prob_uncertain)
        self._transfer_trajectories.append(obs.trajectory_confidence)

        if transfer_duration < cfg.minimum_transfer_s - _EPS:
            return None, reason

        # Determine label from accumulated evidence
        label = self._determine_transfer_label()
        if not label:
            self._transition_to(InteractionState.WITHDRAWING, obs.timestamp_s)
            self._withdrawal_distances = []
            self._withdrawal_trajectories = [obs.trajectory_confidence]
            return None, "transfer evidence became inconsistent"

        # If still in region with persistent evidence, try to emit
        if not self._event_emitted_this_cycle:
            event = self._build_event(label, obs.timestamp_s)
            if event:
                self._event_emitted_this_cycle = True
                self._last_event_s = obs.timestamp_s
                return event, f"emitted {label} event"

        return None, reason

    def _handle_withdrawing(
        self, obs: TrackAObservation, reason: str
    ) -> tuple[StateMachineEvent | None, str]:
        cfg = self.config

        self._withdrawal_distances.append(
            obs.wrist_to_region_distance_px if obs.wrist_to_region_distance_px is not None else 0.0
        )
        self._withdrawal_trajectories.append(obs.trajectory_confidence)

        if self._state_enter_s is None:
            self._state_enter_s = obs.timestamp_s

        withdrawal_duration = obs.timestamp_s - self._state_enter_s

        # If wrist goes back inside region, return to contact
        if obs.inside_region or (
            obs.wrist_to_region_distance_px is not None
            and obs.wrist_to_region_distance_px <= cfg.region_entry_distance_px
        ):
            self._transition_to(InteractionState.CONTACT, obs.timestamp_s)
            return None, "wrist re-entered region during withdrawal"

        # Minimum withdrawal duration before returning to OUTSIDE
        if withdrawal_duration < cfg.minimum_withdrawal_s - _EPS:
            return None, reason

        # Try to emit event if transfer was detected but not yet emitted
        if not self._event_emitted_this_cycle:
            label = self._determine_transfer_label()
            if label:
                event = self._build_event(label, obs.timestamp_s)
                if event:
                    self._event_emitted_this_cycle = True
                    self._last_event_s = obs.timestamp_s
                    self._reset_cycle(f"completed {label} cycle")
                    return event, f"emitted {label} event at end of withdrawal"

        # Return to outside
        self._reset_cycle("withdrawal complete, back to outside")
        return None, "withdrawal complete"

    # ------------------------------------------------------------------
    # Evidence evaluation
    # ------------------------------------------------------------------

    def _check_transfer_evidence(self, obs: TrackAObservation) -> str | None:
        """Check current observation + accumulated evidence for transfer.

        Returns "pickup", "putdown", or None.
        """
        cfg = self.config

        # Check uncertainty
        total_uncertain = obs.hand_prob_uncertain + obs.shelf_prob_uncertain
        if total_uncertain / 2 > cfg.maximum_uncertainty_ratio:
            return None

        # For pickup: pre-transfer evidence shows empty, current shows carrying
        pre_empty_mean = _mean_probs(self._pre_transfer_hand_empty)
        pre_carrying_mean = _mean_probs(self._pre_transfer_hand_carrying)

        if (
            pre_empty_mean >= cfg.hand_probability_threshold
            and obs.hand_prob_carrying >= cfg.hand_probability_threshold
            and obs.shelf_prob_object_removed >= cfg.shelf_probability_threshold
        ):
            return "pickup"

        # Putdown: hand carrying->empty AND shelf object_placed
        if (
            pre_carrying_mean >= cfg.hand_probability_threshold
            and obs.hand_prob_empty >= cfg.hand_probability_threshold
            and obs.shelf_prob_object_placed >= cfg.shelf_probability_threshold
        ):
            return "putdown"

        return None

    def _determine_transfer_label(self) -> str | None:
        """Determine label from accumulated pre/post transfer evidence."""
        cfg = self.config

        if not self._pre_transfer_hand_empty or not self._post_transfer_hand_empty:
            return None

        pre_empty = _mean_probs(self._pre_transfer_hand_empty)
        pre_carrying = _mean_probs(self._pre_transfer_hand_carrying)
        post_empty = _mean_probs(self._post_transfer_hand_empty)
        post_carrying = _mean_probs(self._post_transfer_hand_carrying)

        # Check uncertainty in accumulated evidence
        pre_uncertain = _mean_probs(self._pre_transfer_hand_uncertain)
        post_uncertain = _mean_probs(self._post_transfer_hand_uncertain)
        transfer_uncertain = _mean_probs(self._transfer_shelf_uncertain)
        avg_uncertain = (pre_uncertain + post_uncertain + transfer_uncertain) / 3
        if avg_uncertain > cfg.maximum_uncertainty_ratio:
            return None

        shelf_removed = _mean_probs(self._transfer_shelf_removed)
        shelf_placed = _mean_probs(self._transfer_shelf_placed)

        # Pickup: pre-empty, post-carrying, shelf removed
        if (
            pre_empty >= cfg.hand_probability_threshold
            and post_carrying >= cfg.hand_probability_threshold
            and shelf_removed >= cfg.shelf_probability_threshold
        ):
            return "pickup"

        # Putdown: pre-carrying, post-empty, shelf placed
        if (
            pre_carrying >= cfg.hand_probability_threshold
            and post_empty >= cfg.hand_probability_threshold
            and shelf_placed >= cfg.shelf_probability_threshold
        ):
            return "putdown"

        return None

    # ------------------------------------------------------------------
    # Event building
    # ------------------------------------------------------------------

    def _build_event(self, label: str, current_s: float) -> StateMachineEvent | None:
        """Build an event from accumulated evidence."""
        cfg = self.config

        # Minimum event separation — still mark attempt to block withdrawal retry
        if (
            self._last_event_s is not None
            and current_s - self._last_event_s < cfg.minimum_event_separation_s - _EPS
        ):
            self._last_event_s = current_s
            return None

        # Build evidence summary
        evidence = self._build_evidence_summary(label)

        # Compute confidence
        confidence = self._compute_confidence(label, evidence)

        if confidence < cfg.event_confidence_threshold:
            return None

        cycle_start = self._cycle_start_s or current_s
        transfer_s = self._state_enter_s or current_s

        return StateMachineEvent(
            clip_id="",
            candidate_id="",
            actor_id="",
            hand_side="",
            region_id="",
            label=label,
            start_s=cycle_start,
            end_s=current_s,
            transfer_timestamp_s=transfer_s,
            confidence=round(confidence, 4),
            evidence=evidence,
            cycle_id=self.cycle_id,
        )

    def _build_evidence_summary(self, label: str) -> EvidenceSummary:
        pre_empty = _mean_probs(self._pre_transfer_hand_empty)
        pre_carrying = _mean_probs(self._pre_transfer_hand_carrying)
        post_empty = _mean_probs(self._post_transfer_hand_empty)
        post_carrying = _mean_probs(self._post_transfer_hand_carrying)

        if label == "pickup":
            shelf_prob = _mean_probs(self._transfer_shelf_removed)
        else:
            shelf_prob = _mean_probs(self._transfer_shelf_placed)

        traj_conf = _mean_probs(self._transfer_trajectories)

        n_support = (
            len(self._pre_transfer_hand_empty)
            + len(self._post_transfer_hand_empty)
            + len(self._transfer_shelf_removed)
        )

        pre_uncertain = _mean_probs(self._pre_transfer_hand_uncertain)
        post_uncertain = _mean_probs(self._post_transfer_hand_uncertain)
        shelf_uncertain = _mean_probs(self._transfer_shelf_uncertain)
        uncertainty = (pre_uncertain + post_uncertain + shelf_uncertain) / 3

        evidence_duration = 0.0
        if self._cycle_start_s and self._last_observation_s:
            evidence_duration = self._last_observation_s - self._cycle_start_s

        return EvidenceSummary(
            pre_transfer_hand_empty=round(pre_empty, 4),
            pre_transfer_hand_carrying=round(pre_carrying, 4),
            post_transfer_hand_empty=round(post_empty, 4),
            post_transfer_hand_carrying=round(post_carrying, 4),
            shelf_transition_prob=round(shelf_prob, 4),
            trajectory_confidence=round(traj_conf, 4),
            n_supporting_observations=n_support,
            evidence_duration_s=round(evidence_duration, 4),
            uncertainty_proportion=round(uncertainty, 4),
        )

    def _compute_confidence(self, label: str, evidence: EvidenceSummary) -> float:
        """Compute weighted confidence score."""
        cfg = self.config

        # Hand transition strength
        if label == "pickup":
            hand_strength = min(
                evidence.pre_transfer_hand_empty,
                evidence.post_transfer_hand_carrying,
            )
        else:
            hand_strength = min(
                evidence.pre_transfer_hand_carrying,
                evidence.post_transfer_hand_empty,
            )

        # Shelf strength
        shelf_strength = evidence.shelf_transition_prob

        # Trajectory
        traj_strength = evidence.trajectory_confidence

        # Weighted sum
        raw = (
            cfg.confidence_weight_hand * hand_strength
            + cfg.confidence_weight_shelf * shelf_strength
            + cfg.confidence_weight_trajectory * traj_strength
        )

        # Uncertainty penalty
        uncertainty_penalty = 1.0 - (evidence.uncertainty_proportion * 0.5)
        confidence = raw * uncertainty_penalty

        return max(0.0, min(1.0, round(confidence, 4)))

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def _transition_to(self, new_state: InteractionState, timestamp_s: float) -> None:
        self.state = new_state
        self._state_enter_s = timestamp_s

    def _reset_cycle(self, reason: str) -> None:
        """Reset to OUTSIDE, increment cycle ID for next interaction."""
        if self.state != InteractionState.OUTSIDE:
            self.cycle_id += 1
        self.state = InteractionState.OUTSIDE
        self._cycle_start_s = None
        self._state_enter_s = None
        self._event_emitted_this_cycle = False
        self._approach_distances.clear()
        self._contact_distances.clear()
        self._contact_trajectories.clear()
        self._pre_transfer_hand_empty.clear()
        self._pre_transfer_hand_carrying.clear()
        self._pre_transfer_hand_uncertain.clear()
        self._post_transfer_hand_empty.clear()
        self._post_transfer_hand_carrying.clear()
        self._post_transfer_hand_uncertain.clear()
        self._transfer_shelf_removed.clear()
        self._transfer_shelf_placed.clear()
        self._transfer_shelf_uncertain.clear()
        self._transfer_trajectories.clear()
        self._withdrawal_distances.clear()
        self._withdrawal_trajectories.clear()

    def _record_trace(
        self,
        timestamp_s: float,
        prev_state: str,
        new_state: str,
        reason: str,
        obs: TrackAObservation,
        event_emitted: bool,
        event_reason: str,
    ) -> None:
        trace = DebugTrace(
            timestamp_s=timestamp_s,
            previous_state=prev_state,
            new_state=new_state,
            reason=reason,
            hand_evidence={
                "empty": obs.hand_prob_empty,
                "carrying": obs.hand_prob_carrying,
                "uncertain": obs.hand_prob_uncertain,
            },
            shelf_evidence={
                "object_removed": obs.shelf_prob_object_removed,
                "object_placed": obs.shelf_prob_object_placed,
                "no_change": obs.shelf_prob_no_change,
                "uncertain": obs.shelf_prob_uncertain,
            },
            region_evidence={
                "inside_region": obs.inside_region,
                "distance_px": obs.wrist_to_region_distance_px,
            },
            event_emitted=event_emitted,
            event_reason=event_reason,
        )
        self._debug_traces.append(trace)


# ---------------------------------------------------------------------------
# Public API: RepeatingInteractionStateMachine
# ---------------------------------------------------------------------------


class RepeatingInteractionStateMachine:
    """Deterministic repeating state machine for interaction detection.

    Processes observations grouped by (clip_id, actor_id, hand_side, region_id)
    stream. Each stream runs its own state machine instance.

    Usage:
        machine = RepeatingInteractionStateMachine(config)
        events = machine.process(observations)

    Or incrementally:
        machine = RepeatingInteractionStateMachine(config)
        for obs in observations:
            event = machine.update(obs)
        remaining = machine.finalize()
    """

    def __init__(
        self,
        config: StateMachineConfig | None = None,
        *,
        debug: bool = False,
    ) -> None:
        self.config = config or StateMachineConfig()
        self.debug = debug
        self._streams: dict[tuple[str, str, str, str], _StreamStateMachine] = {}
        self._all_events: list[StateMachineEvent] = []

    def _stream_key(self, obs: TrackAObservation) -> tuple[str, str, str, str]:
        return (obs.clip_id, obs.actor_id, obs.hand_side, obs.region_id)

    def _get_or_create_stream(self, key: tuple[str, str, str, str]) -> _StreamStateMachine:
        if key not in self._streams:
            self._streams[key] = _StreamStateMachine(self.config, debug=self.debug)
        return self._streams[key]

    def update(self, obs: TrackAObservation) -> StateMachineEvent | None:
        """Process a single observation. Returns event if emitted."""
        key = self._stream_key(obs)
        stream = self._get_or_create_stream(key)
        event = stream.update(obs)
        if event:
            event.clip_id = obs.clip_id
            event.candidate_id = obs.candidate_id
            event.actor_id = obs.actor_id
            event.hand_side = obs.hand_side
            event.region_id = obs.region_id
            self._all_events.append(event)
        return event

    def process(self, observations: list[TrackAObservation]) -> list[StateMachineEvent]:
        """Process a batch of observations and return all emitted events.

        Observations are sorted by timestamp within each stream before
        processing. Mixed-stream observations are grouped first.
        """
        self._all_events.clear()

        if not observations:
            return []

        # Group by stream
        groups: dict[tuple[str, str, str, str], list[TrackAObservation]] = {}
        for obs in observations:
            key = self._stream_key(obs)
            groups.setdefault(key, []).append(obs)

        # Sort each group by timestamp
        for key in groups:
            groups[key].sort(key=lambda o: o.timestamp_s)

        # Process each stream
        events: list[StateMachineEvent] = []
        for key, obs_list in groups.items():
            stream = self._get_or_create_stream(key)
            for obs in obs_list:
                event = stream.update(obs)
                if event:
                    event.clip_id = obs.clip_id
                    event.candidate_id = obs.candidate_id
                    event.actor_id = obs.actor_id
                    event.hand_side = obs.hand_side
                    event.region_id = obs.region_id
                    events.append(event)

        self._all_events = events
        return events

    def finalize(self) -> list[StateMachineEvent]:
        """Finalize all streams. Returns any late events (normally empty)."""
        late_events: list[StateMachineEvent] = []
        for stream in self._streams.values():
            late_events.extend(stream.finalize())
        return late_events

    @property
    def debug_traces(self) -> dict[tuple[str, str, str, str], list[DebugTrace]]:
        """Return debug traces for all streams."""
        return {key: stream.debug_traces for key, stream in self._streams.items()}

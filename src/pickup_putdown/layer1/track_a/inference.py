"""Track A inference pipeline.

End-to-end callable that integrates feature extraction, trained classifiers,
and the repeating state machine to produce canonical Track A predictions.

Flow:
    candidates + poses
        -> sliding-window sampling
        -> hand/shelf feature extraction (cached)
        -> classifier probabilities
        -> state-machine observations
        -> state machine per stream
        -> transition-frame grace window
        -> boundary refinement
        -> cross-candidate deduplication
        -> canonical predictions + diagnostics
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from pickup_putdown.layer1.track_a.classifier import (
    ClassifierMetadata,
    TrackAClassifier,
)
from pickup_putdown.layer1.track_a.hand_state import HAND_STATE_CLASS_NAMES
from pickup_putdown.layer1.track_a.shelf_state import SHELF_STATE_CLASS_NAMES
from pickup_putdown.layer1.track_a.state_machine import (
    EvidenceSummary,
    RepeatingInteractionStateMachine,
    StateMachineConfig,
    StateMachineEvent,
    TrackAObservation,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Inference configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SamplingConfig:
    """Sliding-window sampling for inference."""

    sample_fps: float = 4.0
    minimum_valid_samples: int = 3
    maximum_feature_gap_s: float = 0.50
    shelf_reference_from_start_s: float = 0.50

    def __post_init__(self) -> None:
        if self.sample_fps <= 0:
            raise ValueError(f"sample_fps must be positive, got {self.sample_fps}")
        if self.minimum_valid_samples < 0:
            raise ValueError(
                f"minimum_valid_samples must be non-negative, got {self.minimum_valid_samples}"
            )
        if self.maximum_feature_gap_s < 0:
            raise ValueError(
                f"maximum_feature_gap_s must be non-negative, got {self.maximum_feature_gap_s}"
            )


@dataclass(frozen=True)
class BoundaryRefinementConfig:
    """Event boundary refinement settings."""

    use_contact_start: bool = True
    stabilization_window_s: float = 0.25
    maximum_candidate_extension_s: float = 0.50
    minimum_event_duration_s: float = 0.10

    def __post_init__(self) -> None:
        if self.stabilization_window_s < 0:
            raise ValueError("stabilization_window_s must be non-negative")
        if self.maximum_candidate_extension_s < 0:
            raise ValueError("maximum_candidate_extension_s must be non-negative")
        if self.minimum_event_duration_s < 0:
            raise ValueError("minimum_event_duration_s must be non-negative")


@dataclass(frozen=True)
class DeduplicationConfig:
    """Cross-candidate deduplication settings."""

    temporal_iou_threshold: float = 0.50
    transfer_time_tolerance_s: float = 0.50
    require_same_actor: bool = True
    require_same_hand: bool = True
    require_same_region: bool = False

    def __post_init__(self) -> None:
        if not (0.0 <= self.temporal_iou_threshold <= 1.0):
            raise ValueError("temporal_iou_threshold must be in [0, 1]")
        if self.transfer_time_tolerance_s < 0:
            raise ValueError("transfer_time_tolerance_s must be non-negative")


@dataclass(frozen=True)
class InferenceConfig:
    """Top-level inference configuration."""

    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    boundary_refinement: BoundaryRefinementConfig = field(default_factory=BoundaryRefinementConfig)
    deduplication: DeduplicationConfig = field(default_factory=DeduplicationConfig)
    transition_grace_s: float = 0.25
    debug_traces: bool = False

    def __post_init__(self) -> None:
        if self.transition_grace_s < 0:
            raise ValueError("transition_grace_s must be non-negative")


def load_inference_config(path: Path | str) -> InferenceConfig:
    """Load inference config from YAML file.

    Reads the `inference` section from the YAML. Falls back to defaults
    for any missing keys.
    """
    import yaml

    path = Path(path)
    if not path.exists():
        logger.warning("Config file not found: %s, using defaults", path)
        return InferenceConfig()

    data = yaml.safe_load(path.read_text()) or {}
    inf = data.get("inference", {})

    sampling = SamplingConfig(**inf.get("sampling", {}))
    boundary = BoundaryRefinementConfig(**inf.get("boundary_refinement", {}))
    dedup = DeduplicationConfig(**inf.get("deduplication", {}))

    return InferenceConfig(
        sampling=sampling,
        boundary_refinement=boundary,
        deduplication=dedup,
        transition_grace_s=inf.get("transition_grace_s", 0.25),
        debug_traces=inf.get("debug_traces", False),
    )


# ---------------------------------------------------------------------------
# Evidence and result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class HandStateEvidence:
    """Timestamped hand-state classifier output."""

    timestamp_s: float
    probability_empty: float = 0.0
    probability_carrying: float = 0.0
    probability_uncertain: float = 0.0
    predicted_state: str = "uncertain"
    confidence: float = 0.0
    margin: float = 0.0
    raw_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class ShelfStateEvidence:
    """Timestamped shelf-state classifier output."""

    timestamp_s: float
    probability_object_removed: float = 0.0
    probability_object_placed: float = 0.0
    probability_no_change: float = 0.0
    probability_uncertain: float = 0.0
    predicted_state: str = "uncertain"
    confidence: float = 0.0
    margin: float = 0.0
    raw_probabilities: dict[str, float] = field(default_factory=dict)


@dataclass
class CandidateDiagnostics:
    """Per-candidate diagnostic trace."""

    candidate_id: str
    clip_id: str
    actor_id: str
    hand_side: str
    region_id: str
    skipped: bool = False
    skip_reason: str = ""
    n_samples: int = 0
    n_hand_evidence: int = 0
    n_shelf_evidence: int = 0
    n_events: int = 0
    observations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class DedupAuditEntry:
    """One deduplication decision."""

    kept_prediction_id: str
    kept_candidate_id: str
    kept_confidence: float
    suppressed_prediction_ids: list[str]
    suppressed_candidate_ids: list[str]
    suppressed_confidences: list[float]
    temporal_iou: float
    transfer_time_diff_s: float
    selection_reason: str


@dataclass
class InferenceSummary:
    """High-level inference summary statistics."""

    candidates_total: int = 0
    candidates_processed: int = 0
    candidates_skipped: int = 0
    skip_reasons: dict[str, int] = field(default_factory=dict)
    feature_cache_hits: int = 0
    feature_cache_misses: int = 0
    total_samples: int = 0
    raw_events_emitted: int = 0
    final_events_after_dedup: int = 0
    pickup_count: int = 0
    putdown_count: int = 0
    mean_confidence: float = 0.0
    uncertain_proportion: float = 0.0
    confidence_distribution: dict[str, int] = field(default_factory=dict)


@dataclass
class CanonicalPrediction:
    """Canonical prediction record matching the evaluator schema."""

    clip_id: str
    pred_id: str
    label: str  # "pickup" or "putdown"
    start_s: float
    end_s: float
    confidence: float
    actor_id: str = ""
    hand_side: str = ""
    region_id: str = ""
    model: str = "track_a"
    candidate_ids: list[str] = field(default_factory=list)


@dataclass
class InferenceResult:
    """Complete inference pipeline output."""

    predictions: list[CanonicalPrediction]
    raw_events: list[StateMachineEvent]
    dedup_audit: list[DedupAuditEntry]
    diagnostics: list[CandidateDiagnostics]
    summary: InferenceSummary
    output_paths: dict[str, Path] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Artifact loading and validation
# ---------------------------------------------------------------------------


class ArtifactCompatibilityError(ValueError):
    """Raised when classifier artifact is incompatible with feature extractor."""


def load_classifier_artifact(
    classifier_path: Path | str,
    metadata_path: Path | str | None = None,
) -> tuple[TrackAClassifier, ClassifierMetadata]:
    """Load a classifier artifact and its metadata.

    Raises FileNotFoundError if paths don't exist.
    Raises ArtifactCompatibilityError if metadata is inconsistent.
    """
    classifier_path = Path(classifier_path)
    clf, meta = TrackAClassifier.load_pipeline(classifier_path, metadata_path)

    if meta is None:
        raise ArtifactCompatibilityError(
            f"No metadata found for classifier at {classifier_path}. "
            "Metadata JSON is required for inference."
        )

    if meta.artifact_version != "1.0":
        raise ArtifactCompatibilityError(
            f"Unsupported artifact version {meta.artifact_version!r} in {classifier_path}"
        )

    if meta.embedding_dim <= 0:
        raise ArtifactCompatibilityError(
            f"Invalid embedding_dim {meta.embedding_dim} in {classifier_path}"
        )

    return clf, meta


def validate_classifier_classes(
    clf: TrackAClassifier,
    meta: ClassifierMetadata,
    expected_classes: list[str],
    classifier_kind: str,
) -> None:
    """Validate that the classifier contains the required classes.

    Raises ArtifactCompatibilityError if required classes are missing.
    """
    actual = set(meta.class_names)
    required = set(expected_classes)
    missing = required - actual
    if missing:
        raise ArtifactCompatibilityError(
            f"{classifier_kind} classifier missing required classes {sorted(missing)}. "
            f"Has: {meta.class_names}"
        )


def validate_artifact_compatibility(
    hand_meta: ClassifierMetadata,
    shelf_meta: ClassifierMetadata,
) -> None:
    """Validate that hand and shelf artifacts share compatible feature settings.

    Raises ArtifactCompatibilityError on mismatch.
    """
    if hand_meta.embedding_dim != shelf_meta.embedding_dim:
        raise ArtifactCompatibilityError(
            f"Embedding dimension mismatch: hand={hand_meta.embedding_dim}, "
            f"shelf={shelf_meta.embedding_dim}"
        )

    if hand_meta.encoder_name != shelf_meta.encoder_name:
        raise ArtifactCompatibilityError(
            f"Encoder mismatch: hand={hand_meta.encoder_name!r}, shelf={shelf_meta.encoder_name!r}"
        )

    if hand_meta.encoder_version != shelf_meta.encoder_version:
        raise ArtifactCompatibilityError(
            f"Encoder version mismatch: hand={hand_meta.encoder_version!r}, "
            f"shelf={shelf_meta.encoder_version!r}"
        )


# ---------------------------------------------------------------------------
# Sliding-window sampling
# ---------------------------------------------------------------------------


def compute_inference_sample_times(
    t_start: float,
    t_end: float,
    sample_fps: float,
) -> list[float]:
    """Compute deterministic sliding-window sample timestamps.

    Produces uniform samples at `sample_fps` across [t_start, t_end].
    Always includes endpoints.

    Args:
        t_start: Start of candidate window (source-video seconds).
        t_end: End of candidate window (source-video seconds).
        sample_fps: Sampling frequency in Hz.

    Returns:
        Sorted list of timestamps.
    """
    if t_end <= t_start:
        return [t_start]

    interval = 1.0 / sample_fps
    times = [t_start]
    t = t_start + interval
    while t < t_end - 1e-9:
        times.append(t)
        t += interval
    times.append(t_end)

    # Deduplicate near-duplicate timestamps
    unique: list[float] = [times[0]]
    for ts in times[1:]:
        if ts - unique[-1] > 1e-6:
            unique.append(ts)

    return unique


# ---------------------------------------------------------------------------
# Feature extraction interface
# ---------------------------------------------------------------------------


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction for one candidate."""

    hand_embeddings: list[tuple[float, np.ndarray]]  # (timestamp_s, embedding)
    shelf_embeddings: list[tuple[float, np.ndarray]]  # (timestamp_s, embedding)
    cache_hits: int = 0
    cache_misses: int = 0
    skipped: bool = False
    skip_reason: str = ""


def extract_features_for_candidate(
    candidate_id: str,
    clip_id: str,
    video_path: Path,
    t_start: float,
    t_end: float,
    pose_observations: list[PoseObservationLike],
    shelf_region: list[tuple[float, float]] | None,
    embedder: Any,
    cache_dir: Path,
    config: SamplingConfig,
    hand_crop_size: int = 224,
    shelf_patch_size: int = 224,
) -> FeatureExtractionResult:
    """Extract hand and shelf features for one candidate.

    Uses sliding-window sampling and the existing cache infrastructure.

    Args:
        candidate_id: Candidate identifier.
        clip_id: Clip identifier.
        video_path: Path to source video.
        t_start: Window start (source-video seconds).
        t_end: Window end (source-video seconds).
        pose_observations: Pose observations for this candidate's actor/hand.
        shelf_region: Shelf region polygon points, or None.
        embedder: Image embedder instance.
        cache_dir: Cache directory for embeddings.
        config: Sampling configuration.
        hand_crop_size: Hand crop size in pixels.
        shelf_patch_size: Shelf patch size in pixels.

    Returns:
        FeatureExtractionResult with embeddings and cache stats.
    """
    from pickup_putdown.layer1.track_a.cache import (
        compute_crop_cache_key,
        compute_embedding_cache_key,
        get_video_checksum,
        is_embedding_cached,
        load_embedding,
        save_embedding,
    )
    from pickup_putdown.layer1.track_a.contracts import CropGeometry
    from pickup_putdown.layer1.track_a.crop_extractor import (
        extract_hand_crop,
        extract_shelf_patch,
        find_nearest_pose_observation,
        load_frame_at_timestamp,
    )

    result = FeatureExtractionResult(
        hand_embeddings=[],
        shelf_embeddings=[],
    )

    if not video_path.exists():
        result.skipped = True
        result.skip_reason = "missing_source_video"
        return result

    if not pose_observations:
        result.skipped = True
        result.skip_reason = "missing_pose_observations"
        return result

    if shelf_region is None or len(shelf_region) < 3:
        result.skipped = True
        result.skip_reason = "missing_shelf_region"
        return result

    sample_times = compute_inference_sample_times(t_start, t_end, config.sample_fps)
    if len(sample_times) < config.minimum_valid_samples:
        result.skipped = True
        result.skip_reason = "too_few_samples"
        return result

    video_checksum = get_video_checksum(video_path)
    cache_hits = 0
    cache_misses = 0

    for ts in sample_times:
        nearest = find_nearest_pose_observation(ts, pose_observations, max_tolerance_s=0.3)
        if nearest is None:
            continue

        frame = load_frame_at_timestamp(video_path, ts)
        if frame is None:
            continue

        # --- Hand crop ---
        wrist_x = getattr(nearest, "wrist_x", 0.0)
        wrist_y = getattr(nearest, "wrist_y", 0.0)

        hand_geom = CropGeometry(
            x=max(0, int(wrist_x) - hand_crop_size // 2),
            y=max(0, int(wrist_y) - hand_crop_size // 2),
            width=hand_crop_size,
            height=hand_crop_size,
        )
        hand_cache_key = compute_crop_cache_key(video_checksum, ts, hand_geom, "hand")
        hand_emb_key = compute_embedding_cache_key(
            hand_cache_key, embedder.model_name, embedder.version
        )

        if is_embedding_cached(cache_dir, hand_emb_key):
            emb = load_embedding(cache_dir, hand_emb_key)
            if emb is not None:
                result.hand_embeddings.append((ts, emb))
                cache_hits += 1
                continue

        hand_crop, hand_actual_geom = extract_hand_crop(frame, wrist_x, wrist_y, hand_crop_size)
        hand_cache_key = compute_crop_cache_key(video_checksum, ts, hand_actual_geom, "hand")
        hand_emb_key = compute_embedding_cache_key(
            hand_cache_key, embedder.model_name, embedder.version
        )
        hand_emb = embedder.embed(hand_crop)
        save_embedding(hand_emb, cache_dir, hand_emb_key)
        result.hand_embeddings.append((ts, hand_emb))
        cache_misses += 1

        # --- Shelf patch ---
        contact_pt = (wrist_x, wrist_y)
        shelf_crop, shelf_geom = extract_shelf_patch(
            frame, shelf_region, contact_pt, shelf_patch_size
        )
        shelf_cache_key = compute_crop_cache_key(video_checksum, ts, shelf_geom, "shelf")
        shelf_emb_key = compute_embedding_cache_key(
            shelf_cache_key, embedder.model_name, embedder.version
        )

        if is_embedding_cached(cache_dir, shelf_emb_key):
            emb = load_embedding(cache_dir, shelf_emb_key)
            if emb is not None:
                result.shelf_embeddings.append((ts, emb))
                cache_hits += 1
                continue

        shelf_emb = embedder.embed(shelf_crop)
        save_embedding(shelf_emb, cache_dir, shelf_emb_key)
        result.shelf_embeddings.append((ts, shelf_emb))
        cache_misses += 1

    result.cache_hits = cache_hits
    result.cache_misses = cache_misses
    return result


# ---------------------------------------------------------------------------
# Pose observation protocol (duck-typed)
# ---------------------------------------------------------------------------


class PoseObservationLike:
    """Minimal protocol for pose observations.

    Accepts both Pydantic PoseObservation and dicts/dataclasses.
    """

    timestamp_s: float
    wrist_x: float
    wrist_y: float
    wrist_confidence: float = 0.5

    @classmethod
    def from_any(cls, obs: Any) -> PoseObservationLike:
        """Convert any observation-like object to this protocol."""
        if hasattr(obs, "timestamp_s"):
            instance = cls()
            instance.timestamp_s = obs.timestamp_s
            instance.wrist_x = getattr(obs, "wrist_x", 0.0)
            instance.wrist_y = getattr(obs, "wrist_y", 0.0)
            instance.wrist_confidence = getattr(obs, "wrist_confidence", 0.5)
            return instance
        if isinstance(obs, dict):
            instance = cls()
            instance.timestamp_s = obs.get("timestamp_s", 0.0)
            instance.wrist_x = obs.get("wrist_x", 0.0)
            instance.wrist_y = obs.get("wrist_y", 0.0)
            instance.wrist_confidence = obs.get("wrist_confidence", 0.5)
            return instance
        raise TypeError(f"Cannot convert {type(obs)} to PoseObservationLike")


# ---------------------------------------------------------------------------
# Classifier prediction helpers
# ---------------------------------------------------------------------------


def predict_hand_state(
    clf: TrackAClassifier,
    embedding: np.ndarray,
    timestamp_s: float,
) -> HandStateEvidence:
    """Run hand-state classifier on a single embedding."""
    pred = clf.predict(embedding)

    probs = pred.probabilities
    p_empty = probs.get("empty", 0.0)
    p_carrying = probs.get("carrying", 0.0)

    sorted_probs = sorted([p_empty, p_carrying], reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

    evidence = HandStateEvidence(
        timestamp_s=timestamp_s,
        probability_empty=p_empty,
        probability_carrying=p_carrying,
        predicted_state=pred.state,
        confidence=pred.confidence,
        margin=margin,
        raw_probabilities=probs,
    )

    if pred.is_uncertain:
        evidence.probability_uncertain = 1.0
        evidence.predicted_state = "uncertain"
    else:
        evidence.probability_uncertain = 0.0

    return evidence


def predict_shelf_state(
    clf: TrackAClassifier,
    embedding: np.ndarray,
    timestamp_s: float,
) -> ShelfStateEvidence:
    """Run shelf-state classifier on a single embedding."""
    pred = clf.predict(embedding)

    probs = pred.probabilities
    p_removed = probs.get("object_removed", 0.0)
    p_placed = probs.get("object_placed", 0.0)
    p_no_change = probs.get("no_change", 0.0)

    all_probs = [p_removed, p_placed, p_no_change]
    sorted_probs = sorted(all_probs, reverse=True)
    margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else 0.0

    evidence = ShelfStateEvidence(
        timestamp_s=timestamp_s,
        probability_object_removed=p_removed,
        probability_object_placed=p_placed,
        probability_no_change=p_no_change,
        predicted_state=pred.state,
        confidence=pred.confidence,
        margin=margin,
        raw_probabilities=probs,
    )

    if pred.is_uncertain:
        evidence.probability_uncertain = 1.0
        evidence.predicted_state = "uncertain"
    else:
        evidence.probability_uncertain = 0.0

    return evidence


# ---------------------------------------------------------------------------
# Build state-machine observations
# ---------------------------------------------------------------------------


def build_observations(
    candidate_id: str,
    clip_id: str,
    actor_id: str,
    hand_side: str,
    region_id: str,
    pose_observations: list[Any],
    hand_evidence_list: list[HandStateEvidence],
    shelf_evidence_list: list[ShelfStateEvidence],
    shelf_region: list[tuple[float, float]] | None,
    region_entry_distance_px: float = 40.0,
) -> list[TrackAObservation]:
    """Convert timestamped evidence into state-machine observations.

    Merges pose trajectory info with classifier evidence at each timestamp.
    Returns observations sorted by timestamp.
    """
    hand_by_ts: dict[float, HandStateEvidence] = {}
    for h in hand_evidence_list:
        hand_by_ts[h.timestamp_s] = h

    shelf_by_ts: dict[float, ShelfStateEvidence] = {}
    for s in shelf_evidence_list:
        shelf_by_ts[s.timestamp_s] = s

    all_timestamps = sorted(set(list(hand_by_ts.keys()) + list(shelf_by_ts.keys())))

    if not all_timestamps:
        return []

    observations: list[TrackAObservation] = []

    for ts in all_timestamps:
        hand_ev = hand_by_ts.get(ts)
        shelf_ev = shelf_by_ts.get(ts)

        nearest_pose = _find_nearest_pose(ts, pose_observations, max_tolerance_s=0.3)

        wrist_x = getattr(nearest_pose, "wrist_x", None) if nearest_pose else None
        wrist_y = getattr(nearest_pose, "wrist_y", None) if nearest_pose else None
        traj_conf = getattr(nearest_pose, "wrist_confidence", 0.5) if nearest_pose else 0.5

        inside = False
        dist_px: float | None = None
        if wrist_x is not None and wrist_y is not None and shelf_region:
            inside = _point_in_polygon(float(wrist_x), float(wrist_y), shelf_region)
            dist_px = _point_to_polygon_distance(float(wrist_x), float(wrist_y), shelf_region)

        obs = TrackAObservation(
            clip_id=clip_id,
            candidate_id=candidate_id,
            actor_id=actor_id,
            hand_side=hand_side,
            region_id=region_id,
            timestamp_s=ts,
            wrist_x=wrist_x,
            wrist_y=wrist_y,
            wrist_to_region_distance_px=dist_px,
            inside_region=inside,
            trajectory_confidence=float(traj_conf),
            hand_prob_empty=hand_ev.probability_empty if hand_ev else 0.0,
            hand_prob_carrying=hand_ev.probability_carrying if hand_ev else 0.0,
            hand_prob_uncertain=hand_ev.probability_uncertain if hand_ev else 0.0,
            shelf_prob_object_removed=shelf_ev.probability_object_removed if shelf_ev else 0.0,
            shelf_prob_object_placed=shelf_ev.probability_object_placed if shelf_ev else 0.0,
            shelf_prob_no_change=shelf_ev.probability_no_change if shelf_ev else 0.0,
            shelf_prob_uncertain=shelf_ev.probability_uncertain if shelf_ev else 0.0,
        )
        observations.append(obs)

    observations.sort(key=lambda o: o.timestamp_s)
    return observations


def _find_nearest_pose(
    timestamp_s: float,
    poses: list[Any],
    max_tolerance_s: float = 0.3,
) -> Any | None:
    """Find nearest pose observation to a timestamp."""
    if not poses:
        return None
    nearest = min(poses, key=lambda p: abs(getattr(p, "timestamp_s", 0.0) - timestamp_s))
    diff = abs(getattr(nearest, "timestamp_s", 0.0) - timestamp_s)
    return nearest if diff <= max_tolerance_s else None


def _point_in_polygon(px: float, py: float, polygon: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test."""
    if len(polygon) < 3:
        return False
    inside = False
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        crosses = (y1 > py) != (y2 > py)
        if crosses:
            ix = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if px < ix:
                inside = not inside
    return inside


def _point_to_polygon_distance(px: float, py: float, polygon: list[tuple[float, float]]) -> float:
    """Minimum distance from point to polygon edges."""
    if not polygon:
        return float("inf")
    min_d = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        d = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        min_d = min(min_d, d)
    return min_d


def _point_to_segment_distance(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> float:
    dx = x2 - x1
    dy = y2 - y1
    len_sq = dx * dx + dy * dy
    if len_sq < 1e-12:
        return math.hypot(px - x1, py - y1)
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / len_sq))
    return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))


# ---------------------------------------------------------------------------
# Transition-frame grace window
# ---------------------------------------------------------------------------


def apply_grace_window(
    events: list[StateMachineEvent],
    observations: list[TrackAObservation],
    grace_s: float,
) -> list[StateMachineEvent]:
    """Apply transition-frame grace window to recover events lost at withdrawal.

    When the wrist exits the region on the exact observation containing the
    final transfer evidence, the state machine may reject the event because
    the withdrawal is too brief. The grace window allows the last inside-region
    evidence to complete a transfer if a following outside-region observation
    arrives within `grace_s`.

    Args:
        events: Raw state-machine events.
        observations: All observations for the stream.
        grace_s: Grace window duration in seconds.

    Returns:
        Events list (may add recovered events).
    """
    if grace_s <= 0 or not observations:
        return events

    recovered: list[StateMachineEvent] = []

    for i, obs in enumerate(observations):
        if not obs.inside_region:
            continue

        next_obs = None
        for j in range(i + 1, len(observations)):
            if observations[j].timestamp_s > obs.timestamp_s:
                next_obs = observations[j]
                break

        if next_obs is None:
            continue

        gap = next_obs.timestamp_s - obs.timestamp_s
        if gap > grace_s:
            continue

        if next_obs.inside_region:
            continue

        # This inside obs is followed by an outside obs within grace window.
        # Check if there's transfer evidence at this observation that could
        # form an event. We build a synthetic event from the evidence.
        hand_carries = obs.hand_prob_carrying > 0.55
        hand_empty = obs.hand_prob_empty > 0.55
        shelf_removed = obs.shelf_prob_object_removed > 0.50
        shelf_placed = obs.shelf_prob_object_placed > 0.50

        label: str | None = None
        if hand_carries and shelf_removed:
            label = "pickup"
        elif hand_empty and shelf_placed:
            label = "putdown"

        if label is None:
            continue

        # Avoid duplicate: check if an event already covers this time
        already = any(
            abs(obs.timestamp_s - e.transfer_timestamp_s) < 0.15 for e in events + recovered
        )
        if already:
            continue

        synth = StateMachineEvent(
            clip_id=obs.clip_id,
            candidate_id=obs.candidate_id,
            actor_id=obs.actor_id,
            hand_side=obs.hand_side,
            region_id=obs.region_id,
            label=label,
            start_s=obs.timestamp_s - 0.15,
            end_s=next_obs.timestamp_s,
            transfer_timestamp_s=obs.timestamp_s,
            confidence=0.35,
            evidence=EvidenceSummary(
                n_supporting_observations=1,
                trajectory_confidence=obs.trajectory_confidence,
            ),
            cycle_id=-1,
        )
        recovered.append(synth)

    return events + recovered


# ---------------------------------------------------------------------------
# Event boundary refinement
# ---------------------------------------------------------------------------


def refine_event_boundaries(
    event: StateMachineEvent,
    candidate_start_s: float,
    candidate_end_s: float,
    clip_start_s: float = 0.0,
    clip_end_s: float | None = None,
    config: BoundaryRefinementConfig | None = None,
) -> StateMachineEvent:
    """Refine provisional event boundaries using state trace information.

    Rules:
    - start_s >= max(clip_start, candidate_start - extension)
    - end_s <= min(clip_end, candidate_end + extension)
    - start_s < end_s always
    - Minimum duration enforced when needed

    Args:
        event: Raw state-machine event.
        candidate_start_s: Candidate window start.
        candidate_end_s: Candidate window end.
        clip_start_s: Source clip start (usually 0).
        clip_end_s: Source clip end, or None for no upper bound.
        config: Boundary refinement configuration.

    Returns:
        Event with refined boundaries.
    """
    if config is None:
        config = BoundaryRefinementConfig()

    start = event.start_s
    end = event.end_s
    transfer = event.transfer_timestamp_s

    # Bound by candidate window with extension
    soft_start = max(clip_start_s, candidate_start_s - config.maximum_candidate_extension_s)
    soft_end = candidate_end_s + config.maximum_candidate_extension_s
    if clip_end_s is not None:
        soft_end = min(soft_end, clip_end_s)

    # Refine start toward contact/transfer
    if config.use_contact_start:
        start = max(start, soft_start)
        start = min(start, transfer)
    else:
        start = max(start, soft_start)

    # Refine end toward stabilization/withdrawal
    end = min(end, soft_end)
    end = max(end, transfer)

    # Ensure start < end
    if start >= end:
        mid = (start + end) / 2
        start = mid - config.minimum_event_duration_s / 2
        end = mid + config.minimum_event_duration_s / 2

    # Enforce minimum duration
    duration = end - start
    if duration < config.minimum_event_duration_s:
        center = (start + end) / 2
        start = center - config.minimum_event_duration_s / 2
        end = center + config.minimum_event_duration_s / 2

    # Final clamp
    start = max(start, clip_start_s)
    if clip_end_s is not None:
        end = min(end, clip_end_s)

    # Safety: if still invalid, use minimum duration around transfer
    if start >= end:
        start = transfer - config.minimum_event_duration_s / 2
        end = transfer + config.minimum_event_duration_s / 2

    refined = StateMachineEvent(
        clip_id=event.clip_id,
        candidate_id=event.candidate_id,
        actor_id=event.actor_id,
        hand_side=event.hand_side,
        region_id=event.region_id,
        label=event.label,
        start_s=round(start, 4),
        end_s=round(end, 4),
        transfer_timestamp_s=transfer,
        confidence=event.confidence,
        evidence=event.evidence,
        cycle_id=event.cycle_id,
    )
    return refined


# ---------------------------------------------------------------------------
# Cross-candidate deduplication
# ---------------------------------------------------------------------------


def temporal_iou(start1: float, end1: float, start2: float, end2: float) -> float:
    """Compute temporal IoU between two intervals."""
    overlap_start = max(start1, start2)
    overlap_end = min(end1, end2)
    overlap = max(0.0, overlap_end - overlap_start)
    union = (end1 - start1) + (end2 - start2) - overlap
    if union <= 0:
        return 0.0
    return overlap / union


def deduplicate_predictions(
    events: list[StateMachineEvent],
    config: DeduplicationConfig,
) -> tuple[list[StateMachineEvent], list[DedupAuditEntry]]:
    """Deduplicate events from overlapping candidates.

    Rules:
    - Same clip_id required
    - Same label required (pickup never merged with putdown)
    - Same actor/hand/region when configured
    - Temporal IoU above threshold
    - Keep highest-confidence prediction
    - Separate repeated events are preserved

    Args:
        events: Raw state-machine events (possibly from multiple candidates).
        config: Deduplication configuration.

    Returns:
        Tuple of (deduplicated events, audit entries).
    """
    if len(events) <= 1:
        return list(events), []

    sorted_events = sorted(events, key=lambda e: (-e.confidence, e.start_s, e.clip_id))
    kept: list[StateMachineEvent] = []
    audit: list[DedupAuditEntry] = []
    suppressed_indices: set[int] = set()

    n = len(sorted_events)

    for i in range(n):
        if i in suppressed_indices:
            continue

        kept.append(sorted_events[i])
        group: list[int] = []

        for j in range(i + 1, n):
            if j in suppressed_indices:
                continue

            e1 = sorted_events[i]
            e2 = sorted_events[j]

            # Must be same clip
            if e1.clip_id != e2.clip_id:
                continue

            # Must be same label
            if e1.label != e2.label:
                continue

            # Actor/hand/region checks
            if config.require_same_actor and e1.actor_id != e2.actor_id:
                continue
            if config.require_same_hand and e1.hand_side != e2.hand_side:
                continue
            if config.require_same_region and e1.region_id != e2.region_id:
                continue

            iou = temporal_iou(e1.start_s, e1.end_s, e2.start_s, e2.end_s)
            if iou < config.temporal_iou_threshold:
                continue

            transfer_diff = abs(e1.transfer_timestamp_s - e2.transfer_timestamp_s)
            if transfer_diff > config.transfer_time_tolerance_s:
                continue

            suppressed_indices.add(j)
            group.append(j)

        if group:
            suppressed_events = [sorted_events[j] for j in group]
            audit.append(
                DedupAuditEntry(
                    kept_prediction_id=_event_to_pred_id(sorted_events[i]),
                    kept_candidate_id=sorted_events[i].candidate_id,
                    kept_confidence=sorted_events[i].confidence,
                    suppressed_prediction_ids=[_event_to_pred_id(e) for e in suppressed_events],
                    suppressed_candidate_ids=[e.candidate_id for e in suppressed_events],
                    suppressed_confidences=[e.confidence for e in suppressed_events],
                    temporal_iou=max(
                        temporal_iou(
                            sorted_events[i].start_s, sorted_events[i].end_s, e2.start_s, e2.end_s
                        )
                        for e2 in suppressed_events
                    ),
                    transfer_time_diff_s=max(
                        abs(sorted_events[i].transfer_timestamp_s - e2.transfer_timestamp_s)
                        for e2 in suppressed_events
                    ),
                    selection_reason="highest_confidence",
                )
            )

    return kept, audit


def _event_to_pred_id(event: StateMachineEvent) -> str:
    """Generate a deterministic prediction ID from an event."""
    payload = f"{event.clip_id}:{event.label}:{event.start_s:.4f}:{event.end_s:.4f}"
    return f"pred_{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


# ---------------------------------------------------------------------------
# Canonical prediction output
# ---------------------------------------------------------------------------


def events_to_predictions(
    events: list[StateMachineEvent],
    model_name: str = "track_a",
) -> list[CanonicalPrediction]:
    """Convert state-machine events to canonical prediction records."""
    predictions: list[CanonicalPrediction] = []
    for event in events:
        pred = CanonicalPrediction(
            clip_id=event.clip_id,
            pred_id=_event_to_pred_id(event),
            label=event.label,
            start_s=event.start_s,
            end_s=event.end_s,
            confidence=event.confidence,
            actor_id=event.actor_id,
            hand_side=event.hand_side,
            region_id=event.region_id,
            model=model_name,
            candidate_ids=[event.candidate_id],
        )
        predictions.append(pred)
    return predictions


def merge_dedup_candidate_ids(
    predictions: list[CanonicalPrediction],
    audit: list[DedupAuditEntry],
) -> list[CanonicalPrediction]:
    """Merge suppressed candidate IDs back into kept predictions."""
    pred_id_map: dict[str, CanonicalPrediction] = {}
    for p in predictions:
        pred_id_map[p.pred_id] = p

    for entry in audit:
        if entry.kept_prediction_id in pred_id_map:
            pred = pred_id_map[entry.kept_prediction_id]
            existing = set(pred.candidate_ids)
            for cid in entry.suppressed_candidate_ids:
                existing.add(cid)
            pred.candidate_ids = sorted(existing)

    return predictions


def save_predictions_csv(
    predictions: list[CanonicalPrediction],
    path: Path,
) -> Path:
    """Save predictions to canonical CSV format."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_id",
        "pred_id",
        "type",
        "t_start",
        "t_end",
        "score",
        "model",
        "actor_id",
        "hand_side",
        "region_id",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for p in predictions:
            writer.writerow(
                {
                    "clip_id": p.clip_id,
                    "pred_id": p.pred_id,
                    "type": p.label,
                    "t_start": f"{p.start_s:.4f}",
                    "t_end": f"{p.end_s:.4f}",
                    "score": f"{p.confidence:.4f}",
                    "model": p.model,
                    "actor_id": p.actor_id,
                    "hand_side": p.hand_side,
                    "region_id": p.region_id,
                }
            )
    return path


def save_diagnostics(
    diagnostics: list[CandidateDiagnostics],
    output_dir: Path,
) -> Path:
    """Save per-candidate diagnostics as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "candidate_diagnostics.json"
    data = []
    for d in diagnostics:
        entry: dict[str, Any] = {
            "candidate_id": d.candidate_id,
            "clip_id": d.clip_id,
            "actor_id": d.actor_id,
            "hand_side": d.hand_side,
            "region_id": d.region_id,
            "skipped": d.skipped,
            "skip_reason": d.skip_reason,
            "n_samples": d.n_samples,
            "n_hand_evidence": d.n_hand_evidence,
            "n_shelf_evidence": d.n_shelf_evidence,
            "n_events": d.n_events,
        }
        if d.observations:
            entry["observations"] = d.observations
        data.append(entry)
    path.write_text(json.dumps(data, indent=2, default=str))
    return path


def save_inference_summary(
    summary: InferenceSummary,
    output_dir: Path,
) -> Path:
    """Save inference summary as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "inference_summary.json"
    data = {
        "candidates_total": summary.candidates_total,
        "candidates_processed": summary.candidates_processed,
        "candidates_skipped": summary.candidates_skipped,
        "skip_reasons": summary.skip_reasons,
        "feature_cache_hits": summary.feature_cache_hits,
        "feature_cache_misses": summary.feature_cache_misses,
        "total_samples": summary.total_samples,
        "raw_events_emitted": summary.raw_events_emitted,
        "final_events_after_dedup": summary.final_events_after_dedup,
        "pickup_count": summary.pickup_count,
        "putdown_count": summary.putdown_count,
        "mean_confidence": summary.mean_confidence,
        "uncertain_proportion": summary.uncertain_proportion,
        "confidence_distribution": summary.confidence_distribution,
        "timestamp": datetime.now().isoformat(),
    }
    path.write_text(json.dumps(data, indent=2))
    return path


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


class TrackAInferencePipeline:
    """End-to-end Track A inference pipeline.

    Integrates feature extraction, classifier inference, state-machine
    processing, boundary refinement, and deduplication.

    Usage:
        pipeline = TrackAInferencePipeline(config)
        result = pipeline.run(
            candidates=candidates,
            pose_observations=poses,
            source_videos=video_paths,
            hand_classifier_path=hand_path,
            shelf_classifier_path=shelf_path,
            output_dir=output_dir,
        )
    """

    def __init__(
        self,
        config: InferenceConfig | None = None,
        state_machine_config: StateMachineConfig | None = None,
    ) -> None:
        self.config = config or InferenceConfig()
        self.sm_config = state_machine_config or StateMachineConfig()

    def run(
        self,
        candidates: list[Any],
        pose_observations: list[Any],
        source_videos: dict[str, Path],
        hand_classifier_path: Path | str,
        shelf_classifier_path: Path | str,
        output_dir: Path | str,
        *,
        shelf_regions: dict[str, list[tuple[float, float]]] | None = None,
        embedder: Any = None,
        cache_dir: Path | str | None = None,
        clip_durations: dict[str, float] | None = None,
        hand_crop_size: int = 224,
        shelf_patch_size: int = 224,
    ) -> InferenceResult:
        """Run the full inference pipeline.

        Args:
            candidates: List of Candidate objects.
            pose_observations: All pose observations.
            source_videos: Map of clip_id -> video path.
            hand_classifier_path: Path to hand_state.joblib.
            shelf_classifier_path: Path to shelf_state.joblib.
            output_dir: Directory for output files.
            shelf_regions: Map of region_id -> polygon points.
            embedder: Image embedder (required for feature extraction).
            cache_dir: Cache directory for embeddings.
            clip_durations: Map of clip_id -> duration in seconds.
            hand_crop_size: Hand crop size in pixels.
            shelf_patch_size: Shelf patch size in pixels.

        Returns:
            InferenceResult with predictions, events, audit, diagnostics.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = InferenceSummary(candidates_total=len(candidates))

        # Load and validate classifiers
        hand_clf, hand_meta = load_classifier_artifact(
            hand_classifier_path,
            metadata_path=Path(str(hand_classifier_path).replace(".joblib", "_metadata.json")),
        )
        validate_classifier_classes(hand_clf, hand_meta, HAND_STATE_CLASS_NAMES, "hand")

        shelf_clf, shelf_meta = load_classifier_artifact(
            shelf_classifier_path,
            metadata_path=Path(str(shelf_classifier_path).replace(".joblib", "_metadata.json")),
        )
        validate_classifier_classes(shelf_clf, shelf_meta, SHELF_STATE_CLASS_NAMES, "shelf")

        validate_artifact_compatibility(hand_meta, shelf_meta)

        # Process each candidate
        all_observations: list[TrackAObservation] = []
        all_raw_events: list[StateMachineEvent] = []
        diagnostics: list[CandidateDiagnostics] = []

        for cand in candidates:
            cand_id = getattr(cand, "candidate_id", "unknown")
            clip_id = getattr(cand, "clip_id", "")
            actor_id = getattr(cand, "actor_id", "")
            hand_side = getattr(cand, "hand_side", "") or ""
            region_id = getattr(cand, "region_id", "") or ""
            t_start = getattr(cand, "raw_start_s", 0.0)
            t_end = getattr(cand, "raw_end_s", 0.0)

            diag = CandidateDiagnostics(
                candidate_id=cand_id,
                clip_id=clip_id,
                actor_id=actor_id,
                hand_side=hand_side,
                region_id=region_id,
            )

            # Validate required fields
            if not clip_id or not actor_id:
                diag.skipped = True
                diag.skip_reason = "missing_identity_fields"
                summary.candidates_skipped += 1
                summary.skip_reasons["missing_identity_fields"] = (
                    summary.skip_reasons.get("missing_identity_fields", 0) + 1
                )
                diagnostics.append(diag)
                continue

            # Resolve video
            video_path = source_videos.get(clip_id)
            if video_path is None or not video_path.exists():
                diag.skipped = True
                diag.skip_reason = "missing_source_video"
                summary.candidates_skipped += 1
                summary.skip_reasons["missing_source_video"] = (
                    summary.skip_reasons.get("missing_source_video", 0) + 1
                )
                diagnostics.append(diag)
                continue

            # Resolve pose observations for this candidate
            cand_poses = _filter_poses_for_candidate(cand, pose_observations)
            if not cand_poses:
                diag.skipped = True
                diag.skip_reason = "missing_pose_observations"
                summary.candidates_skipped += 1
                summary.skip_reasons["missing_pose_observations"] = (
                    summary.skip_reasons.get("missing_pose_observations", 0) + 1
                )
                diagnostics.append(diag)
                continue

            # Resolve shelf region
            shelf_region: list[tuple[float, float]] | None = None
            if shelf_regions and region_id:
                shelf_region = shelf_regions.get(region_id)
            elif shelf_regions:
                shelf_region = next(iter(shelf_regions.values()), None)

            if not shelf_region:
                diag.skipped = True
                diag.skip_reason = "missing_shelf_region"
                summary.candidates_skipped += 1
                summary.skip_reasons["missing_shelf_region"] = (
                    summary.skip_reasons.get("missing_shelf_region", 0) + 1
                )
                diagnostics.append(diag)
                continue

            # Feature extraction
            hand_evidence_list: list[HandStateEvidence] = []
            shelf_evidence_list: list[ShelfStateEvidence] = []

            if embedder is not None and cache_dir is not None:
                feat_result = extract_features_for_candidate(
                    candidate_id=cand_id,
                    clip_id=clip_id,
                    video_path=video_path,
                    t_start=t_start,
                    t_end=t_end,
                    pose_observations=cand_poses,
                    shelf_region=shelf_region,
                    embedder=embedder,
                    cache_dir=Path(cache_dir),
                    config=self.config.sampling,
                    hand_crop_size=hand_crop_size,
                    shelf_patch_size=shelf_patch_size,
                )

                if feat_result.skipped:
                    diag.skipped = True
                    diag.skip_reason = feat_result.skip_reason
                    summary.candidates_skipped += 1
                    summary.skip_reasons[feat_result.skip_reason] = (
                        summary.skip_reasons.get(feat_result.skip_reason, 0) + 1
                    )
                    diagnostics.append(diag)
                    continue

                summary.feature_cache_hits += feat_result.cache_hits
                summary.feature_cache_misses += feat_result.cache_misses
                summary.total_samples += len(feat_result.hand_embeddings)

                # Classifier predictions
                for ts, emb in feat_result.hand_embeddings:
                    he = predict_hand_state(hand_clf, emb, ts)
                    hand_evidence_list.append(he)

                for ts, emb in feat_result.shelf_embeddings:
                    se = predict_shelf_state(shelf_clf, emb, ts)
                    shelf_evidence_list.append(se)

                diag.n_samples = len(feat_result.hand_embeddings)
                diag.n_hand_evidence = len(hand_evidence_list)
                diag.n_shelf_evidence = len(shelf_evidence_list)
            else:
                # No embedder provided: skip feature extraction
                # (tests can provide mock evidence directly)
                pass

            # Build observations
            obs_list = build_observations(
                candidate_id=cand_id,
                clip_id=clip_id,
                actor_id=actor_id,
                hand_side=hand_side,
                region_id=region_id,
                pose_observations=cand_poses,
                hand_evidence_list=hand_evidence_list,
                shelf_evidence_list=shelf_evidence_list,
                shelf_region=shelf_region,
                region_entry_distance_px=self.sm_config.region_entry_distance_px,
            )

            if self.config.debug_traces:
                for obs in obs_list:
                    diag.observations.append(
                        {
                            "timestamp_s": obs.timestamp_s,
                            "inside_region": obs.inside_region,
                            "distance_px": obs.wrist_to_region_distance_px,
                            "hand_probs": {
                                "empty": obs.hand_prob_empty,
                                "carrying": obs.hand_prob_carrying,
                                "uncertain": obs.hand_prob_uncertain,
                            },
                            "shelf_probs": {
                                "removed": obs.shelf_prob_object_removed,
                                "placed": obs.shelf_prob_object_placed,
                                "no_change": obs.shelf_prob_no_change,
                                "uncertain": obs.shelf_prob_uncertain,
                            },
                            "trajectory_confidence": obs.trajectory_confidence,
                        }
                    )

            all_observations.extend(obs_list)
            summary.candidates_processed += 1
            diagnostics.append(diag)

        # Run state machine per stream
        if all_observations:
            sm = RepeatingInteractionStateMachine(
                config=self.sm_config,
                debug=self.config.debug_traces,
            )
            raw_events = sm.process(all_observations)
            sm.finalize()

            # Group events by stream for grace window
            stream_events: dict[tuple[str, str, str, str], list[StateMachineEvent]] = {}
            stream_obs: dict[tuple[str, str, str, str], list[TrackAObservation]] = {}

            for obs in all_observations:
                key = (obs.clip_id, obs.actor_id, obs.hand_side, obs.region_id)
                stream_obs.setdefault(key, []).append(obs)

            for event in raw_events:
                key = (event.clip_id, event.actor_id, event.hand_side, event.region_id)
                stream_events.setdefault(key, []).append(event)

            # Apply grace window per stream
            for key, events in stream_events.items():
                obs_for_stream = stream_obs.get(key, [])
                graced = apply_grace_window(events, obs_for_stream, self.config.transition_grace_s)
                stream_events[key] = graced

            all_raw_events = []
            for events in stream_events.values():
                all_raw_events.extend(events)

        summary.raw_events_emitted = len(all_raw_events)

        # Boundary refinement
        refined_events: list[StateMachineEvent] = []
        cand_map: dict[str, Any] = {}
        for cand in candidates:
            cid = getattr(cand, "candidate_id", "")
            cand_map[cid] = cand

        for event in all_raw_events:
            cand = cand_map.get(event.candidate_id)
            if cand:
                t_start = getattr(cand, "raw_start_s", event.start_s)
                t_end = getattr(cand, "raw_end_s", event.end_s)
            else:
                t_start = event.start_s
                t_end = event.end_s

            clip_end = clip_durations.get(event.clip_id) if clip_durations else None
            refined = refine_event_boundaries(
                event,
                candidate_start_s=t_start,
                candidate_end_s=t_end,
                clip_end_s=clip_end,
                config=self.config.boundary_refinement,
            )
            refined_events.append(refined)

        # Deduplication
        deduped_events, dedup_audit = deduplicate_predictions(
            refined_events,
            self.config.deduplication,
        )

        # Convert to predictions
        predictions = events_to_predictions(deduped_events, model_name="track_a")
        predictions = merge_dedup_candidate_ids(predictions, dedup_audit)

        # Update diagnostics event counts
        for event in deduped_events:
            for diag in diagnostics:
                if diag.candidate_id == event.candidate_id:
                    diag.n_events += 1

        # Summary stats
        summary.final_events_after_dedup = len(predictions)
        summary.pickup_count = sum(1 for p in predictions if p.label == "pickup")
        summary.putdown_count = sum(1 for p in predictions if p.label == "putdown")

        if predictions:
            summary.mean_confidence = sum(p.confidence for p in predictions) / len(predictions)

        # Confidence distribution
        if predictions:
            bins = {"low": 0, "medium": 0, "high": 0}
            for p in predictions:
                if p.confidence < 0.4:
                    bins["low"] += 1
                elif p.confidence < 0.7:
                    bins["medium"] += 1
                else:
                    bins["high"] += 1
            summary.confidence_distribution = bins

        # Save outputs
        output_paths: dict[str, Path] = {}

        pred_csv = save_predictions_csv(predictions, output_dir / "predictions.csv")
        output_paths["predictions_csv"] = pred_csv

        # Save raw events
        raw_events_path = output_dir / "raw_state_machine_events.json"
        _save_raw_events(all_raw_events, raw_events_path)
        output_paths["raw_events_json"] = raw_events_path

        # Save dedup audit
        if dedup_audit:
            dedup_path = output_dir / "dedup_audit.json"
            _save_dedup_audit(dedup_audit, dedup_path)
            output_paths["dedup_audit_json"] = dedup_path

        # Save diagnostics
        diag_path = save_diagnostics(diagnostics, output_dir / "diagnostics")
        output_paths["diagnostics_json"] = diag_path

        # Save summary
        summary_path = save_inference_summary(summary, output_dir)
        output_paths["summary_json"] = summary_path

        return InferenceResult(
            predictions=predictions,
            raw_events=all_raw_events,
            dedup_audit=dedup_audit,
            diagnostics=diagnostics,
            summary=summary,
            output_paths=output_paths,
        )


def _filter_poses_for_candidate(
    candidate: Any,
    all_poses: list[Any],
) -> list[Any]:
    """Filter pose observations for a specific candidate."""
    clip_id = getattr(candidate, "clip_id", "")
    actor_id = getattr(candidate, "actor_id", "")
    hand_side = getattr(candidate, "hand_side", "") or ""
    w_start = getattr(candidate, "window_start_s", getattr(candidate, "raw_start_s", 0.0))
    w_end = getattr(candidate, "window_end_s", getattr(candidate, "raw_end_s", 0.0))

    relevant = [
        p
        for p in all_poses
        if getattr(p, "clip_id", "") == clip_id
        and getattr(p, "actor_id", "") == actor_id
        and (not hand_side or getattr(p, "hand_side", "") == hand_side)
        and w_start <= getattr(p, "timestamp_s", 0.0) <= w_end
    ]

    # Fallback for person-tracker actor_id format
    if not relevant and ":" in actor_id:
        relevant = [
            p
            for p in all_poses
            if getattr(p, "clip_id", "") == clip_id
            and (not hand_side or getattr(p, "hand_side", "") == hand_side)
            and w_start <= getattr(p, "timestamp_s", 0.0) <= w_end
        ]

    relevant.sort(key=lambda p: getattr(p, "timestamp_s", 0.0))
    return relevant


def _save_raw_events(events: list[StateMachineEvent], path: Path) -> None:
    """Save raw state-machine events as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for e in events:
        data.append(
            {
                "clip_id": e.clip_id,
                "candidate_id": e.candidate_id,
                "actor_id": e.actor_id,
                "hand_side": e.hand_side,
                "region_id": e.region_id,
                "label": e.label,
                "start_s": e.start_s,
                "end_s": e.end_s,
                "transfer_timestamp_s": e.transfer_timestamp_s,
                "confidence": e.confidence,
                "cycle_id": e.cycle_id,
                "evidence": {
                    "pre_transfer_hand_empty": e.evidence.pre_transfer_hand_empty,
                    "pre_transfer_hand_carrying": e.evidence.pre_transfer_hand_carrying,
                    "post_transfer_hand_empty": e.evidence.post_transfer_hand_empty,
                    "post_transfer_hand_carrying": e.evidence.post_transfer_hand_carrying,
                    "shelf_transition_prob": e.evidence.shelf_transition_prob,
                    "trajectory_confidence": e.evidence.trajectory_confidence,
                    "n_supporting_observations": e.evidence.n_supporting_observations,
                    "evidence_duration_s": e.evidence.evidence_duration_s,
                    "uncertainty_proportion": e.evidence.uncertainty_proportion,
                },
            }
        )
    path.write_text(json.dumps(data, indent=2))


def _save_dedup_audit(audit: list[DedupAuditEntry], path: Path) -> None:
    """Save deduplication audit as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for entry in audit:
        data.append(
            {
                "kept_prediction_id": entry.kept_prediction_id,
                "kept_candidate_id": entry.kept_candidate_id,
                "kept_confidence": entry.kept_confidence,
                "suppressed_prediction_ids": entry.suppressed_prediction_ids,
                "suppressed_candidate_ids": entry.suppressed_candidate_ids,
                "suppressed_confidences": entry.suppressed_confidences,
                "temporal_iou": entry.temporal_iou,
                "transfer_time_diff_s": entry.transfer_time_diff_s,
                "selection_reason": entry.selection_reason,
            }
        )
    path.write_text(json.dumps(data, indent=2))

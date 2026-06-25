"""Track A per-clip inference orchestration (task_10).

Composes the label-free feature wrapper, the trained (or placeholder) state
classifiers, and the repeating state machine into canonical Track A predictions
for one clip, and writes them as a Task 8-consumable ``events.csv``.

The orchestration takes already-deserialised domain objects (candidates, pose,
shelf regions) plus injectable classifiers and a feature function, so it is unit
-testable without parquet I/O or video decoding. Parquet/shelf loading and the
encoder wiring live in the CLI layer (next commit).
"""

from __future__ import annotations

import csv
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from pickup_putdown.config import TrackAConfig
from pickup_putdown.layer1.track_a.features import SampleFeatures
from pickup_putdown.layer1.track_a.state_machine import GroupInput, decode_all
from pickup_putdown.layer1.track_a.state_types import (
    CANONICAL_PREDICTION_COLUMNS,
    ActorHandRegion,
    HandStateClassifier,
    ShelfStateClassifier,
    StateObservation,
    TrackAPrediction,
)

if TYPE_CHECKING:
    from pickup_putdown.common.schemas import Candidate, PoseObservation
    from pickup_putdown.config import TrackAFeaturesConfig
    from pickup_putdown.perception.shelf_regions import Polygon

#: Produces label-free features for one candidate (wraps the feature shim).
FeatureFn = Callable[["CandidateInput"], "list[SampleFeatures]"]


@dataclass(frozen=True)
class CandidateInput:
    """One candidate plus the inputs Track A needs to score it."""

    candidate: Candidate
    pose_observations: tuple[PoseObservation, ...]
    shelf_region: Polygon


def observations_from_features(
    features: Sequence[SampleFeatures],
    hand_classifier: HandStateClassifier,
    shelf_classifier: ShelfStateClassifier,
) -> list[StateObservation]:
    """Score paired embeddings into per-timestamp state observations."""
    return [
        StateObservation(
            timestamp_s=feature.timestamp_s,
            sample_position=feature.sample_position,
            p_hand_holding=hand_classifier.predict_holding(feature.hand_embedding),
            p_shelf_occupied=shelf_classifier.predict_occupied(feature.shelf_embedding),
        )
        for feature in features
    ]


def infer_track_a(
    clip_id: str,
    candidate_inputs: Sequence[CandidateInput],
    config: TrackAConfig,
    *,
    hand_classifier: HandStateClassifier,
    shelf_classifier: ShelfStateClassifier,
    feature_fn: FeatureFn,
) -> list[TrackAPrediction]:
    """Run Track A over one clip's candidates and return canonical predictions."""
    groups: list[GroupInput] = []
    for candidate_input in candidate_inputs:
        features = feature_fn(candidate_input)
        if not features:
            continue
        observations = observations_from_features(features, hand_classifier, shelf_classifier)
        candidate = candidate_input.candidate
        groups.append(
            GroupInput(
                group=ActorHandRegion(
                    actor_id=candidate.actor_id,
                    hand_side=candidate.hand_side or "",
                    region_id=candidate.region_id or "",
                ),
                candidate_id=candidate.candidate_id,
                observations=tuple(observations),
            )
        )
    return decode_all(clip_id, groups, config)


def build_feature_fn(video_path: Path, features_config: TrackAFeaturesConfig) -> FeatureFn:
    """Build the real (encoder-backed) feature function for one video.

    Wires the Task 9 encoder + crop extraction via the TEMPORARY feature shim.
    Lazy-imports keep cv2/torch off the import path until this is actually used
    (the real path is gated behind classifier availability by the caller).
    """
    from pickup_putdown.layer1.track_a.features import (
        extract_inference_features,
        make_crop_embedder,
    )
    from pickup_putdown.layer1.track_a.image_features import create_embedder

    embedder = create_embedder(features_config)
    crop_embed_fn = make_crop_embedder(video_path, embedder)

    def _feature_fn(candidate_input: CandidateInput) -> list[SampleFeatures]:
        return extract_inference_features(
            video_path,
            candidate_input.candidate,
            list(candidate_input.pose_observations),
            candidate_input.shelf_region,
            features_config,
            crop_embed_fn,
        )

    return _feature_fn


def write_events_csv(path: Path, predictions: Sequence[TrackAPrediction]) -> Path:
    """Write canonical Track A predictions (header always present)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CANONICAL_PREDICTION_COLUMNS))
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.to_canonical_row())
    return path

"""TEMPORARY label-free feature wrapper for Track A inference (task_10).

Task 9 only exposes a *training* feature entry (``build_feature_dataset`` /
``process_candidate``), which requires ground-truth events and splits. Inference
needs per-clip embeddings with no labels, so this module composes the existing
label-free Task 9 pieces (sampling -> ``extract_crops_for_candidate`` -> the
encoder) into ordered hand/shelf embeddings per sample point.

This is a thin, deliberately TEMPORARY shim. The proper fix is for Task 9 to
expose a label-free feature core; once it does, this module collapses into a
direct call. The heavy Task 9 imports (cv2 via ``crop_extractor``, torch via the
encoder) are deferred so importing this module — and unit-testing it with mocked
crop extraction / embedding — never decodes video.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from pickup_putdown.layer1.track_a.sampling import (
    compute_sample_times,
    get_contact_time,
    get_wrist_trajectory_for_candidate,
)
from pickup_putdown.layer1.track_a.state_types import Embedding

if TYPE_CHECKING:
    from pickup_putdown.common.schemas import Candidate, PoseObservation
    from pickup_putdown.config import TrackAFeaturesConfig
    from pickup_putdown.layer1.track_a.contracts import CropRecord
    from pickup_putdown.layer1.track_a.image_features import AbstractImageEmbedder
    from pickup_putdown.perception.shelf_regions import Polygon

#: Embeds a single crop record into a frozen embedding vector.
CropEmbedFn = Callable[["CropRecord"], Embedding]

#: Extracts hand+shelf crops for a candidate (Task 9 ``extract_crops_for_candidate``).
CropExtractFn = Callable[..., "list[CropRecord]"]

#: Loads a frame as a BGR array at a timestamp (Task 9 ``load_frame_at_timestamp``).
FrameLoader = Callable[["Path", float], "np.ndarray | None"]


@dataclass(frozen=True)
class SampleFeatures:
    """Paired hand/shelf embeddings at one sampled timestamp (label-free)."""

    timestamp_s: float
    sample_position: str
    hand_embedding: Embedding
    shelf_embedding: Embedding


def pair_and_embed(
    crops: Sequence[CropRecord], crop_embed_fn: CropEmbedFn
) -> list[SampleFeatures]:
    """Pair the hand and shelf crop at each sample point and embed both.

    Sample points missing either crop are dropped. Output is time-ordered.
    """
    by_key: dict[tuple[float, str], dict[str, CropRecord]] = {}
    for crop in crops:
        by_key.setdefault((crop.timestamp_s, crop.sample_position), {})[crop.crop_type] = crop

    features: list[SampleFeatures] = []
    for timestamp_s, sample_position in sorted(by_key):
        pair = by_key[timestamp_s, sample_position]
        hand, shelf = pair.get("hand"), pair.get("shelf")
        if hand is None or shelf is None:
            continue
        features.append(
            SampleFeatures(
                timestamp_s=timestamp_s,
                sample_position=sample_position,
                hand_embedding=crop_embed_fn(hand),
                shelf_embedding=crop_embed_fn(shelf),
            )
        )
    return features


def extract_crops_for_inference(
    video_path: Path,
    candidate: Candidate,
    pose_observations: list[PoseObservation],
    shelf_region: Polygon,
    features_config: TrackAFeaturesConfig,
    *,
    crop_extract_fn: CropExtractFn | None = None,
) -> list[CropRecord]:
    """Label-free crop extraction for one candidate (sampling -> Task 9 crops)."""
    trajectory = get_wrist_trajectory_for_candidate(candidate, pose_observations)
    if not trajectory:
        return []
    contact_t = get_contact_time(candidate, trajectory, shelf_region)
    samples = compute_sample_times(
        candidate.raw_start_s, candidate.raw_end_s, contact_t, features_config
    )

    extractor = crop_extract_fn
    if extractor is None:  # lazy: importing crop_extractor pulls cv2
        from pickup_putdown.layer1.track_a.crop_extractor import extract_crops_for_candidate

        extractor = extract_crops_for_candidate
    return extractor(
        video_path, candidate, samples, pose_observations, shelf_region, features_config
    )


def make_crop_embedder(
    video_path: Path,
    embedder: AbstractImageEmbedder,
    *,
    frame_loader: FrameLoader | None = None,
) -> CropEmbedFn:
    """Build a crop->embedding function: reload the frame, crop, encode.

    TEMPORARY: this re-decodes/re-crops because Task 9 discards crop pixels. A
    label-free Task 9 core would remove the reload entirely.
    """
    loader = frame_loader
    if loader is None:  # lazy: importing crop_extractor pulls cv2
        from pickup_putdown.layer1.track_a.crop_extractor import load_frame_at_timestamp

        loader = load_frame_at_timestamp

    def _embed(crop: CropRecord) -> Embedding:
        frame = loader(video_path, crop.timestamp_s)
        if frame is None:
            raise ValueError(f"no frame at {crop.timestamp_s}s for crop {crop.crop_id}")
        geom = crop.geometry
        patch = frame[geom.y : geom.y + geom.height, geom.x : geom.x + geom.width]
        return np.asarray(embedder.embed(patch), dtype=np.float32)

    return _embed


def extract_inference_features(
    video_path: Path,
    candidate: Candidate,
    pose_observations: list[PoseObservation],
    shelf_region: Polygon,
    features_config: TrackAFeaturesConfig,
    crop_embed_fn: CropEmbedFn,
    *,
    crop_extract_fn: CropExtractFn | None = None,
) -> list[SampleFeatures]:
    """Full label-free path for one candidate: crops -> paired embeddings."""
    crops = extract_crops_for_inference(
        video_path,
        candidate,
        pose_observations,
        shelf_region,
        features_config,
        crop_extract_fn=crop_extract_fn,
    )
    return pair_and_embed(crops, crop_embed_fn)

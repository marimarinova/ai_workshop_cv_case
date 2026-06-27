"""Build the Track A feature dataset from candidates and ground truth.

This module orchestrates the feature extraction pipeline:
1. For each candidate, compute sample timestamps
2. Extract hand and shelf crops at each timestamp
3. Compute embeddings using frozen encoder
4. Assign labels based on ground truth events
5. Cache results for efficiency
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickup_putdown.common.schemas import Candidate, Event, PoseObservation
    from pickup_putdown.config import TrackAFeaturesConfig
    from pickup_putdown.perception.shelf_regions import Polygon

from pickup_putdown.layer1.track_a.cache import (
    compute_crop_cache_key,
    compute_embedding_cache_key,
    get_video_checksum,
    is_embedding_cached,
    load_embedding,
    save_crop,
    save_embedding,
)
from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureDataset,
    FeatureRecord,
)
from pickup_putdown.layer1.track_a.crop_extractor import (
    extract_hand_crop,
    extract_shelf_patch,
    find_nearest_pose_observation,
    generate_crop_id,
    load_frame_at_timestamp,
)
from pickup_putdown.layer1.track_a.image_features import (
    AbstractImageEmbedder,
    create_embedder,
)
from pickup_putdown.layer1.track_a.sampling import (
    SamplePoint,
    compute_sample_times,
    get_contact_time,
    get_wrist_trajectory_for_candidate,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Label assignment
# ---------------------------------------------------------------------------


def assign_label(
    candidate: Candidate,
    events: list[Event],
    min_overlap_ratio: float = 0.5,
) -> tuple[str, str | None, str | None]:
    """Assign a label to a candidate based on ground truth events.

    Args:
        candidate: The candidate interval.
        events: List of ground truth events.
        min_overlap_ratio: Minimum overlap ratio to consider a match.

    Returns:
        Tuple of (label, event_id, confidence):
        - label: "pickup", "putdown", or "negative"
        - event_id: ID of matched event (None if negative)
        - confidence: Confidence of matched event (None if negative)
    """
    candidate_start = candidate.raw_start_s
    candidate_end = candidate.raw_end_s
    candidate_duration = candidate_end - candidate_start

    best_match = None
    best_overlap = 0.0

    for event in events:
        # Check if same clip
        if event.clip_id != candidate.clip_id:
            continue

        # Compute overlap
        overlap_start = max(candidate_start, event.t_start)
        overlap_end = min(candidate_end, event.t_end)
        overlap = max(0.0, overlap_end - overlap_start)

        # Compute overlap ratio (relative to candidate)
        overlap_ratio = overlap / candidate_duration if candidate_duration > 0 else 0.0

        if overlap_ratio >= min_overlap_ratio and overlap > best_overlap:
            best_overlap = overlap
            best_match = event

    if best_match is not None:
        label = str(best_match.type)  # "pickup" or "putdown"
        event_id = best_match.event_id
        confidence = str(best_match.confidence) if hasattr(best_match, "confidence") else None
        return label, event_id, confidence

    return "negative", None, None


def overlaps_ignore_interval(
    candidate: Candidate,
    ignore_intervals: list,
) -> bool:
    """Check if a candidate overlaps any ignore interval.

    Args:
        candidate: The candidate interval.
        ignore_intervals: List of ignore intervals with clip_id, t_start, t_end.

    Returns:
        True if overlaps an ignore interval.
    """
    for ignore in ignore_intervals:
        if ignore.clip_id != candidate.clip_id:
            continue

        # Check overlap
        overlap_start = max(candidate.raw_start_s, ignore.t_start)
        overlap_end = min(candidate.raw_end_s, ignore.t_end)

        if overlap_end > overlap_start:
            return True

    return False


# ---------------------------------------------------------------------------
# Single candidate processing
# ---------------------------------------------------------------------------


def process_candidate(
    candidate: Candidate,
    video_path: Path,
    pose_observations: list[PoseObservation],
    shelf_region: Polygon,
    events: list[Event],
    split: str,
    embedder: AbstractImageEmbedder,
    config: TrackAFeaturesConfig,
    video_checksum: str | None = None,
    label_override: str | None = None,
) -> list[FeatureRecord]:
    """Process a single candidate to extract feature records.

    Args:
        candidate: The candidate interval.
        video_path: Path to the source video.
        pose_observations: All pose observations for this clip.
        shelf_region: Shelf region polygon.
        events: Ground truth events for label assignment.
        split: Dataset split ("train", "val", "test").
        embedder: Image embedder instance.
        config: Track A configuration.
        video_checksum: Pre-computed video checksum (computed if None).
        label_override: When set, bypasses assign_label() and uses this label
            directly. Used for reviewed datasets with known labels.

    Returns:
        List of FeatureRecords for this candidate.
    """
    video_path = Path(video_path)
    cache_dir = Path(config.cache_dir)

    # Get video checksum for caching
    if video_checksum is None:
        video_checksum = get_video_checksum(video_path)

    # Filter pose observations for this candidate
    wrist_trajectory = get_wrist_trajectory_for_candidate(candidate, pose_observations)

    if not wrist_trajectory:
        logger.warning(f"No pose observations for candidate {candidate.candidate_id}, skipping")
        return []

    # Get contact time and compute sample timestamps
    contact_time = get_contact_time(candidate, wrist_trajectory, shelf_region)
    sample_points = compute_sample_times(
        candidate.raw_start_s,
        candidate.raw_end_s,
        contact_time,
        config,
    )

    # Assign label based on ground truth (or use override for reviewed data)
    if label_override is not None:
        label = label_override
        event_id: str | None = None
        confidence: str | None = None
    else:
        label, event_id, confidence = assign_label(candidate, events)

    # Process each sample point
    feature_records: list[FeatureRecord] = []

    for sample in sample_points:
        # Find nearest pose observation
        nearest_pose = find_nearest_pose_observation(sample.timestamp_s, wrist_trajectory)

        if nearest_pose is None:
            logger.debug(
                f"No pose near {sample.timestamp_s}s for candidate "
                f"{candidate.candidate_id}, skipping sample"
            )
            continue

        # Process hand crop
        hand_record = _process_crop(
            video_path=video_path,
            candidate=candidate,
            sample=sample,
            crop_type="hand",
            wrist_x=nearest_pose.wrist_x,
            wrist_y=nearest_pose.wrist_y,
            shelf_region=shelf_region,
            label=label,
            split=split,
            event_id=event_id,
            confidence=confidence,
            embedder=embedder,
            config=config,
            video_checksum=video_checksum,
            cache_dir=cache_dir,
        )

        if hand_record is not None:
            feature_records.append(hand_record)

        # Process shelf crop
        shelf_record = _process_crop(
            video_path=video_path,
            candidate=candidate,
            sample=sample,
            crop_type="shelf",
            wrist_x=nearest_pose.wrist_x,
            wrist_y=nearest_pose.wrist_y,
            shelf_region=shelf_region,
            label=label,
            split=split,
            event_id=event_id,
            confidence=confidence,
            embedder=embedder,
            config=config,
            video_checksum=video_checksum,
            cache_dir=cache_dir,
        )

        if shelf_record is not None:
            feature_records.append(shelf_record)

    return feature_records


def _process_crop(
    video_path: Path,
    candidate: Candidate,
    sample: SamplePoint,
    crop_type: str,
    wrist_x: float,
    wrist_y: float,
    shelf_region: Polygon,
    label: str,
    split: str,
    event_id: str | None,
    confidence: str | None,
    embedder: AbstractImageEmbedder,
    config: TrackAFeaturesConfig,
    video_checksum: str,
    cache_dir: Path,
) -> FeatureRecord | None:
    """Process a single crop (hand or shelf) and return a FeatureRecord."""

    # Determine crop size
    crop_size = config.hand_crop_size if crop_type == "hand" else config.shelf_patch_size

    # Create a placeholder geometry for cache key
    # (actual geometry determined after extraction)
    placeholder_geom = CropGeometry(
        x=max(0, int(wrist_x) - crop_size // 2),
        y=max(0, int(wrist_y) - crop_size // 2),
        width=crop_size,
        height=crop_size,
    )

    # Compute cache keys
    crop_cache_key = compute_crop_cache_key(
        video_checksum,
        sample.timestamp_s,
        placeholder_geom,
        crop_type,
    )

    embedding_cache_key = compute_embedding_cache_key(
        crop_cache_key,
        embedder.model_name,
        embedder.version,
    )

    # Check if embedding is cached
    if is_embedding_cached(cache_dir, embedding_cache_key):
        embedding = load_embedding(cache_dir, embedding_cache_key)
        if embedding is not None:
            # Generate crop ID
            crop_id = generate_crop_id(
                candidate.clip_id,
                candidate.candidate_id,
                sample.timestamp_s,
                crop_type,
            )

            from pickup_putdown.layer1.track_a.cache import get_embedding_cache_path

            embedding_path = get_embedding_cache_path(cache_dir, embedding_cache_key)

            return FeatureRecord(
                crop_id=crop_id,
                clip_id=candidate.clip_id,
                candidate_id=candidate.candidate_id,
                timestamp_s=sample.timestamp_s,
                sample_position=sample.position,
                crop_type=crop_type,
                geometry=placeholder_geom,
                embedding_path=embedding_path,
                encoder_name=embedder.model_name,
                encoder_version=embedder.version,
                label=label,
                split=split,
                actor_id=candidate.actor_id,
                hand_side=candidate.hand_side,
                region_id=candidate.region_id,
                confidence=confidence,
                event_id=event_id,
            )

    # Load frame
    frame = load_frame_at_timestamp(video_path, sample.timestamp_s)
    if frame is None:
        logger.warning(
            f"Failed to load frame at {sample.timestamp_s}s for candidate {candidate.candidate_id}"
        )
        return None

    # Extract crop
    if crop_type == "hand":
        crop, geometry = extract_hand_crop(frame, wrist_x, wrist_y, crop_size)
    else:
        contact_point = (wrist_x, wrist_y)
        crop, geometry = extract_shelf_patch(frame, shelf_region, contact_point, crop_size)

    # Update cache key with actual geometry
    crop_cache_key = compute_crop_cache_key(
        video_checksum,
        sample.timestamp_s,
        geometry,
        crop_type,
    )

    embedding_cache_key = compute_embedding_cache_key(
        crop_cache_key,
        embedder.model_name,
        embedder.version,
    )

    # Save crop if configured
    if config.save_crops:
        save_crop(crop, cache_dir, crop_cache_key)

    # Compute embedding
    embedding = embedder.embed(crop)

    # Save embedding
    embedding_path = save_embedding(embedding, cache_dir, embedding_cache_key)

    # Generate crop ID
    crop_id = generate_crop_id(
        candidate.clip_id,
        candidate.candidate_id,
        sample.timestamp_s,
        crop_type,
    )

    return FeatureRecord(
        crop_id=crop_id,
        clip_id=candidate.clip_id,
        candidate_id=candidate.candidate_id,
        timestamp_s=sample.timestamp_s,
        sample_position=sample.position,
        crop_type=crop_type,
        geometry=geometry,
        embedding_path=embedding_path,
        encoder_name=embedder.model_name,
        encoder_version=embedder.version,
        label=label,
        split=split,
        actor_id=candidate.actor_id,
        hand_side=candidate.hand_side,
        region_id=candidate.region_id,
        confidence=confidence,
        event_id=event_id,
    )


# ---------------------------------------------------------------------------
# Main dataset builder
# ---------------------------------------------------------------------------


def build_feature_dataset(
    candidates: list[Candidate],
    events: list[Event],
    pose_observations: list[PoseObservation],
    shelf_regions: dict[str, Polygon],
    splits: dict[str, str],  # clip_id -> split
    video_paths: dict[str, Path],  # clip_id -> video_path
    config: TrackAFeaturesConfig,
    ignore_intervals: list | None = None,
    embedder: AbstractImageEmbedder | None = None,
    label_overrides: dict[str, str] | None = None,
) -> FeatureDataset:
    """Build the complete feature dataset from candidates.

    Args:
        candidates: List of candidates from Task 5.
        events: List of ground truth events.
        pose_observations: All pose observations.
        shelf_regions: Map of region_id -> polygon.
        splits: Map of clip_id -> split ("train", "val", "test").
        video_paths: Map of clip_id -> video file path.
        config: Track A configuration.
        ignore_intervals: Optional list of ignore intervals.
        embedder: Optional pre-created embedder (created from config if None).
        label_overrides: Optional map of candidate_id -> label. When set, bypasses
            assign_label() for matching candidates. Used for reviewed datasets.

    Returns:
        FeatureDataset containing all extracted feature records.
    """
    if ignore_intervals is None:
        ignore_intervals = []
    if label_overrides is None:
        label_overrides = {}

    # Create embedder if not provided
    if embedder is None:
        embedder = create_embedder(config)

    # Cache video checksums
    video_checksums: dict[str, str] = {}

    # Process all candidates
    all_records: list[FeatureRecord] = []
    skipped_ignore = 0
    skipped_no_video = 0
    skipped_no_split = 0
    skipped_no_region = 0
    skipped_no_pose = 0

    for i, candidate in enumerate(candidates):
        # Skip if overlaps ignore interval
        if overlaps_ignore_interval(candidate, ignore_intervals):
            skipped_ignore += 1
            continue

        # Get video path
        video_path = video_paths.get(candidate.clip_id)
        if video_path is None:
            logger.debug(f"No video path for clip {candidate.clip_id}")
            skipped_no_video += 1
            continue

        # Get split
        split = splits.get(candidate.clip_id)
        if split is None:
            logger.debug(f"No split assigned for clip {candidate.clip_id}")
            skipped_no_split += 1
            continue

        # Get shelf region
        region_id = candidate.region_id
        shelf_region = shelf_regions.get(region_id) if region_id else None
        if shelf_region is None:
            # Use first available region as fallback
            if shelf_regions:
                shelf_region = next(iter(shelf_regions.values()))
            else:
                logger.warning(f"No shelf region for candidate {candidate.candidate_id}")
                skipped_no_region += 1
                continue

        # Get or compute video checksum
        video_checksum = video_checksums.get(candidate.clip_id)
        if video_checksum is None:
            video_checksum = get_video_checksum(video_path)
            video_checksums[candidate.clip_id] = video_checksum

        # Filter pose observations for this clip
        clip_poses = [p for p in pose_observations if p.clip_id == candidate.clip_id]

        # Process candidate
        records = process_candidate(
            candidate=candidate,
            video_path=video_path,
            pose_observations=clip_poses,
            shelf_region=shelf_region,
            events=events,
            split=split,
            embedder=embedder,
            config=config,
            video_checksum=video_checksum,
            label_override=label_overrides.get(candidate.candidate_id),
        )

        if not records:
            skipped_no_pose += 1

        all_records.extend(records)

        # Progress logging
        if (i + 1) % 100 == 0:
            logger.info(f"Processed {i + 1}/{len(candidates)} candidates")

    # Create dataset
    dataset = FeatureDataset(
        records=all_records,
        encoder_name=embedder.model_name,
        encoder_version=embedder.version,
    )
    dataset.compute_stats()

    # Log summary
    logger.info(
        f"Built feature dataset: {len(all_records)} records from {len(candidates)} candidates"
    )
    logger.info(
        f"  Labels: {dataset.n_pickup} pickup, {dataset.n_putdown} putdown, "
        f"{dataset.n_negative} negative"
    )
    logger.info(f"  Splits: {dataset.n_train} train, {dataset.n_val} val, {dataset.n_test} test")
    logger.info(
        f"  Skipped: {skipped_ignore} ignore, {skipped_no_video} no_video, "
        f"{skipped_no_split} no_split, {skipped_no_region} no_region, "
        f"{skipped_no_pose} no_pose"
    )

    return dataset


def validate_split_isolation(dataset: FeatureDataset) -> bool:
    """Validate that no clip appears in multiple splits.

    Args:
        dataset: The feature dataset to validate.

    Returns:
        True if valid, False if there's split leakage.
    """
    clips_by_split: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}

    for record in dataset.records:
        clips_by_split[record.split].add(record.clip_id)

    # Check for overlap
    train_clips = clips_by_split["train"]
    val_clips = clips_by_split["val"]
    test_clips = clips_by_split["test"]

    train_val_overlap = train_clips & val_clips
    train_test_overlap = train_clips & test_clips
    val_test_overlap = val_clips & test_clips

    if train_val_overlap or train_test_overlap or val_test_overlap:
        logger.error(
            f"Split leakage detected! "
            f"train∩val: {len(train_val_overlap)}, "
            f"train∩test: {len(train_test_overlap)}, "
            f"val∩test: {len(val_test_overlap)}"
        )
        return False

    logger.info("Split isolation validated: no clip appears in multiple splits")
    return True

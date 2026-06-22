"""Actor association, region measurements, raw interactions, and candidate generation.

This module ties together:
- pose observations from :mod:`pose_tracker`
- actor tracks from Layer 0A (person observations)
- shelf/surface regions from :mod:`shelf_regions`

It produces:
- actor-assigned pose observations
- raw interactions (wrist inside expanded region for min duration)
- merged candidate intervals
- a proposal-recall API for measuring ground-truth event coverage
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections import defaultdict
from dataclasses import dataclass

from pickup_putdown.common.exceptions import ConfigError, ValidationError
from pickup_putdown.common.schemas import (
    Candidate,
    PersonObservation,
    PoseObservation,
    ProposalRecallResult,
)
from pickup_putdown.config import (
    ActorAssociationConfig,
    ProposalsConfig,
    RegionMeasurementConfig,
)
from pickup_putdown.perception.shelf_regions import (
    CameraShelfConfig,
    Polygon,
    get_expanded_regions,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class RawInteraction:
    """One contiguous confident wrist interaction with one configured region."""

    clip_id: str
    actor_id: str
    hand_side: str
    region_id: str
    start_s: float
    end_s: float
    n_observations: int = 0
    mean_wrist_confidence: float = 0.0
    mean_distance: float = 0.0


@dataclass
class _RegionMeasurement:
    """Measurements for one wrist observation against one region."""

    wrist_x: float
    wrist_y: float
    wrist_confidence: float
    distance: float
    inside_region: bool
    inside_expanded: bool
    entry_event: bool = False
    exit_event: bool = False
    dwell_duration_s: float = 0.0
    speed: float = 0.0
    velocity_reversal: bool = False


@dataclass(frozen=True)
class _PoseDetectionKey:
    """Identity shared by the left/right wrists from one pose detection."""

    clip_id: str
    timestamp_us: int
    source_frame_index: int
    sample_index: int
    bbox_x1_milli: int
    bbox_y1_milli: int
    bbox_x2_milli: int
    bbox_y2_milli: int


# ---------------------------------------------------------------------------
# Actor association
# ---------------------------------------------------------------------------


def associate_poses_with_actors(
    pose_observations: list[PoseObservation],
    person_observations: list[PersonObservation],
    actor_cfg: ActorAssociationConfig,
) -> list[PoseObservation]:
    """Associate pose detections with Layer 0A actor tracks.

    Matching is timestamp-aware and one-to-one per source frame. Left and right
    wrist rows from the same pose detection receive the same actor assignment.
    Unmatched pose detections retain their original ``actor_id``.
    """
    if not pose_observations or not person_observations:
        return pose_observations

    people_by_clip: dict[str, list[PersonObservation]] = defaultdict(list)
    for person in person_observations:
        actor_id = _person_actor_id(person)
        if actor_id is None:
            continue
        people_by_clip[person.clip_id].append(person)

    for observations in people_by_clip.values():
        observations.sort(key=lambda item: item.timestamp_s)

    detections_by_clip: dict[str, dict[_PoseDetectionKey, list[PoseObservation]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for pose in pose_observations:
        detections_by_clip[pose.clip_id][_pose_detection_key(pose)].append(pose)

    for clip_id, detections in detections_by_clip.items():
        clip_people = people_by_clip.get(clip_id, [])
        if not clip_people:
            continue

        detections_by_frame: dict[tuple[int, int], list[_PoseDetectionKey]] = defaultdict(list)
        for detection_key in detections:
            frame_key = (detection_key.source_frame_index, detection_key.timestamp_us)
            detections_by_frame[frame_key].append(detection_key)

        for frame_keys in detections_by_frame.values():
            candidate_pairs: list[
                tuple[float, float, str, _PoseDetectionKey, PersonObservation]
            ] = []

            for detection_key in frame_keys:
                representative = detections[detection_key][0]
                nearest_by_actor = _nearest_person_observation_by_actor(
                    representative,
                    clip_people,
                    actor_cfg.max_gap_s,
                )

                for actor_id, person in nearest_by_actor.items():
                    iou = _compute_bbox_iou(
                        _finite_or_zero(representative.person_bbox_x1),
                        _finite_or_zero(representative.person_bbox_y1),
                        _finite_or_zero(representative.person_bbox_x2),
                        _finite_or_zero(representative.person_bbox_y2),
                        person.bbox_x1,
                        person.bbox_y1,
                        person.bbox_x2,
                        person.bbox_y2,
                    )
                    if iou < actor_cfg.match_iou_threshold:
                        continue

                    time_gap = abs(person.timestamp_s - representative.timestamp_s)
                    candidate_pairs.append((-iou, time_gap, actor_id, detection_key, person))

            candidate_pairs.sort(
                key=lambda item: (
                    item[0],
                    item[1],
                    item[2],
                    item[3].bbox_x1_milli,
                    item[3].bbox_y1_milli,
                )
            )

            assigned_detections: set[_PoseDetectionKey] = set()
            assigned_actors: set[str] = set()

            for neg_iou, _time_gap, actor_id, detection_key, person in candidate_pairs:
                if detection_key in assigned_detections or actor_id in assigned_actors:
                    continue

                association_confidence = -neg_iou
                for pose in detections[detection_key]:
                    pose.actor_id = actor_id
                    pose.pose_association_confidence = association_confidence
                    pose.person_bbox_x1 = person.bbox_x1
                    pose.person_bbox_y1 = person.bbox_y1
                    pose.person_bbox_x2 = person.bbox_x2
                    pose.person_bbox_y2 = person.bbox_y2

                assigned_detections.add(detection_key)
                assigned_actors.add(actor_id)

    return pose_observations


def _nearest_person_observation_by_actor(
    pose: PoseObservation,
    person_observations: list[PersonObservation],
    max_gap_s: float,
) -> dict[str, PersonObservation]:
    """Return the nearest observation for each actor within ``max_gap_s``."""
    nearest: dict[str, tuple[float, PersonObservation]] = {}

    for person in person_observations:
        time_gap = abs(person.timestamp_s - pose.timestamp_s)
        if time_gap > max_gap_s:
            continue

        actor_id = _person_actor_id(person)
        if actor_id is None:
            continue

        previous = nearest.get(actor_id)
        if previous is None or time_gap < previous[0]:
            nearest[actor_id] = (time_gap, person)

    return {actor_id: item[1] for actor_id, item in nearest.items()}


def _find_best_actor_match(
    pose: PoseObservation,
    person_obs: list[PersonObservation],
    actor_cfg: ActorAssociationConfig,
) -> PersonObservation | None:
    """Compatibility helper returning the best single actor match."""
    nearest = _nearest_person_observation_by_actor(
        pose,
        person_obs,
        actor_cfg.max_gap_s,
    )

    best: PersonObservation | None = None
    best_iou = 0.0
    best_gap = float("inf")

    for person in nearest.values():
        iou = _compute_bbox_iou(
            _finite_or_zero(pose.person_bbox_x1),
            _finite_or_zero(pose.person_bbox_y1),
            _finite_or_zero(pose.person_bbox_x2),
            _finite_or_zero(pose.person_bbox_y2),
            person.bbox_x1,
            person.bbox_y1,
            person.bbox_x2,
            person.bbox_y2,
        )
        gap = abs(person.timestamp_s - pose.timestamp_s)
        if iou > best_iou or (math.isclose(iou, best_iou) and gap < best_gap):
            best = person
            best_iou = iou
            best_gap = gap

    if best is not None and best_iou >= actor_cfg.match_iou_threshold:
        return best
    return None


def _person_actor_id(person: PersonObservation) -> str | None:
    """Normalize the public actor identifier from a person observation."""
    actor_id = getattr(person, "person_track_id", None)
    if actor_id is None:
        actor_id = getattr(person, "tracker_track_id", None)
    if actor_id is None:
        return None

    if isinstance(actor_id, float):
        if not math.isfinite(actor_id):
            return None
        if actor_id.is_integer():
            return str(int(actor_id))

    value = str(actor_id).strip()
    return value or None


def _pose_detection_key(pose: PoseObservation) -> _PoseDetectionKey:
    return _PoseDetectionKey(
        clip_id=pose.clip_id,
        timestamp_us=round(float(pose.timestamp_s) * 1_000_000),
        source_frame_index=_optional_int(getattr(pose, "source_frame_index", None)),
        sample_index=_optional_int(getattr(pose, "sample_index", None)),
        bbox_x1_milli=round(_finite_or_zero(pose.person_bbox_x1) * 1_000),
        bbox_y1_milli=round(_finite_or_zero(pose.person_bbox_y1) * 1_000),
        bbox_x2_milli=round(_finite_or_zero(pose.person_bbox_x2) * 1_000),
        bbox_y2_milli=round(_finite_or_zero(pose.person_bbox_y2) * 1_000),
    )


def _optional_int(value: object) -> int:
    if value is None:
        return -1
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return -1


def _finite_or_zero(value: float | None) -> float:
    if value is None:
        return 0.0
    result = float(value)
    return result if math.isfinite(result) else 0.0


def _compute_bbox_iou(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    x3: float,
    y3: float,
    x4: float,
    y4: float,
) -> float:
    """Compute IoU between two axis-aligned bounding boxes."""
    ix1 = max(x1, x3)
    iy1 = max(y1, y3)
    ix2 = min(x2, x4)
    iy2 = min(y2, y4)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0

    area1 = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area2 = max(0.0, x4 - x3) * max(0.0, y4 - y3)
    union = area1 + area2 - inter_area
    return inter_area / union if union > 0.0 else 0.0


# ---------------------------------------------------------------------------
# Region-based measurements
# ---------------------------------------------------------------------------


def compute_region_measurements(
    pose_observations: list[PoseObservation],
    camera_config: CameraShelfConfig,
    region_cfg: RegionMeasurementConfig,
) -> dict[str, list[_RegionMeasurement]]:
    """Compute measurements for every actor, hand, and configured region.

    The complete trajectory is retained for each region. Observations outside
    the expanded polygon are not discarded, which preserves approach, entry,
    exit, speed, dwell, and reversal evidence.
    """
    original_regions = {region.region_id: region.points for region in camera_config.regions}
    expanded_regions = _get_expanded_regions(camera_config, region_cfg)

    groups: dict[tuple[str, str, str], list[PoseObservation]] = defaultdict(list)
    for pose in pose_observations:
        actor_id = _pose_actor_id(pose)
        if actor_id is None:
            continue
        groups[(pose.clip_id, actor_id, pose.hand_side)].append(pose)

    result: dict[str, list[_RegionMeasurement]] = {}
    reversal_window = max(1, int(getattr(region_cfg, "velocity_window_frames", 1)))

    for (clip_id, actor_id, hand_side), poses in groups.items():
        poses.sort(key=lambda item: (item.timestamp_s, item.source_frame_index))

        for region_id, original_polygon in original_regions.items():
            expanded_polygon = expanded_regions[region_id]
            measurements: list[_RegionMeasurement] = []
            entry_time: float | None = None
            previous_inside_expanded: bool | None = None

            for index, pose in enumerate(poses):
                inside_original = _point_in_polygon(
                    pose.wrist_x,
                    pose.wrist_y,
                    original_polygon,
                )
                inside_expanded = _point_in_polygon(
                    pose.wrist_x,
                    pose.wrist_y,
                    expanded_polygon,
                )
                distance = _point_to_polygon_distance(
                    pose.wrist_x,
                    pose.wrist_y,
                    original_polygon,
                )

                entry_event = inside_expanded and previous_inside_expanded is not True
                exit_event = not inside_expanded and previous_inside_expanded is True

                if entry_event:
                    entry_time = pose.timestamp_s
                if exit_event:
                    entry_time = None

                dwell_duration = (
                    max(0.0, pose.timestamp_s - entry_time)
                    if inside_expanded and entry_time is not None
                    else 0.0
                )

                speed = 0.0
                if index > 0:
                    previous = poses[index - 1]
                    dt = pose.timestamp_s - previous.timestamp_s
                    if dt > 0.0:
                        speed = (
                            math.hypot(
                                pose.wrist_x - previous.wrist_x,
                                pose.wrist_y - previous.wrist_y,
                            )
                            / dt
                        )

                reversal = _has_velocity_reversal(
                    poses,
                    index,
                    reversal_window,
                    float(region_cfg.reversal_threshold),
                )

                measurements.append(
                    _RegionMeasurement(
                        wrist_x=pose.wrist_x,
                        wrist_y=pose.wrist_y,
                        wrist_confidence=pose.wrist_confidence,
                        distance=distance,
                        inside_region=inside_original,
                        inside_expanded=inside_expanded,
                        entry_event=entry_event,
                        exit_event=exit_event,
                        dwell_duration_s=dwell_duration,
                        speed=speed,
                        velocity_reversal=reversal,
                    )
                )
                previous_inside_expanded = inside_expanded

            result[f"{clip_id}:{actor_id}:{hand_side}:{region_id}"] = measurements

    return result


def _has_velocity_reversal(
    poses: list[PoseObservation],
    index: int,
    window: int,
    reversal_threshold: float,
) -> bool:
    if index < 2 * window:
        return False

    first = poses[index - 2 * window]
    middle = poses[index - window]
    current = poses[index]

    previous_dx = middle.wrist_x - first.wrist_x
    previous_dy = middle.wrist_y - first.wrist_y
    current_dx = current.wrist_x - middle.wrist_x
    current_dy = current.wrist_y - middle.wrist_y

    previous_magnitude = math.hypot(previous_dx, previous_dy)
    current_magnitude = math.hypot(current_dx, current_dy)
    if previous_magnitude <= 1e-9 or current_magnitude <= 1e-9:
        return False

    cosine = (previous_dx * current_dx + previous_dy * current_dy) / (
        previous_magnitude * current_magnitude
    )
    cosine = max(-1.0, min(1.0, cosine))
    return cosine < -reversal_threshold


def _get_expanded_regions(
    camera_config: CameraShelfConfig,
    region_cfg: RegionMeasurementConfig,
) -> dict[str, Polygon]:
    override = getattr(region_cfg, "expanded_margin_override", None)
    if override is None:
        return get_expanded_regions(camera_config)

    copied_config = camera_config.model_copy(deep=True)
    copied_config.expansion.value = float(override)
    return get_expanded_regions(copied_config)


# ---------------------------------------------------------------------------
# Raw interaction detection
# ---------------------------------------------------------------------------


def detect_raw_interactions(
    pose_observations: list[PoseObservation],
    camera_config: CameraShelfConfig,
    proposals_cfg: ProposalsConfig,
    region_cfg: RegionMeasurementConfig,
) -> list[RawInteraction]:
    """Detect contiguous confident wrist interactions with expanded regions.

    Every trajectory is evaluated independently for each region. This supports
    overlapping expanded polygons and preserves outside observations needed to
    close interactions correctly.
    """
    if not pose_observations:
        return []

    original_regions = {region.region_id: region.points for region in camera_config.regions}
    expanded_regions = _get_expanded_regions(camera_config, region_cfg)

    grouped_poses: dict[tuple[str, str, str], list[PoseObservation]] = defaultdict(list)
    for pose in pose_observations:
        actor_id = _pose_actor_id(pose)
        if actor_id is None:
            continue
        grouped_poses[(pose.clip_id, actor_id, pose.hand_side)].append(pose)

    minimum_confidence = float(proposals_cfg.minimum_wrist_confidence)
    minimum_duration = float(proposals_cfg.minimum_interaction_duration_s)
    configured_gap_tolerance = max(
        0.0,
        float(getattr(region_cfg, "gap_tolerance_s", 0.0)),
    )
    sampling_gap_tolerance = 1.5 / max(
        float(proposals_cfg.target_fps),
        1e-9,
    )

    interactions: list[RawInteraction] = []

    for (clip_id, actor_id, hand_side), poses in grouped_poses.items():
        poses.sort(key=lambda item: (item.timestamp_s, item.source_frame_index))

        # Tests, imported fixtures, and lower-rate acceptance runs may have a
        # cadence different from proposals_cfg.target_fps. Infer the normal
        # cadence from the trajectory while capping the tolerated gap so a
        # genuinely long observation outage still terminates an interaction.
        positive_deltas = sorted(
            current.timestamp_s - previous.timestamp_s
            for previous, current in zip(poses, poses[1:], strict=False)
            if current.timestamp_s > previous.timestamp_s
        )
        if positive_deltas:
            median_delta = positive_deltas[len(positive_deltas) // 2]
            observed_gap_tolerance = min(2.0, 1.5 * median_delta)
        else:
            observed_gap_tolerance = 0.0

        maximum_observation_gap = max(
            configured_gap_tolerance,
            sampling_gap_tolerance,
            observed_gap_tolerance,
        )

        for region_id, expanded_polygon in expanded_regions.items():
            original_polygon = original_regions[region_id]

            start_s: float | None = None
            last_qualifying_s: float | None = None
            previous_timestamp: float | None = None
            confidence_sum = 0.0
            distance_sum = 0.0
            observation_count = 0

            def flush(
                current_clip_id: str = clip_id,
                current_actor_id: str = actor_id,
                current_hand_side: str = hand_side,
                current_region_id: str = region_id,
            ) -> None:
                nonlocal start_s
                nonlocal last_qualifying_s
                nonlocal confidence_sum
                nonlocal distance_sum
                nonlocal observation_count

                if start_s is not None and last_qualifying_s is not None:
                    duration = last_qualifying_s - start_s
                    if duration >= minimum_duration:
                        interactions.append(
                            RawInteraction(
                                clip_id=current_clip_id,
                                actor_id=current_actor_id,
                                hand_side=current_hand_side,
                                region_id=current_region_id,
                                start_s=start_s,
                                end_s=last_qualifying_s,
                                n_observations=observation_count,
                                mean_wrist_confidence=(confidence_sum / observation_count),
                                mean_distance=distance_sum / observation_count,
                            )
                        )

                start_s = None
                last_qualifying_s = None
                confidence_sum = 0.0
                distance_sum = 0.0
                observation_count = 0

            # Multiple pose rows may share an actor, hand, and timestamp
            # (for example, synthetic fixtures or duplicate detections). For
            # each region, collapse such rows into one timestamp-level state.
            # A non-qualifying row at the same timestamp must not terminate a
            # qualifying interaction from another row.
            poses_by_timestamp: dict[float, list[PoseObservation]] = defaultdict(list)
            for pose in poses:
                poses_by_timestamp[float(pose.timestamp_s)].append(pose)

            for timestamp in sorted(poses_by_timestamp):
                timestamp_poses = poses_by_timestamp[timestamp]

                if (
                    start_s is not None
                    and previous_timestamp is not None
                    and timestamp - previous_timestamp > maximum_observation_gap
                ):
                    flush()

                qualifying_poses = [
                    pose
                    for pose in timestamp_poses
                    if getattr(pose, "is_valid", True) is not False
                    and pose.wrist_confidence >= minimum_confidence
                    and _point_in_polygon(
                        pose.wrist_x,
                        pose.wrist_y,
                        expanded_polygon,
                    )
                ]

                if qualifying_poses:
                    # Prefer the most confident observation. Distance is a
                    # deterministic tie-breaker that favors the region-local
                    # wrist when duplicate detections exist.
                    pose = max(
                        qualifying_poses,
                        key=lambda item: (
                            item.wrist_confidence,
                            -_point_to_polygon_distance(
                                item.wrist_x,
                                item.wrist_y,
                                original_polygon,
                            ),
                            -(getattr(item, "source_frame_index", 0) or 0),
                        ),
                    )

                    if start_s is None:
                        start_s = timestamp
                    last_qualifying_s = timestamp
                    observation_count += 1
                    confidence_sum += pose.wrist_confidence
                    distance_sum += _point_to_polygon_distance(
                        pose.wrist_x,
                        pose.wrist_y,
                        original_polygon,
                    )
                elif (
                    start_s is not None
                    and last_qualifying_s is not None
                    and timestamp - last_qualifying_s > configured_gap_tolerance
                ):
                    flush()

                previous_timestamp = timestamp

            flush()

    interactions.sort(
        key=lambda item: (
            item.clip_id,
            item.actor_id,
            item.hand_side,
            item.region_id,
            item.start_s,
            item.end_s,
        )
    )
    return interactions


def _pose_actor_id(pose: PoseObservation) -> str | None:
    actor_id = getattr(pose, "actor_id", None)
    if actor_id is None:
        return None
    value = str(actor_id).strip()
    return value or None


# Kept for compatibility with callers that imported this private helper.
def _original_polygon_from_expanded(
    wx: float,
    wy: float,
    expanded_pts: Polygon,
    img_w: int,
    img_h: int,
) -> Polygon:
    """Deprecated compatibility helper.

    The proposal implementation now uses the configured original polygon
    directly. This function returns a copy of its input only for compatibility.
    """
    del wx, wy, img_w, img_h
    return list(expanded_pts)


# ---------------------------------------------------------------------------
# Candidate generation
# ---------------------------------------------------------------------------


def generate_candidates(
    raw_interactions: list[RawInteraction],
    clip_durations: dict[str, float],
    proposals_cfg: ProposalsConfig,
) -> list[Candidate]:
    """Merge same-actor/hand/region interactions and add bounded context."""
    if not raw_interactions:
        return []

    groups: dict[tuple[str, str, str, str], list[RawInteraction]] = defaultdict(list)
    for interaction in raw_interactions:
        groups[
            (
                interaction.clip_id,
                interaction.actor_id,
                interaction.hand_side,
                interaction.region_id,
            )
        ].append(interaction)

    candidates: list[Candidate] = []

    for group_key, interactions in groups.items():
        interactions.sort(key=lambda item: (item.start_s, item.end_s))

        merged_groups: list[list[RawInteraction]] = []
        current_group = [interactions[0]]

        for interaction in interactions[1:]:
            previous_end = max(item.end_s for item in current_group)
            gap = interaction.start_s - previous_end
            if gap <= proposals_cfg.merge_gap_s:
                current_group.append(interaction)
            else:
                merged_groups.append(current_group)
                current_group = [interaction]
        merged_groups.append(current_group)

        for merged in merged_groups:
            clip_id, actor_id, hand_side, region_id = group_key
            clip_duration = clip_durations.get(clip_id)
            if clip_duration is None or not math.isfinite(float(clip_duration)):
                raise ValidationError(f"Missing finite clip duration for clip_id={clip_id!r}")
            clip_duration = float(clip_duration)
            if clip_duration <= 0.0:
                raise ValidationError(
                    f"Clip duration must be positive for clip_id={clip_id!r}: {clip_duration}"
                )

            raw_start = max(0.0, min(item.start_s for item in merged))
            raw_end = min(clip_duration, max(item.end_s for item in merged))
            if raw_end <= raw_start:
                logger.warning(
                    "Skipping degenerate raw interaction group %s: [%.6f, %.6f]",
                    group_key,
                    raw_start,
                    raw_end,
                )
                continue

            desired_start = max(
                0.0,
                raw_start - proposals_cfg.context_before_s,
            )
            desired_end = min(
                clip_duration,
                raw_end + proposals_cfg.context_after_s,
            )
            window_start, window_end = _bounded_context_window(
                raw_start=raw_start,
                raw_end=raw_end,
                desired_start=desired_start,
                desired_end=desired_end,
                clip_duration=clip_duration,
                maximum_duration=float(proposals_cfg.maximum_candidate_duration_s),
            )

            minimum_distance = min(item.mean_distance for item in merged)
            maximum_confidence = max(item.mean_wrist_confidence for item in merged)
            total_dwell = sum(item.end_s - item.start_s for item in merged)

            raw_start_us = round(raw_start * 1_000_000)
            raw_end_us = round(raw_end * 1_000_000)
            identifier_payload = "\x1f".join(
                [
                    clip_id,
                    actor_id,
                    hand_side,
                    region_id,
                    str(raw_start_us),
                    str(raw_end_us),
                ]
            )
            candidate_id = (
                "cand_" + hashlib.sha256(identifier_payload.encode("utf-8")).hexdigest()[:12]
            )

            config_fingerprint = _config_fingerprint(proposals_cfg)
            candidate = Candidate(
                candidate_id=candidate_id,
                clip_id=clip_id,
                actor_id=actor_id,
                hand_side=hand_side,
                region_id=region_id,
                raw_start_s=round(raw_start, 4),
                raw_end_s=round(raw_end, 4),
                window_start_s=round(window_start, 4),
                window_end_s=round(window_end, 4),
                n_raw_interactions=len(merged),
                min_region_distance=round(minimum_distance, 4),
                max_wrist_confidence=round(maximum_confidence, 4),
                total_dwell_duration_s=round(total_dwell, 4),
                config_fingerprint=config_fingerprint,
                proposal_reason="wrist_in_expanded_region",
                proposal_score=maximum_confidence,
                review_status="pending",
            )
            validate_candidate(candidate, clip_duration)
            candidates.append(candidate)

    candidates.sort(
        key=lambda candidate: (
            candidate.clip_id,
            candidate.actor_id,
            candidate.hand_side or "",
            candidate.region_id or "",
            candidate.raw_start_s,
            candidate.raw_end_s,
            candidate.candidate_id,
        )
    )
    return candidates


def _bounded_context_window(
    *,
    raw_start: float,
    raw_end: float,
    desired_start: float,
    desired_end: float,
    clip_duration: float,
    maximum_duration: float,
) -> tuple[float, float]:
    """Limit context without ever cutting the raw interaction interval."""
    raw_duration = raw_end - raw_start
    if raw_duration >= maximum_duration:
        logger.warning(
            "Raw interaction duration %.3fs exceeds maximum candidate duration "
            "%.3fs; preserving the raw interval without context.",
            raw_duration,
            maximum_duration,
        )
        return raw_start, raw_end

    desired_before = raw_start - desired_start
    desired_after = desired_end - raw_end
    context_budget = maximum_duration - raw_duration

    total_desired_context = desired_before + desired_after
    if total_desired_context <= context_budget:
        return desired_start, desired_end

    if total_desired_context <= 0.0:
        return raw_start, raw_end

    before = min(
        desired_before,
        context_budget * desired_before / total_desired_context,
    )
    after = min(
        desired_after,
        context_budget - before,
    )

    remaining = context_budget - before - after
    if remaining > 0.0:
        additional_before = min(remaining, desired_before - before)
        before += additional_before
        remaining -= additional_before
        after += min(remaining, desired_after - after)

    return (
        max(0.0, raw_start - before),
        min(clip_duration, raw_end + after),
    )


def _config_fingerprint(proposals_cfg: ProposalsConfig) -> str:
    values = (
        ("target_fps", proposals_cfg.target_fps),
        ("minimum_wrist_confidence", proposals_cfg.minimum_wrist_confidence),
        (
            "minimum_interaction_duration_s",
            proposals_cfg.minimum_interaction_duration_s,
        ),
        ("merge_gap_s", proposals_cfg.merge_gap_s),
        ("context_before_s", proposals_cfg.context_before_s),
        ("context_after_s", proposals_cfg.context_after_s),
        (
            "maximum_candidate_duration_s",
            proposals_cfg.maximum_candidate_duration_s,
        ),
    )
    payload = ";".join(f"{key}={value}" for key, value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# Proposal recall API
# ---------------------------------------------------------------------------


def compute_proposal_recall(
    candidates: list[Candidate],
    ground_truth_events: list[dict],
    *,
    actor_aware: bool = False,
    region_aware: bool = False,
) -> tuple[list[ProposalRecallResult], dict]:
    """Measure whether each ground-truth interval overlaps a candidate."""
    results: list[ProposalRecallResult] = []

    for event in ground_truth_events:
        event_id = str(event.get("event_id", "unknown"))
        clip_id = event["clip_id"]
        gt_start = float(event["t_start"])
        gt_end = float(event["t_end"])
        gt_type = event["type"]
        gt_actor = event.get("actor_id")
        gt_region = event.get("region_id")

        covered = False
        coverage_method: str | None = None
        matching_candidate_id: str | None = None
        # Actor matching is evaluated first. Region matching remains None
        # when an actor mismatch short-circuits the candidate, because the
        # region dimension was not evaluated for that candidate.
        matched_actor: bool | None = False if actor_aware else None
        matched_region: bool | None = None

        for candidate in candidates:
            if candidate.clip_id != clip_id:
                continue

            actor_matches = not actor_aware or gt_actor is None or candidate.actor_id == gt_actor
            if actor_aware:
                matched_actor = actor_matches
            if not actor_matches:
                continue

            region_matches = (
                not region_aware or gt_region is None or candidate.region_id == gt_region
            )
            if region_aware:
                matched_region = region_matches
            if not region_matches:
                continue

            if _intervals_overlap(
                gt_start,
                gt_end,
                candidate.raw_start_s,
                candidate.raw_end_s,
            ):
                covered = True
                coverage_method = "raw"
            elif _intervals_overlap(
                gt_start,
                gt_end,
                candidate.window_start_s,
                candidate.window_end_s,
            ):
                covered = True
                coverage_method = "padded"

            if covered:
                matching_candidate_id = candidate.candidate_id
                if actor_aware:
                    matched_actor = True
                if region_aware:
                    matched_region = True
                break

        results.append(
            ProposalRecallResult(
                event_id=event_id,
                clip_id=clip_id,
                gt_type=gt_type,
                gt_t_start=gt_start,
                gt_t_end=gt_end,
                covered=covered,
                coverage_method=coverage_method,
                matching_candidate_id=matching_candidate_id,
                actor_match=matched_actor,
                region_match=matched_region,
            )
        )

    total_events = len(results)
    covered_events = sum(result.covered for result in results)
    recall = covered_events / total_events if total_events else 0.0

    aggregate = {
        "total_events": total_events,
        "covered_events": covered_events,
        "uncovered_events": total_events - covered_events,
        "proposal_recall": round(recall, 4),
        "actor_aware": actor_aware,
        "region_aware": region_aware,
    }
    return results, aggregate


def _intervals_overlap(
    start_a: float,
    end_a: float,
    start_b: float,
    end_b: float,
) -> bool:
    return min(end_a, end_b) >= max(start_a, start_b)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:
    """Return whether a point is inside or on the boundary of a polygon."""
    if len(polygon) < 3:
        return False

    for index in range(len(polygon)):
        next_index = (index + 1) % len(polygon)
        if (
            _point_to_segment_distance(
                px,
                py,
                polygon[index][0],
                polygon[index][1],
                polygon[next_index][0],
                polygon[next_index][1],
            )
            <= 1e-9
        ):
            return True

    inside = False
    previous_index = len(polygon) - 1
    for index, (current_x, current_y) in enumerate(polygon):
        previous_x, previous_y = polygon[previous_index]
        crosses = (current_y > py) != (previous_y > py)
        if crosses:
            x_at_y = (previous_x - current_x) * (py - current_y) / (
                previous_y - current_y
            ) + current_x
            if px < x_at_y:
                inside = not inside
        previous_index = index
    return inside


def _point_to_polygon_distance(px: float, py: float, polygon: Polygon) -> float:
    """Compute minimum distance from a point to polygon edges."""
    if not polygon:
        return float("inf")

    return min(
        _point_to_segment_distance(
            px,
            py,
            polygon[index][0],
            polygon[index][1],
            polygon[(index + 1) % len(polygon)][0],
            polygon[(index + 1) % len(polygon)][1],
        )
        for index in range(len(polygon))
    )


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Minimum distance from a point to a line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_squared = dx * dx + dy * dy
    if length_squared <= 0.0:
        return math.hypot(px - x1, py - y1)

    projection = ((px - x1) * dx + (py - y1) * dy) / length_squared
    projection = max(0.0, min(1.0, projection))
    projected_x = x1 + projection * dx
    projected_y = y1 + projection * dy
    return math.hypot(px - projected_x, py - projected_y)


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_proposals_config(
    proposals_cfg: ProposalsConfig,
    region_cfg: RegionMeasurementConfig,
) -> None:
    """Validate proposal configuration and raise ConfigError if invalid."""
    if proposals_cfg.target_fps <= 0:
        raise ConfigError(f"target_fps must be positive, got {proposals_cfg.target_fps}")
    if proposals_cfg.target_fps > 120:
        raise ConfigError(f"target_fps too high: {proposals_cfg.target_fps}")

    if not (0.0 <= proposals_cfg.minimum_wrist_confidence <= 1.0):
        raise ConfigError(
            "minimum_wrist_confidence must be in [0, 1], got "
            f"{proposals_cfg.minimum_wrist_confidence}"
        )

    if proposals_cfg.minimum_interaction_duration_s < 0:
        raise ConfigError(
            "minimum_interaction_duration_s must be non-negative, got "
            f"{proposals_cfg.minimum_interaction_duration_s}"
        )

    if proposals_cfg.merge_gap_s < 0:
        raise ConfigError(f"merge_gap_s must be non-negative, got {proposals_cfg.merge_gap_s}")
    if proposals_cfg.context_before_s < 0:
        raise ConfigError(
            f"context_before_s must be non-negative, got {proposals_cfg.context_before_s}"
        )
    if proposals_cfg.context_after_s < 0:
        raise ConfigError(
            f"context_after_s must be non-negative, got {proposals_cfg.context_after_s}"
        )
    if proposals_cfg.maximum_candidate_duration_s <= 0:
        raise ConfigError(
            "maximum_candidate_duration_s must be positive, got "
            f"{proposals_cfg.maximum_candidate_duration_s}"
        )

    if region_cfg.velocity_window_frames < 1:
        raise ConfigError(
            f"velocity_window_frames must be >= 1, got {region_cfg.velocity_window_frames}"
        )
    if not (0.0 <= region_cfg.reversal_threshold <= 1.0):
        raise ConfigError(
            f"reversal_threshold must be in [0, 1], got {region_cfg.reversal_threshold}"
        )

    gap_tolerance = float(getattr(region_cfg, "gap_tolerance_s", 0.0))
    if gap_tolerance < 0.0:
        raise ConfigError(f"gap_tolerance_s must be non-negative, got {gap_tolerance}")


def validate_candidate(candidate: Candidate, clip_duration: float) -> None:
    """Validate a candidate's timestamps against its clip duration."""
    if not (0.0 <= candidate.raw_start_s <= candidate.raw_end_s <= clip_duration + 1e-6):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: raw timestamps out of range "
            f"[0, {clip_duration}]: "
            f"[{candidate.raw_start_s}, {candidate.raw_end_s}]"
        )
    if not (0.0 <= candidate.window_start_s <= candidate.window_end_s <= clip_duration + 1e-6):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: window timestamps out of "
            f"range [0, {clip_duration}]: "
            f"[{candidate.window_start_s}, {candidate.window_end_s}]"
        )
    if not (candidate.raw_start_s < candidate.raw_end_s):
        raise ValidationError(f"Candidate {candidate.candidate_id}: raw_start must be < raw_end")
    if not (candidate.window_start_s < candidate.window_end_s):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: window_start must be < window_end"
        )
    if not (
        candidate.window_start_s <= candidate.raw_start_s
        and candidate.window_end_s >= candidate.raw_end_s
    ):
        raise ValidationError(
            f"Candidate {candidate.candidate_id}: padded interval must contain the raw interval"
        )

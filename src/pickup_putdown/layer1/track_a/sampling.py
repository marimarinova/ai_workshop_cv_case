"""Timestamp selection with adaptive sampling for Track A feature extraction.

This module determines which timestamps to sample from a candidate interval
for hand/shelf crop extraction. It supports adaptive splitting for longer
intervals while always preserving the key moments: pre-contact, contact,
and post-contact.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pickup_putdown.common.schemas import Candidate, PoseObservation
    from pickup_putdown.config import TrackAFeaturesConfig
    from pickup_putdown.perception.shelf_regions import Polygon

logger = logging.getLogger(__name__)


@dataclass
class SamplePoint:
    """A single sample point with metadata."""

    timestamp_s: float
    position: str  # "pre", "mid", "contact", "post"
    index: int  # ordering index within the candidate


def compute_sample_times(
    t_start: float,
    t_end: float,
    contact_t: float,
    config: TrackAFeaturesConfig,
) -> list[SamplePoint]:
    """Compute sample timestamps for a candidate interval with adaptive splitting.

    Always includes t_start (pre), contact_t, and t_end (post). If intervals
    between these points exceed config.max_interval_s, intermediate samples
    are added.

    Args:
        t_start: Start of candidate interval (pre-contact moment).
        t_end: End of candidate interval (post-contact moment).
        contact_t: Estimated moment of contact/transfer.
        config: Track A features configuration.

    Returns:
        List of SamplePoint objects sorted by timestamp.

    Raises:
        ValueError: If timestamps are invalid (e.g., t_end <= t_start).
    """
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be greater than t_start ({t_start})")

    if not (t_start <= contact_t <= t_end):
        raise ValueError(f"contact_t ({contact_t}) must be within [{t_start}, {t_end}]")

    samples: list[SamplePoint] = []
    index = 0

    # Always add PRE (t_start)
    samples.append(SamplePoint(timestamp_s=t_start, position="pre", index=index))
    index += 1

    # Add intermediate samples between PRE and CONTACT if needed
    pre_to_contact_mids = adaptive_split_interval(t_start, contact_t, config.max_interval_s)
    for mid_t in pre_to_contact_mids:
        samples.append(SamplePoint(timestamp_s=mid_t, position="mid", index=index))
        index += 1

    # Always add CONTACT
    # Avoid duplicate if contact_t equals t_start
    if not math.isclose(contact_t, t_start, abs_tol=1e-6):
        samples.append(SamplePoint(timestamp_s=contact_t, position="contact", index=index))
        index += 1

    # Add intermediate samples between CONTACT and POST if needed
    contact_to_post_mids = adaptive_split_interval(contact_t, t_end, config.max_interval_s)
    for mid_t in contact_to_post_mids:
        samples.append(SamplePoint(timestamp_s=mid_t, position="mid", index=index))
        index += 1

    # Always add POST (t_end)
    # Avoid duplicate if contact_t equals t_end
    if not math.isclose(contact_t, t_end, abs_tol=1e-6):
        samples.append(SamplePoint(timestamp_s=t_end, position="post", index=index))

    # Sort by timestamp and re-index
    samples.sort(key=lambda s: s.timestamp_s)
    for i, sample in enumerate(samples):
        sample.index = i

    return samples


def adaptive_split_interval(
    t_a: float,
    t_b: float,
    max_interval_s: float,
) -> list[float]:
    """Split an interval into segments of at most max_interval_s, returning midpoints.

    Does NOT include the endpoints t_a and t_b in the result.

    Args:
        t_a: Start of interval.
        t_b: End of interval.
        max_interval_s: Maximum allowed gap between samples.

    Returns:
        List of intermediate timestamps (excluding t_a and t_b).

    Example:
        >>> adaptive_split_interval(2.0, 6.0, 1.0)
        [3.0, 4.0, 5.0]  # 4 segments of 1.0s each, 3 intermediate points
    """
    if max_interval_s <= 0:
        raise ValueError(f"max_interval_s must be positive, got {max_interval_s}")

    gap = t_b - t_a
    if gap <= 0:
        return []

    if gap <= max_interval_s:
        # No splitting needed
        return []

    # Calculate number of segments needed
    n_splits = math.ceil(gap / max_interval_s)
    step = gap / n_splits

    # Generate intermediate points (excluding endpoints)
    midpoints = []
    for i in range(1, n_splits):
        midpoints.append(t_a + i * step)

    return midpoints


def get_contact_time(
    candidate: Candidate,
    wrist_trajectory: list[PoseObservation],
    shelf_region: Polygon,
) -> float:
    """Estimate the contact time for a candidate based on wrist trajectory.

    The contact time is the moment when the wrist enters or is closest to
    the shelf region during the candidate interval.

    Args:
        candidate: The candidate interval.
        wrist_trajectory: List of pose observations for the relevant actor/hand.
        shelf_region: Polygon defining the shelf region.

    Returns:
        Estimated contact timestamp in seconds.

    Falls back to the midpoint of raw interval if no clear entry point is found.
    """
    entry_time = find_shelf_entry_point(
        wrist_trajectory,
        shelf_region,
        candidate.raw_start_s,
        candidate.raw_end_s,
    )

    if entry_time is not None:
        return entry_time

    # Fallback: midpoint of raw interaction interval
    return (candidate.raw_start_s + candidate.raw_end_s) / 2


def find_shelf_entry_point(
    trajectory: list[PoseObservation],
    shelf_region: Polygon,
    t_start: float,
    t_end: float,
) -> float | None:
    """Find the timestamp when the wrist first enters the shelf region.

    Searches for the first observation where the wrist transitions from
    outside to inside the shelf region polygon.

    Args:
        trajectory: Sorted list of pose observations.
        shelf_region: Polygon defining the shelf region boundary.
        t_start: Start of search window.
        t_end: End of search window.

    Returns:
        Timestamp of shelf entry, or None if no clear entry found.
    """
    if not trajectory or not shelf_region:
        return None

    # Filter to observations within the candidate window
    window_obs = [obs for obs in trajectory if t_start <= obs.timestamp_s <= t_end]

    if not window_obs:
        return None

    # Sort by timestamp
    window_obs.sort(key=lambda obs: obs.timestamp_s)

    # Look for transition from outside to inside
    previous_inside = None
    for obs in window_obs:
        inside = _point_in_polygon(obs.wrist_x, obs.wrist_y, shelf_region)

        if previous_inside is False and inside is True:
            # Found entry point
            return obs.timestamp_s

        previous_inside = inside

    # No clear entry transition found
    # Return timestamp of first observation inside the region
    for obs in window_obs:
        if _point_in_polygon(obs.wrist_x, obs.wrist_y, shelf_region):
            return obs.timestamp_s

    # No observation inside the region - return the observation closest to region
    min_dist = float("inf")
    closest_time = None

    for obs in window_obs:
        dist = _point_to_polygon_distance(obs.wrist_x, obs.wrist_y, shelf_region)
        if dist < min_dist:
            min_dist = dist
            closest_time = obs.timestamp_s

    return closest_time


def _point_in_polygon(px: float, py: float, polygon: Polygon) -> bool:

    # If we imagine Polygon as set of edges, a point would be outside if for edge_i we have edge_j
    # Where edge_i and edge_j are intersected by the same y axis of the ray but x_i < x_j
    # Note realistically for "un-logical" polygons we would again have even number of pairs |{e_i1,  e_j1}, ..... {e_ik, e_jk})| MOD2 == 0

    if len(polygon) < 3:
        return False

    inside = False
    previous_index = len(polygon) - 1

    for current_index in range(len(polygon)):
        current_x, current_y = polygon[current_index]
        previous_x, previous_y = polygon[previous_index]

        crosses_horizontal_line = (current_y > py) != (previous_y > py)

        if crosses_horizontal_line:
            intersection_x = current_x + (py - current_y) * (previous_x - current_x) / (
                previous_y - current_y
            )

            if px < intersection_x:
                inside = not inside

        previous_index = current_index

    return inside


def _point_to_polygon_distance(px: float, py: float, polygon: Polygon) -> float:
    """Calculate minimum distance from a point to a polygon's edges."""

    # Let Euclidian distance between point x and edge u inside the polygon be dist[x, u];
    # Return min(ALL: dist[x, u_i])

    if not polygon:
        return float("inf")

    min_dist = float("inf")
    n = len(polygon)

    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]

        dist = _point_to_segment_distance(px, py, x1, y1, x2, y2)
        min_dist = min(min_dist, dist)

    return min_dist


def _point_to_segment_distance(
    px: float,
    py: float,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    """Calculate the distance from a point to a line segment."""
    dx = x2 - x1
    dy = y2 - y1
    length_sq = dx * dx + dy * dy

    if length_sq < 1e-12:
        # Segment is essentially a point
        return math.hypot(px - x1, py - y1)

    # Project point onto line, clamped to segment
    t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / length_sq))

    proj_x = x1 + t * dx
    proj_y = y1 + t * dy

    return math.hypot(px - proj_x, py - proj_y)


def get_wrist_trajectory_for_candidate(
    candidate: Candidate,
    pose_observations: list[PoseObservation],
) -> list[PoseObservation]:
    """Filter pose observations to those relevant to a candidate.

    Filters by clip_id, actor_id, hand_side, and time window.

    Args:
        candidate: The candidate to filter for.
        pose_observations: All pose observations.

    Returns:
        Sorted list of pose observations matching the candidate.
    """
    relevant = [
        obs
        for obs in pose_observations
        if (
            obs.clip_id == candidate.clip_id
            and obs.actor_id == candidate.actor_id
            and obs.hand_side == candidate.hand_side
            and candidate.window_start_s <= obs.timestamp_s <= candidate.window_end_s
        )
    ]

    # Fallback: proposals use two actor_id formats (actor_N and
    # clip_D2_S...:person:N) while the pose tracker only emits actor_N.
    # When exact match fails AND the candidate uses person-tracker format,
    # fall back to clip_id + hand_side + window.
    if not relevant and ":" in candidate.actor_id:
        relevant = [
            obs
            for obs in pose_observations
            if (
                obs.clip_id == candidate.clip_id
                and obs.hand_side == candidate.hand_side
                and candidate.window_start_s <= obs.timestamp_s <= candidate.window_end_s
            )
        ]
        if relevant:
            logger.debug(
                "Pose fallback match for %s: actor=%s not found, "
                "matched %d poses by clip+hand+window",
                candidate.candidate_id,
                candidate.actor_id,
                len(relevant),
            )

    if not relevant and pose_observations:
        clip_poses = [o for o in pose_observations if o.clip_id == candidate.clip_id]
        logger.debug(
            "Pose filter miss for %s: clip=%s actor=%s hand=%s window=[%.1f,%.1f]s clip_poses=%d",
            candidate.candidate_id,
            candidate.clip_id,
            candidate.actor_id,
            candidate.hand_side,
            candidate.window_start_s,
            candidate.window_end_s,
            len(clip_poses),
        )
        if clip_poses:
            first = clip_poses[0]
            logger.debug(
                "  First clip pose: actor=%s hand=%s ts=%.1fs",
                first.actor_id,
                first.hand_side,
                first.timestamp_s,
            )

    relevant.sort(key=lambda obs: obs.timestamp_s)
    return relevant

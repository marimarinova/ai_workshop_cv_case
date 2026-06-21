"""Active-span derivation from person track observations."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from pickup_putdown.common.schemas import ActiveSpan, PersonObservation

logger = logging.getLogger(__name__)


@dataclass
class _Interval:
    """Internal interval representation for merging."""

    t_start: float
    t_end: float

    def overlaps(self, other: _Interval) -> bool:
        return self.t_start < other.t_end and other.t_start < self.t_end

    def merges_with(self, other: _Interval, gap: float) -> bool:
        """True if this interval and other can be merged (overlap or within gap)."""
        return self.t_start <= other.t_end + gap and other.t_start <= self.t_end + gap


def derive_active_spans(
    observations: list[PersonObservation],
    clip_id: str,
    clip_duration_s: float,
    merge_gap_s: float,
    effective_sample_fps: float,
) -> list[ActiveSpan]:
    """Derive active spans from stable person observations.

    Parameters
    ----------
    observations : list[PersonObservation]
        Flat observations from PersonTracker.run().
    clip_id : str
        The clip identifier.
    clip_duration_s : float
        Total source clip duration in seconds.
    merge_gap_s : float
        Maximum gap (seconds) between intervals to merge.
    effective_sample_fps : float
        The actual sampling rate used (source_fps / vid_stride).

    Returns
    -------
    list[ActiveSpan]
        Sorted, non-overlapping active spans within [0, clip_duration_s].
    """
    # Filter to stable observations only
    stable = [o for o in observations if o.is_stable]
    if not stable:
        return []

    # Compute observation intervals
    observation_radius_s = 0.5 / effective_sample_fps if effective_sample_fps > 0 else 0.5
    intervals: list[_Interval] = []
    for obs in stable:
        t = obs.timestamp_s
        lo = max(0.0, t - observation_radius_s)
        hi = min(clip_duration_s, t + observation_radius_s)
        if lo < hi:
            intervals.append(_Interval(lo, hi))

    if not intervals:
        return []

    # Sort by start time, then end time
    intervals.sort(key=lambda iv: (iv.t_start, iv.t_end))

    # Merge overlapping or close intervals
    merged: list[_Interval] = [intervals[0]]
    for iv in intervals[1:]:
        last = merged[-1]
        if iv.t_start <= last.t_end + merge_gap_s:
            # Merge
            last.t_start = min(last.t_start, iv.t_start)
            last.t_end = max(last.t_end, iv.t_end)
        else:
            merged.append(iv)

    # Clamp to source duration
    for iv in merged:
        iv.t_start = max(0.0, iv.t_start)
        iv.t_end = min(clip_duration_s, iv.t_end)

    # Remove any degenerate intervals
    merged = [iv for iv in merged if iv.t_end > iv.t_start]

    # Generate deterministic IDs and return ActiveSpan objects
    spans: list[ActiveSpan] = []
    for i, iv in enumerate(merged):
        span_id = f"{clip_id}:active:{i:03d}"
        spans.append(
            ActiveSpan(
                clip_id=clip_id,
                active_span_id=span_id,
                t_start=iv.t_start,
                t_end=iv.t_end,
                n_person_tracks=0,  # filled by caller from track summaries
            )
        )

    logger.info(
        "Active spans for %s: %d spans, merge_gap=%.2fs",
        clip_id,
        len(spans),
        merge_gap_s,
    )
    return spans


def compute_clip_summary(
    observations: list[PersonObservation],
    spans: list[ActiveSpan],
) -> dict:
    """Compute clip-level summary fields for manifest update.

    Parameters
    ----------
    observations : list[PersonObservation]
        All observations (stable and unstable).
    spans : list[ActiveSpan]
        Derived active spans.

    Returns
    -------
    dict
        Keys: n_person_tracks, has_person, active_start_s, active_end_s
    """
    stable_tracks = set()
    for obs in observations:
        if obs.is_stable and obs.tracker_track_id is not None:
            stable_tracks.add(obs.tracker_track_id)

    n_person_tracks = len(stable_tracks)
    has_person = len(spans) > 0

    if spans:
        active_start_s = spans[0].t_start
        active_end_s = spans[-1].t_end
    else:
        active_start_s = None
        active_end_s = None

    return {
        "n_person_tracks": n_person_tracks,
        "has_person": has_person,
        "active_start_s": active_start_s,
        "active_end_s": active_end_s,
    }

"""Perception modules for person detection, tracking, and active-span extraction."""

from pickup_putdown.perception.active_spans import compute_clip_summary, derive_active_spans
from pickup_putdown.perception.person_tracker import PersonObservation, PersonTracker, TrackSummary
from pickup_putdown.perception.previews import OverlayConfig, draw_overlay, render_triage_preview

__all__ = [
    "compute_clip_summary",
    "derive_active_spans",
    "OverlayConfig",
    "PersonTracker",
    "PersonObservation",
    "TrackSummary",
    "draw_overlay",
    "render_triage_preview",
]

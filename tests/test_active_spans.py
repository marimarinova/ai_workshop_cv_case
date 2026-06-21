"""Tests for active-span derivation from person track observations."""

from __future__ import annotations

from pickup_putdown.common.schemas import PersonObservation
from pickup_putdown.perception.active_spans import (
    compute_clip_summary,
    derive_active_spans,
)


def _make_obs(
    clip_id: str = "clip_001",
    tracker_id: int = 1,
    timestamp_s: float = 5.0,
    is_stable: bool = True,
    source_frame: int = 150,
    sample_index: int = 5,
) -> PersonObservation:
    tid_str = str(tracker_id) if tracker_id is not None else None
    return PersonObservation(
        clip_id=clip_id,
        person_track_id=f"{clip_id}:person:{tid_str}"
        if tid_str
        else f"{clip_id}:person:untracked",
        tracker_track_id=tracker_id,
        sample_index=sample_index,
        source_frame_index=source_frame,
        timestamp_s=timestamp_s,
        bbox_x1=100.0,
        bbox_y1=50.0,
        bbox_x2=300.0,
        bbox_y2=400.0,
        confidence=0.85,
        is_stable=is_stable,
    )


class TestDeriveActiveSpans:
    """Tests for derive_active_spans function."""

    def test_single_contiguous_track(self):
        """A single stable track produces one active span."""
        obs = [_make_obs(timestamp_s=t) for t in [5.0, 6.0, 7.0, 8.0]]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 1
        assert spans[0].active_span_id == "clip_001:active:000"
        assert spans[0].t_start >= 4.5
        assert spans[0].t_end <= 8.5

    def test_overlapping_tracks(self):
        """Overlapping tracks from different IDs produce merged spans."""
        obs = [
            _make_obs(tracker_id=1, timestamp_s=5.0),
            _make_obs(tracker_id=1, timestamp_s=7.0),
            _make_obs(tracker_id=2, timestamp_s=6.0),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 1

    def test_mergeable_short_gap(self):
        """Observations separated by a gap <= merge_gap_s are merged."""
        obs = [
            _make_obs(timestamp_s=5.0),
            _make_obs(timestamp_s=5.5),
            _make_obs(
                timestamp_s=7.0
            ),  # 1.5s gap from 5.5, but with radius 0.5, intervals are [4.5,5.5] and [6.5,7.5] - not mergeable at 1.0 gap
            _make_obs(
                timestamp_s=6.5
            ),  # this one is 1.0 from 5.5, intervals [6.0,7.0] merges with [4.5,5.5]
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        # With radius=0.5: [4.5,5.5], [5.0,6.0], [6.5,7.5], [6.0,7.0]
        # After sort: [4.5,5.5], [5.0,6.0], [6.0,7.0], [6.5,7.5]
        # [4.5,5.5] merges with [5.0,6.0] -> [4.5,6.0]
        # [6.0,7.0] starts at 6.0, last end is 6.0, gap=0, merges -> [4.5,7.0]
        # [6.5,7.5] starts at 6.5, last end is 7.0, gap=0, merges -> [4.5,7.5]
        assert len(spans) == 1

    def test_non_mergeable_long_gap(self):
        """Observations separated by a gap > merge_gap_s produce separate spans."""
        obs = [
            _make_obs(timestamp_s=5.0),
            _make_obs(timestamp_s=6.0),
            _make_obs(timestamp_s=15.0),
            _make_obs(timestamp_s=16.0),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 2
        assert spans[0].active_span_id == "clip_001:active:000"
        assert spans[1].active_span_id == "clip_001:active:001"

    def test_multiple_runs_under_one_tracker_id(self):
        """Multiple runs under one tracker ID produce separate spans."""
        obs = [
            _make_obs(tracker_id=1, timestamp_s=5.0, is_stable=True),
            _make_obs(tracker_id=1, timestamp_s=6.0, is_stable=True),
            _make_obs(tracker_id=1, timestamp_s=20.0, is_stable=True),
            _make_obs(tracker_id=1, timestamp_s=21.0, is_stable=True),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 2

    def test_clamping_at_zero(self):
        """Spans are clamped to start at 0."""
        obs = [_make_obs(timestamp_s=0.0)]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 1
        assert spans[0].t_start >= 0.0

    def test_clamping_at_source_duration(self):
        """Spans are clamped to not exceed clip duration."""
        obs = [_make_obs(timestamp_s=29.5)]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 1
        assert spans[0].t_end <= 30.0

    def test_deterministic_active_span_ids(self):
        """Active span IDs are deterministic and sequential."""
        obs = [
            _make_obs(timestamp_s=5.0),
            _make_obs(timestamp_s=15.0),
            _make_obs(timestamp_s=25.0),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        ids = [s.active_span_id for s in spans]
        assert ids == ["clip_001:active:000", "clip_001:active:001", "clip_001:active:002"]

    def test_no_person_clip_produces_zero_spans(self):
        """No-person clip (empty observations) produces zero spans."""
        spans = derive_active_spans(
            [], "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 0

    def test_unstable_observations_excluded(self):
        """Unstable observations do not contribute to active spans."""
        obs = [_make_obs(timestamp_s=5.0, is_stable=False)]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        assert len(spans) == 0

    def test_span_within_source_duration(self):
        """All spans remain within [0, clip_duration_s]."""
        obs = [
            _make_obs(timestamp_s=2.0),
            _make_obs(timestamp_s=28.0),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        for span in spans:
            assert span.t_start >= 0.0
            assert span.t_end <= 30.0


class TestComputeClipSummary:
    """Tests for compute_clip_summary function."""

    def test_person_clip_summary(self):
        """Person clip summary has correct fields."""
        obs = [_make_obs(timestamp_s=t) for t in [5.0, 6.0, 7.0]]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        summary = compute_clip_summary(obs, spans)
        assert summary["has_person"] is True
        assert summary["n_person_tracks"] >= 1
        assert summary["active_start_s"] is not None
        assert summary["active_end_s"] is not None

    def test_no_person_clip_summary(self):
        """No-person clip summary has correct fields."""
        summary = compute_clip_summary([], [])
        assert summary["has_person"] is False
        assert summary["n_person_tracks"] == 0
        assert summary["active_start_s"] is None
        assert summary["active_end_s"] is None

    def test_n_person_tracks_counts_stable_only(self):
        """n_person_tracks counts only stable tracker IDs."""
        obs = [
            _make_obs(tracker_id=1, timestamp_s=5.0, is_stable=True),
            _make_obs(tracker_id=2, timestamp_s=6.0, is_stable=False),
            _make_obs(tracker_id=3, timestamp_s=7.0, is_stable=True),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        summary = compute_clip_summary(obs, spans)
        assert summary["n_person_tracks"] == 2

    def test_active_span_bounds_reflect_span_data(self):
        """active_start_s and active_end_s reflect span boundaries."""
        obs = [
            _make_obs(timestamp_s=5.0),
            _make_obs(timestamp_s=15.0),
        ]
        spans = derive_active_spans(
            obs, "clip_001", 30.0, merge_gap_s=1.0, effective_sample_fps=1.0
        )
        summary = compute_clip_summary(obs, spans)
        assert summary["active_start_s"] == spans[0].t_start
        assert summary["active_end_s"] == spans[-1].t_end

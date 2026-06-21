"""Tests for canonical schema validation."""

import pytest
from pydantic import ValidationError

from pickup_putdown.common.schemas import (
    ActiveSpan,
    Candidate,
    Clip,
    Confidence,
    Event,
    EventType,
    IgnoreInterval,
    Prediction,
)

# -----------------------------------------------------------------------
# Clip
# -----------------------------------------------------------------------


class TestClip:
    def test_valid_clip(self, valid_clip_data):
        clip = Clip(**valid_clip_data)
        assert clip.clip_id == "clip_001"

    def test_negative_duration_fails(self):
        with pytest.raises(ValidationError):
            Clip(
                clip_id="x",
                s3_key="x",
                duration_s=-1.0,
                fps=30.0,
                width=1920,
                height=1080,
            )

    def test_negative_fps_fails(self):
        with pytest.raises(ValidationError):
            Clip(
                clip_id="x",
                s3_key="x",
                duration_s=10.0,
                fps=-1.0,
                width=1920,
                height=1080,
            )

    def test_negative_active_start_fails(self):
        with pytest.raises(ValidationError):
            Clip(
                clip_id="x",
                s3_key="x",
                duration_s=10.0,
                fps=30.0,
                width=1920,
                height=1080,
                active_start_s=-1.0,
            )


# -----------------------------------------------------------------------
# Event
# -----------------------------------------------------------------------


class TestEvent:
    def test_valid_event(self, valid_event_data):
        event = Event(**valid_event_data)
        assert event.type == EventType.PICKUP
        assert event.confidence == Confidence.HIGH

    def test_putdown_event(self):
        event = Event(
            event_id="e1",
            clip_id="c1",
            type="putdown",
            t_start=5.0,
            t_end=7.0,
        )
        assert event.type == EventType.PUTDOWN

    def test_invalid_event_type_fails(self):
        with pytest.raises(ValidationError):
            Event(
                event_id="e1",
                clip_id="c1",
                type="touch",
                t_start=5.0,
                t_end=7.0,
            )

    def test_invalid_confidence_fails(self):
        with pytest.raises(ValidationError):
            Event(
                event_id="e1",
                clip_id="c1",
                type="pickup",
                t_start=5.0,
                t_end=7.0,
                confidence="critical",
            )

    def test_t_end_before_t_start_fails(self):
        with pytest.raises(ValidationError):
            Event(
                event_id="e1",
                clip_id="c1",
                type="pickup",
                t_start=10.0,
                t_end=5.0,
            )

    def test_t_end_equal_t_start_fails(self):
        with pytest.raises(ValidationError):
            Event(
                event_id="e1",
                clip_id="c1",
                type="pickup",
                t_start=5.0,
                t_end=5.0,
            )

    def test_negative_timestamp_fails(self):
        with pytest.raises(ValidationError):
            Event(
                event_id="e1",
                clip_id="c1",
                type="pickup",
                t_start=-1.0,
                t_end=5.0,
            )


# -----------------------------------------------------------------------
# Prediction
# -----------------------------------------------------------------------


class TestPrediction:
    def test_valid_prediction(self, valid_prediction_data):
        pred = Prediction(**valid_prediction_data)
        assert pred.score == 0.85

    def test_score_zero_ok(self):
        pred = Prediction(
            pred_id="p1",
            clip_id="c1",
            type="pickup",
            t_start=0.0,
            t_end=2.0,
            score=0.0,
            model="test",
        )
        assert pred.score == 0.0

    def test_score_one_ok(self):
        pred = Prediction(
            pred_id="p1",
            clip_id="c1",
            type="pickup",
            t_start=0.0,
            t_end=2.0,
            score=1.0,
            model="test",
        )
        assert pred.score == 1.0

    def test_score_below_zero_fails(self):
        with pytest.raises(ValidationError):
            Prediction(
                pred_id="p1",
                clip_id="c1",
                type="pickup",
                t_start=0.0,
                t_end=2.0,
                score=-0.1,
                model="test",
            )

    def test_score_above_one_fails(self):
        with pytest.raises(ValidationError):
            Prediction(
                pred_id="p1",
                clip_id="c1",
                type="pickup",
                t_start=0.0,
                t_end=2.0,
                score=1.1,
                model="test",
            )

    def test_t_end_before_t_start_fails(self):
        with pytest.raises(ValidationError):
            Prediction(
                pred_id="p1",
                clip_id="c1",
                type="pickup",
                t_start=10.0,
                t_end=5.0,
                score=0.9,
                model="test",
            )


# -----------------------------------------------------------------------
# ActiveSpan
# -----------------------------------------------------------------------


class TestActiveSpan:
    def test_valid_span(self, valid_active_span_data):
        span = ActiveSpan(**valid_active_span_data)
        assert span.t_start == 5.0

    def test_t_end_before_t_start_fails(self):
        with pytest.raises(ValidationError):
            ActiveSpan(
                clip_id="c1",
                active_span_id="s1",
                t_start=10.0,
                t_end=5.0,
            )

    def test_negative_timestamp_fails(self):
        with pytest.raises(ValidationError):
            ActiveSpan(
                clip_id="c1",
                active_span_id="s1",
                t_start=-1.0,
                t_end=5.0,
            )


# -----------------------------------------------------------------------
# IgnoreInterval
# -----------------------------------------------------------------------


class TestIgnoreInterval:
    def test_valid_interval(self, valid_ignore_interval_data):
        ig = IgnoreInterval(**valid_ignore_interval_data)
        assert ig.reason == "ACTION_OCCLUDED"

    def test_t_end_before_t_start_fails(self):
        with pytest.raises(ValidationError):
            IgnoreInterval(
                ignore_id="i1",
                clip_id="c1",
                t_start=10.0,
                t_end=5.0,
                reason="ACTION_OCCLUDED",
            )


# -----------------------------------------------------------------------
# Candidate
# -----------------------------------------------------------------------


class TestCandidate:
    def test_valid_candidate(self, valid_candidate_data):
        cand = Candidate(**valid_candidate_data)
        assert cand.actor_id == "actor_3"

    def test_negative_timestamp_fails(self):
        with pytest.raises(ValidationError):
            Candidate(
                candidate_id="c1",
                clip_id="c1",
                actor_id="a1",
                raw_start_s=-1.0,
                raw_end_s=5.0,
                window_start_s=0.0,
                window_end_s=10.0,
            )

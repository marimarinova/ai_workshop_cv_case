"""Shared fixtures for schema tests."""

from __future__ import annotations

import pytest


@pytest.fixture
def valid_clip_data() -> dict:
    return {
        "clip_id": "clip_001",
        "s3_key": "s3://bucket/clip_001.mp4",
        "duration_s": 30.0,
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "n_person_tracks": 2,
        "usable": True,
        "active_start_s": 5.0,
        "active_end_s": 25.0,
        "split": "train",
        "session_id": "session_01",
        "notes": None,
    }


@pytest.fixture
def valid_event_data() -> dict:
    return {
        "event_id": "evt_001",
        "clip_id": "clip_001",
        "type": "pickup",
        "t_start": 10.0,
        "t_end": 12.5,
        "hard_case": False,
        "annotator": "alice",
        "confidence": "high",
        "notes": None,
    }


@pytest.fixture
def valid_prediction_data() -> dict:
    return {
        "pred_id": "pred_001",
        "clip_id": "clip_001",
        "type": "pickup",
        "t_start": 10.0,
        "t_end": 12.5,
        "score": 0.85,
        "model": "layer1_track_a_pose_state_v1",
    }


@pytest.fixture
def valid_active_span_data() -> dict:
    return {
        "clip_id": "clip_001",
        "active_span_id": "span_001",
        "t_start": 5.0,
        "t_end": 25.0,
        "n_person_tracks": 2,
    }


@pytest.fixture
def valid_ignore_interval_data() -> dict:
    return {
        "ignore_id": "ign_001",
        "clip_id": "clip_001",
        "t_start": 1.0,
        "t_end": 3.0,
        "reason": "ACTION_OCCLUDED",
        "annotator": "bob",
        "notes": None,
    }


@pytest.fixture
def valid_candidate_data() -> dict:
    return {
        "candidate_id": "cand_001",
        "clip_id": "clip_001",
        "actor_id": "actor_3",
        "hand_side": "right",
        "region_id": "shelf_left",
        "raw_start_s": 9.0,
        "raw_end_s": 13.0,
        "window_start_s": 7.0,
        "window_end_s": 15.0,
        "proposal_reason": "wrist_entered_region",
        "proposal_score": 0.72,
        "review_status": "pending",
    }

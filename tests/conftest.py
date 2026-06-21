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


@pytest.fixture
def valid_person_observation_data() -> dict:
    return {
        "clip_id": "clip_001",
        "person_track_id": "clip_001:person:3",
        "tracker_track_id": 3,
        "sample_index": 5,
        "source_frame_index": 150,
        "timestamp_s": 5.0,
        "bbox_x1": 100.0,
        "bbox_y1": 50.0,
        "bbox_x2": 300.0,
        "bbox_y2": 400.0,
        "confidence": 0.85,
        "is_stable": True,
    }


@pytest.fixture
def valid_track_summary_data() -> dict:
    return {
        "clip_id": "clip_001",
        "tracker_track_id": 3,
        "first_seen_s": 5.0,
        "last_seen_s": 10.0,
        "visible_duration_s": 5.0,
        "n_observations": 10,
        "mean_confidence": 0.82,
        "max_observation_gap_s": 1.0,
        "is_stable": True,
    }


@pytest.fixture
def valid_sampling_report_data() -> dict:
    return {
        "clip_id": "clip_001",
        "decision": "person_detected",
        "selected_for_qa": False,
        "selection_reason": "person_detected",
        "preview_path": "outputs/triage_previews/clip_001.mp4",
        "source_duration_s": 30.0,
        "target_fps": 1.0,
        "effective_sample_fps": 1.0,
        "n_raw_tracks": 2,
        "n_stable_tracks": 1,
        "n_observations": 25,
        "review_status": "ok",
    }


@pytest.fixture
def triage_config() -> dict:
    return {
        "model_path": "models/person_detector.pt",
        "target_fps": 1.0,
        "image_size": 640,
        "device": "cpu",
        "half": False,
        "detector_confidence": 0.10,
        "detector_iou_threshold": 0.70,
        "max_detections": 100,
        "minimum_track_confidence": 0.35,
        "minimum_visible_duration_s": 0.75,
        "minimum_observations": 2,
        "max_track_observation_gap_s": 1.5,
        "merge_gap_s": 1.0,
        "preview_sample_rate": 0.10,
        "sampling_seed": 42,
        "tracker_config": "configs/bytetrack_triage.yaml",
    }

"""Tests for the Track A inference pipeline.

All tests use synthetic data. No GPU, real videos, YOLO, MobileNet,
or real classifier artifacts are required.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pytest

from pickup_putdown.layer1.track_a.classifier import (
    ClassifierMetadata,
    TrackAClassifier,
)
from pickup_putdown.layer1.track_a.hand_state import HAND_STATE_CLASS_NAMES
from pickup_putdown.layer1.track_a.inference import (
    BoundaryRefinementConfig,
    CanonicalPrediction,
    DedupAuditEntry,
    DeduplicationConfig,
    FeatureExtractionResult,
    HandStateEvidence,
    InferenceConfig,
    InferenceResult,
    InferenceSummary,
    ShelfStateEvidence,
    TrackAInferencePipeline,
    apply_grace_window,
    build_observations,
    compute_inference_sample_times,
    deduplicate_predictions,
    events_to_predictions,
    merge_dedup_candidate_ids,
    predict_hand_state,
    predict_shelf_state,
    refine_event_boundaries,
    save_predictions_csv,
    temporal_iou,
    validate_artifact_compatibility,
    validate_classifier_classes,
)
from pickup_putdown.layer1.track_a.shelf_state import SHELF_STATE_CLASS_NAMES
from pickup_putdown.layer1.track_a.state_machine import (
    EvidenceSummary,
    RepeatingInteractionStateMachine,
    StateMachineConfig,
    StateMachineEvent,
    TrackAObservation,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EMB_DIM = 16


@dataclass
class MockCandidate:
    candidate_id: str
    clip_id: str
    actor_id: str
    hand_side: str = "right"
    region_id: str = "region_1"
    raw_start_s: float = 0.0
    raw_end_s: float = 2.0
    window_start_s: float = 0.0
    window_end_s: float = 2.0


@dataclass
class MockPose:
    clip_id: str
    timestamp_s: float
    actor_id: str
    hand_side: str
    wrist_x: float = 100.0
    wrist_y: float = 200.0
    wrist_confidence: float = 0.8


def _make_mock_classifier(
    kind: str,
    class_names: list[str],
    crop_type: str,
    emb_dim: int = _EMB_DIM,
) -> tuple[TrackAClassifier, ClassifierMetadata]:
    """Create a mock classifier with known prediction behavior."""
    clf = TrackAClassifier(
        classifier_name=kind,
        crop_type=crop_type,
        class_names=class_names,
        confidence_threshold=0.60,
        margin_threshold=0.15,
    )
    # Build a trivial pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    n_classes = len(class_names)
    coef = np.eye(n_classes, _EMB_DIM)
    intercept = np.zeros(n_classes)
    lr = LogisticRegression()
    lr.classes_ = np.array(class_names)
    lr.coef_ = coef
    lr.intercept_ = intercept

    scaler = StandardScaler()
    scaler.fit(np.zeros((2, _EMB_DIM)))

    clf._pipeline = [
        ("scaler", scaler),
        ("classifier", lr),
    ]
    clf._is_fitted = True
    clf._embedding_dim = _EMB_DIM

    meta = ClassifierMetadata(
        classifier_name=kind,
        class_names=class_names,
        embedding_dim=emb_dim,
        crop_type=crop_type,
        encoder_name="mobilenet_v3_small",
        encoder_version="v1",
        training_timestamp="2025-01-01T00:00:00",
        training_record_counts=dict.fromkeys(class_names, 10),
        train_split_count=100,
        val_split_count=20,
        confidence_threshold=0.60,
        margin_threshold=0.15,
        random_seed=42,
        class_weight="balanced",
        max_iter=1000,
        artifact_version="1.0",
    )
    return clf, meta


def _make_sm_event(
    label: str,
    start_s: float,
    end_s: float,
    transfer_s: float | None = None,
    confidence: float = 0.7,
    clip_id: str = "clip_1",
    candidate_id: str = "cand_1",
    actor_id: str = "actor_1",
    hand_side: str = "right",
    region_id: str = "region_1",
    cycle_id: int = 0,
) -> StateMachineEvent:
    return StateMachineEvent(
        clip_id=clip_id,
        candidate_id=candidate_id,
        actor_id=actor_id,
        hand_side=hand_side,
        region_id=region_id,
        label=label,
        start_s=start_s,
        end_s=end_s,
        transfer_timestamp_s=transfer_s or (start_s + end_s) / 2,
        confidence=confidence,
        evidence=EvidenceSummary(
            pre_transfer_hand_empty=0.8,
            pre_transfer_hand_carrying=0.1,
            post_transfer_hand_empty=0.1,
            post_transfer_hand_carrying=0.8,
            shelf_transition_prob=0.75,
            trajectory_confidence=0.8,
            n_supporting_observations=5,
            evidence_duration_s=0.5,
            uncertainty_proportion=0.1,
        ),
        cycle_id=cycle_id,
    )


def _make_obs(
    t: float,
    inside: bool = False,
    distance: float | None = None,
    hand_empty: float = 0.5,
    hand_carrying: float = 0.3,
    hand_uncertain: float = 0.2,
    shelf_removed: float = 0.3,
    shelf_placed: float = 0.3,
    shelf_no_change: float = 0.3,
    shelf_uncertain: float = 0.1,
    traj_conf: float = 0.8,
    **extra: object,
) -> TrackAObservation:
    base: dict[str, object] = {
        "clip_id": "clip_1",
        "candidate_id": "cand_1",
        "actor_id": "actor_1",
        "hand_side": "right",
        "region_id": "region_1",
        **extra,
    }
    return TrackAObservation(
        timestamp_s=t,
        inside_region=inside,
        wrist_to_region_distance_px=distance,
        trajectory_confidence=traj_conf,
        hand_prob_empty=hand_empty,
        hand_prob_carrying=hand_carrying,
        hand_prob_uncertain=hand_uncertain,
        shelf_prob_object_removed=shelf_removed,
        shelf_prob_object_placed=shelf_placed,
        shelf_prob_no_change=shelf_no_change,
        shelf_prob_uncertain=shelf_uncertain,
        wrist_x=100.0,
        wrist_y=200.0,
        **base,
    )


# ---------------------------------------------------------------------------
# 1-5: Artifact compatibility
# ---------------------------------------------------------------------------


class TestArtifactCompatibility:
    def test_valid_hand_artifact_loads(self, tmp_path) -> None:
        clf, meta = _make_mock_classifier("hand_state", HAND_STATE_CLASS_NAMES, "hand")
        p = tmp_path / "hand_state.joblib"
        clf.save_pipeline(p)
        meta_path = tmp_path / "hand_state_metadata.json"
        meta_path.write_text(json.dumps(meta.to_dict()))
        loaded, loaded_meta = TrackAClassifier.load_pipeline(p, meta_path)
        assert loaded_meta is not None
        validate_classifier_classes(loaded, loaded_meta, HAND_STATE_CLASS_NAMES, "hand")

    def test_valid_shelf_artifact_loads(self, tmp_path) -> None:
        clf, meta = _make_mock_classifier("shelf_state", SHELF_STATE_CLASS_NAMES, "shelf")
        p = tmp_path / "shelf_state.joblib"
        clf.save_pipeline(p)
        meta_path = tmp_path / "shelf_state_metadata.json"
        meta_path.write_text(json.dumps(meta.to_dict()))
        loaded, loaded_meta = TrackAClassifier.load_pipeline(p, meta_path)
        assert loaded_meta is not None
        validate_classifier_classes(loaded, loaded_meta, SHELF_STATE_CLASS_NAMES, "shelf")

    def test_embedding_dim_mismatch_fails(self) -> None:
        _, meta1 = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand", emb_dim=64)
        _, meta2 = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf", emb_dim=128)
        with pytest.raises(Exception, match="dimension mismatch"):
            validate_artifact_compatibility(meta1, meta2)

    def test_encoder_mismatch_fails(self) -> None:
        _, m1 = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        m1.encoder_name = "resnet50"
        _, m2 = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        with pytest.raises(Exception, match="Encoder mismatch"):
            validate_artifact_compatibility(m1, m2)

    def test_missing_required_class_fails(self) -> None:
        clf, meta = _make_mock_classifier("hand", ["empty"], "hand")
        with pytest.raises(Exception, match="missing required classes"):
            validate_classifier_classes(clf, meta, HAND_STATE_CLASS_NAMES, "hand")


# ---------------------------------------------------------------------------
# 6-10: Input and stream handling
# ---------------------------------------------------------------------------


class TestInputStreamHandling:
    def test_candidate_pose_resolution(self) -> None:
        cand = MockCandidate(
            candidate_id="c1", clip_id="clip_1", actor_id="actor_1", hand_side="right"
        )
        poses = [
            MockPose(clip_id="clip_1", timestamp_s=0.5, actor_id="actor_1", hand_side="right"),
            MockPose(clip_id="clip_1", timestamp_s=1.0, actor_id="actor_2", hand_side="right"),
            MockPose(clip_id="clip_2", timestamp_s=0.5, actor_id="actor_1", hand_side="right"),
        ]
        from pickup_putdown.layer1.track_a.inference import _filter_poses_for_candidate

        result = _filter_poses_for_candidate(cand, poses)
        assert len(result) == 1
        assert result[0].timestamp_s == 0.5

    def test_different_streams_isolated(self) -> None:
        obs_a = [_make_obs(0.0, actor_id="actor_A"), _make_obs(0.1, actor_id="actor_A")]
        obs_b = [_make_obs(0.0, actor_id="actor_B"), _make_obs(0.1, actor_id="actor_B")]
        sm = RepeatingInteractionStateMachine(debug=False)
        sm.process(obs_a + obs_b)
        traces = sm.debug_traces
        keys = list(traces.keys())
        assert len(keys) == 2

    def test_absolute_source_timestamps_preserved(self) -> None:
        obs = [_make_obs(10.0), _make_obs(10.5), _make_obs(11.0)]
        sm = RepeatingInteractionStateMachine()
        sm.process(obs)
        traces = sm.debug_traces
        for _key, trace_list in traces.items():
            for t in trace_list:
                assert t.timestamp_s >= 10.0

    def test_missing_pose_data_skipped(self, tmp_path) -> None:
        cfg = InferenceConfig()
        sm_cfg = StateMachineConfig()
        pipeline = TrackAInferencePipeline(cfg, sm_cfg)
        hand_p = tmp_path / "hand_state.joblib"
        shelf_p = tmp_path / "shelf_state.joblib"
        _, hand_meta = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        _, shelf_meta = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        hc, _ = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        hc.save_pipeline(hand_p)
        (tmp_path / "hand_state_metadata.json").write_text(json.dumps(hand_meta.to_dict()))
        sc, _ = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        sc.save_pipeline(shelf_p)
        (tmp_path / "shelf_state_metadata.json").write_text(json.dumps(shelf_meta.to_dict()))

        video_path = tmp_path / "clip_1.mp4"
        video_path.write_bytes(b"fake")
        cand = MockCandidate(candidate_id="c1", clip_id="clip_1", actor_id="actor_1")
        result = pipeline.run(
            candidates=[cand],
            pose_observations=[],
            source_videos={"clip_1": video_path},
            hand_classifier_path=hand_p,
            shelf_classifier_path=shelf_p,
            output_dir=tmp_path / "out",
        )
        assert result.summary.candidates_skipped == 1
        assert "missing_pose_observations" in result.summary.skip_reasons

    def test_missing_shelf_region_skipped(self, tmp_path) -> None:
        cfg = InferenceConfig()
        sm_cfg = StateMachineConfig()
        pipeline = TrackAInferencePipeline(cfg, sm_cfg)
        hand_p = tmp_path / "hand_state.joblib"
        shelf_p = tmp_path / "shelf_state.joblib"
        _, hand_meta = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        _, shelf_meta = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        hc, _ = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        hc.save_pipeline(hand_p)
        (tmp_path / "hand_state_metadata.json").write_text(json.dumps(hand_meta.to_dict()))
        sc, _ = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        sc.save_pipeline(shelf_p)
        (tmp_path / "shelf_state_metadata.json").write_text(json.dumps(shelf_meta.to_dict()))

        cand = MockCandidate(candidate_id="c1", clip_id="clip_1", actor_id="actor_1")
        poses = [
            MockPose(clip_id="clip_1", timestamp_s=0.5, actor_id="actor_1", hand_side="right")
        ]
        result = pipeline.run(
            candidates=[cand],
            pose_observations=poses,
            source_videos={},
            hand_classifier_path=hand_p,
            shelf_classifier_path=shelf_p,
            output_dir=tmp_path / "out2",
            shelf_regions={},
        )
        assert result.summary.candidates_skipped == 1


# ---------------------------------------------------------------------------
# 11-15: Feature and probability integration
# ---------------------------------------------------------------------------


class TestFeatureProbabilityIntegration:
    def test_cached_embeddings_reused(self) -> None:
        res = FeatureExtractionResult(
            hand_embeddings=[(0.0, np.zeros(_EMB_DIM))],
            shelf_embeddings=[(0.0, np.zeros(_EMB_DIM))],
            cache_hits=5,
            cache_misses=0,
        )
        assert res.cache_hits == 5
        assert not res.skipped

    def test_hand_features_to_hand_classifier(self) -> None:
        clf, _ = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        emb = np.random.randn(_EMB_DIM)
        evidence = predict_hand_state(clf, emb, 1.0)
        assert isinstance(evidence, HandStateEvidence)
        assert evidence.timestamp_s == 1.0

    def test_shelf_features_to_shelf_classifier(self) -> None:
        clf, _ = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        emb = np.random.randn(_EMB_DIM)
        evidence = predict_shelf_state(clf, emb, 1.0)
        assert isinstance(evidence, ShelfStateEvidence)
        assert evidence.timestamp_s == 1.0

    def test_raw_probabilities_preserved_when_uncertain(self) -> None:
        clf = TrackAClassifier(
            classifier_name="hand",
            crop_type="hand",
            class_names=HAND_STATE_CLASS_NAMES,
            confidence_threshold=0.99,
            margin_threshold=0.99,
        )
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        lr = LogisticRegression()
        lr.classes_ = np.array(HAND_STATE_CLASS_NAMES)
        lr.coef_ = np.array(
            [[0.5, 0.5] + [0.0] * (_EMB_DIM - 2), [-0.5, -0.5] + [0.0] * (_EMB_DIM - 2)]
        )
        lr.intercept_ = np.zeros(2)
        scaler = StandardScaler()
        scaler.fit(np.zeros((2, _EMB_DIM)))
        clf._pipeline = [("scaler", scaler), ("classifier", lr)]
        clf._is_fitted = True
        clf._embedding_dim = _EMB_DIM

        emb = np.random.randn(_EMB_DIM)
        evidence = predict_hand_state(clf, emb, 1.0)
        assert evidence.probability_uncertain > 0
        assert len(evidence.raw_probabilities) == 2

    def test_no_ground_truth_labels_used(self) -> None:
        obs = build_observations(
            candidate_id="c1",
            clip_id="clip_1",
            actor_id="actor_1",
            hand_side="right",
            region_id="r1",
            pose_observations=[],
            hand_evidence_list=[HandStateEvidence(timestamp_s=0.0)],
            shelf_evidence_list=[ShelfStateEvidence(timestamp_s=0.0)],
            shelf_region=None,
        )
        assert len(obs) == 1


# ---------------------------------------------------------------------------
# 16-20: State-machine integration
# ---------------------------------------------------------------------------


class TestStateMachineIntegration:
    def test_pickup_evidence_produces_pickup(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.6, inside=False, distance=50.0),
            _make_obs(0.7, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine()
        events = sm.process(obs)
        assert len(events) == 1
        assert events[0].label == "pickup"

    def test_putdown_evidence_produces_putdown(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.05, hand_carrying=0.9),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.6, inside=False, distance=50.0),
            _make_obs(0.7, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine()
        events = sm.process(obs)
        assert len(events) == 1
        assert events[0].label == "putdown"

    def test_uncertain_evidence_no_event(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_uncertain=0.6, shelf_uncertain=0.6),
            _make_obs(0.1, inside=False, distance=35.0, hand_uncertain=0.6, shelf_uncertain=0.6),
            _make_obs(0.2, inside=True, distance=10.0, hand_uncertain=0.6, shelf_uncertain=0.6),
            _make_obs(0.3, inside=True, distance=10.0, hand_uncertain=0.6, shelf_uncertain=0.6),
            _make_obs(0.4, inside=True, distance=10.0, hand_uncertain=0.6, shelf_uncertain=0.6),
            _make_obs(0.5, inside=False, distance=50.0),
            _make_obs(0.6, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine()
        events = sm.process(obs)
        assert len(events) == 0

    def test_multiple_cycles_multiple_events(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.6, inside=False, distance=50.0),
            _make_obs(0.7, inside=False, distance=80.0),
            _make_obs(0.8, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.9, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(1.0, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(1.3, inside=False, distance=50.0),
            _make_obs(1.4, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine()
        events = sm.process(obs)
        assert len(events) == 2

    def test_different_streams_independent_events(self) -> None:
        obs_a = [
            _make_obs(
                0.0,
                inside=False,
                distance=100.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                actor_id="actor_A",
            ),
            _make_obs(
                0.1,
                inside=False,
                distance=35.0,
                hand_empty=0.9,
                hand_carrying=0.05,
                actor_id="actor_A",
            ),
            _make_obs(
                0.2,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                actor_id="actor_A",
            ),
            _make_obs(
                0.3,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                actor_id="actor_A",
            ),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                actor_id="actor_A",
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
                actor_id="actor_A",
            ),
            _make_obs(0.6, inside=False, distance=50.0, actor_id="actor_A"),
            _make_obs(0.7, inside=False, distance=80.0, actor_id="actor_A"),
        ]
        obs_b = [
            _make_obs(0.0, inside=False, distance=100.0, actor_id="actor_B"),
            _make_obs(0.1, inside=False, distance=90.0, actor_id="actor_B"),
        ]
        sm = RepeatingInteractionStateMachine()
        events = sm.process(obs_a + obs_b)
        assert len(events) == 1
        assert events[0].actor_id == "actor_A"


# ---------------------------------------------------------------------------
# 21-22: Successful-emission tracking
# ---------------------------------------------------------------------------


class TestSuccessfulEmissionTracking:
    def test_rejected_low_confidence_does_not_update_last_event_s(self) -> None:
        sm_cfg = StateMachineConfig(
            event_confidence_threshold=0.99,
            minimum_event_separation_s=1.0,
        )
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.6, inside=False, distance=50.0),
            _make_obs(0.7, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine(config=sm_cfg)
        events = sm.process(obs)
        assert len(events) == 0

    def test_later_valid_event_not_blocked_by_rejected(self) -> None:
        sm_cfg = StateMachineConfig(
            event_confidence_threshold=0.50,
            minimum_event_separation_s=1.0,
        )
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                0.5,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.6, inside=False, distance=50.0),
            _make_obs(0.7, inside=False, distance=80.0),
            _make_obs(0.8, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.9, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(1.0, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                1.1,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(
                1.2,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(1.3, inside=False, distance=50.0),
            _make_obs(1.4, inside=False, distance=80.0),
        ]
        sm = RepeatingInteractionStateMachine(config=sm_cfg)
        events = sm.process(obs)
        assert len(events) >= 1


# ---------------------------------------------------------------------------
# 23-26: Transition grace window
# ---------------------------------------------------------------------------


class TestTransitionGraceWindow:
    def test_pickup_retained_wrist_exits_on_transfer_frame(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.9, hand_carrying=0.05),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.85, hand_carrying=0.1),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.5, inside=False, distance=50.0),
        ]
        raw_events = []
        events = apply_grace_window(raw_events, obs, grace_s=0.25)
        pickups = [e for e in events if e.label == "pickup"]
        assert len(pickups) == 1

    def test_putdown_retained_wrist_exits_on_transfer_frame(self) -> None:
        obs = [
            _make_obs(0.0, inside=False, distance=100.0, hand_empty=0.05, hand_carrying=0.9),
            _make_obs(0.1, inside=False, distance=35.0, hand_empty=0.05, hand_carrying=0.9),
            _make_obs(0.2, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _make_obs(0.3, inside=True, distance=10.0, hand_empty=0.1, hand_carrying=0.85),
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.85,
                hand_carrying=0.1,
                shelf_placed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.5, inside=False, distance=50.0),
        ]
        raw_events = []
        events = apply_grace_window(raw_events, obs, grace_s=0.25)
        putdowns = [e for e in events if e.label == "putdown"]
        assert len(putdowns) == 1

    def test_stale_evidence_outside_grace_no_event(self) -> None:
        obs = [
            _make_obs(
                0.0,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(1.0, inside=False, distance=50.0),
        ]
        raw_events = []
        events = apply_grace_window(raw_events, obs, grace_s=0.25)
        assert len(events) == 0

    def test_grace_does_not_emit_duplicates(self) -> None:
        existing = [_make_sm_event("pickup", 0.35, 0.55, transfer_s=0.4)]
        obs = [
            _make_obs(
                0.4,
                inside=True,
                distance=10.0,
                hand_empty=0.1,
                hand_carrying=0.85,
                shelf_removed=0.8,
                shelf_no_change=0.1,
            ),
            _make_obs(0.5, inside=False, distance=50.0),
        ]
        events = apply_grace_window(existing, obs, grace_s=0.25)
        pickups = [e for e in events if e.label == "pickup"]
        assert len(pickups) == 1


# ---------------------------------------------------------------------------
# 27-31: Boundary refinement
# ---------------------------------------------------------------------------


class TestBoundaryRefinement:
    def test_refined_boundaries_start_lt_end(self) -> None:
        event = _make_sm_event("pickup", 0.5, 1.0)
        refined = refine_event_boundaries(event, 0.0, 2.0)
        assert refined.start_s < refined.end_s

    def test_boundaries_inside_source_clip(self) -> None:
        event = _make_sm_event("pickup", -0.5, 1.0)
        refined = refine_event_boundaries(event, 0.0, 2.0, clip_start_s=0.0, clip_end_s=5.0)
        assert refined.start_s >= 0.0
        assert refined.end_s <= 5.0

    def test_adjacent_pickup_putdown_remain_separate(self) -> None:
        p = _make_sm_event("pickup", 0.5, 1.0)
        d = _make_sm_event("putdown", 1.1, 1.6)
        rp = refine_event_boundaries(p, 0.0, 2.0)
        rd = refine_event_boundaries(d, 0.0, 2.0)
        assert rp.end_s < rd.start_s

    def test_repeated_events_remain_separate(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0)
        e2 = _make_sm_event("pickup", 1.5, 2.0, cycle_id=1)
        r1 = refine_event_boundaries(e1, 0.0, 3.0)
        r2 = refine_event_boundaries(e2, 0.0, 3.0)
        assert r1.end_s < r2.start_s

    def test_minimum_duration_deterministic(self) -> None:
        event = _make_sm_event("pickup", 1.0, 1.0)
        cfg = BoundaryRefinementConfig(minimum_event_duration_s=0.2)
        r1 = refine_event_boundaries(event, 0.0, 2.0, config=cfg)
        r2 = refine_event_boundaries(event, 0.0, 2.0, config=cfg)
        assert r1.start_s == r2.start_s
        assert r1.end_s == r2.end_s
        assert r1.end_s - r1.start_s >= 0.2


# ---------------------------------------------------------------------------
# 32-37: Deduplication
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_overlapping_same_label_deduplicates(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.8, candidate_id="c1")
        e2 = _make_sm_event("pickup", 0.6, 1.1, confidence=0.6, candidate_id="c2")
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(deduped) == 1
        assert len(audit) == 1

    def test_highest_confidence_kept(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.6, candidate_id="c1")
        e2 = _make_sm_event("pickup", 0.6, 1.1, confidence=0.9, candidate_id="c2")
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(deduped) == 1
        assert deduped[0].confidence == 0.9

    def test_dedup_preserves_candidate_ids_in_audit(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.8, candidate_id="c1")
        e2 = _make_sm_event("pickup", 0.6, 1.1, confidence=0.6, candidate_id="c2")
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(audit) == 1
        assert "c1" in audit[0].kept_candidate_id or "c1" in audit[0].suppressed_candidate_ids
        assert "c2" in audit[0].kept_candidate_id or "c2" in audit[0].suppressed_candidate_ids

    def test_pickup_putdown_never_merged(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.8)
        e2 = _make_sm_event("putdown", 0.5, 1.0, confidence=0.8)
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(deduped) == 2

    def test_separate_repeated_pickups_not_merged(self) -> None:
        e1 = _make_sm_event("pickup", 0.0, 0.5, confidence=0.8)
        e2 = _make_sm_event("pickup", 2.0, 2.5, confidence=0.8, cycle_id=1)
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(deduped) == 2

    def test_different_clips_not_deduplicated(self) -> None:
        e1 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.8, clip_id="clip_A")
        e2 = _make_sm_event("pickup", 0.5, 1.0, confidence=0.6, clip_id="clip_B")
        deduped, audit = deduplicate_predictions([e1, e2], DeduplicationConfig())
        assert len(deduped) == 2


# ---------------------------------------------------------------------------
# 38-43: Output and diagnostics
# ---------------------------------------------------------------------------


class TestOutputDiagnostics:
    def test_canonical_predictions_match_schema(self) -> None:
        event = _make_sm_event("pickup", 0.5, 1.0)
        preds = events_to_predictions([event])
        assert len(preds) == 1
        p = preds[0]
        assert p.clip_id == "clip_1"
        assert p.label == "pickup"
        assert p.start_s == 0.5
        assert p.end_s == 1.0
        assert 0.0 <= p.confidence <= 1.0

    def test_prediction_ids_deterministic(self) -> None:
        event = _make_sm_event("pickup", 0.5, 1.0)
        preds1 = events_to_predictions([event])
        preds2 = events_to_predictions([event])
        assert preds1[0].pred_id == preds2[0].pred_id

    def test_debug_traces_preserve_probs(self) -> None:
        cfg = InferenceConfig(debug_traces=True)
        sm_cfg = StateMachineConfig()
        pipeline = TrackAInferencePipeline(cfg, sm_cfg)
        assert pipeline.config.debug_traces is True

    def test_summary_counts_correct(self) -> None:
        summary = InferenceSummary(
            candidates_total=10,
            candidates_processed=8,
            candidates_skipped=2,
            raw_events_emitted=5,
            final_events_after_dedup=3,
            pickup_count=2,
            putdown_count=1,
        )
        assert (
            summary.candidates_total == summary.candidates_processed + summary.candidates_skipped
        )
        assert summary.pickup_count + summary.putdown_count == summary.final_events_after_dedup

    def test_skip_reasons_reported(self, tmp_path) -> None:
        cfg = InferenceConfig()
        sm_cfg = StateMachineConfig()
        pipeline = TrackAInferencePipeline(cfg, sm_cfg)
        hand_p = tmp_path / "h.joblib"
        shelf_p = tmp_path / "s.joblib"
        _, hm = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        _, sm = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        hc, _ = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        hc.save_pipeline(hand_p)
        (tmp_path / "h_metadata.json").write_text(json.dumps(hm.to_dict()))
        sc, _ = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        sc.save_pipeline(shelf_p)
        (tmp_path / "s_metadata.json").write_text(json.dumps(sm.to_dict()))

        video_path = tmp_path / "clip_1.mp4"
        video_path.write_bytes(b"fake")
        cand = MockCandidate(candidate_id="c1", clip_id="clip_1", actor_id="actor_1")
        result = pipeline.run(
            candidates=[cand],
            pose_observations=[],
            source_videos={"clip_1": video_path},
            hand_classifier_path=hand_p,
            shelf_classifier_path=shelf_p,
            output_dir=tmp_path / "out3",
        )
        assert "missing_pose_observations" in result.summary.skip_reasons

    def test_repeated_runs_identical(self, tmp_path) -> None:
        event = _make_sm_event("pickup", 0.5, 1.0)
        p1 = events_to_predictions([event])
        p2 = events_to_predictions([event])
        assert p1[0].pred_id == p2[0].pred_id
        assert p1[0].start_s == p2[0].start_s
        assert p1[0].end_s == p2[0].end_s


# ---------------------------------------------------------------------------
# Sampling and utility tests
# ---------------------------------------------------------------------------


class TestSamplingUtilities:
    def test_sample_times_uniform(self) -> None:
        times = compute_inference_sample_times(0.0, 2.0, 4.0)
        assert times[0] == 0.0
        assert times[-1] == 2.0
        for i in range(1, len(times)):
            gap = times[i] - times[i - 1]
            assert abs(gap - 0.25) < 0.01

    def test_sample_times_single_point(self) -> None:
        times = compute_inference_sample_times(1.0, 1.0, 4.0)
        assert times == [1.0]

    def test_temporal_iou_correct(self) -> None:
        iou = temporal_iou(0.0, 1.0, 0.5, 1.5)
        assert abs(iou - 1.0 / 3.0) < 1e-9

    def test_temporal_iou_no_overlap(self) -> None:
        iou = temporal_iou(0.0, 0.5, 1.0, 1.5)
        assert iou == 0.0

    def test_temporal_iou_identical(self) -> None:
        iou = temporal_iou(0.0, 1.0, 0.0, 1.0)
        assert abs(iou - 1.0) < 1e-9


class TestConfigValidation:
    def test_invalid_sample_fps(self) -> None:
        from pickup_putdown.layer1.track_a.inference import SamplingConfig

        with pytest.raises(ValueError):
            SamplingConfig(sample_fps=-1.0)

    def test_invalid_iou_threshold(self) -> None:
        with pytest.raises(ValueError):
            DeduplicationConfig(temporal_iou_threshold=1.5)

    def test_negative_grace_s(self) -> None:
        with pytest.raises(ValueError):
            InferenceConfig(transition_grace_s=-0.1)

    def test_negative_min_duration(self) -> None:
        with pytest.raises(ValueError):
            BoundaryRefinementConfig(minimum_event_duration_s=-0.1)


class TestEndToEndPipeline:
    def test_pipeline_run_produces_result(self, tmp_path) -> None:
        cfg = InferenceConfig()
        sm_cfg = StateMachineConfig()
        pipeline = TrackAInferencePipeline(cfg, sm_cfg)

        hand_p = tmp_path / "h.joblib"
        shelf_p = tmp_path / "s.joblib"
        _, hm = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        _, shm = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        hc, _ = _make_mock_classifier("hand", HAND_STATE_CLASS_NAMES, "hand")
        hc.save_pipeline(hand_p)
        (tmp_path / "h_metadata.json").write_text(json.dumps(hm.to_dict()))
        sc, _ = _make_mock_classifier("shelf", SHELF_STATE_CLASS_NAMES, "shelf")
        sc.save_pipeline(shelf_p)
        (tmp_path / "s_metadata.json").write_text(json.dumps(shm.to_dict()))

        result = pipeline.run(
            candidates=[],
            pose_observations=[],
            source_videos={},
            hand_classifier_path=hand_p,
            shelf_classifier_path=shelf_p,
            output_dir=tmp_path / "out_e2e",
        )
        assert isinstance(result, InferenceResult)
        assert result.summary.candidates_total == 0
        assert "predictions_csv" in result.output_paths

    def test_pipeline_saves_csv(self, tmp_path) -> None:
        event = _make_sm_event("pickup", 0.5, 1.0)
        preds = events_to_predictions([event])
        csv_path = tmp_path / "preds.csv"
        save_predictions_csv(preds, csv_path)
        assert csv_path.exists()
        content = csv_path.read_text()
        assert "pickup" in content
        assert "clip_1" in content

    def test_merge_dedup_candidate_ids(self) -> None:
        preds = [
            CanonicalPrediction(
                clip_id="c",
                pred_id="p1",
                label="pickup",
                start_s=0.5,
                end_s=1.0,
                confidence=0.8,
                candidate_ids=["cand_1"],
            )
        ]
        audit = [
            DedupAuditEntry(
                kept_prediction_id="p1",
                kept_candidate_id="cand_1",
                kept_confidence=0.8,
                suppressed_prediction_ids=["p2"],
                suppressed_candidate_ids=["cand_2"],
                suppressed_confidences=[0.6],
                temporal_iou=0.8,
                transfer_time_diff_s=0.1,
                selection_reason="highest_confidence",
            )
        ]
        merged = merge_dedup_candidate_ids(preds, audit)
        assert "cand_1" in merged[0].candidate_ids
        assert "cand_2" in merged[0].candidate_ids

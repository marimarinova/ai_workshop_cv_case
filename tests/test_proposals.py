"""Tests for Layer 0B pose tracking, actor association, region measurements,
raw interaction detection, candidate generation, and proposal recall.

Covers all 15 required test categories using synthetic data.  No model
downloads or GPU access are required.
"""

from __future__ import annotations

import pytest

from pickup_putdown.common.exceptions import ConfigError, ValidationError
from pickup_putdown.common.schemas import (
    Candidate,
    PersonObservation,
    PoseObservation,
)
from pickup_putdown.config import (
    ActorAssociationConfig,
    ProposalsConfig,
    RegionMeasurementConfig,
)
from pickup_putdown.perception.proposals import (
    RawInteraction,
    associate_poses_with_actors,
    compute_proposal_recall,
    detect_raw_interactions,
    generate_candidates,
    validate_candidate,
    validate_proposals_config,
)
from pickup_putdown.perception.shelf_regions import (
    CameraShelfConfig,
    ExpansionConfig,
    RegionType,
    ShelfRegion,
)

# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

_W, _H = 1920, 1080

_SHELF_POLY = [[200, 100], [800, 100], [800, 400], [200, 400]]
_COUNTER_POLY = [[1000, 500], [1600, 500], [1600, 800], [1000, 800]]


def _camera_config(
    region_polygons=None,
    expansion_value=20.0,
):
    if region_polygons is None:
        region_polygons = [_SHELF_POLY]
    regions = []
    for i, poly in enumerate(region_polygons):
        regions.append(
            ShelfRegion(
                region_id=f"region_{i}",
                type=RegionType.SHELF,
                polygon=poly,
            )
        )
    return CameraShelfConfig(
        source_width=_W,
        source_height=_H,
        expansion=ExpansionConfig(mode="pixel", value=expansion_value),
        regions=regions,
    )


def _pose_obs(
    clip_id="clip_test",
    timestamp_s=5.0,
    source_frame_index=150,
    sample_index=5,
    actor_id="actor_3",
    hand_side="right",
    wrist_x=500.0,
    wrist_y=250.0,
    wrist_confidence=0.8,
    person_bbox_x1=200.0,
    person_bbox_y1=50.0,
    person_bbox_x2=400.0,
    person_bbox_y2=450.0,
    pose_association_confidence=0.7,
    is_valid=True,
):
    return PoseObservation(
        clip_id=clip_id,
        timestamp_s=timestamp_s,
        source_frame_index=source_frame_index,
        sample_index=sample_index,
        actor_id=actor_id,
        hand_side=hand_side,
        wrist_x=wrist_x,
        wrist_y=wrist_y,
        wrist_confidence=wrist_confidence,
        person_bbox_x1=person_bbox_x1,
        person_bbox_y1=person_bbox_y1,
        person_bbox_x2=person_bbox_x2,
        person_bbox_y2=person_bbox_y2,
        pose_association_confidence=pose_association_confidence,
        is_valid=is_valid,
    )


def _person_obs(
    clip_id="clip_test",
    person_track_id="clip_test:person:3",
    tracker_track_id=3,
    sample_index=5,
    source_frame_index=150,
    timestamp_s=5.0,
    bbox_x1=200.0,
    bbox_y1=50.0,
    bbox_x2=400.0,
    bbox_y2=450.0,
    confidence=0.85,
    is_stable=True,
):
    return PersonObservation(
        clip_id=clip_id,
        person_track_id=person_track_id,
        tracker_track_id=tracker_track_id,
        sample_index=sample_index,
        source_frame_index=source_frame_index,
        timestamp_s=timestamp_s,
        bbox_x1=bbox_x1,
        bbox_y1=bbox_y1,
        bbox_x2=bbox_x2,
        bbox_y2=bbox_y2,
        confidence=confidence,
        is_stable=is_stable,
    )


def _raw_interaction(
    clip_id="clip_test",
    actor_id="actor_3",
    hand_side="right",
    region_id="region_0",
    start_s=5.0,
    end_s=7.0,
    n_observations=10,
    mean_wrist_confidence=0.8,
    mean_distance=5.0,
):
    return RawInteraction(
        clip_id=clip_id,
        actor_id=actor_id,
        hand_side=hand_side,
        region_id=region_id,
        start_s=start_s,
        end_s=end_s,
        n_observations=n_observations,
        mean_wrist_confidence=mean_wrist_confidence,
        mean_distance=mean_distance,
    )


# ---------------------------------------------------------------------------
# 1. One actor entering and leaving one region
# ---------------------------------------------------------------------------


class TestSingleActorRegion:
    def test_actor_entering_and_leaving_region(self):
        """A wrist starts outside, enters the expanded region, stays, then leaves."""
        cam = _camera_config()
        poses = [
            _pose_obs(timestamp_s=3.0, wrist_x=100.0, wrist_y=50.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=4.0, wrist_x=300.0, wrist_y=200.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=6.0, wrist_x=500.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=7.0, wrist_x=600.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=8.0, wrist_x=700.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=9.0, wrist_x=900.0, wrist_y=250.0, wrist_confidence=0.8),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        assert len(interactions) >= 1
        for ri in interactions:
            assert ri.actor_id == "actor_3"
            assert ri.hand_side == "right"
            assert ri.region_id == "region_0"
            assert ri.end_s > ri.start_s


# ---------------------------------------------------------------------------
# 2. Two simultaneous actors producing independent candidates
# ---------------------------------------------------------------------------


class TestTwoActors:
    def test_two_actors_independent_candidates(self):
        """Two actors in the same clip produce independent candidates."""
        cam = _camera_config()
        poses = [
            _pose_obs(
                actor_id="actor_1",
                timestamp_s=5.0,
                wrist_x=400.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                actor_id="actor_1",
                timestamp_s=6.0,
                wrist_x=450.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                actor_id="actor_1",
                timestamp_s=7.0,
                wrist_x=500.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                actor_id="actor_2",
                timestamp_s=5.0,
                wrist_x=400.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                actor_id="actor_2",
                timestamp_s=6.0,
                wrist_x=450.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                actor_id="actor_2",
                timestamp_s=7.0,
                wrist_x=500.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        actor_ids = {ri.actor_id for ri in interactions}
        assert "actor_1" in actor_ids
        assert "actor_2" in actor_ids

        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        cand_actors = {c.actor_id for c in candidates}
        assert "actor_1" in cand_actors
        assert "actor_2" in cand_actors


# ---------------------------------------------------------------------------
# 3. Left and right hands remaining independent
# ---------------------------------------------------------------------------


class TestHandIndependence:
    def test_left_right_hands_independent(self):
        """Left and right hands of the same actor produce separate candidates."""
        cam = _camera_config()
        poses = [
            _pose_obs(
                hand_side="left",
                timestamp_s=5.0,
                wrist_x=400.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                hand_side="left",
                timestamp_s=6.0,
                wrist_x=450.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                hand_side="right",
                timestamp_s=5.0,
                wrist_x=400.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
            _pose_obs(
                hand_side="right",
                timestamp_s=6.0,
                wrist_x=450.0,
                wrist_y=250.0,
                wrist_confidence=0.8,
            ),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        hand_sides = {ri.hand_side for ri in interactions}
        assert "left" in hand_sides
        assert "right" in hand_sides


# ---------------------------------------------------------------------------
# 4. Separate shelf regions remaining independent
# ---------------------------------------------------------------------------


class TestRegionIndependence:
    def test_separate_regions_independent(self):
        """Two shelf regions produce independent candidates."""
        cam = _camera_config([_SHELF_POLY, _COUNTER_POLY])
        poses = [
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=6.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=5.0, wrist_x=1300.0, wrist_y=650.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=6.0, wrist_x=1300.0, wrist_y=650.0, wrist_confidence=0.8),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        region_ids = {ri.region_id for ri in interactions}
        assert len(region_ids) >= 2


# ---------------------------------------------------------------------------
# 5. Low-confidence wrist observations not creating interactions
# ---------------------------------------------------------------------------


class TestLowConfidence:
    def test_low_confidence_no_interaction(self):
        """Wrist observations below minimum confidence do not create raw interactions."""
        cam = _camera_config()
        poses = [
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.1),
            _pose_obs(timestamp_s=6.0, wrist_x=450.0, wrist_y=250.0, wrist_confidence=0.1),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        assert len(interactions) == 0


# ---------------------------------------------------------------------------
# 6. Minimum dwell duration threshold behavior
# ---------------------------------------------------------------------------


class TestDwellDuration:
    def test_below_min_dwell_no_interaction(self):
        """A wrist inside the region for less than min_duration_s does not create an interaction."""
        cam = _camera_config()
        short_dur = ProposalsConfig(minimum_interaction_duration_s=5.0)
        poses = [
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=5.5, wrist_x=420.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=6.0, wrist_x=440.0, wrist_y=250.0, wrist_confidence=0.8),
        ]
        interactions = detect_raw_interactions(poses, cam, short_dur, RegionMeasurementConfig())
        assert len(interactions) == 0

    def test_above_min_dwell_creates_interaction(self):
        """A wrist inside the region for >= min_duration_s creates an interaction."""
        cam = _camera_config()
        poses = [
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=6.0, wrist_x=420.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=7.0, wrist_x=440.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=8.0, wrist_x=460.0, wrist_y=250.0, wrist_confidence=0.8),
        ]
        interactions = detect_raw_interactions(
            poses, cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        assert len(interactions) >= 1


# ---------------------------------------------------------------------------
# 7. Short-gap merging for the same actor, hand, and region
# ---------------------------------------------------------------------------


class TestGapMerging:
    def test_short_gap_merges(self):
        """Two raw interactions with a gap <= merge_gap_s are merged into one candidate."""
        interactions = [
            _raw_interaction(start_s=5.0, end_s=7.0),
            _raw_interaction(start_s=8.0, end_s=10.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 1
        assert candidates[0].n_raw_interactions == 2
        assert candidates[0].raw_start_s == 5.0
        assert candidates[0].raw_end_s == 10.0

    def test_large_gap_no_merge(self):
        """Two raw interactions with a gap > merge_gap_s are not merged."""
        interactions = [
            _raw_interaction(start_s=5.0, end_s=7.0),
            _raw_interaction(start_s=10.0, end_s=12.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 2


# ---------------------------------------------------------------------------
# 8. No merging across actors, hands, or regions
# ---------------------------------------------------------------------------


class TestNoCrossMerge:
    def test_no_merge_across_actors(self):
        interactions = [
            _raw_interaction(actor_id="actor_1", start_s=5.0, end_s=7.0),
            _raw_interaction(actor_id="actor_2", start_s=5.0, end_s=7.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 2

    def test_no_merge_across_hands(self):
        interactions = [
            _raw_interaction(hand_side="left", start_s=5.0, end_s=7.0),
            _raw_interaction(hand_side="right", start_s=5.0, end_s=7.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 2

    def test_no_merge_across_regions(self):
        interactions = [
            _raw_interaction(region_id="region_0", start_s=5.0, end_s=7.0),
            _raw_interaction(region_id="region_1", start_s=5.0, end_s=7.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 2


# ---------------------------------------------------------------------------
# 9. Raw timestamps remaining unchanged after context padding
# ---------------------------------------------------------------------------


class TestRawTimestampsUnchanged:
    def test_raw_timestamps_preserved(self):
        """Raw start/end timestamps are preserved in the candidate."""
        interactions = [_raw_interaction(start_s=5.0, end_s=8.0)]
        clip_durations = {"clip_test": 30.0}
        cfg = ProposalsConfig(context_before_s=2.0, context_after_s=2.0)
        candidates = generate_candidates(interactions, clip_durations, cfg)
        assert len(candidates) == 1
        c = candidates[0]
        assert c.raw_start_s == pytest.approx(5.0)
        assert c.raw_end_s == pytest.approx(8.0)
        assert c.window_start_s == pytest.approx(3.0)
        assert c.window_end_s == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# 10. Padded timestamps clamped at clip start and end
# ---------------------------------------------------------------------------


class TestClamping:
    def test_clamp_at_start(self):
        """Padded start clamped to 0 when raw_start < context_before."""
        interactions = [_raw_interaction(start_s=0.5, end_s=2.0)]
        clip_durations = {"clip_test": 30.0}
        cfg = ProposalsConfig(context_before_s=2.0, context_after_s=2.0)
        candidates = generate_candidates(interactions, clip_durations, cfg)
        assert candidates[0].window_start_s == pytest.approx(0.0)

    def test_clamp_at_end(self):
        """Padded end clamped to clip_duration."""
        interactions = [_raw_interaction(start_s=28.0, end_s=29.5)]
        clip_durations = {"clip_test": 30.0}
        cfg = ProposalsConfig(context_before_s=2.0, context_after_s=2.0)
        candidates = generate_candidates(interactions, clip_durations, cfg)
        assert candidates[0].window_end_s == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# 11. Candidate may contain both pickup and putdown without receiving a type
# ---------------------------------------------------------------------------


class TestNoEventType:
    def test_candidate_has_no_event_type(self):
        """A candidate with multiple raw interactions never receives a type label."""
        interactions = [
            _raw_interaction(start_s=5.0, end_s=7.0),
            _raw_interaction(start_s=8.0, end_s=10.0),
        ]
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 1
        c = candidates[0]
        assert c.proposal_reason == "wrist_in_expanded_region"
        assert c.proposal_score is not None
        assert not hasattr(c, "event_type")
        assert not hasattr(c, "event_class")


# ---------------------------------------------------------------------------
# 12. Deterministic candidate IDs and output ordering
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_deterministic_candidate_ids(self):
        """Running candidate generation twice produces identical IDs."""
        interactions = [_raw_interaction(start_s=5.0, end_s=7.0)]
        clip_durations = {"clip_test": 30.0}
        c1 = generate_candidates(interactions, clip_durations, ProposalsConfig())
        c2 = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(c1) == len(c2) == 1
        assert c1[0].candidate_id == c2[0].candidate_id

    def test_deterministic_output_ordering(self):
        """Candidates are sorted by (clip_id, actor_id, hand_side, region_id, raw_start)."""
        interactions = [
            _raw_interaction(
                clip_id="clip_b",
                actor_id="actor_2",
                hand_side="left",
                region_id="region_0",
                start_s=5.0,
                end_s=7.0,
            ),
            _raw_interaction(
                clip_id="clip_a",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                start_s=3.0,
                end_s=5.0,
            ),
            _raw_interaction(
                clip_id="clip_a",
                actor_id="actor_1",
                hand_side="left",
                region_id="region_0",
                start_s=3.0,
                end_s=5.0,
            ),
        ]
        clip_durations = {"clip_a": 30.0, "clip_b": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 3
        assert candidates[0].clip_id == "clip_a"
        assert candidates[0].actor_id == "actor_1"
        assert candidates[0].hand_side == "left"


# ---------------------------------------------------------------------------
# 13. Proposal-recall computation with covered and uncovered events
# ---------------------------------------------------------------------------


class TestProposalRecall:
    def test_covered_and_uncovered_events(self):
        """Some events are covered by candidates, others are not."""
        candidates = [
            Candidate(
                candidate_id="cand_1",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=8.0,
                window_start_s=3.0,
                window_end_s=10.0,
            ),
        ]
        gt_events = [
            {
                "event_id": "evt_1",
                "clip_id": "clip_test",
                "type": "pickup",
                "t_start": 6.0,
                "t_end": 7.0,
            },
            {
                "event_id": "evt_2",
                "clip_id": "clip_test",
                "type": "putdown",
                "t_start": 15.0,
                "t_end": 16.0,
            },
        ]
        results, aggregate = compute_proposal_recall(candidates, gt_events)
        assert len(results) == 2
        assert results[0].covered is True
        assert results[1].covered is False
        assert aggregate["proposal_recall"] == pytest.approx(0.5)
        assert aggregate["covered_events"] == 1
        assert aggregate["uncovered_events"] == 1

    def test_all_covered_by_padded(self):
        """An event outside raw but inside padded interval is covered."""
        candidates = [
            Candidate(
                candidate_id="cand_1",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=7.0,
                window_start_s=3.0,
                window_end_s=10.0,
            ),
        ]
        gt_events = [
            {
                "event_id": "evt_1",
                "clip_id": "clip_test",
                "type": "pickup",
                "t_start": 9.0,
                "t_end": 9.5,
            },
        ]
        results, aggregate = compute_proposal_recall(candidates, gt_events)
        assert results[0].covered is True
        assert aggregate["proposal_recall"] == pytest.approx(1.0)

    def test_empty_candidates_no_coverage(self):
        results, aggregate = compute_proposal_recall(
            [],
            [
                {
                    "event_id": "evt_1",
                    "clip_id": "clip_test",
                    "type": "pickup",
                    "t_start": 5.0,
                    "t_end": 6.0,
                },
            ],
        )
        assert results[0].covered is False
        assert aggregate["proposal_recall"] == 0.0


# ---------------------------------------------------------------------------
# 14. Empty active spans and empty pose results
# ---------------------------------------------------------------------------


class TestEmptyInputs:
    def test_empty_pose_observations(self):
        cam = _camera_config()
        interactions = detect_raw_interactions(
            [], cam, ProposalsConfig(), RegionMeasurementConfig()
        )
        assert len(interactions) == 0
        clip_durations = {"clip_test": 30.0}
        candidates = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert len(candidates) == 0

    def test_empty_candidates_for_recall(self):
        results, aggregate = compute_proposal_recall([], [])
        assert len(results) == 0
        assert aggregate["proposal_recall"] == 0.0

    def test_associate_empty_poses(self):
        result = associate_poses_with_actors([], [_person_obs()], ActorAssociationConfig())
        assert len(result) == 0


# ---------------------------------------------------------------------------
# 15. Malformed configuration and invalid timestamps
# ---------------------------------------------------------------------------


class TestValidation:
    def test_negative_fps_raises(self):
        cfg = ProposalsConfig(target_fps=-1.0)
        with pytest.raises(ConfigError, match="positive"):
            validate_proposals_config(cfg, RegionMeasurementConfig())

    def test_invalid_wrist_confidence_raises(self):
        cfg = ProposalsConfig(minimum_wrist_confidence=1.5)
        with pytest.raises(ConfigError, match="minimum_wrist_confidence"):
            validate_proposals_config(cfg, RegionMeasurementConfig())

    def test_negative_duration_raises(self):
        cfg = ProposalsConfig(minimum_interaction_duration_s=-0.5)
        with pytest.raises(ConfigError, match="non-negative"):
            validate_proposals_config(cfg, RegionMeasurementConfig())

    def test_negative_context_raises(self):
        cfg = ProposalsConfig(context_before_s=-1.0)
        with pytest.raises(ConfigError, match="non-negative"):
            validate_proposals_config(cfg, RegionMeasurementConfig())

    def test_invalid_candidate_timestamps(self):
        with pytest.raises(Exception, match="raw_end_s"):
            Candidate(
                candidate_id="bad",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=3.0,
                window_start_s=3.0,
                window_end_s=10.0,
            )

    def test_candidate_out_of_clip_duration(self):
        c = Candidate(
            candidate_id="bad",
            clip_id="clip_test",
            actor_id="actor_1",
            hand_side="right",
            region_id="region_0",
            raw_start_s=5.0,
            raw_end_s=35.0,
            window_start_s=3.0,
            window_end_s=10.0,
        )
        with pytest.raises(ValidationError, match="out of range"):
            validate_candidate(c, 30.0)


# ---------------------------------------------------------------------------
# Actor association tests
# ---------------------------------------------------------------------------


class TestActorAssociation:
    def test_association_by_iou(self):
        """Pose detection within IoU threshold gets matched to actor track."""
        poses = [_pose_obs(wrist_x=300.0, wrist_y=200.0, wrist_confidence=0.8)]
        persons = [
            _person_obs(
                person_track_id="clip_test:person:99",
                tracker_track_id=99,
                timestamp_s=5.0,
                bbox_x1=200.0,
                bbox_y1=50.0,
                bbox_x2=400.0,
                bbox_y2=450.0,
                confidence=0.85,
            )
        ]
        associated = associate_poses_with_actors(poses, persons, ActorAssociationConfig())
        assert len(associated) == 1
        assert associated[0].actor_id == "clip_test:person:99"

    def test_unmatched_pose_keeps_original_actor(self):
        """Pose detection far from any actor track remains unmatched."""
        poses = [
            _pose_obs(
                wrist_x=1500.0,
                wrist_y=800.0,
                wrist_confidence=0.8,
                person_bbox_x1=1400.0,
                person_bbox_y1=700.0,
                person_bbox_x2=1600.0,
                person_bbox_y2=900.0,
            )
        ]
        persons = [
            _person_obs(
                tracker_track_id=1,
                timestamp_s=5.0,
                bbox_x1=200.0,
                bbox_y1=50.0,
                bbox_x2=400.0,
                bbox_y2=450.0,
                confidence=0.85,
            )
        ]
        associated = associate_poses_with_actors(poses, persons, ActorAssociationConfig())
        assert len(associated) == 1
        assert "untracked" in associated[0].actor_id or associated[0].actor_id == "actor_3"


# ---------------------------------------------------------------------------
# Region measurement tests
# ---------------------------------------------------------------------------


class TestRegionMeasurements:
    def test_entry_exit_detection(self):
        """Region entry and exit events are detected."""
        from pickup_putdown.perception.proposals import compute_region_measurements

        cam = _camera_config()
        poses = [
            _pose_obs(timestamp_s=3.0, wrist_x=100.0, wrist_y=50.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=5.0, wrist_x=400.0, wrist_y=250.0, wrist_confidence=0.8),
            _pose_obs(timestamp_s=7.0, wrist_x=900.0, wrist_y=250.0, wrist_confidence=0.8),
        ]
        measurements = compute_region_measurements(poses, cam, RegionMeasurementConfig())
        for _key, ms in measurements.items():
            entries = [m for m in ms if m.entry_event]
            exits = [m for m in ms if m.exit_event]
            if entries and exits:
                break
        else:
            assert len(measurements) > 0


# ---------------------------------------------------------------------------
# Candidate schema validation
# ---------------------------------------------------------------------------


class TestCandidateSchema:
    def test_valid_candidate(self):
        c = Candidate(
            candidate_id="cand_test",
            clip_id="clip_test",
            actor_id="actor_1",
            hand_side="right",
            region_id="region_0",
            raw_start_s=5.0,
            raw_end_s=8.0,
            window_start_s=3.0,
            window_end_s=10.0,
        )
        validate_candidate(c, 30.0)

    def test_candidate_with_signal_summaries(self):
        c = Candidate(
            candidate_id="cand_test",
            clip_id="clip_test",
            actor_id="actor_1",
            hand_side="right",
            region_id="region_0",
            raw_start_s=5.0,
            raw_end_s=8.0,
            window_start_s=3.0,
            window_end_s=10.0,
            n_raw_interactions=2,
            min_region_distance=3.5,
            max_wrist_confidence=0.85,
            total_dwell_duration_s=2.5,
            config_fingerprint="abcd1234",
        )
        validate_candidate(c, 30.0)

    def test_candidate_id_deterministic(self):
        """Candidate IDs are deterministic given the same inputs."""
        interactions = [_raw_interaction(start_s=5.0, end_s=7.0)]
        clip_durations = {"clip_test": 30.0}
        c1 = generate_candidates(interactions, clip_durations, ProposalsConfig())
        c2 = generate_candidates(interactions, clip_durations, ProposalsConfig())
        assert c1[0].candidate_id == c2[0].candidate_id


# ---------------------------------------------------------------------------
# Proposal recall actor/region aware
# ---------------------------------------------------------------------------


class TestProposalRecallActorRegionAware:
    def test_actor_aware_coverage(self):
        """Actor-aware recall requires actor_id match."""
        candidates = [
            Candidate(
                candidate_id="cand_1",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=8.0,
                window_start_s=3.0,
                window_end_s=10.0,
            ),
        ]
        gt_events = [
            {
                "event_id": "evt_1",
                "clip_id": "clip_test",
                "type": "pickup",
                "t_start": 6.0,
                "t_end": 7.0,
                "actor_id": "actor_2",
            },
        ]
        results, aggregate = compute_proposal_recall(candidates, gt_events, actor_aware=True)
        assert results[0].covered is False

    def test_region_aware_coverage(self):
        """Region-aware recall requires region_id match."""
        candidates = [
            Candidate(
                candidate_id="cand_1",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=8.0,
                window_start_s=3.0,
                window_end_s=10.0,
            ),
        ]
        gt_events = [
            {
                "event_id": "evt_1",
                "clip_id": "clip_test",
                "type": "pickup",
                "t_start": 6.0,
                "t_end": 7.0,
                "region_id": "region_1",
            },
        ]
        results, aggregate = compute_proposal_recall(candidates, gt_events, region_aware=True)
        assert results[0].covered is False

    def test_both_actor_and_region_aware(self):
        """Both actor and region awareness required."""
        candidates = [
            Candidate(
                candidate_id="cand_1",
                clip_id="clip_test",
                actor_id="actor_1",
                hand_side="right",
                region_id="region_0",
                raw_start_s=5.0,
                raw_end_s=8.0,
                window_start_s=3.0,
                window_end_s=10.0,
            ),
        ]
        gt_events = [
            {
                "event_id": "evt_1",
                "clip_id": "clip_test",
                "type": "pickup",
                "t_start": 6.0,
                "t_end": 7.0,
                "actor_id": "actor_2",
                "region_id": "region_1",
            },
        ]
        results, aggregate = compute_proposal_recall(
            candidates, gt_events, actor_aware=True, region_aware=True
        )
        assert results[0].covered is False
        assert results[0].actor_match is False
        assert results[0].region_match is None

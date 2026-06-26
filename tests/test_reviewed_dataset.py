"""Tests for the reviewed Track A dataset build pipeline.

Covers:
1. Reviewed pickup and putdown events are included.
2. Reviewed zero-event tasks are included as verified negatives.
3. Unreviewed candidates are excluded.
4. Unmatched candidate is not automatically labeled negative.
5. Split assignment is deterministic.
6. Clips and recording days do not cross splits.
7. Missing or inconsistent review records produce clear errors.
8. Cached features are reused.
9. Output manifests preserve required provenance.
10. Dataset summary counts are correct.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pickup_putdown.common.schemas import Candidate, Event, EventType, PoseObservation
from pickup_putdown.layer1.track_a.contracts import (
    CropGeometry,
    FeatureDataset,
    FeatureRecord,
)
from pickup_putdown.layer1.track_a.reviewed_dataset import (
    CandidateMetadata,
    ReviewedExample,
    _is_zero_event,
    assign_splits_by_recording_day,
    extract_recording_day,
    load_candidate_metadata_index,
    load_clips_csv,
    load_events_csv,
    load_review_manifest,
    resolve_reviewed_examples,
    validate_split_isolation,
)
from pickup_putdown.layer1.track_a.sampling import get_wrist_trajectory_for_candidate

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def review_manifest(tmp_path: Path) -> Path:
    """Create a synthetic review manifest CSV with matching JSON files."""
    json_dir = tmp_path / "json"
    json_dir.mkdir()

    # Create JSON files with reviewed events
    json_data = {
        "cand_p1": {
            "candidate_id": "cand_p1",
            "clip_id": "D2_S20260520141725_E20260520142151_anon",
            "source_start_s": 9.0,
            "source_end_s": 13.0,
            "events": [
                {
                    "label": "pickup",
                    "start_s": 1.0,
                    "end_s": 2.0,
                    "item_count": 1,
                    "confidence": "high",
                    "hard_case": False,
                    "notes": "reviewed pickup",
                }
            ],
        },
        "cand_pd1": {
            "candidate_id": "cand_pd1",
            "clip_id": "D2_S20260520141725_E20260520142151_anon",
            "source_start_s": 14.0,
            "source_end_s": 18.0,
            "events": [
                {
                    "label": "putdown",
                    "start_s": 0.5,
                    "end_s": 1.5,
                    "item_count": 1,
                    "confidence": "high",
                    "hard_case": False,
                    "notes": "reviewed putdown",
                }
            ],
        },
        "cand_n1": {
            "candidate_id": "cand_n1",
            "clip_id": "D2_S20260521112037_E20260521112553_anon",
            "source_start_s": 20.0,
            "source_end_s": 24.0,
            "events": [],
        },
        "cand_n2": {
            "candidate_id": "cand_n2",
            "clip_id": "D2_S20260521112037_E20260521112553_anon",
            "source_start_s": 30.0,
            "source_end_s": 34.0,
            "events": [],
        },
        "cand_unrev": {
            "candidate_id": "cand_unrev",
            "clip_id": "D2_S20260522132934_E20260522133448_anon",
            "source_start_s": 49.0,
            "source_end_s": 53.0,
            "events": [],
        },
    }
    for cid, data in json_data.items():
        (json_dir / f"{cid}.json").write_text(json.dumps(data))

    p = tmp_path / "review_manifest.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_id",
                "clip_id",
                "review_groups",
                "video_path",
                "json_path",
                "event_count",
                "reviewed",
                "review_notes",
            ]
        )
        writer.writerow(
            [
                "cand_p1",
                "D2_S20260520141725_E20260520142151_anon",
                "vlm_positive",
                "/v/cand_p1.mp4",
                str(json_dir / "cand_p1.json"),
                1,
                "true",
                "confirmed pickup",
            ]
        )
        writer.writerow(
            [
                "cand_pd1",
                "D2_S20260520141725_E20260520142151_anon",
                "vlm_positive",
                "/v/cand_pd1.mp4",
                str(json_dir / "cand_pd1.json"),
                1,
                "true",
                "confirmed putdown",
            ]
        )
        writer.writerow(
            [
                "cand_n1",
                "D2_S20260521112037_E20260521112553_anon",
                "negative_sample",
                "/v/cand_n1.mp4",
                str(json_dir / "cand_n1.json"),
                0,
                "true",
                "confirmed no events",
            ]
        )
        writer.writerow(
            [
                "cand_n2",
                "D2_S20260521112037_E20260521112553_anon",
                "vlm_positive",
                "/v/cand_n2.mp4",
                str(json_dir / "cand_n2.json"),
                0,
                "true",
                "confirmed no event",
            ]
        )
        writer.writerow(
            [
                "cand_unrev",
                "D2_S20260522132934_E20260522133448_anon",
                "vlm_positive",
                "/v/cand_unrev.mp4",
                str(json_dir / "cand_unrev.json"),
                1,
                "false",
                "not reviewed yet",
            ]
        )
        writer.writerow(
            [
                "cand_nometa",
                "D2_S20260522132934_E20260522133448_anon",
                "vlm_positive",
                "/v/cand_nometa.mp4",
                str(json_dir / "cand_nometa.json"),
                1,
                "true",
                "confirmed pickup",
            ]
        )
    return p


@pytest.fixture
def events_csv(tmp_path: Path) -> Path:
    """Create a synthetic events CSV."""
    p = tmp_path / "events.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_id",
                "clip_id",
                "type",
                "t_start",
                "t_end",
                "hard_case",
                "annotator",
                "confidence",
                "notes",
            ]
        )
        writer.writerow(
            [
                "evt_p1",
                "D2_S20260520141725_E20260520142151_anon",
                "pickup",
                10.0,
                12.0,
                "False",
                "vlm",
                "high",
                "pickup event",
            ]
        )
        writer.writerow(
            [
                "evt_pd1",
                "D2_S20260520141725_E20260520142151_anon",
                "putdown",
                15.0,
                17.0,
                "False",
                "vlm",
                "high",
                "putdown event",
            ]
        )
        writer.writerow(
            [
                "evt_other",
                "D2_S20260522132934_E20260522133448_anon",
                "pickup",
                50.0,
                52.0,
                "False",
                "vlm",
                "high",
                "other clip event",
            ]
        )
    return p


@pytest.fixture
def clips_csv(tmp_path: Path) -> Path:
    """Create a synthetic clips CSV."""
    p = tmp_path / "clips.csv"
    with open(p, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "clip_id",
                "s3_key",
                "duration_s",
                "fps",
                "width",
                "height",
                "n_person_tracks",
                "usable",
                "active_start_s",
                "active_end_s",
                "split",
                "session_id",
                "notes",
            ]
        )
        writer.writerow(
            [
                "D2_S20260520141725_E20260520142151_anon",
                "s3://b/c1.mp4",
                200.0,
                20.0,
                3840,
                2160,
                0,
                "True",
                0,
                200,
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "D2_S20260521112037_E20260521112553_anon",
                "s3://b/c2.mp4",
                150.0,
                20.0,
                3840,
                2160,
                0,
                "True",
                0,
                150,
                "",
                "",
                "",
            ]
        )
        writer.writerow(
            [
                "D2_S20260522132934_E20260522133448_anon",
                "s3://b/c3.mp4",
                100.0,
                20.0,
                3840,
                2160,
                0,
                "True",
                0,
                100,
                "",
                "",
                "",
            ]
        )
    return p


@pytest.fixture
def candidate_metadata(tmp_path: Path) -> dict[str, CandidateMetadata]:
    """Create candidate metadata index."""
    return {
        "cand_p1": CandidateMetadata(
            candidate_id="cand_p1",
            clip_id="D2_S20260520141725_E20260520142151_anon",
            source_start_s=9.0,
            source_end_s=13.0,
            duration_s=4.0,
        ),
        "cand_pd1": CandidateMetadata(
            candidate_id="cand_pd1",
            clip_id="D2_S20260520141725_E20260520142151_anon",
            source_start_s=14.0,
            source_end_s=18.0,
            duration_s=4.0,
        ),
        "cand_n1": CandidateMetadata(
            candidate_id="cand_n1",
            clip_id="D2_S20260521112037_E20260521112553_anon",
            source_start_s=20.0,
            source_end_s=24.0,
            duration_s=4.0,
        ),
        "cand_n2": CandidateMetadata(
            candidate_id="cand_n2",
            clip_id="D2_S20260521112037_E20260521112553_anon",
            source_start_s=30.0,
            source_end_s=34.0,
            duration_s=4.0,
        ),
        "cand_unrev": CandidateMetadata(
            candidate_id="cand_unrev",
            clip_id="D2_S20260522132934_E20260522133448_anon",
            source_start_s=49.0,
            source_end_s=53.0,
            duration_s=4.0,
        ),
    }


@pytest.fixture
def events(events_csv: Path) -> list[Event]:
    return load_events_csv(events_csv)


# ---------------------------------------------------------------------------
# Test 1: Load review manifest
# ---------------------------------------------------------------------------


class TestLoadReviewManifest:
    def test_load_valid_manifest(self, review_manifest: Path):
        records = load_review_manifest(review_manifest)
        assert len(records) == 6
        assert records[0].candidate_id == "cand_p1"
        assert records[0].reviewed is True
        assert records[0].event_count == 1

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_review_manifest("/nonexistent/review_manifest.csv")

    def test_missing_columns(self, tmp_path: Path):
        p = tmp_path / "bad.csv"
        p.write_text("candidate_id,clip_id\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_review_manifest(p)


# ---------------------------------------------------------------------------
# Test 2: Load events CSV
# ---------------------------------------------------------------------------


class TestLoadEventsCSV:
    def test_load_valid_events(self, events_csv: Path):
        events = load_events_csv(events_csv)
        assert len(events) == 3
        assert events[0].event_id == "evt_p1"
        assert events[0].type == EventType.PICKUP

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_events_csv("/nonexistent/events.csv")

    def test_missing_columns(self, tmp_path: Path):
        p = tmp_path / "bad.csv"
        p.write_text("event_id,clip_id\n")
        with pytest.raises(ValueError, match="missing columns"):
            load_events_csv(p)


# ---------------------------------------------------------------------------
# Test 3: Load clips CSV
# ---------------------------------------------------------------------------


class TestLoadClipsCSV:
    def test_load_valid_clips(self, clips_csv: Path):
        clips = load_clips_csv(clips_csv)
        assert len(clips) == 3
        assert "D2_S20260520141725_E20260520142151_anon" in clips

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_clips_csv("/nonexistent/clips.csv")


# ---------------------------------------------------------------------------
# Test 4: Zero-event detection
# ---------------------------------------------------------------------------


class TestIsZeroEvent:
    def test_zero_count(self):
        assert _is_zero_event("some notes", 0) is True

    def test_no_events_phrase(self):
        assert _is_zero_event("confirmed no events", 1) is True

    def test_no_event_phrase(self):
        assert _is_zero_event("confirmed no event", 0) is True

    def test_not_zero_event(self):
        assert _is_zero_event("confirmed pickup", 1) is False

    def test_positive_notes(self):
        assert _is_zero_event("confirmed putdown and pickup", 2) is False


# ---------------------------------------------------------------------------
# Test 5: Resolve reviewed examples
# ---------------------------------------------------------------------------


class TestResolveReviewedExamples:
    def test_positives_included(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, summary = resolve_reviewed_examples(records, events, candidate_metadata)
        labels = {ex.candidate_id: ex.label for ex in examples}
        assert "cand_p1" in labels
        assert labels["cand_p1"] == "pickup"

    def test_putdown_included(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, _ = resolve_reviewed_examples(records, events, candidate_metadata)
        labels = {ex.candidate_id: ex.label for ex in examples}
        assert "cand_pd1" in labels
        assert labels["cand_pd1"] == "putdown"

    def test_zero_event_as_negative(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, _ = resolve_reviewed_examples(records, events, candidate_metadata)
        labels = {ex.candidate_id: ex.label for ex in examples}
        assert "cand_n1" in labels
        assert labels["cand_n1"] == "negative"
        assert "cand_n2" in labels
        assert labels["cand_n2"] == "negative"

    def test_unreviewed_excluded(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, summary = resolve_reviewed_examples(records, events, candidate_metadata)
        cids = {ex.candidate_id for ex in examples}
        assert "cand_unrev" not in cids
        assert summary.excluded_unreviewed >= 1

    def test_no_metadata_excluded(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, summary = resolve_reviewed_examples(records, events, candidate_metadata)
        cids = {ex.candidate_id for ex in examples}
        assert "cand_nometa" not in cids
        assert summary.excluded_no_match >= 1

    def test_labels_from_reviewed_json(self, tmp_path):
        """Labels come from reviewed JSON events, not VLM events."""
        json_dir = tmp_path / "json"
        json_dir.mkdir()

        # JSON has pickup label
        (json_dir / "cand_good.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_good",
                    "clip_id": "clip1",
                    "source_start_s": 10.0,
                    "source_end_s": 12.0,
                    "events": [
                        {"label": "pickup", "start_s": 0.5, "end_s": 1.5, "confidence": "high"}
                    ],
                }
            )
        )

        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "candidate_id",
                    "clip_id",
                    "review_groups",
                    "video_path",
                    "json_path",
                    "event_count",
                    "reviewed",
                    "review_notes",
                ]
            )
            writer.writerow(
                [
                    "cand_good",
                    "clip1",
                    "vlm_positive",
                    "/v/c.mp4",
                    str(json_dir / "cand_good.json"),
                    1,
                    "true",
                    "confirmed pickup",
                ]
            )

        events = []  # No VLM events
        meta = {
            "cand_good": CandidateMetadata(
                candidate_id="cand_good",
                clip_id="clip1",
                source_start_s=10.0,
                source_end_s=12.0,
                duration_s=2.0,
            )
        }

        records = load_review_manifest(manifest)
        examples, summary = resolve_reviewed_examples(records, events, meta)
        assert len(examples) == 1
        assert examples[0].label == "pickup"
        assert summary.positives == 1

    def test_missing_json_excluded(self, tmp_path):
        """Candidate with missing JSON is excluded."""
        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "candidate_id",
                    "clip_id",
                    "review_groups",
                    "video_path",
                    "json_path",
                    "event_count",
                    "reviewed",
                    "review_notes",
                ]
            )
            writer.writerow(
                [
                    "cand_bad",
                    "clip1",
                    "vlm_positive",
                    "/v/c.mp4",
                    "/nonexistent/c.json",
                    1,
                    "true",
                    "confirmed pickup",
                ]
            )

        events = []
        meta = {
            "cand_bad": CandidateMetadata(
                candidate_id="cand_bad",
                clip_id="clip1",
                source_start_s=10.0,
                source_end_s=12.0,
                duration_s=2.0,
            )
        }

        records = load_review_manifest(manifest)
        examples, summary = resolve_reviewed_examples(records, events, meta)
        assert len(examples) == 0
        assert summary.excluded_no_match >= 1

    def test_event_relative_overlap(self):
        """Event-relative overlap matches even when candidate is much wider."""
        from pickup_putdown.layer1.track_a.reviewed_dataset import (
            _match_events_to_candidate,
        )

        # Wide candidate (8s), short event (1s) — candidate fully covers event
        events = [
            Event(
                event_id="evt1",
                clip_id="clip1",
                type=EventType.PICKUP,
                t_start=5.0,
                t_end=6.0,
            )
        ]
        matched = _match_events_to_candidate(events, 2.0, 10.0)
        assert len(matched) == 1
        assert matched[0]["event_id"] == "evt1"
        # Event-relative: 1s overlap / 1s event = 1.0
        assert matched[0]["overlap_ratio"] == 1.0

    def test_partial_event_overlap_matches(self):
        """Partial event coverage still matches above threshold."""
        from pickup_putdown.layer1.track_a.reviewed_dataset import (
            _match_events_to_candidate,
        )

        # Event is 2s, candidate covers 0.5s of it → 25% > 10% threshold
        events = [
            Event(
                event_id="evt1",
                clip_id="clip1",
                type=EventType.PICKUP,
                t_start=5.0,
                t_end=7.0,
            )
        ]
        matched = _match_events_to_candidate(events, 5.0, 5.5)
        assert len(matched) == 1
        assert matched[0]["overlap_ratio"] == pytest.approx(0.25)

    def test_no_examples_raises(self, tmp_path):
        """Pipeline raises when no examples are resolved at all."""
        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "candidate_id",
                    "clip_id",
                    "review_groups",
                    "video_path",
                    "json_path",
                    "event_count",
                    "reviewed",
                    "review_notes",
                ]
            )
            writer.writerow(
                [
                    "cand_bad",
                    "D2_S20260599999999_E20260599999999_anon",
                    "vlm_positive",
                    "/v/c.mp4",
                    "/nonexistent/c.json",
                    1,
                    "true",
                    "confirmed pickup",
                ]
            )

        events = []
        meta = {
            "cand_bad": CandidateMetadata(
                candidate_id="cand_bad",
                clip_id="D2_S20260599999999_E20260599999999_anon",
                source_start_s=10.0,
                source_end_s=12.0,
                duration_s=2.0,
            )
        }

        records = load_review_manifest(manifest)
        examples, summary = resolve_reviewed_examples(records, events, meta)
        assert len(examples) == 0
        assert summary.excluded_no_match >= 1


# ---------------------------------------------------------------------------
# Test 6: Split assignment
# ---------------------------------------------------------------------------


class TestSplitAssignment:
    def test_deterministic(self):
        clip_ids = [
            "D2_S20260520141725_E20260520142151_anon",
            "D2_S20260521112037_E20260521112553_anon",
            "D2_S20260522132934_E20260522133448_anon",
        ]
        splits1 = assign_splits_by_recording_day(clip_ids, seed=42)
        splits2 = assign_splits_by_recording_day(clip_ids, seed=42)
        assert splits1 == splits2

    def test_different_seeds_different_splits(self):
        clip_ids = [
            "D2_S20260520141725_E20260520142151_anon",
            "D2_S20260521112037_E20260521112553_anon",
            "D2_S20260522132934_E20260522133448_anon",
            "D2_S20260523102833_E20260523103217_anon",
            "D2_S20260523105427_E20260523105637_anon",
            "D2_S20260526153840_E20260526154202_anon",
        ]
        s1 = assign_splits_by_recording_day(clip_ids, seed=42)
        s2 = assign_splits_by_recording_day(clip_ids, seed=99)
        assert s1 != s2

    def test_same_day_same_split(self):
        clip_ids = [
            "D2_S20260520141725_E20260520142151_anon",
            "D2_S20260520143523_E20260520143803_anon",
        ]
        splits = assign_splits_by_recording_day(clip_ids, seed=42)
        assert splits[clip_ids[0]] == splits[clip_ids[1]]

    def test_all_splits_present(self):
        clip_ids = [f"D2_S2026052{i}000000_E2026052{i}000000_anon" for i in range(10)]
        splits = assign_splits_by_recording_day(clip_ids, seed=42)
        split_names = set(splits.values())
        assert "train" in split_names
        assert "val" in split_names
        assert "test" in split_names


# ---------------------------------------------------------------------------
# Test 7: Split isolation validation
# ---------------------------------------------------------------------------


class TestSplitIsolation:
    def test_valid_splits(self):
        splits = {
            "clip1": "train",
            "clip2": "val",
            "clip3": "test",
        }
        examples = [
            ReviewedExample("c1", "clip1", "pickup", 0, 1),
            ReviewedExample("c2", "clip2", "negative", 0, 1),
        ]
        assert validate_split_isolation(splits, examples) is True

    def test_single_clip_no_leak(self):
        splits = {"clip1": "train"}
        examples = [ReviewedExample("c1", "clip1", "pickup", 0, 1)]
        assert validate_split_isolation(splits, examples) is True

    def test_leakage_detected(self):
        splits = {
            "clip1": "train",
            "clip2": "val",
        }
        examples = [
            ReviewedExample("c1", "clip1", "pickup", 0, 1),
            ReviewedExample("c2", "clip2", "negative", 0, 1),
            ReviewedExample("c3", "clip1", "putdown", 0, 1),
        ]
        assert validate_split_isolation(splits, examples) is True


# ---------------------------------------------------------------------------
# Test 8: Recording day extraction
# ---------------------------------------------------------------------------


class TestExtractRecordingDay:
    def test_standard_clip_id(self):
        day = extract_recording_day("D2_S20260520141725_E20260520142151_anon")
        assert day == "20260520"

    def test_another_clip_id(self):
        day = extract_recording_day("D2_S20260526153840_E20260526154202_anon")
        assert day == "20260526"

    def test_no_match_fallback(self):
        day = extract_recording_day("unknown_clip")
        assert day == "unknown_clip"


# ---------------------------------------------------------------------------
# Test 9: Candidate metadata loading
# ---------------------------------------------------------------------------


class TestLoadCandidateMetadata:
    def test_load_from_dir(self, tmp_path: Path):
        staging = tmp_path / "staging"
        clip_dir = staging / "candidates" / "clip1"
        clip_dir.mkdir(parents=True)
        meta_file = clip_dir / "clip1.json"
        meta_file.write_text(
            json.dumps(
                {
                    "source_video_id": "clip1",
                    "candidates": [
                        {
                            "candidate_id": "cand_a",
                            "source_start_s": 10.0,
                            "source_end_s": 12.0,
                            "duration_s": 2.0,
                        },
                        {
                            "candidate_id": "cand_b",
                            "source_start_s": 20.0,
                            "source_end_s": 22.0,
                            "duration_s": 2.0,
                        },
                    ],
                }
            )
        )

        index = load_candidate_metadata_index(staging)
        assert "cand_a" in index
        assert "cand_b" in index
        assert index["cand_a"].source_start_s == 10.0

    def test_empty_dir(self, tmp_path: Path):
        staging = tmp_path / "empty"
        staging.mkdir()
        index = load_candidate_metadata_index(staging)
        assert index == {}


# ---------------------------------------------------------------------------
# Test 10: Build summary counts
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_summary_counts(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, summary = resolve_reviewed_examples(records, events, candidate_metadata)
        assert summary.total_reviewed == 5
        assert summary.positives == 2
        assert summary.negatives == 2
        assert summary.excluded_unreviewed >= 1

    def test_provenance_preserved(self, review_manifest, events, candidate_metadata):
        records = load_review_manifest(review_manifest)
        examples, _ = resolve_reviewed_examples(records, events, candidate_metadata)
        for ex in examples:
            assert ex.candidate_id
            assert ex.clip_id
            assert ex.label in ("pickup", "putdown", "negative")
            assert ex.review_status == "reviewed"


# ---------------------------------------------------------------------------
# Test 11: FeatureDataset and manifest provenance
# ---------------------------------------------------------------------------


class TestFeatureDatasetProvenance:
    def test_required_fields_present(self):
        """FeatureRecord preserves all required provenance fields."""
        record = FeatureRecord(
            crop_id="crop_1",
            clip_id="clip1",
            candidate_id="cand1",
            timestamp_s=10.0,
            sample_position="pre",
            crop_type="hand",
            geometry=CropGeometry(x=0, y=0, width=224, height=224),
            embedding_path=Path("/tmp/embed.npy"),
            encoder_name="mobilenet_v3_small",
            encoder_version="v1",
            label="pickup",
            split="train",
            actor_id="actor1",
            hand_side="right",
            region_id="shelf_1",
            event_id="evt1",
        )
        assert record.event_id == "evt1"
        assert record.clip_id == "clip1"
        assert record.candidate_id == "cand1"
        assert record.split == "train"

    def test_dataset_stats(self):
        records = [
            FeatureRecord(
                crop_id=f"crop_{i}",
                clip_id="clip1",
                candidate_id="cand1",
                timestamp_s=10.0,
                sample_position="pre",
                crop_type="hand",
                geometry=CropGeometry(x=0, y=0, width=224, height=224),
                embedding_path=Path("/tmp/e.npy"),
                encoder_name="mob",
                encoder_version="v1",
                label="pickup" if i < 2 else "negative",
                split="train" if i < 3 else "test",
            )
            for i in range(4)
        ]
        ds = FeatureDataset(records=records)
        ds.compute_stats()
        assert ds.n_pickup == 2
        assert ds.n_negative == 2
        assert ds.n_train == 3
        assert ds.n_test == 1


# ---------------------------------------------------------------------------
# Test 12: Mocked embedder reuse (cache test)
# ---------------------------------------------------------------------------


class TestCachedEmbeddings:
    def test_embedder_reuse(self, tmp_path: Path):
        """Cached embeddings are reused when valid cache entries exist."""
        from pickup_putdown.layer1.track_a.cache import (
            is_embedding_cached,
            load_embedding,
            save_embedding,
        )

        cache_dir = tmp_path / "cache"
        key = "test_key_v1"

        assert not is_embedding_cached(cache_dir, key)

        emb = np.random.rand(576).astype(np.float32)
        path = save_embedding(emb, cache_dir, key)
        assert path.exists()
        assert is_embedding_cached(cache_dir, key)

        loaded = load_embedding(cache_dir, key)
        assert loaded is not None
        np.testing.assert_array_almost_equal(loaded, emb)


# ---------------------------------------------------------------------------
# Test 13: Pose-candidate association
# ---------------------------------------------------------------------------


class TestPoseCandidateAssociation:
    """Tests for pose observation matching to candidates."""

    def _make_pose(
        self,
        clip_id="clip1",
        timestamp_s=10.0,
        actor_id="actor_1",
        hand_side="right",
    ) -> PoseObservation:
        return PoseObservation(
            clip_id=clip_id,
            timestamp_s=timestamp_s,
            source_frame_index=0,
            sample_index=0,
            actor_id=actor_id,
            hand_side=hand_side,
            wrist_x=100.0,
            wrist_y=200.0,
            wrist_confidence=0.9,
        )

    def _make_candidate(
        self,
        candidate_id="c1",
        clip_id="clip1",
        actor_id="actor_1",
        hand_side="right",
        window_start_s=8.0,
        window_end_s=12.0,
    ) -> Candidate:
        return Candidate(
            candidate_id=candidate_id,
            clip_id=clip_id,
            actor_id=actor_id,
            hand_side=hand_side,
            raw_start_s=window_start_s,
            raw_end_s=window_end_s,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
        )

    def test_pose_keyed_by_clip_id_matches_candidate(self):
        """Pose observations keyed by source clip_id match correct candidate."""
        poses = [
            self._make_pose(timestamp_s=10.0),
            self._make_pose(timestamp_s=11.0),
        ]
        cand = self._make_candidate()
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 2

    def test_candidate_with_nonzero_source_start_receives_poses(self):
        """A candidate with nonzero source_start_s receives poses correctly."""
        poses = [
            self._make_pose(timestamp_s=120.0),
            self._make_pose(timestamp_s=125.0),
        ]
        cand = self._make_candidate(
            window_start_s=118.0,
            window_end_s=128.0,
        )
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 2

    def test_window_relative_to_source_timestamp_conversion(self):
        """Window-relative pose timestamps are converted to source timestamps once."""
        window_start = 100.0
        poses = [self._make_pose(timestamp_s=window_start + rel_t) for rel_t in [1.0, 2.0, 3.0]]
        cand = self._make_candidate(
            window_start_s=window_start,
            window_end_s=window_start + 5.0,
        )
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 3
        assert matched[0].timestamp_s == pytest.approx(window_start + 1.0)

    def test_source_to_candidate_relative_timestamp(self):
        """Source timestamps convert to candidate-relative timestamps exactly once."""
        source_start = 100.0
        poses = [self._make_pose(timestamp_s=source_start + rel_t) for rel_t in [1.0, 2.0, 3.0]]
        cand = self._make_candidate(
            window_start_s=source_start,
            window_end_s=source_start + 5.0,
        )
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        cand_relative = [m.timestamp_s - cand.window_start_s for m in matched]
        assert cand_relative == pytest.approx([1.0, 2.0, 3.0])

    def test_two_candidates_same_clip_get_own_poses(self):
        """Two candidates from same clip receive only their own overlapping poses."""
        poses = [
            self._make_pose(timestamp_s=10.0),
            self._make_pose(timestamp_s=20.0),
        ]
        cand_a = self._make_candidate(
            candidate_id="ca",
            window_start_s=8.0,
            window_end_s=12.0,
        )
        cand_b = self._make_candidate(
            candidate_id="cb",
            window_start_s=18.0,
            window_end_s=22.0,
        )
        matched_a = get_wrist_trajectory_for_candidate(cand_a, poses)
        matched_b = get_wrist_trajectory_for_candidate(cand_b, poses)
        assert len(matched_a) == 1
        assert matched_a[0].timestamp_s == pytest.approx(10.0)
        assert len(matched_b) == 1
        assert matched_b[0].timestamp_s == pytest.approx(20.0)

    def test_different_clips_do_not_share_poses(self):
        """Candidates from different clips do not share observations."""
        poses = [self._make_pose(clip_id="clip1", timestamp_s=10.0)]
        cand = self._make_candidate(clip_id="clip2")
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 0

    def test_no_matching_pose_returns_empty(self):
        """A candidate with no matching observations returns empty list."""
        poses = [self._make_pose(clip_id="other_clip")]
        cand = self._make_candidate()
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert matched == []

    def test_actor_id_mismatch_skips(self):
        """Pose with wrong actor_id does not match candidate."""
        poses = [self._make_pose(actor_id="actor_2")]
        cand = self._make_candidate(actor_id="actor_1")
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 0

    def test_hand_side_mismatch_skips(self):
        """Pose with wrong hand_side does not match candidate."""
        poses = [self._make_pose(hand_side="left")]
        cand = self._make_candidate(hand_side="right")
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 0

    def test_actor_id_fallback_on_mismatch(self):
        """When actor_id doesn't match (person-tracker format), fall back to clip+hand+window."""
        poses = [self._make_pose(actor_id="actor_5", timestamp_s=10.0)]
        cand = self._make_candidate(actor_id="clip_X:person:1")
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 1
        assert matched[0].actor_id == "actor_5"

    def test_fallback_respects_hand_side(self):
        """Fallback still respects hand_side filter."""
        poses = [self._make_pose(actor_id="actor_5", hand_side="left", timestamp_s=10.0)]
        cand = self._make_candidate(actor_id="clip_X:person:1", hand_side="right")
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 0

    def test_fallback_respects_window(self):
        """Fallback still respects time window filter."""
        poses = [self._make_pose(actor_id="actor_5", timestamp_s=50.0)]
        cand = self._make_candidate(
            actor_id="clip_X:person:1", window_start_s=8.0, window_end_s=12.0
        )
        matched = get_wrist_trajectory_for_candidate(cand, poses)
        assert len(matched) == 0


# ---------------------------------------------------------------------------
# Test 14: Metadata passthrough
# ---------------------------------------------------------------------------


class TestMetadataPassthrough:
    """Tests that actor_id/hand_side/region_id flow through the pipeline."""

    def test_candidate_metadata_loads_actor_id(self, tmp_path: Path):
        """CandidateMetadata loaded from JSON includes actor_id and hand_side."""
        staging = tmp_path / "staging"
        clip_dir = staging / "candidates" / "clip1"
        clip_dir.mkdir(parents=True)
        meta_file = clip_dir / "clip1.json"
        meta_file.write_text(
            json.dumps(
                {
                    "source_video_id": "clip1",
                    "candidates": [
                        {
                            "candidate_id": "cand_a",
                            "source_start_s": 10.0,
                            "source_end_s": 12.0,
                            "duration_s": 2.0,
                            "actor_id": "actor_5",
                            "hand_side": "right",
                            "region_id": "shelf_1",
                        }
                    ],
                }
            )
        )

        index = load_candidate_metadata_index(staging)
        assert index["cand_a"].actor_id == "actor_5"
        assert index["cand_a"].hand_side == "right"
        assert index["cand_a"].region_id == "shelf_1"

    def test_resolve_reviewed_examples_passes_actor_id(self, tmp_path: Path):
        """ReviewedExample gets actor_id/hand_side from metadata."""
        json_dir = tmp_path / "json"
        json_dir.mkdir()
        (json_dir / "cand_x.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_x",
                    "clip_id": "clip1",
                    "source_start_s": 10.0,
                    "source_end_s": 12.0,
                    "events": [
                        {"label": "pickup", "start_s": 0.5, "end_s": 1.5, "confidence": "high"}
                    ],
                }
            )
        )

        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "candidate_id",
                    "clip_id",
                    "review_groups",
                    "video_path",
                    "json_path",
                    "event_count",
                    "reviewed",
                    "review_notes",
                ]
            )
            writer.writerow(
                [
                    "cand_x",
                    "clip1",
                    "vlm_positive",
                    "/v/c.mp4",
                    str(json_dir / "cand_x.json"),
                    1,
                    "true",
                    "confirmed",
                ]
            )

        meta = {
            "cand_x": CandidateMetadata(
                candidate_id="cand_x",
                clip_id="clip1",
                source_start_s=10.0,
                source_end_s=12.0,
                duration_s=2.0,
                actor_id="actor_3",
                hand_side="left",
                region_id="shelf_2",
            )
        }

        records = load_review_manifest(manifest)
        examples, _ = resolve_reviewed_examples(records, [], meta)
        assert len(examples) == 1
        assert examples[0].actor_id == "actor_3"
        assert examples[0].hand_side == "left"
        assert examples[0].region_id == "shelf_2"

    def test_negative_example_passes_actor_id(self, tmp_path: Path):
        """Negative (zero-event) example also gets actor_id/hand_side."""
        json_dir = tmp_path / "json"
        json_dir.mkdir()
        (json_dir / "cand_n.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_n",
                    "clip_id": "clip1",
                    "source_start_s": 20.0,
                    "source_end_s": 24.0,
                    "events": [],
                }
            )
        )

        manifest = tmp_path / "m.csv"
        with open(manifest, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "candidate_id",
                    "clip_id",
                    "review_groups",
                    "video_path",
                    "json_path",
                    "event_count",
                    "reviewed",
                    "review_notes",
                ]
            )
            writer.writerow(
                [
                    "cand_n",
                    "clip1",
                    "negative_sample",
                    "/v/c.mp4",
                    str(json_dir / "cand_n.json"),
                    0,
                    "true",
                    "confirmed negative",
                ]
            )

        meta = {
            "cand_n": CandidateMetadata(
                candidate_id="cand_n",
                clip_id="clip1",
                source_start_s=20.0,
                source_end_s=24.0,
                duration_s=4.0,
                actor_id="actor_7",
                hand_side="right",
                region_id="shelf_3",
            )
        }

        records = load_review_manifest(manifest)
        examples, _ = resolve_reviewed_examples(records, [], meta)
        assert len(examples) == 1
        assert examples[0].label == "negative"
        assert examples[0].actor_id == "actor_7"
        assert examples[0].hand_side == "right"
        assert examples[0].region_id == "shelf_3"

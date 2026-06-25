"""Tests for vlm_annotate pipeline module.

Covers: normalization, timestamp conversion, resume behavior,
no-event candidates, and validation.
"""

from __future__ import annotations

import csv
import json
from unittest.mock import patch

import pytest

from pickup_putdown.annotation.schemas import (
    ConfidenceLevel,
    EventLabel,
)
from pickup_putdown.annotation.vlm_annotate import (
    PipelineConfig,
    VlMCandidateResult,
    VlMEventAnnotation,
    discover_candidates,
    frame_index_to_time,
    normalize_candidate_result,
    run_pipeline,
    validate_candidate_annotation,
)


class TestTimestampConversion:
    """Test candidate-relative to source-video timestamp conversion."""

    def test_zero_offset_preserves_timestamps(self):
        result = VlMCandidateResult(
            candidate_id="c1",
            clip_id="clip_1",
            video_path="/v/c1.mp4",
            candidate_duration_s=10.0,
            source_start_s=0.0,
            source_end_s=10.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=2.3,
                    end_s=3.7,
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        events, _ = normalize_candidate_result(result)
        assert len(events) == 1
        assert events[0]["t_start"] == pytest.approx(2.3)
        assert events[0]["t_end"] == pytest.approx(3.7)

    def test_nonzero_offset_added_correctly(self):
        result = VlMCandidateResult(
            candidate_id="c2",
            clip_id="clip_2",
            video_path="/v/c2.mp4",
            candidate_duration_s=8.0,
            source_start_s=102.0,
            source_end_s=110.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PUTDOWN,
                    start_s=1.5,
                    end_s=3.0,
                    confidence=ConfidenceLevel.MED,
                )
            ],
        )
        events, _ = normalize_candidate_result(result)
        assert len(events) == 1
        assert events[0]["t_start"] == pytest.approx(103.5)
        assert events[0]["t_end"] == pytest.approx(105.0)

    def test_frame_index_to_time(self):
        cand_rel, source_abs = frame_index_to_time(60, fps=30.0, source_start_s=100.0)
        assert cand_rel == pytest.approx(2.0)
        assert source_abs == pytest.approx(102.0)

    def test_frame_index_to_time_zero_fps_fallback(self):
        cand_rel, source_abs = frame_index_to_time(30, fps=0.0, source_start_s=50.0)
        assert cand_rel == pytest.approx(1.0)
        assert source_abs == pytest.approx(51.0)


class TestNormalization:
    """Test normalization of candidate results into canonical events."""

    def test_single_event_normalizes(self):
        result = VlMCandidateResult(
            candidate_id="c1",
            clip_id="clip_1",
            video_path="/v/c1.mp4",
            candidate_duration_s=10.0,
            source_start_s=50.0,
            source_end_s=60.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=1.0,
                    end_s=3.0,
                    item_count=1,
                    confidence=ConfidenceLevel.HIGH,
                    notes="hand removes item from shelf",
                )
            ],
        )
        events, ignores = normalize_candidate_result(result)
        assert len(events) == 1
        assert len(ignores) == 0
        ev = events[0]
        assert ev["type"] == "pickup"
        assert ev["t_start"] == pytest.approx(51.0)
        assert ev["t_end"] == pytest.approx(53.0)
        assert ev["confidence"] == "high"
        assert ev["annotator"] == "vlm_pipeline"
        assert ev["notes"] == "hand removes item from shelf"

    def test_multi_item_expands_to_separate_rows(self):
        result = VlMCandidateResult(
            candidate_id="c2",
            clip_id="clip_2",
            video_path="/v/c2.mp4",
            candidate_duration_s=10.0,
            source_start_s=0.0,
            source_end_s=10.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=2.0,
                    end_s=4.0,
                    item_count=2,
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        events, _ = normalize_candidate_result(result)
        assert len(events) == 2
        assert events[0]["event_id"] != events[1]["event_id"]
        assert events[0]["event_id"].startswith("evt_")
        assert events[1]["event_id"].startswith("evt_")
        assert events[0]["t_start"] == pytest.approx(2.0)
        assert events[1]["t_start"] == pytest.approx(2.0)
        assert events[0]["t_end"] == pytest.approx(4.0)
        assert events[1]["t_end"] == pytest.approx(4.0)

    def test_no_events_produces_empty_list(self):
        result = VlMCandidateResult(
            candidate_id="c3",
            clip_id="clip_3",
            video_path="/v/c3.mp4",
            candidate_duration_s=5.0,
            source_start_s=100.0,
            source_end_s=105.0,
            events=[],
        )
        events, ignores = normalize_candidate_result(result)
        assert events == []
        assert ignores == []

    def test_multiple_events_in_one_candidate(self):
        result = VlMCandidateResult(
            candidate_id="c4",
            clip_id="clip_4",
            video_path="/v/c4.mp4",
            candidate_duration_s=15.0,
            source_start_s=200.0,
            source_end_s=215.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=1.0,
                    end_s=2.5,
                    confidence=ConfidenceLevel.HIGH,
                ),
                VlMEventAnnotation(
                    label=EventLabel.PUTDOWN,
                    start_s=10.0,
                    end_s=11.5,
                    confidence=ConfidenceLevel.MED,
                ),
            ],
        )
        events, _ = normalize_candidate_result(result)
        assert len(events) == 2
        assert events[0]["type"] == "pickup"
        assert events[0]["t_start"] == pytest.approx(201.0)
        assert events[1]["type"] == "putdown"
        assert events[1]["t_start"] == pytest.approx(210.0)

    def test_ignore_intervals_normalized(self):
        result = VlMCandidateResult(
            candidate_id="c5",
            clip_id="clip_5",
            video_path="/v/c5.mp4",
            candidate_duration_s=10.0,
            source_start_s=300.0,
            source_end_s=310.0,
            ignore_intervals=[
                {"start_s": 2.0, "end_s": 4.0, "reason": "ACTION_OCCLUDED", "notes": "hand hidden"}
            ],
        )
        _, ignores = normalize_candidate_result(result)
        assert len(ignores) == 1
        assert ignores[0]["t_start"] == pytest.approx(302.0)
        assert ignores[0]["t_end"] == pytest.approx(304.0)
        assert ignores[0]["reason"] == "ACTION_OCCLUDED"
        assert ignores[0]["ignore_id"].startswith("ign_")

    def test_deterministic_event_ids(self):
        result = VlMCandidateResult(
            candidate_id="c6",
            clip_id="clip_6",
            video_path="/v/c6.mp4",
            candidate_duration_s=10.0,
            source_start_s=0.0,
            source_end_s=10.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=2.0,
                    end_s=3.0,
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        ev1, _ = normalize_candidate_result(result)
        ev2, _ = normalize_candidate_result(result)
        assert ev1[0]["event_id"] == ev2[0]["event_id"]


class TestValidation:
    """Test annotation validation logic."""

    def test_valid_annotation_passes(self):
        result = VlMCandidateResult(
            candidate_id="c1",
            clip_id="clip_1",
            video_path="/v/c1.mp4",
            candidate_duration_s=10.0,
            source_start_s=0.0,
            source_end_s=10.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=1.0,
                    end_s=3.0,
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        errors = validate_candidate_annotation(result)
        assert errors == []

    def test_negative_start_rejected(self):
        result = VlMCandidateResult(
            candidate_id="c1",
            clip_id="clip_1",
            video_path="/v/c1.mp4",
            candidate_duration_s=10.0,
            source_start_s=0.0,
            source_end_s=10.0,
        )
        # Manually set negative start (bypass Pydantic for test)
        evt = VlMEventAnnotation(
            label=EventLabel.PICKUP,
            start_s=1.0,
            end_s=3.0,
            confidence=ConfidenceLevel.HIGH,
        )
        result.events.append(evt)
        # Override to negative for test
        result.events[0].start_s = -1.0  # type: ignore[assignment]
        errors = validate_candidate_annotation(result)
        assert any("negative" in e.lower() for e in errors)

    def test_end_exceeds_duration_rejected(self):
        result = VlMCandidateResult(
            candidate_id="c1",
            clip_id="clip_1",
            video_path="/v/c1.mp4",
            candidate_duration_s=5.0,
            source_start_s=0.0,
            source_end_s=5.0,
            events=[
                VlMEventAnnotation(
                    label=EventLabel.PICKUP,
                    start_s=4.0,
                    end_s=10.0,
                    confidence=ConfidenceLevel.HIGH,
                )
            ],
        )
        errors = validate_candidate_annotation(result)
        assert any("exceeds" in e.lower() for e in errors)


class TestCandidateDiscovery:
    """Test candidate discovery from metadata directory."""

    def test_discovers_nested_candidates(self, tmp_path):
        meta = {
            "source_video_id": "src_001",
            "candidates": [
                {
                    "candidate_id": "cand_a",
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "candidate_key": str(tmp_path / "cand_a.mp4"),
                },
                {
                    "candidate_id": "cand_b",
                    "source_start_s": 20.0,
                    "source_end_s": 25.0,
                    "candidate_key": str(tmp_path / "cand_b.mp4"),
                },
            ],
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        candidates = discover_candidates(tmp_path)
        assert len(candidates) == 2
        assert candidates[0]["candidate_id"] == "cand_a"
        assert candidates[1]["candidate_id"] == "cand_b"
        assert candidates[0]["clip_id"] == "src_001"

    def test_discovers_flat_array(self, tmp_path):
        meta = [
            {
                "candidate_id": "cand_x",
                "clip_id": "clip_x",
                "source_start_s": 0.0,
                "source_end_s": 5.0,
                "candidate_key": str(tmp_path / "cand_x.mp4"),
            },
        ]
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        candidates = discover_candidates(tmp_path)
        assert len(candidates) == 1
        assert candidates[0]["candidate_id"] == "cand_x"

    def test_sorted_deterministically(self, tmp_path):
        meta = {
            "source_video_id": "src_001",
            "candidates": [
                {
                    "candidate_id": "cand_z",
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "candidate_key": "z.mp4",
                },
                {
                    "candidate_id": "cand_a",
                    "source_start_s": 20.0,
                    "source_end_s": 25.0,
                    "candidate_key": "a.mp4",
                },
            ],
        }
        (tmp_path / "meta.json").write_text(json.dumps(meta))
        candidates = discover_candidates(tmp_path)
        assert candidates[0]["candidate_id"] == "cand_a"
        assert candidates[1]["candidate_id"] == "cand_z"


class TestResumeBehavior:
    """Test that --force and resume work correctly."""

    def test_resume_skips_completed(self, tmp_path):
        """Candidates with existing normalized output are skipped."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {
            "source_video_id": "src_1",
            "candidates": [
                {
                    "candidate_id": "done_1",
                    "source_start_s": 0.0,
                    "source_end_s": 5.0,
                    "candidate_key": str(tmp_path / "v.mp4"),
                },
                {
                    "candidate_id": "done_2",
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "candidate_key": str(tmp_path / "v.mp4"),
                },
            ],
        }
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        normalized_dir = output_dir / "normalized"
        normalized_dir.mkdir(parents=True)
        # Pre-create normalized output for first candidate
        (normalized_dir / "done_1.json").write_text("{}")

        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
            force=False,
        )

        with patch(
            "pickup_putdown.annotation.vlm_annotate.probe_candidate_video",
            side_effect=FileNotFoundError("no video"),
        ):
            summary = run_pipeline(config)

        assert summary.skipped == 1
        assert summary.failed == 1  # done_2 fails (no video)

    def test_force_reprocesses_completed(self, tmp_path):
        """--force reprocesses candidates with existing output."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {
            "source_video_id": "src_1",
            "candidates": [
                {
                    "candidate_id": "redo_1",
                    "source_start_s": 0.0,
                    "source_end_s": 5.0,
                    "candidate_key": str(tmp_path / "v.mp4"),
                },
            ],
        }
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        normalized_dir = output_dir / "normalized"
        normalized_dir.mkdir(parents=True)
        (normalized_dir / "redo_1.json").write_text("{}")

        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
            force=True,
        )

        with patch(
            "pickup_putdown.annotation.vlm_annotate.probe_candidate_video",
            side_effect=FileNotFoundError("no video"),
        ):
            summary = run_pipeline(config)

        assert summary.skipped == 0
        assert summary.failed == 1  # redo_1 fails but was NOT skipped


class TestPipelineOutput:
    """Test that pipeline writes correct output files."""

    def test_pipeline_writes_required_files(self, tmp_path):
        """Pipeline writes events.csv, processing.csv, summary.json."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {
            "source_video_id": "src_1",
            "candidates": [],
        }
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
        )
        summary = run_pipeline(config)

        assert (output_dir / "events.csv").exists()
        assert (output_dir / "processing.csv").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "raw").is_dir()
        assert (output_dir / "normalized").is_dir()
        assert (output_dir / "review_frames").is_dir()
        assert summary.total_candidates == 0

    def test_events_csv_has_correct_columns(self, tmp_path):
        """events.csv uses canonical column names."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {
            "source_video_id": "src_1",
            "candidates": [],
        }
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
        )
        run_pipeline(config)

        with (output_dir / "events.csv").open() as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == [
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

    def test_processing_csv_has_all_candidates(self, tmp_path):
        """processing.csv has one row per candidate."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {
            "source_video_id": "src_1",
            "candidates": [
                {
                    "candidate_id": "c1",
                    "source_start_s": 0.0,
                    "source_end_s": 5.0,
                    "candidate_key": str(tmp_path / "v.mp4"),
                },
                {
                    "candidate_id": "c2",
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "candidate_key": str(tmp_path / "v.mp4"),
                },
            ],
        }
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
        )

        with patch(
            "pickup_putdown.annotation.vlm_annotate.probe_candidate_video",
            side_effect=FileNotFoundError("no video"),
        ):
            run_pipeline(config)

        with (output_dir / "processing.csv").open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2

    def test_summary_json_has_required_fields(self, tmp_path):
        """summary.json contains all required summary fields."""
        cand_dir = tmp_path / "candidates"
        cand_dir.mkdir()
        meta = {"source_video_id": "src_1", "candidates": []}
        (cand_dir / "meta.json").write_text(json.dumps(meta))

        output_dir = tmp_path / "output"
        config = PipelineConfig(
            candidates_dir=str(cand_dir),
            output_dir=str(output_dir),
        )
        run_pipeline(config)

        summary_data = json.loads((output_dir / "summary.json").read_text())
        required_keys = {
            "total_candidates",
            "processed",
            "skipped",
            "failed",
            "review_required",
            "events_found",
            "processing_time_s",
            "annotator",
            "review_fps",
            "force",
            "timestamp",
        }
        assert required_keys.issubset(summary_data.keys())


class TestNoEventCandidate:
    """Test handling of candidates with no events."""

    def test_no_event_candidate_has_empty_events(self):
        """A candidate with no events has empty events array."""
        result = VlMCandidateResult(
            candidate_id="no_evt",
            clip_id="clip_ne",
            video_path="/v/no_evt.mp4",
            candidate_duration_s=5.0,
            source_start_s=100.0,
            source_end_s=105.0,
            review_status="complete",
            events=[],
            complete_active_span_reviewed=True,
        )
        events, ignores = normalize_candidate_result(result)
        assert events == []
        assert ignores == []
        assert result.complete_active_span_reviewed is True

    def test_no_event_still_validates(self):
        """A no-event candidate passes validation."""
        result = VlMCandidateResult(
            candidate_id="no_evt",
            clip_id="clip_ne",
            video_path="/v/no_evt.mp4",
            candidate_duration_s=5.0,
            source_start_s=100.0,
            source_end_s=105.0,
            events=[],
        )
        errors = validate_candidate_annotation(result)
        assert errors == []

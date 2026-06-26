"""Tests for Task 7 finalization pipeline.

Covers: normalized candidate loading, event extraction, clip metadata
discovery, referential integrity validation, provenance generation,
and the full finalization pipeline.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pickup_putdown.annotation.finalize_task7 import (
    collect_normalized_candidates,
    extract_events_from_candidates,
    finalize_task_7,
    link_subdirectory,
    load_normalized_candidate,
    validate_final_artifacts,
    validate_symlink,
    write_events_csv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_normalized_dir(tmp_path: Path) -> Path:
    """Create a normalized directory with test candidate files."""
    norm_dir = tmp_path / "normalized"
    norm_dir.mkdir()

    # Candidate with events
    (norm_dir / "cand_a.json").write_text(
        json.dumps(
            {
                "candidate_id": "cand_a",
                "clip_id": "clip_001",
                "video_path": "/v/cand_a.mp4",
                "candidate_duration_s": 5.0,
                "source_start_s": 100.0,
                "source_end_s": 105.0,
                "review_status": "complete",
                "vlm_status": "success",
                "events": [
                    {
                        "label": "pickup",
                        "start_s": 1.0,
                        "end_s": 2.5,
                        "item_count": 1,
                        "confidence": "high",
                        "hard_case": False,
                        "notes": "clear pickup",
                    }
                ],
                "ignore_intervals": [],
                "fps": 5.0,
            }
        )
    )

    # Candidate with no events
    (norm_dir / "cand_b.json").write_text(
        json.dumps(
            {
                "candidate_id": "cand_b",
                "clip_id": "clip_001",
                "video_path": "/v/cand_b.mp4",
                "candidate_duration_s": 5.0,
                "source_start_s": 110.0,
                "source_end_s": 115.0,
                "review_status": "complete",
                "vlm_status": "success",
                "events": [],
                "ignore_intervals": [],
                "fps": 5.0,
            }
        )
    )

    # Candidate with failed VLM -- must be excluded
    (norm_dir / "cand_c.json").write_text(
        json.dumps(
            {
                "candidate_id": "cand_c",
                "clip_id": "clip_002",
                "video_path": "/v/cand_c.mp4",
                "candidate_duration_s": 5.0,
                "source_start_s": 0.0,
                "source_end_s": 5.0,
                "review_status": "failed",
                "vlm_status": "failed",
                "vlm_error": "VLM timeout",
                "events": [],
                "ignore_intervals": [],
                "fps": 5.0,
            }
        )
    )

    return norm_dir


@pytest.fixture
def clip_durations() -> dict[str, float]:
    return {
        "clip_001": 300.0,
        "clip_002": 200.0,
    }


# ---------------------------------------------------------------------------
# load_normalized_candidate
# ---------------------------------------------------------------------------


class TestLoadNormalizedCandidate:
    def test_valid_file(self, sample_normalized_dir: Path):
        data = load_normalized_candidate(sample_normalized_dir / "cand_a.json")
        assert data["candidate_id"] == "cand_a"
        assert data["clip_id"] == "clip_001"

    def test_malformed_json(self, tmp_path: Path):
        p = tmp_path / "bad.json"
        p.write_text("{invalid json")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_normalized_candidate(p)

    def test_missing_candidate_id(self, tmp_path: Path):
        p = tmp_path / "no_id.json"
        p.write_text(json.dumps({"clip_id": "clip_1"}))
        with pytest.raises(ValueError, match="Missing candidate_id"):
            load_normalized_candidate(p)

    def test_missing_clip_id(self, tmp_path: Path):
        p = tmp_path / "no_clip.json"
        p.write_text(json.dumps({"candidate_id": "c1"}))
        with pytest.raises(ValueError, match="Missing clip_id"):
            load_normalized_candidate(p)


# ---------------------------------------------------------------------------
# collect_normalized_candidates
# ---------------------------------------------------------------------------


class TestCollectNormalizedCandidates:
    def test_collects_all_valid(self, sample_normalized_dir: Path):
        cands = collect_normalized_candidates(sample_normalized_dir)
        assert len(cands) == 3
        ids = {c["candidate_id"] for c in cands}
        assert ids == {"cand_a", "cand_b", "cand_c"}

    def test_raises_on_malformed(self, tmp_path: Path):
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "good.json").write_text(
            json.dumps({"candidate_id": "c1", "clip_id": "clip_1"})
        )
        (norm_dir / "bad.json").write_text("{bad")
        with pytest.raises(ValueError, match="Failed to load"):
            collect_normalized_candidates(norm_dir)

    def test_empty_dir(self, tmp_path: Path):
        norm_dir = tmp_path / "empty"
        norm_dir.mkdir()
        cands = collect_normalized_candidates(norm_dir)
        assert cands == []


# ---------------------------------------------------------------------------
# extract_events_from_candidates
# ---------------------------------------------------------------------------


class TestExtractEventsFromCandidates:
    def test_extracts_events_from_success(
        self, sample_normalized_dir: Path, clip_durations: dict[str, float]
    ):
        cands = collect_normalized_candidates(sample_normalized_dir)
        events, errors, dedup = extract_events_from_candidates(cands, clip_durations)
        assert len(errors) == 0
        assert len(events) == 1
        assert len(dedup) == 0
        evt = events[0]
        assert evt["clip_id"] == "clip_001"
        assert evt["type"] == "pickup"
        assert evt["t_start"] == pytest.approx(101.0)
        assert evt["t_end"] == pytest.approx(102.5)
        assert evt["confidence"] == "high"

    def test_excludes_failed_vlm(
        self, sample_normalized_dir: Path, clip_durations: dict[str, float]
    ):
        cands = collect_normalized_candidates(sample_normalized_dir)
        events, errors, dedup = extract_events_from_candidates(cands, clip_durations)
        # cand_c has vlm_status=failed, must not contribute
        assert len(events) == 1

    def test_no_events_candidate_contributes_nothing(
        self, sample_normalized_dir: Path, clip_durations: dict[str, float]
    ):
        cands = collect_normalized_candidates(sample_normalized_dir)
        events, _, _ = extract_events_from_candidates(cands, clip_durations)
        assert len(events) == 1  # Only cand_a has events

    def test_multi_item_expands(self, tmp_path: Path):
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "cand_multi.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_multi",
                    "clip_id": "clip_001",
                    "video_path": "/v/multi.mp4",
                    "candidate_duration_s": 10.0,
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "pickup",
                            "start_s": 2.0,
                            "end_s": 4.0,
                            "item_count": 2,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "two items",
                        }
                    ],
                    "fps": 5.0,
                }
            )
        )
        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_001": 300.0})
        assert len(errors) == 0
        assert len(events) == 2
        assert events[0]["event_id"] != events[1]["event_id"]

    def test_event_exceeding_duration_flagged(self, tmp_path: Path):
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "cand_bad.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_bad",
                    "clip_id": "clip_short",
                    "video_path": "/v/bad.mp4",
                    "candidate_duration_s": 5.0,
                    "source_start_s": 0.0,
                    "source_end_s": 5.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "pickup",
                            "start_s": 100.0,
                            "end_s": 105.0,
                            "item_count": 1,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "",
                        }
                    ],
                    "fps": 5.0,
                }
            )
        )
        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_short": 10.0})
        assert len(events) == 0
        assert len(errors) == 1
        assert "exceeds" in errors[0].message.lower()

    def test_invalid_label_flagged(self, tmp_path: Path):
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "cand_bad_label.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_bl",
                    "clip_id": "clip_001",
                    "video_path": "/v/bl.mp4",
                    "candidate_duration_s": 5.0,
                    "source_start_s": 0.0,
                    "source_end_s": 5.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "invalid_label",
                            "start_s": 1.0,
                            "end_s": 2.0,
                            "item_count": 1,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "",
                        }
                    ],
                    "fps": 5.0,
                }
            )
        )
        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_001": 300.0})
        assert len(events) == 0
        assert len(errors) == 1
        assert "label" in errors[0].message.lower()

    def test_deterministic_event_ids(self, sample_normalized_dir: Path):
        cands = collect_normalized_candidates(sample_normalized_dir)
        events1, _, _ = extract_events_from_candidates(cands, {"clip_001": 300.0})
        events2, _, _ = extract_events_from_candidates(cands, {"clip_001": 300.0})
        assert events1[0]["event_id"] == events2[0]["event_id"]

    def test_events_sorted_deterministically(self, tmp_path: Path):
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        for i, (label, ss) in enumerate(
            [
                ("putdown", 50.0),
                ("pickup", 30.0),
                ("pickup", 40.0),
            ]
        ):
            (norm_dir / f"cand_{i}.json").write_text(
                json.dumps(
                    {
                        "candidate_id": f"cand_{i}",
                        "clip_id": "clip_001",
                        "video_path": "/v/c.mp4",
                        "candidate_duration_s": 5.0,
                        "source_start_s": ss,
                        "source_end_s": ss + 5.0,
                        "review_status": "complete",
                        "vlm_status": "success",
                        "events": [
                            {
                                "label": label,
                                "start_s": 1.0,
                                "end_s": 2.0,
                                "item_count": 1,
                                "confidence": "high",
                                "hard_case": False,
                                "notes": "",
                            }
                        ],
                        "fps": 5.0,
                    }
                )
            )
        cands = collect_normalized_candidates(norm_dir)
        events, _, _ = extract_events_from_candidates(cands, {"clip_001": 300.0})
        assert len(events) == 3
        assert events[0]["t_start"] == pytest.approx(31.0)
        assert events[1]["t_start"] == pytest.approx(41.0)
        assert events[2]["t_start"] == pytest.approx(51.0)


# ---------------------------------------------------------------------------
# validate_final_artifacts
# ---------------------------------------------------------------------------


class TestValidateFinalArtifacts:
    def test_valid_artifacts_pass(self):
        events = [
            {
                "event_id": "evt_001",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 10.0,
                "t_end": 12.0,
            }
        ]
        clips = {"clip_001": {"clip_id": "clip_001", "duration_s": 300.0}}
        errors = validate_final_artifacts(events, clips, [])
        assert errors == []

    def test_duplicate_event_ids_detected(self):
        events = [
            {
                "event_id": "evt_dup",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 10.0,
                "t_end": 12.0,
            },
            {
                "event_id": "evt_dup",
                "clip_id": "clip_001",
                "type": "putdown",
                "t_start": 20.0,
                "t_end": 22.0,
            },
        ]
        clips = {"clip_001": {"clip_id": "clip_001", "duration_s": 300.0}}
        errors = validate_final_artifacts(events, clips, [])
        assert len(errors) == 1
        assert "Duplicate" in errors[0].message

    def test_missing_clip_reference_detected(self):
        events = [
            {
                "event_id": "evt_001",
                "clip_id": "clip_missing",
                "type": "pickup",
                "t_start": 10.0,
                "t_end": 12.0,
            }
        ]
        clips = {"clip_001": {"clip_id": "clip_001", "duration_s": 300.0}}
        errors = validate_final_artifacts(events, clips, [])
        assert len(errors) == 1
        assert "missing" in errors[0].message.lower()

    def test_event_exceeding_duration_detected(self):
        events = [
            {
                "event_id": "evt_001",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 290.0,
                "t_end": 310.0,
            }
        ]
        clips = {"clip_001": {"clip_id": "clip_001", "duration_s": 300.0}}
        errors = validate_final_artifacts(events, clips, [])
        assert len(errors) == 1
        assert "exceeds" in errors[0].message.lower()

    def test_invalid_ordering_detected(self):
        events = [
            {
                "event_id": "evt_002",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 20.0,
                "t_end": 22.0,
            },
            {
                "event_id": "evt_001",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 10.0,
                "t_end": 12.0,
            },
        ]
        clips = {"clip_001": {"clip_id": "clip_001", "duration_s": 300.0}}
        errors = validate_final_artifacts(events, clips, [])
        assert len(errors) == 1
        assert "order" in errors[0].message.lower()


# ---------------------------------------------------------------------------
# write_events_csv
# ---------------------------------------------------------------------------


class TestWriteEventsCsv:
    def test_writes_correct_columns(self, tmp_path: Path):
        events = [
            {
                "event_id": "evt_001",
                "clip_id": "clip_001",
                "type": "pickup",
                "t_start": 10.0,
                "t_end": 12.0,
                "hard_case": False,
                "annotator": "vlm_pipeline",
                "confidence": "high",
                "notes": "test",
            }
        ]
        out = tmp_path / "events.csv"
        write_events_csv(events, out)

        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 1
        assert rows[0]["event_id"] == "evt_001"
        assert rows[0]["type"] == "pickup"

    def test_empty_events_writes_header_only(self, tmp_path: Path):
        out = tmp_path / "events.csv"
        write_events_csv([], out)
        with out.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert rows == []


# ---------------------------------------------------------------------------
# Full pipeline integration
# ---------------------------------------------------------------------------


class TestFinalizeTask7Integration:
    def test_full_pipeline(self, tmp_path: Path):
        """End-to-end test of finalize_task_7 with mock data."""
        # Set up VLM output structure
        vlm_dir = tmp_path / "vlm_annotations"
        norm_dir = vlm_dir / "normalized"
        norm_dir.mkdir(parents=True)

        (norm_dir / "cand_1.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_1",
                    "clip_id": "clip_A",
                    "video_path": "/v/c1.mp4",
                    "candidate_duration_s": 5.0,
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "pickup",
                            "start_s": 1.0,
                            "end_s": 2.0,
                            "item_count": 1,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "",
                        }
                    ],
                    "ignore_intervals": [],
                    "fps": 5.0,
                }
            )
        )

        # Create processing.csv
        proc_csv = vlm_dir / "processing.csv"
        proc_csv.write_text(
            "candidate_id,video_path,status,error,processed_at,frames_extracted,events_found,vlm_status,vlm_attempts,vlm_finish_reason,prompt_tokens,completion_tokens\n"
            "cand_1,/v/c1.mp4,success,,2026-01-01T00:00:00,10,1,success,1,stop,1000,200\n"
        )

        # Create summary.json
        (vlm_dir / "summary.json").write_text(
            json.dumps(
                {
                    "total_candidates": 1,
                    "processed": 1,
                    "skipped": 0,
                    "failed": 0,
                    "review_required": 0,
                    "events_found": 1,
                    "processing_time_s": 1.0,
                    "errors": [],
                    "annotator": "vlm_pipeline",
                    "review_fps_target": 5.0,
                    "force": False,
                    "timestamp": "2026-01-01T00:00:00+00:00",
                }
            )
        )

        # Create candidate metadata so clip_A is discovered
        cand_meta_dir = tmp_path / "candidate_staging" / "candidates" / "clip_A"
        cand_meta_dir.mkdir(parents=True)
        (cand_meta_dir / "clip_A.json").write_text(
            json.dumps(
                {
                    "source_video_id": "clip_A",
                    "candidates": [
                        {
                            "candidate_id": "cand_1",
                            "source_start_s": 10.0,
                            "source_end_s": 15.0,
                            "candidate_key": "/v/c1.mp4",
                        }
                    ],
                }
            )
        )

        output_dir = tmp_path / "task_7_output"
        result = finalize_task_7(
            vlm_output_dir=vlm_dir,
            output_dir=output_dir,
            candidate_metadata_dir=tmp_path / "candidate_staging" / "candidates",
        )

        assert result.is_valid
        assert result.candidates_count == 1
        assert result.events_count == 1
        assert (output_dir / "events.csv").exists()
        assert (output_dir / "clips.csv").exists()
        assert (output_dir / "processing.csv").exists()
        assert (output_dir / "summary.json").exists()
        assert (output_dir / "provenance.json").exists()

        # Check provenance is provisional
        prov = json.loads((output_dir / "provenance.json").read_text())
        assert prov["status"] == "provisional"
        assert prov["complete_active_span_reviewed"] is False
        assert "pseudo-labels" in prov["provisional_notice"]

    def test_raises_on_missing_normalized_dir(self, tmp_path: Path):
        vlm_dir = tmp_path / "vlm"
        vlm_dir.mkdir()
        with pytest.raises(FileNotFoundError, match="Normalized"):
            finalize_task_7(
                vlm_output_dir=vlm_dir,
                output_dir=tmp_path / "out",
            )

    def test_raises_on_missing_vlm_dir(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError, match="VLM output"):
            finalize_task_7(
                vlm_output_dir=tmp_path / "nonexistent",
                output_dir=tmp_path / "out",
            )


# ---------------------------------------------------------------------------
# Deduplication audit
# ---------------------------------------------------------------------------


class TestDeduplicationAudit:
    def test_dedup_audit_records_overlap(self, tmp_path: Path):
        """Overlapping candidates at same timestamp produce dedup audit entry."""
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()

        # Both candidates produce t_start = 102.0 (100+2 and 102+0)
        for cid, ss, es in [("cand_first", 100.0, 2.0), ("cand_second", 102.0, 0.0)]:
            (norm_dir / f"{cid}.json").write_text(
                json.dumps(
                    {
                        "candidate_id": cid,
                        "clip_id": "clip_X",
                        "video_path": "/v/c.mp4",
                        "candidate_duration_s": 5.0,
                        "source_start_s": ss,
                        "source_end_s": ss + 5.0,
                        "review_status": "complete",
                        "vlm_status": "success",
                        "events": [
                            {
                                "label": "pickup",
                                "start_s": es,
                                "end_s": es + 1.0,
                                "item_count": 1,
                                "confidence": "high",
                                "hard_case": False,
                                "notes": f"from {cid}",
                            }
                        ],
                        "fps": 5.0,
                    }
                )
            )

        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_X": 300.0})
        assert len(events) == 1
        assert len(dedup) == 1
        entry = dedup[0]
        assert entry["kept_candidate"] == "cand_first"
        assert entry["skipped_candidate"] == "cand_second"
        assert entry["kept_t_start"] == pytest.approx(102.0)
        assert entry["skipped_t_start"] == pytest.approx(102.0)
        assert "Overlapping candidates" in entry["reason"]

    def test_distinct_events_not_deduped(self, tmp_path: Path):
        """Events at different timestamps are NOT deduplicated."""
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()

        for cid, ss in [("cand_a", 100.0), ("cand_b", 200.0)]:
            (norm_dir / f"{cid}.json").write_text(
                json.dumps(
                    {
                        "candidate_id": cid,
                        "clip_id": "clip_X",
                        "video_path": "/v/c.mp4",
                        "candidate_duration_s": 5.0,
                        "source_start_s": ss,
                        "source_end_s": ss + 5.0,
                        "review_status": "complete",
                        "vlm_status": "success",
                        "events": [
                            {
                                "label": "pickup",
                                "start_s": 1.0,
                                "end_s": 2.0,
                                "item_count": 1,
                                "confidence": "high",
                                "hard_case": False,
                                "notes": "",
                            }
                        ],
                        "fps": 5.0,
                    }
                )
            )

        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_X": 300.0})
        assert len(events) == 2
        assert len(dedup) == 0

    def test_multi_item_not_collapsed(self, tmp_path: Path):
        """Multi-item events produce separate rows, not collapsed."""
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "cand_m.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_m",
                    "clip_id": "clip_X",
                    "video_path": "/v/c.mp4",
                    "candidate_duration_s": 10.0,
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "pickup",
                            "start_s": 2.0,
                            "end_s": 4.0,
                            "item_count": 3,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "three items",
                        }
                    ],
                    "fps": 5.0,
                }
            )
        )
        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_X": 300.0})
        assert len(events) == 3
        assert len(dedup) == 0
        # All three have same t_start/t_end but different event_ids
        eids = {e["event_id"] for e in events}
        assert len(eids) == 3

    def test_pickup_putdown_pair_not_collapsed(self, tmp_path: Path):
        """Pickup and putdown at same timestamp are separate events."""
        norm_dir = tmp_path / "normalized"
        norm_dir.mkdir()
        (norm_dir / "cand_pp.json").write_text(
            json.dumps(
                {
                    "candidate_id": "cand_pp",
                    "clip_id": "clip_X",
                    "video_path": "/v/c.mp4",
                    "candidate_duration_s": 10.0,
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [
                        {
                            "label": "pickup",
                            "start_s": 2.0,
                            "end_s": 3.0,
                            "item_count": 1,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "",
                        },
                        {
                            "label": "putdown",
                            "start_s": 2.0,
                            "end_s": 3.0,
                            "item_count": 1,
                            "confidence": "high",
                            "hard_case": False,
                            "notes": "",
                        },
                    ],
                    "fps": 5.0,
                }
            )
        )
        cands = collect_normalized_candidates(norm_dir)
        events, errors, dedup = extract_events_from_candidates(cands, {"clip_X": 300.0})
        assert len(events) == 2
        assert len(dedup) == 0
        types = {e["type"] for e in events}
        assert types == {"pickup", "putdown"}


# ---------------------------------------------------------------------------
# Symlink and portability
# ---------------------------------------------------------------------------


class TestSymlinkAndPortability:
    def test_link_subdirectory_creates_relative_symlink(self, tmp_path: Path):
        src = tmp_path / "source"
        src.mkdir()
        (src / "file.txt").write_text("data")
        dest = tmp_path / "output" / "linked"
        link_subdirectory(src, dest)
        assert dest.is_symlink()
        assert dest.resolve().is_dir()
        assert (dest / "file.txt").read_text() == "data"

    def test_validate_symlink_valid(self, tmp_path: Path):
        src = tmp_path / "source"
        src.mkdir()
        dest = tmp_path / "link"
        dest.symlink_to(src)
        assert validate_symlink(dest) is True

    def test_validate_symlink_broken(self, tmp_path: Path):
        dest = tmp_path / "link"
        dest.symlink_to(tmp_path / "nonexistent")
        assert validate_symlink(dest) is False

    def test_provenance_records_source_paths(self, tmp_path: Path):
        """Provenance.json contains source path and symlink info."""
        vlm_dir = tmp_path / "vlm"
        norm_dir = vlm_dir / "normalized"
        norm_dir.mkdir(parents=True)
        (norm_dir / "c1.json").write_text(
            json.dumps(
                {
                    "candidate_id": "c1",
                    "clip_id": "clip_A",
                    "video_path": "/v/c1.mp4",
                    "candidate_duration_s": 5.0,
                    "source_start_s": 10.0,
                    "source_end_s": 15.0,
                    "review_status": "complete",
                    "vlm_status": "success",
                    "events": [],
                    "ignore_intervals": [],
                    "fps": 5.0,
                }
            )
        )
        (vlm_dir / "summary.json").write_text(
            json.dumps(
                {
                    "review_fps_target": 5.0,
                }
            )
        )
        cand_meta = tmp_path / "cand_meta" / "clip_A"
        cand_meta.mkdir(parents=True)
        (cand_meta / "clip_A.json").write_text(
            json.dumps(
                {
                    "source_video_id": "clip_A",
                    "candidates": [
                        {"candidate_id": "c1", "source_start_s": 10.0, "source_end_s": 15.0}
                    ],
                }
            )
        )
        output_dir = tmp_path / "out"
        finalize_task_7(
            vlm_output_dir=vlm_dir,
            output_dir=output_dir,
            candidate_metadata_dir=tmp_path / "cand_meta",
        )
        prov = json.loads((output_dir / "provenance.json").read_text())
        assert "source_vlm_output_dir_resolved" in prov
        assert "output_dir_resolved" in prov
        assert prov["uses_symlinks"] is True
        assert prov["not_self_contained"] is True
        assert "self_contained_note" in prov

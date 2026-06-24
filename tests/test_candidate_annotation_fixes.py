"""Tests for Task 6.2 follow-up fixes.

Covers:
- Fix 1: Active-span confirmation semantics for candidate-backed tasks
- Fix 2: Official-schema-compatible canonical export + provenance export
- Fix 3: Deterministic pilot sampling
- Fix 4: S3 video playback verification
"""

from __future__ import annotations

import csv
import json

import pyarrow.parquet as pq
import pytest

from pickup_putdown.annotation.import_export import (
    _generate_video_url,
    _is_candidate_backed_task,
    check_media_references,
    export_candidate_annotations,
    select_candidate_pilot,
)
from pickup_putdown.annotation.schemas import (
    AnnotationUnit,
    CanonicalEvent,
    IgnoreIntervalExport,
    VideoUrlMode,
)

# ---------------------------------------------------------------------------
# Fix 1: Active-span confirmation semantics
# ---------------------------------------------------------------------------


class TestCandidateBackedAnnotationUnit:
    """Candidate-backed tasks must not require full active-span review."""

    def test_candidate_task_passes_without_full_span_review(self):
        """Candidate task with candidate_clip_reviewed=true exports events."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "annotation_unit": "candidate_clip",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        assert result.canonical_events[0].t_start == 102.0
        assert result.canonical_events[0].t_end == 104.0

    def test_candidate_task_fails_without_any_review(self):
        """Candidate task with no review confirmation fails validation."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "annotation_unit": "candidate_clip",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert not result.is_valid
        assert len(result.canonical_events) == 0
        errors = [e for e in result.validation.errors if "candidate_clip_reviewed" in e.field_name]
        assert len(errors) >= 1

    def test_candidate_task_backward_compat_with_full_span_review(self):
        """Candidate task with complete_active_span_reviewed still works
        (backward compatibility)."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "annotation_unit": "candidate_clip",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1

    def test_legacy_task_requires_full_span_review(self):
        """Legacy (non-candidate) task still requires full-span review."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/clip.mp4",
                    "clip_id": "clip_001",
                    "fps": 30.0,
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert not result.is_valid
        errors = [
            e for e in result.validation.errors if "complete_active_span_reviewed" in e.field_name
        ]
        assert len(errors) >= 1

    def test_candidate_task_cannot_be_interpreted_as_full_span_reviewed(self):
        """Candidate-backed export must not falsely mark full span as reviewed."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "annotation_unit": "candidate_clip",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        assert evt.candidate_id == "cand_001"
        assert evt.clip_id == "clip_001"

    def test_is_candidate_backed_task_detection(self):
        """_is_candidate_backed_task correctly identifies task types."""
        assert _is_candidate_backed_task({"candidate_id": "cand_001"}) is True
        assert _is_candidate_backed_task({"candidate_id": ""}) is False
        assert _is_candidate_backed_task({}) is False
        assert _is_candidate_backed_task({"clip_id": "clip_001"}) is False

    def test_annotation_unit_in_task_data(self):
        """Candidate tasks must carry annotation_unit=candidate_clip."""
        from pickup_putdown.annotation.import_export import build_candidate_tasks

        metadata = [
            {
                "candidate_id": "cand_001",
                "clip_id": "clip_001",
                "source_start_s": 0.0,
                "source_end_s": 10.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert not errors
        assert tasks[0].data.get("annotation_unit") == AnnotationUnit.CANDIDATE_CLIP


# ---------------------------------------------------------------------------
# Fix 2: Official-schema-compatible canonical export
# ---------------------------------------------------------------------------


class TestOfficialCanonicalExport:
    """Official events.csv must contain only approved columns."""

    def test_official_events_csv_columns(self, tmp_path):
        """events.csv contains exactly the approved columns."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "actor_id": "track_3",
                    "hand_side": "right",
                    "region_id": "shelf_2",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        events_path = tmp_path / "events.csv"
        result = export_candidate_annotations(
            export,
            events_output=str(events_path),
        )
        assert result.is_valid
        assert events_path.exists()

        with events_path.open() as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames
            rows = list(reader)

        # Official columns only — no provenance fields
        expected_columns = [
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
        assert columns == expected_columns
        assert len(rows) == 1
        assert "candidate_id" not in rows[0]
        assert "actor_id" not in rows[0]
        assert "hand_side" not in rows[0]
        assert "region_id" not in rows[0]

    def test_official_ignore_parquet_columns(self, tmp_path):
        """ignore_intervals.parquet contains exactly the approved schema."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["ignore"],
                                    "reason": "ACTION_OCCLUDED",
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        ignore_path = tmp_path / "ignore.parquet"
        result = export_candidate_annotations(
            export,
            ignore_output=str(ignore_path),
        )
        assert result.is_valid
        assert ignore_path.exists()

        table = pq.read_table(str(ignore_path))
        columns = table.column_names
        expected_columns = [
            "ignore_id",
            "clip_id",
            "t_start",
            "t_end",
            "reason",
            "annotator",
            "notes",
        ]
        assert columns == expected_columns
        assert "candidate_id" not in columns

    def test_provenance_parquet_retains_candidate_metadata(self, tmp_path):
        """event_provenance.parquet retains candidate traceability."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "actor_id": "track_3",
                    "hand_side": "right",
                    "region_id": "shelf_2",
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        prov_path = tmp_path / "provenance.parquet"
        result = export_candidate_annotations(
            export,
            provenance_output=str(prov_path),
        )
        assert result.is_valid
        assert prov_path.exists()

        table = pq.read_table(str(prov_path))
        columns = table.column_names
        assert "event_id" in columns
        assert "candidate_id" in columns
        assert "actor_id" in columns
        assert "hand_side" in columns
        assert "region_id" in columns

        rows = table.to_pydict()
        assert rows["candidate_id"][0] == "cand_001"
        assert rows["actor_id"][0] == "track_3"
        assert rows["hand_side"][0] == "right"
        assert rows["region_id"][0] == "shelf_2"

    def test_task8_canonical_events_compatible(self, tmp_path):
        """Exported events.csv is readable by Task 8 evaluator."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        events_path = tmp_path / "events.csv"
        result = export_candidate_annotations(
            export,
            events_output=str(events_path),
        )
        assert result.is_valid

        # Task 8 evaluator reads via evaluation/io.py
        from pickup_putdown.evaluation.io import load_events_csv

        events = load_events_csv(str(events_path))
        assert len(events) == 1
        assert events[0].clip_id == "clip_001"
        assert events[0].t_start == pytest.approx(102.0)
        assert events[0].t_end == pytest.approx(104.0)

    def test_canonical_event_dict_methods(self):
        """CanonicalEvent.canonical_dict and provenance_dict work correctly."""
        evt = CanonicalEvent(
            event_id="evt_001",
            clip_id="clip_001",
            type="pickup",
            t_start=1.0,
            t_end=2.0,
            hard_case=False,
            annotator="alice",
            confidence="high",
            notes="test",
            event_group_id="group_001",
            candidate_id="cand_001",
            actor_id="track_3",
            hand_side="right",
            region_id="shelf_2",
        )
        canonical = evt.canonical_dict()
        assert "candidate_id" not in canonical
        assert "actor_id" not in canonical
        assert "hand_side" not in canonical
        assert "region_id" not in canonical
        assert "event_group_id" not in canonical
        assert canonical["event_id"] == "evt_001"
        assert canonical["type"] == "pickup"

        provenance = evt.provenance_dict()
        assert provenance["candidate_id"] == "cand_001"
        assert provenance["actor_id"] == "track_3"
        assert "type" not in provenance
        assert "t_start" not in provenance

    def test_ignore_interval_dict_methods(self):
        """IgnoreIntervalExport.canonical_dict and provenance_dict work."""
        from pickup_putdown.annotation.schemas import IgnoreReason

        ig = IgnoreIntervalExport(
            ignore_id="ign_001",
            clip_id="clip_001",
            t_start=1.0,
            t_end=2.0,
            reason=IgnoreReason.ACTION_OCCLUDED,
            annotator="alice",
            notes="test",
            candidate_id="cand_001",
        )
        canonical = ig.canonical_dict()
        assert "candidate_id" not in canonical
        assert canonical["ignore_id"] == "ign_001"

        provenance = ig.provenance_dict()
        assert provenance["candidate_id"] == "cand_001"

    def test_repeated_export_deterministic(self, tmp_path):
        """Repeated exports produce identical output."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "candidate_clip_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        p1 = tmp_path / "events1.csv"
        p2 = tmp_path / "events2.csv"
        export_candidate_annotations(export, events_output=str(p1))
        export_candidate_annotations(export, events_output=str(p2))
        assert p1.read_text() == p2.read_text()

    def test_legacy_export_still_compatible(self, tmp_path):
        """Legacy (non-candidate) export remains compatible."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/clip.mp4",
                    "clip_id": "clip_001",
                    "fps": 30.0,
                },
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 3.0,
                                    "end": 5.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        events_path = tmp_path / "events.csv"
        result = export_candidate_annotations(
            export,
            events_output=str(events_path),
        )
        assert result.is_valid
        assert len(result.canonical_events) == 1
        assert result.canonical_events[0].t_start == 3.0


# ---------------------------------------------------------------------------
# Fix 3: Deterministic pilot sampling
# ---------------------------------------------------------------------------


class TestPilotSelection:
    """Pilot selection must be deterministic and respect limits."""

    @pytest.fixture
    def sample_candidates(self):
        """Generate 100 valid candidate metadata records."""
        candidates = []
        for i in range(100):
            candidates.append(
                {
                    "candidate_id": f"cand_{i:04d}",
                    "clip_id": f"clip_{i % 10:03d}",
                    "source_start_s": float(i * 5),
                    "source_end_s": float(i * 5 + 10),
                    "candidate_video": f"/videos/cand_{i:04d}.mp4",
                    "actor_id": f"track_{i % 5}",
                    "hand_side": "left" if i % 2 == 0 else "right",
                    "region_id": f"shelf_{i % 3}",
                    "proposal_score": 0.5 + (i % 50) / 100,
                }
            )
        return candidates

    def test_deterministic_selection(self, sample_candidates):
        """Same seed produces identical results."""
        r1 = select_candidate_pilot(sample_candidates, limit=40, seed=42)
        r2 = select_candidate_pilot(sample_candidates, limit=40, seed=42)
        assert len(r1) == 40
        assert r1 == r2
        ids1 = [c["candidate_id"] for c in r1]
        ids2 = [c["candidate_id"] for c in r2]
        assert ids1 == ids2

    def test_different_seeds_different_samples(self, sample_candidates):
        """Different seeds produce different samples (when enough data)."""
        r1 = select_candidate_pilot(sample_candidates, limit=40, seed=42)
        r2 = select_candidate_pilot(sample_candidates, limit=40, seed=99)
        ids1 = {c["candidate_id"] for c in r1}
        ids2 = {c["candidate_id"] for c in r2}
        assert ids1 != ids2

    def test_exact_count(self, sample_candidates):
        """Selection returns exactly the requested count."""
        for limit in [1, 10, 30, 50, 100]:
            selected = select_candidate_pilot(sample_candidates, limit=limit, seed=42)
            assert len(selected) == limit

    def test_stable_ordering(self, sample_candidates):
        """Selected candidates are sorted by candidate_id."""
        selected = select_candidate_pilot(sample_candidates, limit=40, seed=42)
        ids = [c["candidate_id"] for c in selected]
        assert ids == sorted(ids)

    def test_invalid_limit_rejected(self, sample_candidates):
        """Non-positive limit raises ValueError."""
        with pytest.raises(ValueError, match="positive"):
            select_candidate_pilot(sample_candidates, limit=0)
        with pytest.raises(ValueError, match="positive"):
            select_candidate_pilot(sample_candidates, limit=-1)

    def test_fewer_candidates_than_requested(self):
        """When fewer candidates exist than requested, export all."""
        candidates = [
            {
                "candidate_id": f"cand_{i:04d}",
                "clip_id": "clip_001",
                "source_start_s": float(i * 5),
                "source_end_s": float(i * 5 + 10),
                "candidate_video": f"/videos/cand_{i:04d}.mp4",
            }
            for i in range(5)
        ]
        selected = select_candidate_pilot(candidates, limit=50, seed=42)
        assert len(selected) == 5

    def test_no_valid_candidates_raises(self):
        """Empty or all-invalid candidates raises ValueError."""
        with pytest.raises(ValueError, match="No valid candidates"):
            select_candidate_pilot([], limit=10)
        with pytest.raises(ValueError, match="No valid candidates"):
            select_candidate_pilot(
                [{"candidate_id": "", "clip_id": "x"}],
                limit=10,
            )

    def test_no_limit_returns_all(self, sample_candidates):
        """Omitting limit returns all valid candidates."""
        selected = select_candidate_pilot(sample_candidates)
        assert len(selected) == 100

    def test_no_mutation_of_source(self, sample_candidates):
        """Source metadata is not mutated."""
        import copy

        original = copy.deepcopy(sample_candidates)
        select_candidate_pilot(sample_candidates, limit=40, seed=42)
        assert sample_candidates == original


# ---------------------------------------------------------------------------
# Fix 4: Video URL generation and media verification
# ---------------------------------------------------------------------------


class TestVideoUrlGeneration:
    """Video URL generation must match configured mode."""

    def test_s3_key_mode_passthrough(self):
        """S3_KEY mode passes through the raw key."""
        url = _generate_video_url(
            "anon/candidates/videos/camera_01/cand_001.mp4",
            VideoUrlMode.S3_KEY,
        )
        assert url == "anon/candidates/videos/camera_01/cand_001.mp4"

    def test_s3_storage_mode_format(self):
        """S3_STORAGE mode generates s3://bucket/key format."""
        url = _generate_video_url(
            "anon/candidates/videos/camera_01/cand_001.mp4",
            VideoUrlMode.S3_STORAGE,
            s3_bucket="chillnbite-cameras",
        )
        assert url == "s3://chillnbite-cameras/anon/candidates/videos/camera_01/cand_001.mp4"

    def test_s3_storage_mode_preserves_s3_uri(self):
        """S3_STORAGE mode preserves existing s3:// URIs."""
        url = _generate_video_url(
            "s3://my-bucket/path/video.mp4",
            VideoUrlMode.S3_STORAGE,
            s3_bucket="chillnbite-cameras",
        )
        assert url == "s3://my-bucket/path/video.mp4"

    def test_s3_storage_mode_requires_bucket(self):
        """S3_STORAGE mode fails without s3_bucket."""
        with pytest.raises(ValueError, match="s3_bucket is required"):
            _generate_video_url(
                "anon/candidates/videos/cand.mp4",
                VideoUrlMode.S3_STORAGE,
            )

    def test_local_mode_resolves_path(self):
        """LOCAL mode resolves filename under local_video_dir."""
        url = _generate_video_url(
            "anon/candidates/videos/camera_01/cand_001.mp4",
            VideoUrlMode.LOCAL,
            local_video_dir="/data/videos",
        )
        assert url == "/data/videos/cand_001.mp4"

    def test_local_mode_requires_dir(self):
        """LOCAL mode fails without local_video_dir."""
        with pytest.raises(ValueError, match="local_video_dir is required"):
            _generate_video_url(
                "anon/candidates/videos/cand.mp4",
                VideoUrlMode.LOCAL,
            )

    def test_presigned_mode_requires_http(self):
        """PRESIGNED mode requires http(s) URL."""
        with pytest.raises(ValueError, match="presigned mode requires http"):
            _generate_video_url(
                "anon/candidates/videos/cand.mp4",
                VideoUrlMode.PRESIGNED,
            )

    def test_presigned_mode_passes_through(self):
        """PRESIGNED mode passes through http(s) URLs."""
        url = _generate_video_url(
            "https://example.com/presigned-url",
            VideoUrlMode.PRESIGNED,
        )
        assert url == "https://example.com/presigned-url"


class TestMediaCheck:
    """Media reference verification."""

    def test_check_media_local_pass(self, tmp_path):
        """Local mode check passes when file exists."""
        video_file = tmp_path / "cand_001.mp4"
        video_file.touch()
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": str(video_file),
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(str(tasks_path), video_url_mode=VideoUrlMode.LOCAL)
        assert report.total == 1
        assert report.passed == 1
        assert report.failed == 0
        assert report.is_all_ok

    def test_check_media_local_fail(self, tmp_path):
        """Local mode check fails when file missing."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": "/nonexistent/cand_001.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(str(tasks_path), video_url_mode=VideoUrlMode.LOCAL)
        assert report.total == 1
        assert report.passed == 0
        assert report.failed == 1

    def test_check_media_missing_video_ref(self, tmp_path):
        """Missing video reference is reported as failed."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(str(tasks_path), video_url_mode=VideoUrlMode.S3_KEY)
        assert report.failed == 1
        assert "Missing video reference" in report.results[0].message

    def test_check_media_s3_key_format_ok(self, tmp_path):
        """S3_KEY mode validates format only."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": "anon/candidates/videos/camera_01/cand_001.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(str(tasks_path), video_url_mode=VideoUrlMode.S3_KEY)
        assert report.passed == 1
        assert report.failed == 0

    def test_check_media_s3_storage_format_mismatch(self, tmp_path):
        """S3_STORAGE mode fails when URL is not s3://."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": "anon/candidates/videos/cand_001.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(
            str(tasks_path),
            video_url_mode=VideoUrlMode.S3_STORAGE,
        )
        assert report.failed == 1
        assert "Expected s3://" in report.results[0].message

    def test_check_media_presigned_format_ok(self, tmp_path):
        """PRESIGNED mode validates http(s) format."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": "https://example.com/presigned-url",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(
            str(tasks_path),
            video_url_mode=VideoUrlMode.PRESIGNED,
        )
        assert report.passed == 1

    def test_check_media_presigned_format_mismatch(self, tmp_path):
        """PRESIGNED mode fails for non-http URLs."""
        tasks = [
            {
                "id": 1,
                "data": {
                    "video": "anon/candidates/videos/cand_001.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                },
            }
        ]
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps(tasks))

        report = check_media_references(
            str(tasks_path),
            video_url_mode=VideoUrlMode.PRESIGNED,
        )
        assert report.failed == 1
        assert "Expected http" in report.results[0].message

    def test_check_media_invalid_json_array(self, tmp_path):
        """Non-array task file is reported as failed."""
        tasks_path = tmp_path / "tasks.json"
        tasks_path.write_text(json.dumps({"not": "an array"}))

        report = check_media_references(str(tasks_path), video_url_mode=VideoUrlMode.S3_KEY)
        assert report.failed == 1
        assert "JSON array" in report.results[0].message

"""Tests for Task 6.2: Production Annotation Handoff.

Covers candidate metadata validation, Label Studio task generation from
candidates, source-offset timestamp conversion during export, and the full
end-to-end round trip.
"""

from __future__ import annotations

import csv
import json

import pyarrow.parquet as pq
import pytest

from pickup_putdown.annotation.import_export import (
    build_candidate_tasks,
    export_candidate_annotations,
    export_events_csv,
)
from pickup_putdown.annotation.schemas import (
    EventLabel,
    IgnoreReason,
)

# ---------------------------------------------------------------------------
# Test 1: Required metadata appears in generated Label Studio task JSON
# ---------------------------------------------------------------------------


class TestRequiredMetadataInTask:
    def test_all_required_fields_present(self):
        metadata = [
            {
                "candidate_id": "candidate_001",
                "clip_id": "source_clip_001",
                "source_start_s": 102.0,
                "source_end_s": 110.0,
                "candidate_video": "/videos/candidate_001.mp4",
                "actor_id": "track_3",
                "hand_side": "right",
                "region_id": "shelf_2",
                "proposal_score": 0.75,
                "config_fingerprint": "fp_v1",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert not errors
        assert len(tasks) == 1
        data = tasks[0].data
        assert data["candidate_id"] == "candidate_001"
        assert data["clip_id"] == "source_clip_001"
        assert data["source_start_s"] == 102.0
        assert data["source_end_s"] == 110.0
        assert data["video"] == "/videos/candidate_001.mp4"
        assert data["actor_id"] == "track_3"
        assert data["hand_side"] == "right"
        assert data["region_id"] == "shelf_2"
        assert data["proposal_score"] == 0.75
        assert data["config_fingerprint"] == "fp_v1"

    def test_optional_fields_absent_ok(self):
        metadata = [
            {
                "candidate_id": "candidate_002",
                "clip_id": "source_clip_002",
                "source_start_s": 0.0,
                "source_end_s": 8.0,
                "candidate_video": "/videos/candidate_002.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert not errors
        assert len(tasks) == 1
        data = tasks[0].data
        assert "actor_id" not in data
        assert "hand_side" not in data
        assert "region_id" not in data

    def test_candidate_key_fallback(self):
        metadata = [
            {
                "candidate_id": "candidate_003",
                "clip_id": "source_clip_003",
                "source_start_s": 5.0,
                "source_end_s": 15.0,
                "candidate_key": "s3://bucket/candidates/candidate_003.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert not errors
        assert len(tasks) == 1
        assert tasks[0].data["video"] == "s3://bucket/candidates/candidate_003.mp4"


# ---------------------------------------------------------------------------
# Test 2: No default event label is generated
# ---------------------------------------------------------------------------


class TestNoDefaultEventLabel:
    def test_no_predictions_in_task(self):
        metadata = [
            {
                "candidate_id": "candidate_001",
                "clip_id": "source_clip_001",
                "source_start_s": 102.0,
                "source_end_s": 110.0,
                "candidate_video": "/videos/candidate_001.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert not errors
        assert len(tasks[0].predictions) == 0

    def test_serialized_task_has_no_predictions(self):
        metadata = [
            {
                "candidate_id": "candidate_001",
                "clip_id": "source_clip_001",
                "source_start_s": 102.0,
                "source_end_s": 110.0,
                "candidate_video": "/videos/candidate_001.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        payload = json.dumps([t.model_dump() for t in tasks], default=str)
        parsed = json.loads(payload)
        assert "predictions" not in parsed[0] or parsed[0].get("predictions") == []


# ---------------------------------------------------------------------------
# Test 3: Zero source offset preserves annotation timestamps
# ---------------------------------------------------------------------------


class TestZeroSourceOffset:
    def test_zero_offset_preserves_timestamps(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 0.0,
                    "source_end_s": 10.0,
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
        result = export_candidate_annotations(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        assert evt.t_start == 3.0
        assert evt.t_end == 5.0


# ---------------------------------------------------------------------------
# Test 4: Non-zero source offset is added correctly
# ---------------------------------------------------------------------------


class TestNonZeroSourceOffset:
    def test_offset_added_correctly(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 102.0,
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
                                    "start": 2.3,
                                    "end": 3.7,
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
        evt = result.canonical_events[0]
        assert evt.t_start == pytest.approx(104.3)
        assert evt.t_end == pytest.approx(105.7)

    def test_offset_example_from_spec(self):
        """Candidate interval 102.0-110.0, annotation 2.3-3.7 -> source 104.3-105.7."""
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_001",
                    "clip_id": "clip_001",
                    "source_start_s": 102.0,
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
                                    "start": 2.3,
                                    "end": 3.7,
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
        evt = result.canonical_events[0]
        assert evt.t_start == 104.3
        assert evt.t_end == 105.7


# ---------------------------------------------------------------------------
# Test 5: Pickup export conversion
# ---------------------------------------------------------------------------


class TestPickupExportConversion:
    def test_pickup_with_offset(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_pickup",
                    "clip_id": "clip_pickup",
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
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
                                    "start": 1.0,
                                    "end": 3.0,
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
        evt = result.canonical_events[0]
        assert evt.type == EventLabel.PICKUP
        assert evt.t_start == 51.0
        assert evt.t_end == 53.0
        assert evt.candidate_id == "cand_pickup"


# ---------------------------------------------------------------------------
# Test 6: Putdown export conversion
# ---------------------------------------------------------------------------


class TestPutdownExportConversion:
    def test_putdown_with_offset(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_putdown",
                    "clip_id": "clip_putdown",
                    "source_start_s": 200.0,
                    "source_end_s": 210.0,
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
                                    "start": 4.0,
                                    "end": 6.0,
                                    "labels": ["putdown"],
                                    "confidence": "med",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            }
                        ],
                        "who": "bob",
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
        evt = result.canonical_events[0]
        assert evt.type == EventLabel.PUTDOWN
        assert evt.t_start == 204.0
        assert evt.t_end == 206.0
        assert evt.candidate_id == "cand_putdown"


# ---------------------------------------------------------------------------
# Test 7: Ignore interval export conversion
# ---------------------------------------------------------------------------


class TestIgnoreIntervalExportConversion:
    def test_ignore_with_offset(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_ignore",
                    "clip_id": "clip_ignore",
                    "source_start_s": 150.0,
                    "source_end_s": 160.0,
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
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        result = export_candidate_annotations(export)
        assert len(result.canonical_events) == 0
        assert len(result.ignore_intervals) == 1
        ig = result.ignore_intervals[0]
        assert ig.t_start == 152.0
        assert ig.t_end == 154.0
        assert ig.candidate_id == "cand_ignore"


# ---------------------------------------------------------------------------
# Test 8: Candidate-boundary annotations
# ---------------------------------------------------------------------------


class TestCandidateBoundaryAnnotations:
    def test_at_start_boundary(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_boundary",
                    "clip_id": "clip_boundary",
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
                                    "start": 0.0,
                                    "end": 2.0,
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
        evt = result.canonical_events[0]
        assert evt.t_start == 100.0
        assert evt.t_end == 102.0

    def test_at_end_boundary(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_boundary",
                    "clip_id": "clip_boundary",
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
                                    "start": 8.0,
                                    "end": 10.0,
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
        evt = result.canonical_events[0]
        assert evt.t_start == 108.0
        assert evt.t_end == 110.0


# ---------------------------------------------------------------------------
# Test 9: Negative relative timestamps are rejected
# ---------------------------------------------------------------------------


class TestNegativeRelativeTimestamps:
    def test_negative_start_rejected(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_neg",
                    "clip_id": "clip_neg",
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
                                    "start": -1.0,
                                    "end": 2.0,
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
        assert not result.is_valid
        assert len(result.canonical_events) == 0
        neg_errors = [e for e in result.validation.errors if "negative" in e.message.lower()]
        assert len(neg_errors) >= 1


# ---------------------------------------------------------------------------
# Test 10: Relative timestamps beyond candidate duration are rejected
# ---------------------------------------------------------------------------


class TestBeyondDurationTimestamps:
    def test_beyond_duration_rejected(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_beyond",
                    "clip_id": "clip_beyond",
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
                                    "start": 5.0,
                                    "end": 15.0,
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
        assert not result.is_valid
        assert len(result.canonical_events) == 0
        beyond_errors = [
            e for e in result.validation.errors if "exceeds candidate duration" in e.message
        ]
        assert len(beyond_errors) >= 1


# ---------------------------------------------------------------------------
# Test 11: Missing required metadata produces a clear error
# ---------------------------------------------------------------------------


class TestMissingRequiredMetadata:
    def test_missing_candidate_id(self):
        metadata = [
            {
                "clip_id": "clip_001",
                "source_start_s": 0.0,
                "source_end_s": 10.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert len(errors) >= 1
        assert any(e.field_name == "candidate_id" for e in errors)

    def test_missing_clip_id(self):
        metadata = [
            {
                "candidate_id": "cand_001",
                "source_start_s": 0.0,
                "source_end_s": 10.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert any(e.field_name == "clip_id" for e in errors)

    def test_missing_source_start_s(self):
        metadata = [
            {
                "candidate_id": "cand_001",
                "clip_id": "clip_001",
                "source_end_s": 10.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert any(e.field_name == "source_start_s" for e in errors)

    def test_missing_video_location(self):
        metadata = [
            {
                "candidate_id": "cand_001",
                "clip_id": "clip_001",
                "source_start_s": 0.0,
                "source_end_s": 10.0,
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert any(e.field_name == "candidate_video" for e in errors)

    def test_invalid_source_interval(self):
        metadata = [
            {
                "candidate_id": "cand_001",
                "clip_id": "clip_001",
                "source_start_s": 10.0,
                "source_end_s": 5.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert any("source_start_s" in e.field_name for e in errors)


# ---------------------------------------------------------------------------
# Test 12: Invalid source intervals are rejected
# ---------------------------------------------------------------------------


class TestInvalidSourceIntervals:
    def test_source_start_equals_end(self):
        metadata = [
            {
                "candidate_id": "cand_001",
                "clip_id": "clip_001",
                "source_start_s": 10.0,
                "source_end_s": 10.0,
                "candidate_video": "/videos/cand.mp4",
            }
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 0
        assert len(errors) >= 1

    def test_source_start_greater_than_end_in_export(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_inv",
                    "clip_id": "clip_inv",
                    "source_start_s": 110.0,
                    "source_end_s": 100.0,
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
                                    "start": 1.0,
                                    "end": 3.0,
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
        assert not result.is_valid
        inv_errors = [e for e in result.validation.errors if "source_start_s" in e.field_name]
        assert len(inv_errors) >= 1


# ---------------------------------------------------------------------------
# Test 13: Optional actor, hand, and region metadata survive round trip
# ---------------------------------------------------------------------------


class TestOptionalMetadataRoundTrip:
    def test_actor_hand_region_survive(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_meta",
                    "clip_id": "clip_meta",
                    "source_start_s": 100.0,
                    "source_end_s": 110.0,
                    "actor_id": "track_5",
                    "hand_side": "left",
                    "region_id": "shelf_1",
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
        evt = result.canonical_events[0]
        assert evt.candidate_id == "cand_meta"
        assert evt.actor_id == "track_5"
        assert evt.hand_side == "left"
        assert evt.region_id == "shelf_1"


# ---------------------------------------------------------------------------
# Test 14: Multi-item row expansion remains deterministic
# ---------------------------------------------------------------------------


class TestMultiItemDeterminism:
    def test_multi_item_deterministic(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_multi",
                    "clip_id": "clip_multi",
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
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
                                    "item_count": 2,
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
        assert len(result.canonical_events) == 2
        assert (
            result.canonical_events[0].event_group_id == result.canonical_events[1].event_group_id
        )
        assert result.canonical_events[0].event_id != result.canonical_events[1].event_id
        # Both share the same candidate traceability
        assert result.canonical_events[0].candidate_id == "cand_multi"
        assert result.canonical_events[1].candidate_id == "cand_multi"

    def test_multi_item_same_source_timestamps(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_multi",
                    "clip_id": "clip_multi",
                    "source_start_s": 50.0,
                    "source_end_s": 60.0,
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
                                    "item_count": 2,
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
        assert result.canonical_events[0].t_start == 52.0
        assert result.canonical_events[1].t_start == 52.0
        assert result.canonical_events[0].t_end == 54.0
        assert result.canonical_events[1].t_end == 54.0


# ---------------------------------------------------------------------------
# Test 15: Existing non-candidate Label Studio exports remain compatible
# ---------------------------------------------------------------------------


class TestLegacyExportCompatibility:
    def test_legacy_export_still_works(self):
        """Legacy tasks without source_start_s still export correctly."""
        export = [
            {
                "id": 1,
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
                "task": {
                    "id": "task_legacy",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_legacy",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_legacy",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        result = export_candidate_annotations(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        assert evt.t_start == 3.0
        assert evt.t_end == 5.0
        assert evt.candidate_id is None

    def test_legacy_export_events_csv_still_works(self):
        """The existing export_events_csv function still works."""
        export = [
            {
                "id": 1,
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
                "task": {
                    "id": "task_legacy",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_legacy",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_legacy",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        result = export_events_csv(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        assert result.canonical_events[0].t_start == 3.0


# ---------------------------------------------------------------------------
# Test 16: Repeated exports produce identical results
# ---------------------------------------------------------------------------


class TestDeterministicExport:
    def test_repeated_candidate_export_identical(self):
        export = [
            {
                "id": 1,
                "data": {
                    "video": "/videos/cand.mp4",
                    "candidate_id": "cand_det",
                    "clip_id": "clip_det",
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
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            }
        ]
        r1 = export_candidate_annotations(export)
        r2 = export_candidate_annotations(export)
        assert len(r1.canonical_events) == len(r2.canonical_events)
        for e1, e2 in zip(r1.canonical_events, r2.canonical_events, strict=True):
            assert e1.event_id == e2.event_id
            assert e1.t_start == e2.t_start
            assert e1.t_end == e2.t_end
            assert e1.type == e2.type
            assert e1.candidate_id == e2.candidate_id


# ---------------------------------------------------------------------------
# End-to-end fixture: candidate metadata -> task JSON -> simulated export
# -> canonical events.csv and ignore output
# ---------------------------------------------------------------------------


class TestEndToEndFixture:
    """Full pipeline: candidate metadata -> tasks -> annotations -> export."""

    def test_full_round_trip(self, tmp_path):
        """End-to-end test with non-zero source_start_s."""
        # Step 1: Candidate metadata
        candidate_metadata = [
            {
                "candidate_id": "e2e_cand_001",
                "clip_id": "e2e_source_clip",
                "source_start_s": 102.0,
                "source_end_s": 112.0,
                "candidate_video": str(tmp_path / "candidate_001.mp4"),
                "actor_id": "track_3",
                "hand_side": "right",
                "region_id": "shelf_2",
                "proposal_score": 0.82,
                "config_fingerprint": "fp_v1",
            },
            {
                "candidate_id": "e2e_cand_002",
                "clip_id": "e2e_source_clip",
                "source_start_s": 0.0,
                "source_end_s": 8.0,
                "candidate_video": str(tmp_path / "candidate_002.mp4"),
                "actor_id": "track_1",
                "hand_side": "left",
                "region_id": "shelf_1",
            },
        ]

        # Step 2: Build Label Studio tasks
        tasks, errors = build_candidate_tasks(candidate_metadata)
        assert not errors
        assert len(tasks) == 2

        # Verify no default event label
        for task in tasks:
            assert len(task.predictions) == 0

        # Verify task ordering is deterministic (sorted by candidate_id)
        assert tasks[0].data["candidate_id"] == "e2e_cand_001"
        assert tasks[1].data["candidate_id"] == "e2e_cand_002"

        # Step 3: Simulate Label Studio export (annotator adds events)
        simulated_export = [
            {
                "id": 1,
                "data": {
                    "video": str(tmp_path / "candidate_001.mp4"),
                    "candidate_id": "e2e_cand_001",
                    "clip_id": "e2e_source_clip",
                    "source_start_s": 102.0,
                    "source_end_s": 112.0,
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
                                    "start": 2.3,
                                    "end": 3.7,
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
            },
            {
                "id": 2,
                "data": {
                    "video": str(tmp_path / "candidate_002.mp4"),
                    "candidate_id": "e2e_cand_002",
                    "clip_id": "e2e_source_clip",
                    "source_start_s": 0.0,
                    "source_end_s": 8.0,
                    "actor_id": "track_1",
                    "hand_side": "left",
                    "region_id": "shelf_1",
                },
                "annotations": [
                    {
                        "id": 2,
                        "result": [
                            {
                                "id": "r2",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 1.0,
                                    "end": 2.5,
                                    "labels": ["putdown"],
                                    "confidence": "med",
                                    "hard_case": "false",
                                    "item_count": 1,
                                },
                            },
                            {
                                "id": "r3",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 3.0,
                                    "end": 5.0,
                                    "labels": ["ignore"],
                                    "reason": "ACTION_OCCLUDED",
                                },
                            },
                        ],
                        "who": "bob",
                        "state": [
                            {
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["true"]},
                            }
                        ],
                    }
                ],
            },
        ]

        # Step 4: Export with source offset conversion
        events_path = tmp_path / "events.csv"
        ignore_path = tmp_path / "ignore_intervals.parquet"
        result = export_candidate_annotations(
            simulated_export,
            events_output=str(events_path),
            ignore_output=str(ignore_path),
        )

        # Step 5: Validate events
        assert result.is_valid
        assert len(result.canonical_events) == 2

        # Events sorted by (clip_id, t_start, type, event_id)
        # putdown (1.0) comes before pickup (104.3)
        putdown = result.canonical_events[0]
        assert putdown.type == EventLabel.PUTDOWN
        assert putdown.t_start == pytest.approx(1.0)
        assert putdown.t_end == pytest.approx(2.5)
        assert putdown.clip_id == "e2e_source_clip"
        assert putdown.candidate_id == "e2e_cand_002"

        pickup = result.canonical_events[1]
        assert pickup.type == EventLabel.PICKUP
        assert pickup.t_start == pytest.approx(104.3)
        assert pickup.t_end == pytest.approx(105.7)
        assert pickup.clip_id == "e2e_source_clip"
        assert pickup.candidate_id == "e2e_cand_001"
        assert pickup.actor_id == "track_3"
        assert pickup.hand_side == "right"
        assert pickup.region_id == "shelf_2"

        # Step 6: Validate ignore intervals
        assert len(result.ignore_intervals) == 1
        ig = result.ignore_intervals[0]
        assert ig.t_start == pytest.approx(3.0)
        assert ig.t_end == pytest.approx(5.0)
        assert ig.candidate_id == "e2e_cand_002"
        assert ig.reason == IgnoreReason.ACTION_OCCLUDED

        # Step 7: Validate CSV file
        assert events_path.exists()
        with events_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        # Sorted by t_start: putdown (1.0) before pickup (104.3)
        assert float(rows[0]["t_start"]) == pytest.approx(1.0)
        assert float(rows[1]["t_start"]) == pytest.approx(104.3)

        # Step 8: Validate Parquet file
        assert ignore_path.exists()
        table = pq.read_table(str(ignore_path))
        assert len(table) == 1

    def test_candidate_validation_rejects_invalid(self, tmp_path):
        """Invalid candidate metadata is rejected with clear errors."""
        metadata = [
            {
                "candidate_id": "valid_cand",
                "clip_id": "clip_001",
                "source_start_s": 10.0,
                "source_end_s": 20.0,
                "candidate_video": str(tmp_path / "valid.mp4"),
            },
            {
                "candidate_id": "",
                "clip_id": "clip_002",
                "source_start_s": 0.0,
                "source_end_s": 10.0,
                "candidate_video": str(tmp_path / "invalid.mp4"),
            },
        ]
        tasks, errors = build_candidate_tasks(metadata)
        assert len(tasks) == 1
        assert len(errors) >= 1
        assert tasks[0].data["candidate_id"] == "valid_cand"

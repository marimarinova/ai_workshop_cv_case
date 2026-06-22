"""Tests for annotation import/export: schema validation, conversion, and round-trip fidelity.

Tests cover:
1. No-event reviewed clip exports zero official events.
2. One visible pickup exports one row.
3. Immediate pickup then putdown exports two ordered rows.
4. Two-item pickup exports two rows sharing one event_group_id.
5. Fully occluded action exports only an ignore interval.
6. Ignore intervals never appear in events.csv.
7. Low-confidence visible event remains in events.csv.
8. Malformed labels or attributes fail validation.
10. Invalid item_count fails validation.
11. Candidate suggestions are imported as predictions, not annotations.
12. Candidates can be absent.
13. A corrected candidate changes the final export.
14. A deleted candidate produces no official event.
15. A manually added event absent from candidates is exported.
16. Frame or timestamp round-trip fidelity.
17. Deterministic repeated export.
18. Canonical CSV columns and allowed values match the case schema.
19. Parquet ignore schema is stable.
20. Final export requires complete-active-span review confirmation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from pickup_putdown.annotation.import_export import (
    build_label_studio_tasks,
    convert_candidates_to_predictions,
    export_events_csv,
    export_ignore_intervals_parquet,
    round_trip_check,
    validate_export,
)
from pickup_putdown.annotation.schemas import (
    AnnotationEvent,
    AnnotationRegion,
    CanonicalEvent,
    ConfidenceLevel,
    ConversionResult,
    EventLabel,
    IgnoreIntervalExport,
    IgnoreReason,
    LabelStudioPrediction,
    ReviewMetadata,
    ReviewStatus,
    ValidationError,
    ValidationErrors,
)

FIXTURES = Path(__file__).parent / "fixtures" / "annotation"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def acceptance_fixture():
    path = FIXTURES / "acceptance_fixture.json"
    return json.loads(path.read_text())


@pytest.fixture
def label_studio_export():
    path = FIXTURES / "label_studio_export.json"
    return json.loads(path.read_text())


@pytest.fixture
def empty_reviewed_clip():
    """A reviewed clip with zero events."""
    return [
        {
            "id": 1,
            "annotations": [
                {
                    "id": 1,
                    "result": [],
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
                "id": "task_no_events",
                "data": {
                    "video": "/videos/clip_no_events.mp4",
                    "clip_id": "clip_no_events",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_no_events",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "alice",
                },
            },
        }
    ]


@pytest.fixture
def single_pickup_export():
    """A single visible pickup."""
    return [
        {
            "id": 2,
            "annotations": [
                {
                    "id": 2,
                    "result": [
                        {
                            "id": "region_pickup_1",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 4.0,
                                "end": 6.0,
                                "labels": ["pickup"],
                                "confidence": "high",
                                "hard_case": "false",
                                "item_count": 1,
                                "review_status": "accepted",
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
                "id": "task_single_pickup",
                "data": {
                    "video": "/videos/clip_single_pickup.mp4",
                    "clip_id": "clip_single_pickup",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_single_pickup",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "alice",
                },
            },
        }
    ]


@pytest.fixture
def pickup_putdown_export():
    """Immediate pickup then putdown."""
    return [
        {
            "id": 3,
            "annotations": [
                {
                    "id": 3,
                    "result": [
                        {
                            "id": "region_pickup_2",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 3.0,
                                "end": 4.5,
                                "labels": ["pickup"],
                                "confidence": "high",
                                "hard_case": "false",
                                "item_count": 1,
                                "review_status": "accepted",
                            },
                        },
                        {
                            "id": "region_putdown_2",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 5.0,
                                "end": 6.5,
                                "labels": ["putdown"],
                                "confidence": "high",
                                "hard_case": "false",
                                "item_count": 1,
                                "review_status": "accepted",
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
            "task": {
                "id": "task_pickup_putdown",
                "data": {
                    "video": "/videos/clip_pickup_putdown.mp4",
                    "clip_id": "clip_pickup_putdown",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_pickup_putdown",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "bob",
                },
            },
        }
    ]


@pytest.fixture
def two_item_pickup_export():
    """Two-item pickup."""
    return [
        {
            "id": 4,
            "annotations": [
                {
                    "id": 4,
                    "result": [
                        {
                            "id": "region_two_item",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 4.0,
                                "end": 6.0,
                                "labels": ["pickup"],
                                "confidence": "high",
                                "hard_case": "false",
                                "item_count": 2,
                                "review_status": "accepted",
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
                "id": "task_two_item_pickup",
                "data": {
                    "video": "/videos/clip_two_item_pickup.mp4",
                    "clip_id": "clip_two_item_pickup",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_two_item_pickup",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "alice",
                },
            },
        }
    ]


@pytest.fixture
def ignore_export():
    """Fully occluded ignore interval."""
    return [
        {
            "id": 6,
            "annotations": [
                {
                    "id": 6,
                    "result": [
                        {
                            "id": "region_ignore_1",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 2.0,
                                "end": 4.0,
                                "labels": ["ignore"],
                                "reason": "ACTION_OCCLUDED",
                                "notes": "Hand fully occluded",
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
                "id": "task_ignore",
                "data": {
                    "video": "/videos/clip_ignore.mp4",
                    "clip_id": "clip_ignore",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_ignore",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "alice",
                },
            },
        }
    ]


@pytest.fixture
def low_confidence_export():
    """Low-confidence visible event."""
    return [
        {
            "id": 5,
            "annotations": [
                {
                    "id": 5,
                    "result": [
                        {
                            "id": "region_low_conf",
                            "from_name": "labels",
                            "to_name": "video",
                            "type": "timelabels",
                            "value": {
                                "start": 5.0,
                                "end": 7.0,
                                "labels": ["pickup"],
                                "confidence": "low",
                                "hard_case": "true",
                                "item_count": 1,
                                "review_status": "reviewed",
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
            "task": {
                "id": "task_low_confidence",
                "data": {
                    "video": "/videos/clip_low_confidence.mp4",
                    "clip_id": "clip_low_confidence",
                    "fps": 30.0,
                },
                "meta": {
                    "clip_id": "clip_low_confidence",
                    "fps": 30.0,
                    "complete_active_span_reviewed": True,
                    "annotator": "bob",
                },
            },
        }
    ]


# ---------------------------------------------------------------------------
# Test 1: No-event reviewed clip exports zero official events
# ---------------------------------------------------------------------------


class TestNoEventClip:
    def test_exports_zero_events(self, empty_reviewed_clip, tmp_path):
        result = export_events_csv(empty_reviewed_clip, tmp_path / "events.csv")
        assert result.is_valid
        assert len(result.canonical_events) == 0


# ---------------------------------------------------------------------------
# Test 2: One visible pickup exports one row
# ---------------------------------------------------------------------------


class TestSinglePickup:
    def test_exports_one_row(self, single_pickup_export, tmp_path):
        result = export_events_csv(single_pickup_export, tmp_path / "events.csv")
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        assert evt.type == EventLabel.PICKUP
        assert evt.t_start == 4.0
        assert evt.t_end == 6.0
        assert evt.confidence == ConfidenceLevel.HIGH


# ---------------------------------------------------------------------------
# Test 3: Immediate pickup then putdown exports two ordered rows
# ---------------------------------------------------------------------------


class TestPickupPutdown:
    def test_exports_two_ordered_rows(self, pickup_putdown_export, tmp_path):
        result = export_events_csv(pickup_putdown_export, tmp_path / "events.csv")
        assert result.is_valid
        assert len(result.canonical_events) == 2
        assert result.canonical_events[0].type == EventLabel.PICKUP
        assert result.canonical_events[1].type == EventLabel.PUTDOWN
        # Pickup starts before putdown
        assert result.canonical_events[0].t_start < result.canonical_events[1].t_start


# ---------------------------------------------------------------------------
# Test 4: Two-item pickup exports two rows sharing one event_group_id
# ---------------------------------------------------------------------------


class TestTwoItemPickup:
    def test_exports_two_rows_same_group_id(self, two_item_pickup_export, tmp_path):
        result = export_events_csv(two_item_pickup_export, tmp_path / "events.csv")
        assert result.is_valid
        assert len(result.canonical_events) == 2
        assert (
            result.canonical_events[0].event_group_id == result.canonical_events[1].event_group_id
        )
        assert result.canonical_events[0].event_group_id != ""


# ---------------------------------------------------------------------------
# Test 5: Fully occluded action exports only an ignore interval
# ---------------------------------------------------------------------------


class TestOccludedIgnore:
    def test_exports_ignore_interval(self, ignore_export, tmp_path):
        events_result = export_events_csv(ignore_export, tmp_path / "events.csv")
        assert events_result.is_valid
        assert len(events_result.canonical_events) == 0

        ignore_result = export_ignore_intervals_parquet(ignore_export, tmp_path / "ignore.parquet")
        assert len(ignore_result.ignore_intervals) == 1
        ig = ignore_result.ignore_intervals[0]
        assert ig.t_start == 2.0
        assert ig.t_end == 4.0
        assert ig.reason == IgnoreReason.ACTION_OCCLUDED


# ---------------------------------------------------------------------------
# Test 6: Ignore intervals never appear in events.csv
# ---------------------------------------------------------------------------


class TestIgnoreNotInEvents:
    def test_ignore_not_in_events(self, ignore_export, tmp_path):
        result = export_events_csv(ignore_export, tmp_path / "events.csv")
        for evt in result.canonical_events:
            assert evt.type != EventLabel.IGNORE


# ---------------------------------------------------------------------------
# Test 7: Low-confidence visible event remains in events.csv
# ---------------------------------------------------------------------------


class TestLowConfidence:
    def test_low_confidence_in_events(self, low_confidence_export, tmp_path):
        result = export_events_csv(low_confidence_export, tmp_path / "events.csv")
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        assert evt.type == EventLabel.PICKUP
        assert evt.confidence == ConfidenceLevel.LOW


# ---------------------------------------------------------------------------
# Test 8: Malformed labels or attributes fail validation
# ---------------------------------------------------------------------------


class TestValidationFailures:
    def test_unknown_label_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["touch"],
                                    "start": 1.0,
                                    "end": 3.0,
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid
        label_errors = [e for e in errors.errors if "Unknown event label" in e.message]
        assert len(label_errors) >= 1

    def test_invalid_item_count_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "item_count": 0,
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid
        ic_errors = [e for e in errors.errors if "item_count" in e.field_name]
        assert len(ic_errors) >= 1

    def test_negative_timestamp_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": -1.0,
                                    "end": 3.0,
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid

    def test_invalid_confidence_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "confidence": "critical",
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid
        conf_errors = [e for e in errors.errors if "confidence" in e.field_name]
        assert len(conf_errors) >= 1

    def test_invalid_review_status_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "review_status": "approved",
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid
        rs_errors = [e for e in errors.errors if "review_status" in e.field_name]
        assert len(rs_errors) >= 1


# ---------------------------------------------------------------------------
# Test 10: Invalid item_count fails validation
# ---------------------------------------------------------------------------


class TestInvalidItemCount:
    def test_item_count_zero_fails(self):
        with pytest.raises(ValueError):
            AnnotationEvent(
                region_id="r1",
                clip_id="c1",
                label=EventLabel.PICKUP,
                start_frame=10,
                end_frame=20,
                start_time=0.333,
                end_time=0.666,
                item_count=0,
            )

    def test_item_count_negative_fails(self):
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "item_count": -1,
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        errors = validate_export(export)
        assert not errors.is_valid
        ic_errors = [e for e in errors.errors if "item_count" in e.field_name]
        assert len(ic_errors) >= 1


# ---------------------------------------------------------------------------
# Test 11: Candidate suggestions are imported as predictions, not annotations
# ---------------------------------------------------------------------------


class TestCandidateAsPrediction:
    def test_candidates_become_predictions(self, acceptance_fixture):
        candidates = acceptance_fixture.get("candidates", [])
        predictions = convert_candidates_to_predictions(candidates)
        assert len(predictions) == len(candidates)
        for pred in predictions:
            assert isinstance(pred, LabelStudioPrediction)
            assert pred.candidate_id is not None

    def test_prediction_has_candidate_metadata(self, acceptance_fixture):
        candidates = acceptance_fixture.get("candidates", [])
        predictions = convert_candidates_to_predictions(candidates)
        pred = predictions[0]
        assert pred.candidate_source == "wrist_entered_region"
        assert pred.candidate_score == 0.55


# ---------------------------------------------------------------------------
# Test 12: Candidates can be absent
# ---------------------------------------------------------------------------


class TestNoCandidates:
    def test_tasks_without_candidates(self):
        clips = [
            {
                "clip_id": "clip_no_cands",
                "video_path": "/videos/clip.mp4",
                "fps": 30.0,
                "duration_s": 10.0,
                "active_start_s": 1.0,
                "active_end_s": 9.0,
            }
        ]
        tasks = build_label_studio_tasks(clips, candidates=None)
        assert len(tasks) == 1
        assert tasks[0].predictions == []

    def test_empty_candidates_list(self):
        clips = [
            {
                "clip_id": "clip_empty",
                "video_path": "/videos/clip.mp4",
                "fps": 30.0,
                "duration_s": 10.0,
                "active_start_s": 1.0,
                "active_end_s": 9.0,
            }
        ]
        tasks = build_label_studio_tasks(clips, candidates=[])
        assert len(tasks) == 1
        assert tasks[0].predictions == []


# ---------------------------------------------------------------------------
# Test 13: A corrected candidate changes the final export
# ---------------------------------------------------------------------------


class TestCorrectedCandidate:
    def test_corrected_candidate_reflected_in_export(self):
        """A corrected annotation replaces the candidate suggestion.
        The export reflects the corrected annotation, not the original candidate.
        """
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_corrected",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 5.0,
                                    "end": 7.0,
                                    "labels": ["putdown"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
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
                    "id": "task_corrected",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_corrected",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_corrected",
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
        assert result.canonical_events[0].type == EventLabel.PUTDOWN
        assert result.canonical_events[0].t_start == 5.0


# ---------------------------------------------------------------------------
# Test 14: A deleted candidate produces no official event
# ---------------------------------------------------------------------------


class TestDeletedCandidate:
    def test_deleted_candidate_no_event(self):
        """When an annotator deletes a candidate suggestion, no event is exported."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [],
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
                    "id": "task_deleted",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_deleted",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_deleted",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        result = export_events_csv(export)
        assert result.is_valid
        assert len(result.canonical_events) == 0


# ---------------------------------------------------------------------------
# Test 15: Manually added event absent from candidates is exported
# ---------------------------------------------------------------------------


class TestManuallyAddedEvent:
    def test_manually_added_event_exported(self):
        """An event added by the annotator (not from candidates) is exported."""
        export = [
            {
                "id": 9,
                "annotations": [
                    {
                        "id": 9,
                        "result": [
                            {
                                "id": "region_manually_added",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 6.0,
                                    "end": 8.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
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
                "task": {
                    "id": "task_manually_added",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_manually_added",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_manually_added",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "bob",
                    },
                },
            }
        ]
        result = export_events_csv(export)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        assert result.canonical_events[0].type == EventLabel.PICKUP
        assert result.canonical_events[0].t_start == 6.0


# ---------------------------------------------------------------------------
# Test 16: Frame or timestamp round-trip fidelity
# ---------------------------------------------------------------------------


class TestRoundTripFidelity:
    def test_frame_to_time_round_trip(self):
        """Frame indices round-trip to seconds and back within tolerance."""
        fps = 30.0
        original_frames = [0, 15, 30, 45, 60, 90, 120]
        for frame in original_frames:
            time_s = frame / fps
            back_frame = int(time_s * fps)
            assert back_frame == frame, (
                f"Frame {frame} -> {time_s}s -> {back_frame} (expected {frame})"
            )

    def test_round_trip_check_function(self, single_pickup_export):
        """The round_trip_check function correctly validates fidelity."""
        original = [
            CanonicalEvent(
                event_id="test_evt",
                clip_id="clip_single_pickup",
                type=EventLabel.PICKUP,
                t_start=4.0,
                t_end=6.0,
            )
        ]
        assert round_trip_check(original, single_pickup_export, fps=30.0) is True

    def test_round_trip_fails_on_mismatch(self):
        """Round-trip fails when timestamps differ beyond tolerance."""
        original = [
            CanonicalEvent(
                event_id="test_evt",
                clip_id="c1",
                type=EventLabel.PICKUP,
                t_start=4.0,
                t_end=6.0,
            )
        ]
        mismatched = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 10.0,
                                    "end": 12.0,
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        assert round_trip_check(original, mismatched, fps=30.0) is False


# ---------------------------------------------------------------------------
# Test 17: Deterministic repeated export
# ---------------------------------------------------------------------------


class TestDeterministicExport:
    def test_repeated_export_same_result(self, pickup_putdown_export):
        """Running export twice produces identical canonical events."""
        result1 = export_events_csv(pickup_putdown_export)
        result2 = export_events_csv(pickup_putdown_export)
        assert len(result1.canonical_events) == len(result2.canonical_events)
        for e1, e2 in zip(result1.canonical_events, result2.canonical_events, strict=True):
            assert e1.event_id == e2.event_id
            assert e1.t_start == e2.t_start
            assert e1.t_end == e2.t_end
            assert e1.type == e2.type


# ---------------------------------------------------------------------------
# Test 18: Canonical CSV columns and allowed values match the case schema
# ---------------------------------------------------------------------------


class TestCanonicalSchema:
    def test_csv_columns_match(self, tmp_path):
        """Exported CSV has exactly the canonical columns in order."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {"clip_id": "c1", "fps": 30.0},
                },
            }
        ]
        path = tmp_path / "events.csv"
        export_events_csv(export, path)
        with path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == [
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

    def test_allowed_event_types(self):
        """Only pickup and putdown are valid event types."""
        for label in ["pickup", "putdown"]:
            evt = CanonicalEvent(
                event_id="e1",
                clip_id="c1",
                type=EventLabel(label),
                t_start=1.0,
                t_end=3.0,
            )
            assert evt.type == EventLabel(label)

    def test_allowed_confidence_values(self):
        """Only high, med, low are valid confidence values."""
        for conf in ["high", "med", "low"]:
            cv = ConfidenceLevel(conf)
            assert cv in (ConfidenceLevel.HIGH, ConfidenceLevel.MED, ConfidenceLevel.LOW)


# ---------------------------------------------------------------------------
# Test 19: Parquet ignore schema is stable
# ---------------------------------------------------------------------------


class TestParquetIgnoreSchema:
    def test_parquet_columns(self, ignore_export, tmp_path):
        """Parquet ignore file has the expected columns."""
        result = export_ignore_intervals_parquet(ignore_export, tmp_path / "ignore.parquet")
        assert len(result.ignore_intervals) == 1

        table = pq.read_table(str(tmp_path / "ignore.parquet"))
        column_names = table.column_names
        expected = [
            "ignore_id",
            "clip_id",
            "t_start",
            "t_end",
            "reason",
            "annotator",
            "notes",
        ]
        assert column_names == expected

    def test_parquet_types(self, ignore_export, tmp_path):
        """Parquet column types are stable."""
        export_ignore_intervals_parquet(ignore_export, tmp_path / "ignore.parquet")
        table = pq.read_table(str(tmp_path / "ignore.parquet"))
        schema = table.schema
        assert schema.field("ignore_id").type == pa.string()
        assert schema.field("clip_id").type == pa.string()
        assert schema.field("t_start").type == pa.float64()
        assert schema.field("t_end").type == pa.float64()
        assert schema.field("reason").type == pa.string()


# ---------------------------------------------------------------------------
# Test 20: Final export requires complete-active-span review confirmation
# ---------------------------------------------------------------------------


class TestReviewConfirmation:
    def test_unreviewed_clip_no_events(self):
        """Clips without complete_active_span_reviewed=true produce no events."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r1",
                                "value": {
                                    "labels": ["pickup"],
                                    "start": 1.0,
                                    "end": 3.0,
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            }
                        ],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {
                        "clip_id": "c1",
                        "fps": 30.0,
                        "complete_active_span_reviewed": False,
                    },
                },
            }
        ]
        result = export_events_csv(export)
        assert len(result.canonical_events) == 0
        assert not result.is_valid
        review_errors = [
            e for e in result.validation.errors if "complete_active_span_reviewed" in e.message
        ]
        assert len(review_errors) >= 1

    def test_unreviewed_clip_with_zero_events_still_has_flag(self):
        """Zero-event clips must still have the confirmation flag set."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [],
                        "annotation_id": "a1",
                    }
                ],
                "task": {
                    "id": "t1",
                    "meta": {
                        "clip_id": "c1",
                        "fps": 30.0,
                        "complete_active_span_reviewed": False,
                    },
                },
            }
        ]
        errors = validate_export(export)
        # The flag should be checked even for zero-event clips
        assert not errors.is_valid


# ---------------------------------------------------------------------------
# Schema model tests
# ---------------------------------------------------------------------------


class TestSchemaModels:
    def test_annotation_region_validation(self):
        """AnnotationRegion enforces end > start."""
        with pytest.raises(ValueError):
            AnnotationRegion(
                start_frame=20,
                end_frame=10,
                start_time=0.666,
                end_time=0.333,
                labels=[EventLabel.PICKUP],
            )

    def test_canonical_event_validation(self):
        """CanonicalEvent enforces t_end > t_start."""
        with pytest.raises(ValueError):
            CanonicalEvent(
                event_id="e1",
                clip_id="c1",
                type=EventLabel.PICKUP,
                t_start=5.0,
                t_end=3.0,
            )

    def test_ignore_interval_validation(self):
        """IgnoreIntervalExport enforces t_end > t_start."""
        with pytest.raises(ValueError):
            IgnoreIntervalExport(
                ignore_id="i1",
                clip_id="c1",
                t_start=5.0,
                t_end=3.0,
                reason=IgnoreReason.ACTION_OCCLUDED,
            )

    def test_review_metadata_defaults(self):
        """ReviewMetadata has sensible defaults."""
        rm = ReviewMetadata()
        assert rm.complete_active_span_reviewed is False
        assert rm.review_status == ReviewStatus.DRAFT

    def test_conversion_result_validity(self):
        """ConversionResult.is_valid reflects validation state."""
        cr = ConversionResult()
        assert cr.is_valid is True
        cr.validation.add(ValidationError(message="test error"))
        assert cr.is_valid is False

    def test_validation_errors_container(self):
        """ValidationErrors.add_generic works."""
        ve = ValidationErrors()
        ve.add_generic("t1", "something went wrong")
        assert len(ve.errors) == 1
        assert ve.errors[0].task_id == "t1"

    def test_event_label_enum(self):
        """EventLabel has all required values."""
        assert EventLabel.PICKUP == "pickup"
        assert EventLabel.PUTDOWN == "putdown"
        assert EventLabel.IGNORE == "ignore"

    def test_confidence_level_enum(self):
        """ConfidenceLevel has all required values."""
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MED == "med"
        assert ConfidenceLevel.LOW == "low"

    def test_review_status_enum(self):
        """ReviewStatus has all required values."""
        assert ReviewStatus.DRAFT == "draft"
        assert ReviewStatus.REVIEWED == "reviewed"
        assert ReviewStatus.ACCEPTED == "accepted"
        assert ReviewStatus.NEEDS_ADJUDICATION == "needs_adjudication"

    def test_ignore_reason_enum(self):
        """IgnoreReason has all required values."""
        assert IgnoreReason.ACTION_OCCLUDED == "ACTION_OCCLUDED"
        assert IgnoreReason.ACTION_OUT_OF_FRAME == "ACTION_OUT_OF_FRAME"
        assert IgnoreReason.CLIP_BOUNDARY == "CLIP_BOUNDARY"
        assert IgnoreReason.UNLABELABLE == "UNLABELABLE"
        assert IgnoreReason.CORRUPT_SECTION == "CORRUPT_SECTION"


# ---------------------------------------------------------------------------
# Acceptance fixture integration test
# ---------------------------------------------------------------------------


class TestAcceptanceFixture:
    def test_acceptance_fixture_exports(self, acceptance_fixture, label_studio_export):
        """Run the full acceptance fixture through the export pipeline."""
        result = export_events_csv(label_studio_export)
        # Should have events from: single_pickup, pickup_putdown(2),
        # two_item_pickup(2), low_confidence, corrected(1), manually_added(1)
        assert result.is_valid
        assert len(result.canonical_events) >= 7

        ignore_result = export_ignore_intervals_parquet(label_studio_export)
        assert len(ignore_result.ignore_intervals) >= 1

    def test_acceptance_fixture_deterministic(self, label_studio_export):
        """Acceptance fixture export is deterministic."""
        r1 = export_events_csv(label_studio_export)
        r2 = export_events_csv(label_studio_export)
        assert len(r1.canonical_events) == len(r2.canonical_events)
        for e1, e2 in zip(r1.canonical_events, r2.canonical_events, strict=True):
            assert e1.event_id == e2.event_id


# ---------------------------------------------------------------------------
# Task 6 Acceptance Test Suite
# ---------------------------------------------------------------------------


@pytest.mark.annotation_acceptance
class TestTask6Acceptance:
    """Acceptance tests proving the Task 6 handoff contract.

    These tests cover the full Task 6 export pipeline:
    1.  Exact canonical columns and values
    2.  Immediate pickup followed by putdown
    3.  Two-item pickup
    4.  Ignore intervals excluded from official events
    5.  Candidate correction (human annotation overrides candidate)
    6.  Candidate deletion (no event for deleted candidate)
    7.  Candidate supplementation (manually added event exported)
    8.  Full active-span review (unconfirmed vs confirmed zero-event)
    9.  Timestamp round-trip fidelity
    10. Deterministic export
    """

    # ------------------------------------------------------------------
    # 1. Exact canonical columns and values
    # ------------------------------------------------------------------

    def test_01_exact_canonical_columns_and_values(self, tmp_path):
        """Assert that the CSV header exactly matches the canonical schema
        and that exported values use only allowed canonical values."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "r_pickup",
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
                                    "review_status": "accepted",
                                },
                            },
                            {
                                "id": "r_putdown",
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
                                    "review_status": "accepted",
                                },
                            },
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_canonical",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_canonical",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 2

        # Verify exact canonical column order
        with csv_path.open() as fh:
            reader = csv.reader(fh)
            header = next(reader)
        assert header == [
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

        # Verify exported values use only allowed canonical values
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                assert row["type"] in ("pickup", "putdown")
                assert row["confidence"] in ("high", "med", "low")

    # ------------------------------------------------------------------
    # 2. Immediate pickup followed by putdown
    # ------------------------------------------------------------------

    def test_02_immediate_pickup_then_putdown(self, tmp_path):
        """Two separate events: pickup then putdown, chronologically ordered."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_pickup",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 3.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            },
                            {
                                "id": "region_putdown",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 3.0,
                                    "end": 4.0,
                                    "labels": ["putdown"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            },
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_immediate",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_immediate",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 2
        assert result.canonical_events[0].type == EventLabel.PICKUP
        assert result.canonical_events[1].type == EventLabel.PUTDOWN
        assert result.canonical_events[0].t_start < result.canonical_events[1].t_start

    # ------------------------------------------------------------------
    # 3. Two-item pickup
    # ------------------------------------------------------------------

    def test_03_two_item_pickup(self, tmp_path):
        """item_count=2 produces two rows with same event_group_id."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_two_item",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 4.0,
                                    "end": 6.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 2,
                                    "review_status": "accepted",
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_two_item",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_two_item",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 2
        assert result.canonical_events[0].type == EventLabel.PICKUP
        assert result.canonical_events[1].type == EventLabel.PICKUP
        group_id = result.canonical_events[0].event_group_id
        assert group_id != ""
        assert result.canonical_events[1].event_group_id == group_id
        assert result.canonical_events[0].event_id != result.canonical_events[1].event_id

        # Determinism: exporting same input produces same IDs
        result2 = export_events_csv(export)
        assert result2.canonical_events[0].event_id == result.canonical_events[0].event_id
        assert result2.canonical_events[1].event_id == result.canonical_events[1].event_id
        assert result2.canonical_events[0].event_group_id == group_id

    # ------------------------------------------------------------------
    # 4. Ignore intervals excluded from official events
    # ------------------------------------------------------------------

    def test_04_ignore_intervals_excluded(self, tmp_path):
        """Ignore intervals produce no events.csv rows but appear in ignore parquet."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_ignore",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 2.0,
                                    "end": 4.0,
                                    "labels": ["ignore"],
                                    "reason": "ACTION_OCCLUDED",
                                    "notes": "Hand occluded",
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_ignore",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_ignore",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        events_result = export_events_csv(export, csv_path)
        assert events_result.is_valid
        assert len(events_result.canonical_events) == 0

        # No pickup or putdown in events.csv
        with csv_path.open() as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                assert row["type"] not in ("pickup", "putdown")

        # Ignore appears in parquet
        ignore_path = tmp_path / "ignore.parquet"
        ignore_result = export_ignore_intervals_parquet(export, ignore_path)
        assert len(ignore_result.ignore_intervals) == 1
        ig = ignore_result.ignore_intervals[0]
        assert ig.t_start == 2.0
        assert ig.t_end == 4.0
        assert ig.reason == IgnoreReason.ACTION_OCCLUDED

        # Verify stable Parquet columns
        table = pq.read_table(str(ignore_path))
        assert table.column_names == [
            "ignore_id",
            "clip_id",
            "t_start",
            "t_end",
            "reason",
            "annotator",
            "notes",
        ]

    # ------------------------------------------------------------------
    # 5. Candidate correction
    # ------------------------------------------------------------------

    def test_05_candidate_correction(self, tmp_path):
        """Human-corrected annotation overrides candidate values."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_corrected",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 5.0,
                                    "end": 7.0,
                                    "labels": ["putdown"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_corrected",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_corrected",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        evt = result.canonical_events[0]
        # Human annotation: putdown at 5.0-7.0, NOT the candidate pickup at 2.0-6.0
        assert evt.type == EventLabel.PUTDOWN
        assert evt.t_start == 5.0
        assert evt.t_end == 7.0

    # ------------------------------------------------------------------
    # 6. Candidate deletion
    # ------------------------------------------------------------------

    def test_06_candidate_deletion(self, tmp_path):
        """Deleted candidate produces no official event."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [],
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_deleted",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_deleted",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 0

    # ------------------------------------------------------------------
    # 7. Candidate supplementation
    # ------------------------------------------------------------------

    def test_07_candidate_supplementation(self, tmp_path):
        """Manually added event (not from candidates) is exported."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_manually_added",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 6.0,
                                    "end": 8.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
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
                "task": {
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_supplemented",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_supplemented",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "bob",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert result.is_valid
        assert len(result.canonical_events) == 1
        assert result.canonical_events[0].type == EventLabel.PICKUP
        assert result.canonical_events[0].t_start == 6.0

    # ------------------------------------------------------------------
    # 8. Full active-span review
    # ------------------------------------------------------------------

    def test_08a_unconfirmed_no_events(self, tmp_path):
        """Unconfirmed clip (complete_active_span_reviewed != true) emits no events."""
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
                                    "start": 1.0,
                                    "end": 3.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            }
                        ],
                        "who": "alice",
                        "state": [
                            {
                                "name": "complete_active_span_reviewed",
                                "value": {"selected": ["false"]},
                            }
                        ],
                    }
                ],
                "task": {
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_unconfirmed",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_unconfirmed",
                        "fps": 30.0,
                        "complete_active_span_reviewed": False,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"
        result = export_events_csv(export, csv_path)
        assert not result.is_valid
        assert len(result.canonical_events) == 0
        review_errors = [
            e for e in result.validation.errors if "complete_active_span_reviewed" in e.message
        ]
        assert len(review_errors) >= 1

    def test_08b_confirmed_zero_events(self):
        """Confirmed clip with no event regions is valid and emits zero events."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [],
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_zero",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_zero",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        result = export_events_csv(export)
        assert result.is_valid
        assert len(result.canonical_events) == 0

    # ------------------------------------------------------------------
    # 9. Timestamp round trip
    # ------------------------------------------------------------------

    def test_09_timestamp_round_trip(self):
        """Event boundaries survive canonical->Label Studio->canonical path."""
        original = [
            CanonicalEvent(
                event_id="evt_orig",
                clip_id="clip_rt",
                type=EventLabel.PICKUP,
                t_start=4.0,
                t_end=6.0,
            ),
            CanonicalEvent(
                event_id="evt_orig2",
                clip_id="clip_rt",
                type=EventLabel.PUTDOWN,
                t_start=7.0,
                t_end=9.0,
            ),
        ]
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_pickup",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 4.0,
                                    "end": 6.0,
                                    "labels": ["pickup"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            },
                            {
                                "id": "region_putdown",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 7.0,
                                    "end": 9.0,
                                    "labels": ["putdown"],
                                    "confidence": "high",
                                    "hard_case": "false",
                                    "item_count": 1,
                                    "review_status": "accepted",
                                },
                            },
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_rt",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_rt",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        assert round_trip_check(original, export, fps=30.0) is True

    # ------------------------------------------------------------------
    # 10. Deterministic export
    # ------------------------------------------------------------------

    def test_10_deterministic_export(self, tmp_path):
        """Exporting the same input twice produces identical results."""
        export = [
            {
                "id": 1,
                "annotations": [
                    {
                        "id": 1,
                        "result": [
                            {
                                "id": "region_pickup",
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
                                    "review_status": "accepted",
                                },
                            },
                            {
                                "id": "region_putdown",
                                "from_name": "labels",
                                "to_name": "video",
                                "type": "timelabels",
                                "value": {
                                    "start": 5.0,
                                    "end": 7.0,
                                    "labels": ["putdown"],
                                    "confidence": "med",
                                    "hard_case": "true",
                                    "item_count": 1,
                                    "review_status": "reviewed",
                                },
                            },
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
                    "id": "t1",
                    "data": {
                        "video": "/videos/clip.mp4",
                        "clip_id": "clip_det",
                        "fps": 30.0,
                    },
                    "meta": {
                        "clip_id": "clip_det",
                        "fps": 30.0,
                        "complete_active_span_reviewed": True,
                        "annotator": "alice",
                    },
                },
            }
        ]
        csv_path = tmp_path / "events.csv"

        # First export
        result1 = export_events_csv(export, csv_path)
        csv_bytes_1 = csv_path.read_bytes()

        # Second export (overwrite)
        result2 = export_events_csv(export, csv_path)
        csv_bytes_2 = csv_path.read_bytes()

        # Canonical objects identical
        assert len(result1.canonical_events) == len(result2.canonical_events)
        for e1, e2 in zip(result1.canonical_events, result2.canonical_events, strict=True):
            assert e1.event_id == e2.event_id
            assert e1.clip_id == e2.clip_id
            assert e1.type == e2.type
            assert e1.t_start == e2.t_start
            assert e1.t_end == e2.t_end
            assert e1.hard_case == e2.hard_case
            assert e1.annotator == e2.annotator
            assert e1.confidence == e2.confidence
            assert e1.notes == e2.notes
            assert e1.event_group_id == e2.event_group_id

        # Ordering identical (already checked via zip above)

        # Generated CSV bytes identical
        assert csv_bytes_1 == csv_bytes_2

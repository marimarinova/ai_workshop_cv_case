"""Integration test against the ACTUAL Task 1 schemas (pickup_putdown.common.schemas).

Runs in the repository where common.schemas exists; skips in the standalone package.
Confirms the evaluator consumes the real class names / field names / enums directly.
"""

from __future__ import annotations

import pytest

schemas = pytest.importorskip("pickup_putdown.common.schemas")
from pickup_putdown.evaluation import (  # noqa: E402
    aggregate_metrics,
    metrics_to_json,
    slice_metrics,
)


def test_evaluator_runs_on_canonical_task1_schemas():
    gts = [
        schemas.Event(event_id="e1", clip_id="c", type="pickup", t_start=1.0, t_end=2.0),
        schemas.Event(event_id="e2", clip_id="c", type="putdown", t_start=5.0, t_end=6.0),
    ]
    pr = [
        schemas.Prediction(
            pred_id="p1", clip_id="c", type="pickup", t_start=1.1, t_end=2.1, score=0.9, model="m"
        ),
        schemas.Prediction(
            pred_id="p2", clip_id="c", type="pickup", t_start=5.0, t_end=6.0, score=0.7, model="m"
        ),
    ]
    m = aggregate_metrics(gts, pr, {"c": 600.0})
    # one pickup matches; the second pickup pred lands on a putdown GT -> FP, and the
    # putdown GT is unmatched -> FN. This is exactly the type-partition that str(enum) breaks.
    assert m["tiou@0.5"]["tp"] == 1
    assert m["per_type"]["pickup"]["tp"] == 1  # per-type filter survives the real enum
    assert m["per_type"]["putdown"]["fn"] == 1
    assert m["confusion"]["putdown"]["pickup"] == 1
    assert m["mAP"]["mAP@0.5"] is not None  # AP per-type filter exercised
    assert "confusion" in metrics_to_json(m)


def test_slice_metrics_with_real_confidence_enum():
    gts = [
        schemas.Event(
            event_id="e1", clip_id="c", type="pickup", t_start=1.0, t_end=2.0, confidence="low"
        ),
        schemas.Event(event_id="e2", clip_id="c", type="putdown", t_start=5.0, t_end=6.0),
    ]
    pr = [
        schemas.Prediction(
            pred_id="p1", clip_id="c", type="pickup", t_start=1.0, t_end=2.0, score=0.9, model="m"
        ),
    ]
    sl = slice_metrics(gts, pr, {"c": 600.0})
    assert "low_confidence" in sl
    assert sl["low_confidence"]["recall"] == 1.0

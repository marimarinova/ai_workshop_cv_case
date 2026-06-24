"""Regression guard: a non-str ``Enum`` type must not collapse type-partitioned metrics.

The canonical Task 1 schema may expose ``type`` as a plain ``Enum`` whose
``str(member)`` is ``"EventType.pickup"`` (not ``"pickup"``). Every type-partitioned
computation (block matching, per-type P/R, AP/mAP, confusion, multi-item) routes
through ``type_name`` so it keeps working. This fails loudly if any path regresses
to raw ``str(member)`` comparison.
"""

from enum import Enum

from pickup_putdown.evaluation import (
    Criterion,
    aggregate_metrics,
    average_precision,
    evaluate_class_aware,
    evaluate_confusion,
)
from pickup_putdown.evaluation import EvaluationEvent as Event
from pickup_putdown.evaluation import EvaluationPrediction as Prediction


class EventType(Enum):  # plain Enum -> str(EventType.pickup) == "EventType.pickup"
    pickup = "pickup"
    putdown = "putdown"


def test_plain_enum_str_is_not_value():
    assert str(EventType.pickup) != "pickup"  # the premise this guard protects


def test_block_matching_works_with_plain_enum():
    gts = [Event("c", EventType.pickup, 1.0, 2.0), Event("c", EventType.putdown, 5.0, 6.0)]
    pr = [
        Prediction("c", EventType.pickup, 1.0, 2.0),
        Prediction("c", EventType.putdown, 5.0, 6.0),
    ]
    r = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)


def test_per_type_confusion_ap_with_plain_enum():
    gts = [Event("c", EventType.pickup, 1.0, 2.0)]
    pr = [
        Prediction("c", EventType.pickup, 1.0, 2.0, score=0.9),
        Prediction("c", EventType.putdown, 5.0, 6.0, score=0.8),
    ]
    m = aggregate_metrics(gts, pr, {"c": 100.0})
    assert m["per_type"]["pickup"]["tp"] == 1
    assert m["per_type"]["putdown"]["fp"] == 1
    conf = evaluate_confusion(gts, pr, Criterion("tiou", 0.5))
    assert conf["pickup"]["pickup"] == 1
    assert average_precision(gts, pr, "pickup", 0.5) == 1.0

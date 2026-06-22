from __future__ import annotations

import itertools
import json
import random

from pickup_putdown.evaluation import (
    Criterion,
    aggregate_metrics,
    average_precision,
    evaluate_class_aware,
    evaluate_confusion,
    events_from_rows,
    failure_gallery,
    match_one_to_one,
    match_ranked,
    metrics_to_json,
    midpoint_distance,
    predictions_from_rows,
    render_html,
    render_markdown,
    slice_metrics,
    tiou,
)
from pickup_putdown.evaluation import EvaluationEvent as Event
from pickup_putdown.evaluation import EvaluationIgnoreInterval as IgnoreInterval
from pickup_putdown.evaluation import EvaluationPrediction as Prediction


# --- interval math ---
def test_tiou_identical():
    assert tiou(Event("c", "pickup", 1.0, 3.0), Event("c", "pickup", 1.0, 3.0)) == 1.0


def test_tiou_disjoint_is_zero():
    assert tiou(Event("c", "pickup", 0.0, 1.0), Event("c", "pickup", 5.0, 6.0)) == 0.0


def test_tiou_half_overlap():
    assert abs(tiou(Event("c", "pickup", 0.0, 2.0), Event("c", "pickup", 1.0, 3.0)) - 1 / 3) < 1e-9


def test_midpoint_distance():
    assert midpoint_distance(Event("c", "pickup", 0.0, 2.0), Event("c", "pickup", 2.0, 4.0)) == 2.0


# --- matching ---
def test_no_predictions_all_fn():
    r = match_one_to_one([Event("c", "pickup", 1.0, 2.0)], [], Criterion())
    assert (r.tp, r.fp, r.fn) == (0, 0, 1)


def test_no_ground_truth_all_fp():
    r = match_one_to_one([], [Prediction("c", "pickup", 1.0, 2.0)], Criterion())
    assert (r.tp, r.fp, r.fn) == (0, 1, 0)


def test_order_invariance():
    gts = [Event("c", "pickup", 0.0, 1.0, "g1"), Event("c", "pickup", 10.0, 11.0, "g2")]
    pr = [Prediction("c", "pickup", 10.0, 11.0, "p2"), Prediction("c", "pickup", 0.0, 1.0, "p1")]
    a = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))
    b = evaluate_class_aware(list(reversed(gts)), list(reversed(pr)), Criterion("tiou", 0.5))
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn) == (2, 0, 0)


def test_lexicographic_matches_brute_force():
    """match_one_to_one TP must equal the brute-force maximum accepted matching."""
    crit = Criterion("tiou", 0.5)

    def brute(gts, preds):
        n, m = len(gts), len(preds)
        for r in range(min(n, m), 0, -1):
            for gi in itertools.permutations(range(n), r):
                for pj in itertools.permutations(range(m), r):
                    if all(crit.accepts(gts[gi[k]], preds[pj[k]]) for k in range(r)):
                        return r
        return 0

    rng = random.Random(0)
    for _ in range(3000):

        def mk(i, cls):
            s = round(rng.uniform(0, 3), 2)
            e = s + round(rng.uniform(0.5, 2), 2)
            return (Event if cls == "g" else Prediction)("c", "pickup", s, e, f"{cls}{i}")

        gts = [mk(i, "g") for i in range(rng.randint(1, 3))]
        pr = [mk(i, "p") for i in range(rng.randint(1, 3))]
        assert match_one_to_one(gts, pr, crit).tp == brute(gts, pr)


def test_type_flip_is_fp_fn_and_confusion():
    gts = [Event("c", "pickup", 1.0, 2.0)]
    pr = [Prediction("c", "putdown", 1.0, 2.0)]
    r = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (0, 1, 1)
    conf = evaluate_confusion(gts, pr, Criterion("tiou", 0.5))
    assert conf["pickup"]["putdown"] == 1  # nested, JSON-safe


def test_two_item_needs_two_predictions():
    gts = [Event("c", "pickup", 1.0, 2.0, "g1"), Event("c", "pickup", 1.0, 2.0, "g2")]
    r1 = evaluate_class_aware(gts, [Prediction("c", "pickup", 1.0, 2.0)], Criterion("tiou", 0.5))
    assert (r1.tp, r1.fn) == (1, 1)
    two = [Prediction("c", "pickup", 1.0, 2.0, "p1"), Prediction("c", "pickup", 1.0, 2.0, "p2")]
    r2 = evaluate_class_aware(gts, two, Criterion("tiou", 0.5))
    assert (r2.tp, r2.fn) == (2, 0)


def test_immediate_pickup_then_putdown():
    gts = [Event("c", "pickup", 1.0, 2.0), Event("c", "putdown", 2.0, 3.0)]
    pr = [Prediction("c", "pickup", 1.0, 2.0), Prediction("c", "putdown", 2.0, 3.0)]
    r = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)


def test_overlapping_events_one_to_one():
    gts = [Event("c", "pickup", 0.0, 2.0, "g1"), Event("c", "pickup", 1.0, 3.0, "g2")]
    pr = [Prediction("c", "pickup", 0.0, 2.0, "p1"), Prediction("c", "pickup", 1.0, 3.0, "p2")]
    r = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (2, 0, 0)


def test_ignore_any_overlap_excluded():
    # pred overlaps the ignore span only partially -> still dropped
    r = evaluate_class_aware(
        [Event("c", "pickup", 1.0, 2.0)],
        [Prediction("c", "pickup", 1.0, 2.0)],
        Criterion("tiou", 0.5),
        ignores=[IgnoreInterval("c", 1.5, 5.0)],
    )
    assert (r.tp, r.fp, r.fn) == (0, 0, 0)


def test_midpoint_criterion_matches_when_tiou_low():
    gts = [Event("c", "pickup", 0.0, 1.0)]
    pr = [Prediction("c", "pickup", 0.6, 1.6)]
    assert evaluate_class_aware(gts, pr, Criterion("tiou", 0.5)).tp == 0
    assert evaluate_class_aware(gts, pr, Criterion("midpoint", midpoint_tolerance_s=1.0)).tp == 1


def test_greedy_matcher_order_invariant_and_matches_hungarian():
    gts = [Event("c", "pickup", 0.0, 1.0, "g1"), Event("c", "pickup", 10.0, 11.0, "g2")]
    pr = [
        Prediction("c", "pickup", 0.0, 1.0, "p1", score=0.7),
        Prediction("c", "pickup", 10.0, 11.0, "p2", score=0.9),
    ]
    a = match_ranked(gts, pr, Criterion("tiou", 0.5))
    b = match_ranked(list(reversed(gts)), list(reversed(pr)), Criterion("tiou", 0.5))
    assert (a.tp, a.fp, a.fn) == (b.tp, b.fp, b.fn) == (2, 0, 0)
    h = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5), matcher="hungarian")
    g = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5), matcher="greedy")
    assert (h.tp, h.fp, h.fn) == (g.tp, g.fp, g.fn) == (2, 0, 0)


# --- multi-item (default group_id / exact-dup; overlap opt-in) ---
def test_multi_item_default_exact_duplicate():
    gts = [Event("c", "pickup", 1.0, 2.0, "g1"), Event("c", "pickup", 1.0, 2.0, "g2")]
    pr = [Prediction("c", "pickup", 1.0, 2.0, "p1")]
    assert aggregate_metrics(gts, pr, {"c": 100.0})["multi_item_recall"] == 0.5


def test_multi_item_default_group_id():
    gts = [
        Event("c", "pickup", 1.0, 2.0, "g1", group_id="A"),
        Event("c", "pickup", 5.0, 6.0, "g2", group_id="A"),
    ]  # different times, shared group
    pr = [Prediction("c", "pickup", 1.0, 2.0, "p1")]
    m = aggregate_metrics(gts, pr, {"c": 100.0})
    assert m["multi_item_support"] == 2 and m["multi_item_recall"] == 0.5


def test_multi_item_overlap_is_opt_in():
    gts = [
        Event("c", "pickup", 0.0, 2.0, "g1"),
        Event("c", "pickup", 1.0, 3.0, "g2"),
    ]  # tIoU=1/3, no group
    pr = [Prediction("c", "pickup", 0.0, 2.0, "p1")]
    assert (
        aggregate_metrics(gts, pr, {"c": 100.0})["multi_item_recall"] is None
    )  # default: not multi
    assert (
        aggregate_metrics(gts, pr, {"c": 100.0}, multi_item_overlap_thr=0.3)["multi_item_recall"]
        == 0.5
    )


# --- aggregate / AP ---
def test_aggregate_metrics_bundle():
    gts = [Event("c", "pickup", 1.0, 2.0), Event("c", "putdown", 5.0, 6.0)]
    pr = [
        Prediction("c", "pickup", 1.0, 2.0, score=0.9),
        Prediction("c", "putdown", 5.1, 6.1, score=0.8),
    ]
    m = aggregate_metrics(gts, pr, {"c": 600.0})
    assert m["tiou@0.5"]["recall"] == 1.0
    assert m["event_count_error_per_clip"] == 0 and m["event_count_error_absolute"] == 0
    assert m["fp_per_hour"] == 0.0
    assert m["confusion"]["pickup"]["pickup"] == 1
    assert m["mAP"]["mAP@0.5"] == 1.0
    assert m["per_type"]["pickup"]["recall"] == 1.0


def test_ap_perfect_and_ranking_and_undefined():
    gt = [Event("c", "pickup", 1.0, 2.0)]
    good = [
        Prediction("c", "pickup", 1.0, 2.0, score=0.9),
        Prediction("c", "pickup", 50.0, 51.0, score=0.5),
    ]
    bad = [
        Prediction("c", "pickup", 50.0, 51.0, score=0.9),
        Prediction("c", "pickup", 1.0, 2.0, score=0.5),
    ]
    assert average_precision(gt, good, "pickup", 0.5) == 1.0
    assert average_precision(gt, bad, "pickup", 0.5) == 0.5
    assert average_precision([], good, "pickup", 0.5) is None


# --- reports / json / slices / io ---
def test_reports_and_json_serializable():
    gts = [Event("c", "pickup", 1.0, 2.0), Event("c", "putdown", 5.0, 6.0)]
    pr = [
        Prediction("c", "pickup", 1.0, 2.0, score=0.9),
        Prediction("c", "putdown", 5.0, 6.0, score=0.8),
    ]
    m = aggregate_metrics(gts, pr, {"c": 600.0})
    assert render_html(m, "t<a>").startswith("<!doctype html>")
    assert render_markdown(m, "t").startswith("# Evaluation report")
    s = metrics_to_json(m)  # must not raise
    assert json.loads(s)["confusion"]["pickup"]["pickup"] == 1


def test_slice_metrics_recall_oriented():
    gts = [
        Event("c", "pickup", 1.0, 2.0, confidence="low"),
        Event("c", "putdown", 5.0, 6.0, confidence="high"),
    ]
    pr = [Prediction("c", "pickup", 1.0, 2.0)]
    sl = slice_metrics(gts, pr, {"c": 600.0})
    assert "all" in sl
    assert set(sl["low_confidence"]) == {
        "recall",
        "tp",
        "fn",
        "support",
        "threshold",
    }  # recall-only + the threshold used
    assert sl["low_confidence"]["recall"] == 1.0 and sl["low_confidence"]["support"] == 1


def test_multiple_person_slice():
    gts = [
        Event("c", "pickup", 1.0, 2.0, "g1", n_person=2),
        Event("c", "putdown", 5.0, 6.0, "g2", n_person=2),
        Event("c", "pickup", 9.0, 10.0, "g3", n_person=1),
    ]
    pr = [Prediction("c", "pickup", 1.0, 2.0)]
    sl = slice_metrics(gts, pr, {"c": 600.0})
    assert "multiple_person" in sl
    assert sl["multiple_person"]["support"] == 2  # only the two n_person>1 events
    assert sl["multiple_person"]["tp"] == 1 and sl["multiple_person"]["recall"] == 0.5


def test_io_preserves_zero_score_and_column_map():
    rows = [{"clip": "c", "label": "pickup", "start": "1.0", "end": "2.0"}]
    cmap = {"clip_id": "clip", "type": "label", "t_start": "start", "t_end": "end"}
    assert events_from_rows(rows, column_map=cmap)[0].t_end == 2.0
    p = predictions_from_rows(
        [{"clip_id": "c", "type": "pickup", "t_start": "1", "t_end": "2", "score": "0"}]
    )
    assert p[0].score == 0.0  # 0.0 preserved, not coerced to 1.0


def test_runtime_per_video_minute():
    gts = [Event("c", "pickup", 1.0, 2.0)]
    pr = [Prediction("c", "pickup", 1.0, 2.0)]
    assert (
        aggregate_metrics(gts, pr, {"c": 600.0}, runtime_s=30.0)["runtime_per_video_minute"] == 3.0
    )
    assert aggregate_metrics(gts, pr, {"c": 600.0})["runtime_per_video_minute"] is None


def test_failure_gallery_type_confusions():
    gts = [Event("c", "pickup", 1.0, 2.0)]
    pr = [Prediction("c", "putdown", 1.0, 2.0)]
    g = failure_gallery(gts, pr, Criterion("tiou", 0.5))
    assert (
        g["type_confusions"][0]["gt_type"] == "pickup"
        and g["type_confusions"][0]["pred_type"] == "putdown"
    )


# --- regression tests for the strict review (blockers + hardening) ---
from enum import StrEnum  # noqa: E402


def _raises(fn):
    try:
        fn()
        return False
    except Exception:
        return True


def test_matcher_exact_adversarial_case():
    gts = [Event("c", "pickup", 1.3, 3.8, "g1"), Event("c", "pickup", 2.8, 5.3, "g2")]
    pr = [Prediction("c", "pickup", 1.0, 2.0, "p1"), Prediction("c", "pickup", 2.2, 4.2, "p2")]
    r = match_one_to_one(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (1, 1, 1)
    assert r.matched[0][0].event_id == "g1" and r.matched[0][1].pred_id == "p2"


def test_map_order_invariant():
    g = [Event("c", "pickup", 1.3, 3.8, "g1"), Event("c", "pickup", 2.8, 5.3, "g2")]
    p = [
        Prediction("c", "pickup", 1.0, 2.0, "p1", score=0.6),
        Prediction("c", "pickup", 2.2, 4.2, "p2", score=0.9),
    ]
    assert average_precision(g, p, "pickup", 0.5) == average_precision(
        list(reversed(g)), list(reversed(p)), "pickup", 0.5
    )


def test_minimal_objects_do_not_crash():
    class Min:
        def __init__(self, c, t, a, b):
            self.clip_id, self.type, self.t_start, self.t_end = c, t, a, b

    gm = [Min("c", "pickup", 1.0, 2.0)]
    pm = [Min("c", "pickup", 1.0, 2.0)]
    aggregate_metrics(gm, pm, {"c": 10.0})  # no confidence/hard_case/score
    slice_metrics(gm, pm, {"c": 10.0})
    failure_gallery(gm, pm, Criterion("tiou", 0.5))


def test_validation_guards():
    assert _raises(lambda: Criterion("bogus", 0.5))
    assert _raises(lambda: Criterion("tiou", 1.5))
    assert _raises(lambda: Prediction("c", "pickup", -1.0, 2.0))
    assert _raises(lambda: IgnoreInterval("c", 5.0, 1.0))
    assert _raises(lambda: aggregate_metrics([Event("c", "pickup", 1.0, 2.0)], [], {"c": -5.0}))


def test_enum_backed_types_accepted():
    class T(StrEnum):
        PICKUP = "pickup"

    class Obj:
        def __init__(self):
            self.clip_id, self.type, self.t_start, self.t_end, self.score = (
                "c",
                T.PICKUP,
                1.0,
                2.0,
                0.9,
            )

    assert aggregate_metrics([Obj()], [Obj()], {"c": 10.0})["tiou@0.5"]["tp"] == 1


def test_html_report_escapes_model_name():
    m = aggregate_metrics(
        [Event("c", "pickup", 1.0, 2.0)], [Prediction("c", "pickup", 1.0, 2.0)], {"c": 10.0}
    )
    h = render_html(m, "<script>x")
    assert "&lt;script&gt;" in h and "<script>x" not in h


# --- regression tests for the final review ---
from enum import Enum as _Enum  # noqa: E402


def test_slice_uses_configured_threshold():
    # match holds only at tIoU 0.3, not 0.5; slice must honour the configured threshold
    gts = [Event("c", "pickup", 0.0, 2.0, "g1", confidence="low")]
    pr = [Prediction("c", "pickup", 0.0, 3.0, "p1")]  # tIoU = 2/3 -> matches at 0.3 and 0.5
    # use tiou=0.6 so it should NOT match; overall and slice must agree (both 0 recall)
    sl = slice_metrics(gts, pr, {"c": 100.0}, tiou_thresholds=(0.8,))
    assert sl["all"]["tiou@0.8"]["recall"] == 0.0
    assert sl["low_confidence"]["recall"] == 0.0
    # at tiou 0.3 both should be 1.0
    sl2 = slice_metrics(gts, pr, {"c": 100.0}, tiou_thresholds=(0.3,))
    assert sl2["all"]["tiou@0.3"]["recall"] == 1.0
    assert sl2["low_confidence"]["recall"] == 1.0


def test_plain_enum_type_does_not_crash_sort():
    class T(_Enum):  # NOT a str-Enum -> members are not order-comparable
        PICKUP = "pickup"

    gts = [Event("c", T.PICKUP, 1.0, 2.0, "g1"), Event("c", T.PICKUP, 1.0, 2.0, "g2")]
    pr = [Prediction("c", T.PICKUP, 1.0, 2.0, "p1")]
    r = evaluate_class_aware(gts, pr, Criterion("tiou", 0.5))  # must not raise TypeError
    assert (r.tp, r.fn) == (1, 1)


def test_large_block_cardinality_safe():
    # many accepted pairs; one extra match must always beat quality differences
    n = 60
    gts = [Event("c", "pickup", float(i), float(i) + 0.9, f"g{i}") for i in range(n)]
    pr = [Prediction("c", "pickup", float(i), float(i) + 0.9, f"p{i}") for i in range(n)]
    r = match_one_to_one(gts, pr, Criterion("tiou", 0.5))
    assert (r.tp, r.fp, r.fn) == (n, 0, 0)


def test_unknown_matcher_raises_valueerror():
    import pytest

    with pytest.raises(ValueError, match="unknown matcher"):
        evaluate_class_aware(
            [Event("c", "pickup", 1.0, 2.0)],
            [Prediction("c", "pickup", 1.0, 2.0)],
            Criterion("tiou", 0.5),
            matcher="bogus",
        )


def test_empty_tiou_thresholds_raises_valueerror():
    import pytest

    with pytest.raises(ValueError, match="tiou_thresholds must be a non-empty sequence"):
        aggregate_metrics(
            [Event("c", "pickup", 1.0, 2.0)],
            [Prediction("c", "pickup", 1.0, 2.0)],
            {"c": 10.0},
            tiou_thresholds=(),
        )

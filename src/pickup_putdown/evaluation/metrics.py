"""Metric aggregation, per-type metrics, recall-oriented slices, JSON output.

Ignore intervals are filtered ONCE at the top and the filtered collections are
used for every official metric (matching, counts, confusion, mAP, multi-item,
slices). Multi-item detection defaults to shared ``group_id`` or exact-duplicate
GT intervals; overlap clustering is explicit opt-in via ``multi_item_overlap_thr``.
Optional metadata (confidence / hard_case / n_person) is accessed defensively and a
slice is only emitted when that metadata is actually present.

Performance: dominated by the per-(clip, type) Hungarian matching in
``evaluate_class_aware``. For typical data (a few events per clip across many
clips) a full run is sub-second; for very dense clips the assignment grows
cubically in events-per-block, so pre-filter by time window if needed.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from .ap import mean_ap
from .class_aware_matching import by_clip, drop_ignored, evaluate_class_aware
from .confusion_matching import evaluate_confusion
from .contracts import VALID_TYPES, type_name
from .intervals import Criterion, tiou

logger = logging.getLogger(__name__)


def _multi_item_events(events: list[Any], overlap_thr: float | None = None) -> list[Any]:
    """Events in a multi-item action. Default: shared group_id, else duplicate
    clip/type with start/end rounded to 1e-6 (tolerant to float noise).
    ``overlap_thr`` enables transitive overlap clustering."""
    out: list[Any] = []
    bucket: dict[tuple[str, str], list[Any]] = {}
    for e in events:
        bucket.setdefault((e.clip_id, type_name(e.type)), []).append(e)
    for group in bucket.values():
        if overlap_thr is None:
            clusters: dict[tuple[Any, ...], list[Any]] = {}
            for e in group:
                gid = getattr(e, "event_group_id", "") or getattr(e, "group_id", "") or ""
                key: tuple[Any, ...] = (
                    ("gid", gid) if gid else ("dup", round(e.t_start, 6), round(e.t_end, 6))
                )
                clusters.setdefault(key, []).append(e)
            for c in clusters.values():
                if len(c) > 1:
                    out.extend(c)
        else:
            n = len(group)
            parent = list(range(n))

            def find(x: int, _p: list[int] = parent) -> int:
                while _p[x] != x:
                    _p[x] = _p[_p[x]]
                    x = _p[x]
                return x

            for i in range(n):
                for j in range(i + 1, n):
                    if tiou(group[i], group[j]) >= overlap_thr:
                        parent[find(i)] = find(j)
            comps: dict[int, list[Any]] = {}
            for i in range(n):
                comps.setdefault(find(i), []).append(group[i])
            for c in comps.values():
                if len(c) > 1:
                    out.extend(c)
    return out


def _prf(tp: int, fp: int, fn: int) -> dict[str, Any]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def aggregate_metrics(
    events: Any,
    preds: Any,
    clip_durations: dict[str, float],
    ignores: Any = (),
    tiou_thresholds: tuple[float, ...] = (0.3, 0.5),
    midpoint_tol_s: float = 1.0,
    matcher: str = "hungarian",
    map_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    multi_item_overlap_thr: float | None = None,
    runtime_s: float | None = None,
    confusion_tiou: float = 0.5,
) -> dict[str, Any]:
    """Full metric bundle for one model run. Ignore-filtered consistently."""
    if clip_durations and any((d is not None and d < 0) for d in clip_durations.values()):
        raise ValueError("clip_durations must be non-negative")
    if not tiou_thresholds:
        raise ValueError("tiou_thresholds must be a non-empty sequence")
    ev = drop_ignored(events, ignores)
    pr = drop_ignored(preds, ignores)
    out: dict[str, Any] = {}

    for thr in tiou_thresholds:
        r = evaluate_class_aware(ev, pr, Criterion("tiou", tiou_threshold=thr), (), matcher)
        out[f"tiou@{thr}"] = _prf(r.tp, r.fp, r.fn)

    rmid = evaluate_class_aware(
        ev, pr, Criterion("midpoint", midpoint_tolerance_s=midpoint_tol_s), (), matcher
    )
    out[f"midpoint@{midpoint_tol_s}s"] = _prf(rmid.tp, rmid.fp, rmid.fn)
    if rmid.matched:
        se = np.array(
            [[abs(g.t_start - p.t_start), abs(g.t_end - p.t_end)] for g, p in rmid.matched]
        )
        out["start_mae_s"] = float(se[:, 0].mean())
        out["end_mae_s"] = float(se[:, 1].mean())
    else:
        out["start_mae_s"] = out["end_mae_s"] = None

    base_thr = 0.5 if 0.5 in tiou_thresholds else min(tiou_thresholds)
    crit = Criterion("tiou", tiou_threshold=base_thr)
    out["per_type"] = {}
    for t in VALID_TYPES:
        rt = evaluate_class_aware(
            [e for e in ev if type_name(e.type) == t],
            [p for p in pr if type_name(p.type) == t],
            crit,
            (),
            matcher,
        )
        out["per_type"][t] = _prf(rt.tp, rt.fp, rt.fn)

    total_seconds = sum(d for d in clip_durations.values() if d) if clip_durations else 0.0
    fp_key = "tiou@0.5" if "tiou@0.5" in out else f"tiou@{base_thr}"
    out["fp_per_hour"] = (
        (out[fp_key]["fp"] / (total_seconds / 3600.0))
        if (total_seconds and fp_key in out)
        else None
    )
    out["runtime_per_video_minute"] = (
        (runtime_s / (total_seconds / 60.0)) if (runtime_s is not None and total_seconds) else None
    )

    ge, gp = by_clip(ev), by_clip(pr)
    out["event_count_error_per_clip"] = sum(
        abs(len(gp.get(c, [])) - len(ge.get(c, []))) for c in set(ge) | set(gp)
    )
    out["event_count_error_absolute"] = abs(len(pr) - len(ev))

    out["confusion"] = evaluate_confusion(
        ev, pr, Criterion("tiou", tiou_threshold=confusion_tiou), ()
    )
    out["confusion_tiou"] = confusion_tiou

    out["mAP"] = mean_ap(ev, pr, map_thresholds, ())

    multi = _multi_item_events(ev, multi_item_overlap_thr)
    if multi:
        rm = evaluate_class_aware(
            multi, pr, Criterion("tiou", tiou_threshold=base_thr), (), matcher
        )
        out["multi_item_recall"] = rm.tp / (rm.tp + rm.fn) if (rm.tp + rm.fn) else 0.0
        out["multi_item_support"] = len(multi)
    else:
        out["multi_item_recall"] = None
        out["multi_item_support"] = 0

    logger.debug("aggregate_metrics: %d gt, %d pred, matcher=%s", len(ev), len(pr), matcher)
    return out


def slice_metrics(
    events: Any,
    preds: Any,
    clip_durations: dict[str, float],
    ignores: Any = (),
    short_max_s: float = 1.0,
    slice_tiou_threshold: float | None = None,
    **kw: Any,
) -> dict[str, Any]:
    """Full metrics on ``all``; recall/TP/FN/support on GT-metadata slices.

    The slice criterion and matcher are derived from the SAME resolved configuration
    as the overall run (never hardcoded), honouring the threshold-input contract.
    Metadata-gated slices (confidence, hard_case, n_person) are emitted only when at
    least one event carries that metadata; the duration-based slices (short_events /
    long_events) are always emitted since start/end are always present. Empty slices
    are skipped and precision is not reported per slice.
    """
    out: dict[str, Any] = {"all": aggregate_metrics(events, preds, clip_durations, ignores, **kw)}
    ev = drop_ignored(events, ignores)
    pr = drop_ignored(preds, ignores)
    tiou_thresholds: tuple[float, ...] = kw.get("tiou_thresholds", (0.3, 0.5))
    if not tiou_thresholds:
        raise ValueError("tiou_thresholds must be a non-empty sequence")
    slice_tiou = (
        slice_tiou_threshold
        if slice_tiou_threshold is not None
        else (0.5 if 0.5 in tiou_thresholds else min(tiou_thresholds))
    )
    matcher: str = kw.get("matcher", "hungarian")
    crit = Criterion("tiou", tiou_threshold=slice_tiou)

    slices: dict[str, list[Any]] = {}
    if any(hasattr(e, "confidence") for e in ev):
        slices["high_med_only"] = [
            e for e in ev if type_name(getattr(e, "confidence", "high")) in ("high", "med")
        ]
        slices["low_confidence"] = [
            e for e in ev if type_name(getattr(e, "confidence", "high")) == "low"
        ]
    if any(hasattr(e, "hard_case") for e in ev):
        slices["hard_cases"] = [e for e in ev if getattr(e, "hard_case", False)]
    slices["short_events"] = [e for e in ev if (e.t_end - e.t_start) < short_max_s]
    slices["long_events"] = [e for e in ev if (e.t_end - e.t_start) >= short_max_s]
    if any(hasattr(e, "n_person") for e in ev):
        slices["multiple_person"] = [e for e in ev if getattr(e, "n_person", 1) > 1]

    for name, evs in slices.items():
        if not evs:
            continue
        r = evaluate_class_aware(evs, pr, crit, (), matcher)
        recall = r.tp / (r.tp + r.fn) if (r.tp + r.fn) else 0.0
        out[name] = {
            "recall": recall,
            "tp": r.tp,
            "fn": r.fn,
            "support": len(evs),
            "threshold": slice_tiou,
        }
    return out

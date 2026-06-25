"""Pass 1 matching with a correct lexicographic objective.

``match_one_to_one`` maximises the NUMBER of accepted (above-threshold) matches
first, then temporal quality — it does NOT maximise raw tIoU and threshold after
(that earlier objective could leave a valid match unmatched). ``match_ranked`` is
the score-ranked greedy variant used for mAP. Both are order-invariant.

Ignore handling: ``drop_ignored`` excludes any row with ANY positive temporal
overlap with an ignore interval, applied consistently before every official metric.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment  # type: ignore[import-untyped]

from .contracts import VALID_TYPES, MatchResult, type_name
from .intervals import Criterion, overlaps


def _canonical(items: list[Any]) -> list[Any]:
    return sorted(
        items,
        key=lambda x: (
            x.clip_id,
            x.t_start,
            x.t_end,
            type_name(x.type),
            getattr(x, "event_id", "") or getattr(x, "pred_id", ""),
        ),
    )


def match_one_to_one(gts: list[Any], preds: list[Any], criterion: Criterion) -> MatchResult:
    """Maximum-cardinality accepted matching, tie-broken by quality. Order-invariant."""
    gts = _canonical(list(gts))
    preds = _canonical(list(preds))
    if not gts or not preds:
        return MatchResult([], list(gts), list(preds))
    accept_bonus = min(len(gts), len(preds)) + 1.0
    s = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if criterion.accepts(g, p):
                s[i, j] = accept_bonus + criterion.score(g, p)
    rows, cols = linear_sum_assignment(s, maximize=True)
    matched: list[tuple[Any, Any]] = []
    ug: set[int] = set()
    up: set[int] = set()
    for i, j in zip(rows, cols, strict=True):
        if criterion.accepts(gts[i], preds[j]):
            matched.append((gts[i], preds[j]))
            ug.add(i)
            up.add(j)
    return MatchResult(
        matched,
        [g for i, g in enumerate(gts) if i not in ug],
        [p for j, p in enumerate(preds) if j not in up],
    )


def match_ranked(gts: list[Any], preds: list[Any], criterion: Criterion) -> MatchResult:
    """Score-ranked greedy matching (mAP convention). Order-invariant."""
    gts = _canonical(list(gts))
    preds = sorted(
        preds,
        key=lambda x: (
            -getattr(x, "score", 1.0),
            x.clip_id,
            x.t_start,
            x.t_end,
            getattr(x, "pred_id", ""),
        ),
    )
    used = [False] * len(gts)
    matched: list[tuple[Any, Any]] = []
    up: list[Any] = []
    for p in preds:
        best, best_score = -1, -1.0
        for i, g in enumerate(gts):
            if used[i] or not criterion.accepts(g, p):
                continue
            sc = criterion.score(g, p)
            if sc > best_score:
                best, best_score = i, sc
        if best >= 0:
            used[best] = True
            matched.append((gts[best], p))
        else:
            up.append(p)
    return MatchResult(matched, [g for i, g in enumerate(gts) if not used[i]], up)


_MATCHERS: dict[str, Any] = {"hungarian": match_one_to_one, "greedy": match_ranked}


def drop_ignored(items: list[Any], ignores: Any) -> list[Any]:
    """Exclude items with ANY positive temporal overlap with an ignore interval."""
    by: dict[str, list[Any]] = {}
    for ig in ignores:
        by.setdefault(ig.clip_id, []).append(ig)
    return [it for it in items if not any(overlaps(it, sp) for sp in by.get(it.clip_id, []))]


def by_clip(items: list[Any]) -> dict[str, list[Any]]:
    d: dict[str, list[Any]] = {}
    for it in items:
        d.setdefault(it.clip_id, []).append(it)
    return d


def validate_types(items: Any, *, kind: str = "record") -> None:
    """Fail fast if any record's ``type_name`` is not in ``VALID_TYPES``.

    Raises ``ValueError`` naming the offending type so a mislabeled input is
    caught before matching instead of being silently dropped per (clip, type).
    """
    for it in items:
        name = type_name(it.type)
        if name not in VALID_TYPES:
            raise ValueError(f"invalid {kind} type {name!r}; expected one of {list(VALID_TYPES)}")


def evaluate_class_aware(
    events: Any,
    preds: Any,
    criterion: Criterion,
    ignores: Any = (),
    matcher: str = "hungarian",
) -> MatchResult:
    """Match within each (clip, type) block; aggregate one MatchResult."""
    if matcher not in _MATCHERS:
        raise ValueError(f"unknown matcher {matcher!r}; expected one of {sorted(_MATCHERS)}")
    match_fn = _MATCHERS[matcher]
    events = drop_ignored(events, ignores)
    preds = drop_ignored(preds, ignores)
    validate_types(events, kind="event")
    validate_types(preds, kind="prediction")
    ge, gp = by_clip(events), by_clip(preds)
    agg = MatchResult()
    for clip in sorted(set(ge) | set(gp)):
        for t in VALID_TYPES:
            r: MatchResult = match_fn(
                [e for e in ge.get(clip, []) if type_name(e.type) == t],
                [p for p in gp.get(clip, []) if type_name(p.type) == t],
                criterion,
            )
            agg.matched += r.matched
            agg.unmatched_gt += r.unmatched_gt
            agg.unmatched_pred += r.unmatched_pred
    return agg

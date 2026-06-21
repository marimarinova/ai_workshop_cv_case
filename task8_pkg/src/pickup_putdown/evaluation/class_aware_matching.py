"""Pass 1 matching with a correct lexicographic objective.

``match_one_to_one`` maximises the NUMBER of accepted (above-threshold) matches
first, then temporal quality — it does NOT maximise raw tIoU and threshold after
(that earlier objective could leave a valid match unmatched). ``match_ranked`` is
the score-ranked greedy variant used for mAP. Both are order-invariant.

Ignore handling: ``drop_ignored`` excludes any row with ANY positive temporal
overlap with an ignore interval, applied consistently before every official metric.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment
from .contracts import VALID_TYPES, MatchResult, type_name
from .intervals import Criterion, overlaps



def _canonical(items):
    return sorted(items, key=lambda x: (x.clip_id, x.t_start, x.t_end, type_name(x.type),
                                        getattr(x, "event_id", "") or getattr(x, "pred_id", "")))


def match_one_to_one(gts, preds, criterion):
    """Maximum-cardinality accepted matching, tie-broken by quality. Order-invariant."""
    gts = _canonical(list(gts)); preds = _canonical(list(preds))
    if not gts or not preds:
        return MatchResult([], list(gts), list(preds))
    # Cardinality-safe bonus: one extra accepted match must outweigh all quality
    # differences in the rest of the assignment (quality per pair is bounded by 1.0).
    accept_bonus = min(len(gts), len(preds)) + 1.0
    s = np.zeros((len(gts), len(preds)))
    for i, g in enumerate(gts):
        for j, p in enumerate(preds):
            if criterion.accepts(g, p):
                s[i, j] = accept_bonus + criterion.score(g, p)
    rows, cols = linear_sum_assignment(s, maximize=True)
    matched, ug, up = [], set(), set()
    for i, j in zip(rows, cols):
        if criterion.accepts(gts[i], preds[j]):
            matched.append((gts[i], preds[j])); ug.add(i); up.add(j)
    return MatchResult(matched,
                       [g for i, g in enumerate(gts) if i not in ug],
                       [p for j, p in enumerate(preds) if j not in up])


def match_ranked(gts, preds, criterion):
    """Score-ranked greedy matching (mAP convention). Order-invariant."""
    gts = _canonical(list(gts))
    preds = sorted(preds, key=lambda x: (-getattr(x, "score", 1.0), x.clip_id, x.t_start,
                                         x.t_end, getattr(x, "pred_id", "")))
    used = [False] * len(gts); matched, up = [], []
    for p in preds:
        best, best_score = -1, -1.0
        for i, g in enumerate(gts):
            if used[i] or not criterion.accepts(g, p):
                continue
            sc = criterion.score(g, p)
            if sc > best_score:
                best, best_score = i, sc
        if best >= 0:
            used[best] = True; matched.append((gts[best], p))
        else:
            up.append(p)
    return MatchResult(matched, [g for i, g in enumerate(gts) if not used[i]], up)


_MATCHERS = {"hungarian": match_one_to_one, "greedy": match_ranked}


def drop_ignored(items, ignores):
    """Exclude items with ANY positive temporal overlap with an ignore interval."""
    by = {}
    for ig in ignores:
        by.setdefault(ig.clip_id, []).append(ig)
    return [it for it in items if not any(overlaps(it, sp) for sp in by.get(it.clip_id, []))]


def by_clip(items):
    d = {}
    for it in items:
        d.setdefault(it.clip_id, []).append(it)
    return d


def evaluate_class_aware(events, preds, criterion, ignores=(), matcher="hungarian"):
    """Match within each (clip, type) block; aggregate one MatchResult."""
    match_fn = _MATCHERS[matcher]
    events = drop_ignored(events, ignores); preds = drop_ignored(preds, ignores)
    ge, gp = by_clip(events), by_clip(preds)
    agg = MatchResult()
    for clip in set(ge) | set(gp):
        for t in VALID_TYPES:
            r = match_fn([e for e in ge.get(clip, []) if type_name(e.type) == t],
                         [p for p in gp.get(clip, []) if type_name(p.type) == t], criterion)
            agg.matched += r.matched; agg.unmatched_gt += r.unmatched_gt; agg.unmatched_pred += r.unmatched_pred
    return agg

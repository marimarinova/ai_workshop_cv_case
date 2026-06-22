"""mean Average Precision at a temporal IoU threshold (mAP@tIoU).

Order-invariant: ground-truth rows are canonically sorted, clips iterated in
sorted order, and the global ranking is sorted by (-score, t_start, t_end, id) so
score ties break deterministically regardless of input row order.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .class_aware_matching import _canonical, by_clip, drop_ignored
from .contracts import VALID_TYPES, type_name
from .intervals import tiou


def _scored_hits(
    events: Any, preds: Any, etype: str, tiou_thr: float, ignores: Any
) -> tuple[list[tuple[float, int, float, float]], int]:
    """Per type: list of (score, is_tp, t_start, t_end) over all clips + GT count."""
    events = drop_ignored(events, ignores)
    preds = drop_ignored(preds, ignores)
    ge = by_clip([e for e in events if type_name(e.type) == etype])
    gp = by_clip([p for p in preds if type_name(p.type) == etype])
    n_gt = sum(len(v) for v in ge.values())
    scored: list[tuple[float, int, float, float]] = []
    for clip in sorted(set(ge) | set(gp)):
        gts = _canonical(list(ge.get(clip, [])))
        ps = sorted(
            gp.get(clip, []),
            key=lambda x: (
                -getattr(x, "score", 1.0),
                x.t_start,
                x.t_end,
                getattr(x, "pred_id", ""),
            ),
        )
        used = [False] * len(gts)
        for p in ps:
            best, best_ov = -1, tiou_thr
            for i, g in enumerate(gts):
                if used[i]:
                    continue
                ov = tiou(p, g)
                if ov >= best_ov:
                    best, best_ov = i, ov
            scored.append((getattr(p, "score", 1.0), 1 if best >= 0 else 0, p.t_start, p.t_end))
            if best >= 0:
                used[best] = True
    return scored, n_gt


def average_precision(
    events: Any,
    preds: Any,
    etype: str,
    tiou_thr: float = 0.5,
    ignores: Any = (),
) -> float | None:
    """All-points AP for one type at one tIoU threshold (order-invariant).

    ``None`` when no GT of this type; ``0.0`` when GT exists but no predictions.
    """
    scored, n_gt = _scored_hits(events, preds, etype, tiou_thr, ignores)
    if n_gt == 0:
        return None
    if not scored:
        return 0.0
    scored.sort(key=lambda x: (-x[0], x[2], x[3]))
    tp = fp = 0
    prec: list[float] = []
    rec: list[float] = []
    for _score, is_tp, _ts, _te in scored:
        tp += is_tp
        fp += 1 - is_tp
        prec.append(tp / (tp + fp))
        rec.append(tp / n_gt)
    mrec = np.concatenate(([0.0], rec, [rec[-1]]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def mean_ap(
    events: Any,
    preds: Any,
    tiou_thresholds: tuple[float, ...] = (0.3, 0.5, 0.7),
    ignores: Any = (),
) -> dict[str, float | None]:
    """mAP@tIoU at each threshold plus the average across thresholds."""
    out: dict[str, float | None] = {}
    for thr in tiou_thresholds:
        aps = [average_precision(events, preds, t, thr, ignores) for t in VALID_TYPES]
        valid = [a for a in aps if a is not None]
        out[f"mAP@{thr}"] = float(sum(valid) / len(valid)) if valid else None
    vals = [v for v in out.values() if v is not None]
    out["mAP_avg"] = float(sum(vals) / len(vals)) if vals else None
    return out

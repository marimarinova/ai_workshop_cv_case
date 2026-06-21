"""Interval mathematics and the matching Criterion."""
from __future__ import annotations
from dataclasses import dataclass


def _mid(x):
    return 0.5 * (x.t_start + x.t_end)


def tiou(a, b):
    """Temporal IoU; 0.0 when disjoint or zero-union."""
    inter = max(0.0, min(a.t_end, b.t_end) - max(a.t_start, b.t_start))
    union = (a.t_end - a.t_start) + (b.t_end - b.t_start) - inter
    return inter / union if union > 0 else 0.0


def overlaps(a, b):
    """True if intervals have any positive temporal overlap."""
    return a.t_start < b.t_end and b.t_start < a.t_end


def midpoint_distance(a, b):
    return abs(_mid(a) - _mid(b))


@dataclass(frozen=True)
class Criterion:
    """Matching rule. ``kind`` is 'tiou' or 'midpoint'. Thresholds are inputs."""

    kind: str = "tiou"
    tiou_threshold: float = 0.5
    midpoint_tolerance_s: float = 1.0

    def __post_init__(self):
        if self.kind not in ("tiou", "midpoint"):
            raise ValueError(f"bad criterion kind {self.kind!r}")
        if not 0.0 <= self.tiou_threshold <= 1.0:
            raise ValueError("tiou_threshold out of [0,1]")
        if self.midpoint_tolerance_s < 0:
            raise ValueError("midpoint_tolerance_s must be >= 0")

    def score(self, gt, pred):
        if self.kind == "midpoint":
            return 1.0 / (1.0 + midpoint_distance(gt, pred))
        return tiou(gt, pred)

    def accepts(self, gt, pred):
        if self.kind == "midpoint":
            return midpoint_distance(gt, pred) <= self.midpoint_tolerance_s
        return tiou(gt, pred) >= self.tiou_threshold

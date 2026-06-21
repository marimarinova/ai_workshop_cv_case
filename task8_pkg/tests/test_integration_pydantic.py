"""Integration test: the evaluator consumes canonical-style Pydantic models directly.

Uses (str, Enum) to stand in for the Python 3.11 StrEnum of the real Task 1 schemas,
so it runs on 3.9/3.10 too while exercising the duck-typed path.
"""
from __future__ import annotations

from enum import Enum

import pytest

pydantic = pytest.importorskip("pydantic")
from pydantic import BaseModel, Field  # noqa: E402

from pickup_putdown.evaluation import aggregate_metrics, metrics_to_json  # noqa: E402


class EventType(str, Enum):
    PICKUP = "pickup"
    PUTDOWN = "putdown"


class Event(BaseModel):
    event_id: str
    clip_id: str
    type: EventType
    t_start: float
    t_end: float
    hard_case: bool = False
    confidence: str = "high"


class Prediction(BaseModel):
    pred_id: str
    clip_id: str
    type: EventType
    t_start: float
    t_end: float
    score: float = Field(ge=0.0, le=1.0)
    model: str


def test_evaluator_runs_on_pydantic_models():
    gts = [Event(event_id="e1", clip_id="c", type="pickup", t_start=1, t_end=2),
           Event(event_id="e2", clip_id="c", type="putdown", t_start=5, t_end=6)]
    pr = [Prediction(pred_id="p1", clip_id="c", type="pickup", t_start=1.1, t_end=2.1, score=0.9, model="m"),
          Prediction(pred_id="p2", clip_id="c", type="pickup", t_start=5, t_end=6, score=0.7, model="m")]  # type flip
    m = aggregate_metrics(gts, pr, {"c": 600.0})
    assert m["tiou@0.5"]["tp"] == 1
    assert m["confusion"]["putdown"]["pickup"] == 1     # flip captured
    json_str = metrics_to_json(m)                       # JSON-serializable
    assert "confusion" in json_str

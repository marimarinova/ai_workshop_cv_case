"""CSV / dict-row adapter into evaluation records.

The evaluator is duck-typed, so in the repo you can pass the canonical Pydantic
models directly. These loaders build the standalone records and correctly parse a
score of ``0.0`` (a value the old ``x or default`` idiom silently lost).
"""

from __future__ import annotations

import csv
from typing import Any

from .contracts import EvaluationEvent, EvaluationIgnoreInterval, EvaluationPrediction


def _get(row: dict[str, Any], key: str, cm: dict[str, str], default: Any = None) -> Any:
    return row.get(cm.get(key, key), default)


def _to_bool(v: Any) -> bool:
    return str(v).strip().lower() in ("1", "true", "yes", "y", "t")


def _num(v: Any, default: float) -> float:
    """Parse a number, preserving 0/0.0 (only fall back when missing/blank)."""
    if v is None or (isinstance(v, str) and v.strip() == ""):
        return default
    return float(v)


def events_from_rows(
    rows: list[dict[str, Any]], column_map: dict[str, str] | None = None
) -> list[EvaluationEvent]:
    cm = column_map or {}
    return [
        EvaluationEvent(
            clip_id=str(_get(r, "clip_id", cm)),
            type=str(_get(r, "type", cm)),
            t_start=float(_get(r, "t_start", cm)),
            t_end=float(_get(r, "t_end", cm)),
            event_id=str(_get(r, "event_id", cm, "") or ""),
            confidence=str(_get(r, "confidence", cm, "high") or "high"),
            hard_case=_to_bool(_get(r, "hard_case", cm, False)),
            annotator=str(_get(r, "annotator", cm, "") or ""),
            notes=str(_get(r, "notes", cm, "") or ""),
            n_person=int(_num(_get(r, "n_person", cm), 1)),
            group_id=str(_get(r, "group_id", cm, "") or ""),
        )
        for r in rows
    ]


def predictions_from_rows(
    rows: list[dict[str, Any]], column_map: dict[str, str] | None = None
) -> list[EvaluationPrediction]:
    cm = column_map or {}
    return [
        EvaluationPrediction(
            clip_id=str(_get(r, "clip_id", cm)),
            type=str(_get(r, "type", cm)),
            t_start=float(_get(r, "t_start", cm)),
            t_end=float(_get(r, "t_end", cm)),
            pred_id=str(_get(r, "pred_id", cm, "") or ""),
            score=_num(_get(r, "score", cm), 1.0),  # 0.0 preserved
            model=str(_get(r, "model", cm, "") or ""),
        )
        for r in rows
    ]


def ignores_from_rows(
    rows: list[dict[str, Any]], column_map: dict[str, str] | None = None
) -> list[EvaluationIgnoreInterval]:
    cm = column_map or {}
    return [
        EvaluationIgnoreInterval(
            clip_id=str(_get(r, "clip_id", cm)),
            t_start=float(_get(r, "t_start", cm)),
            t_end=float(_get(r, "t_end", cm)),
        )
        for r in rows
    ]


def read_csv(path: str) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_events_csv(path: str, column_map: dict[str, str] | None = None) -> list[EvaluationEvent]:
    return events_from_rows(read_csv(path), column_map)


def load_predictions_csv(
    path: str, column_map: dict[str, str] | None = None
) -> list[EvaluationPrediction]:
    return predictions_from_rows(read_csv(path), column_map)

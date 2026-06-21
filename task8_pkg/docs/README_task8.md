# task_8 — Shared Evaluator (package overview)

One evaluator for every model (Track A, B1, B2, Layer 2, Layer 3) so all results
are comparable. Works on the canonical `events` / `predictions` tables — no video.

## File map (src/pickup_putdown/evaluation/)

- `intervals.py`            — tIoU, midpoint distance, the `Criterion` (tiou | midpoint)
- `class_aware_matching.py` — Hungarian one-to-one matcher + score-ranked greedy + `evaluate_class_aware`
- `confusion_matching.py`   — class-agnostic pass → pickup/putdown confusion
- `ap.py`                   — average precision + `mean_ap` (mAP@tIoU)
- `metrics.py`              — `aggregate_metrics`, `slice_metrics`
- `report.py`               — `render_markdown`, `failure_gallery`
- `io.py`                   — CSV → objects adapter (with `column_map`)
- `contracts.py`            — lightweight dataclasses for self-contained tests
- `__init__.py`             — public API

## Run

```bash
PYTHONPATH=src pytest -q     # 32 passed
```

See `task_8_COMPLETION.md` for the full handoff and plan-§17 coverage.

# task_8_hard: Shared Two-Pass Temporal Evaluator and Reports — COMPLETION / HANDOFF

> This document mirrors the structure of `docs/tasks/task_8_shared_evaluator_hard.md`
> and records the as-built implementation. Cross-check against
> `docs/IMPLEMENTATION_PLAN.md` §17 (Shared Evaluation) and Gate 7.

**Task ID:** `task_8`
**Difficulty:** `hard`
**Status:** implementation complete; verified by the test suite (38 passed, 1 skipped). Real-data run pending `events.csv` (task_7) and a model `predictions.csv`.
**Dependencies:** Task 1 schemas (`pickup_putdown.common.schemas`). Synthetic fixtures used to build; the evaluator is duck-typed and consumes the canonical Pydantic `Event`/`Prediction` directly.
**Branch:** `feature/task-8-shared-evaluator`

## Objective

One evaluator used by Track A, Track B1, Track B2, standalone Layer 2, and Layer 3.
Given canonical ground-truth events and model predictions (both as typed intervals),
decide matches by temporal alignment and report detection-quality metrics. No
model-specific code.

## Inputs

- Canonical `events.csv` and `predictions.csv` (or in-memory `Event`/`Prediction` objects).
- Clip durations (for false-positives-per-hour).
- Optional `ignore_intervals` and hard-case/confidence metadata.

## Deliverables

- Class-aware one-to-one matcher (Hungarian, maximum-weight, order-invariant).
- Class-agnostic temporal matcher for pickup/putdown confusion.
- tIoU and midpoint metrics; precision/recall/F1; start/end MAE.
- Multi-item and event-count metrics; false-positives-per-hour.
- mAP@tIoU (optional metric from §17.3) + switchable matching convention (hungarian/greedy).
- Markdown report and failure-gallery hooks.

## Expected Files or Modules (as built)

| Plan-expected module | Implemented as |
|---|---|
| `evaluation/class_aware_matching.py` | `class_aware_matching.py` — `match_one_to_one`, `match_ranked`, `evaluate_class_aware` |
| `evaluation/confusion_matching.py` | `confusion_matching.py` — `evaluate_confusion` |
| `evaluation/metrics.py` | `metrics.py` — `aggregate_metrics`, `slice_metrics` |
| `evaluation/report.py` | `report.py` — `render_markdown`, `failure_gallery` |
| `tests/test_evaluation.py` | `tests/test_evaluation.py` — 35 tests + `tests/test_enum_types.py` (plain-Enum pipeline guard) + `tests/test_integration_pydantic.py` + `tests/test_integration_real_schemas.py` (1 skipped: runs in-repo) |
| (supporting) | `intervals.py` (tIoU/midpoint/Criterion), `ap.py` (mAP), `io.py` (CSV adapter), `contracts.py` (test-only dataclasses), `__init__.py` |

## Implementation Steps (done)

1. Interval tIoU and midpoint distance with numerical edge-case tests.
2. Maximum-weight one-to-one matching (scipy `linear_sum_assignment`); inputs canonically sorted so results do not depend on row order.
3. Second temporal-only pass; matched types compared to count pickup→putdown and putdown→pickup confusion.
4. Two-item ground truth counted per row; one prediction cannot satisfy two ground-truth rows.
5. Precision/recall/F1 at tIoU 0.3, tIoU 0.5, and midpoint ±1 s.
6. Start/end MAE, false positives per video hour, per-clip absolute event-count error, multi-item recall.
7. Slices for high/med, low confidence, hard cases, short vs long events.
8. Ignore intervals excluded from official matching.
9. Fixtures for no predictions, no ground truth, overlapping events, type flips, immediate pickup/putdown, two identical-time rows.
10. (Beyond mandatory) mAP@tIoU via score-ranked matching + all-points interpolation; `matcher="hungarian"|"greedy"` switch.

## Acceptance Criteria

- [x] Metrics are invariant to row ordering. (`test_order_invariance`, 100-shuffle property check)
- [x] A type flip appears as FP/FN in class-aware metrics and as explicit confusion in pass two. (`test_type_flip_is_fp_fn_and_confusion`)
- [x] Two-item ground truth requires two matched prediction rows. (`test_two_item_needs_two_predictions`)
- [x] All models can be evaluated without model-specific code. (duck-typed; verified against canonical Pydantic models)
- [x] Thresholds are inputs and never optimized against test labels by the evaluator.

## Plan §17 / Gate 7 coverage

- §17.1 Pass 1 class-aware — done. §17.2 Pass 2 confusion — done (reported separately).
- §17.3 required metrics — P/R/F1 @ tIoU0.3/0.5/midpoint±1s, both confusions, start/end MAE, FP/hour — done; mAP (optional) — done.
- §17.4 multi-item + per-clip event-count error — done; no row collapsing.
- §17.5 slices — all/high-med/low/hard_case/short-long — done.
- §17.6 threshold discipline — thresholds are inputs; never tuned on test.
- Gate 7 (class-aware + class-agnostic + multi-item row-level) — satisfied.

## Out of Scope (by design — belongs to other tasks)

- Prediction generation, model training, annotation.
- **Stage B proposal recall** — computed in Layer 0B / task_7 (candidates vs GT), fed in.
- (Runtime, HTML report, and the multiple-person slice were initially deferred but are now implemented — see "Full task-file compliance" below.)

## Handoff Contract

- PR `feature/task-8-shared-evaluator` → `main` (UI review/merge; not merged locally).
- Resolved config used for the acceptance run: defaults — `tiou_thresholds=(0.3,0.5)`, `midpoint_tol_s=1.0`, `map_thresholds=(0.3,0.5,0.7)`, `matcher="hungarian"`.
- Sample output / fixture: `samples/sample_metrics.json` (machine-readable), regenerable from code.
- Assumptions / limitations: multi-item detected by shared `group_id` or exact-duplicate rows (overlap clustering is explicit opt-in); real-data validation pending.
- No source videos, credentials, or model weights committed.

## How to run

```bash
pip install numpy scipy pytest
PYTHONPATH=src pytest -q          # -> 38 passed, 1 skipped
```

```python
from pickup_putdown.evaluation import aggregate_metrics, render_markdown
m = aggregate_metrics(events, predictions, clip_durations={...})
print(render_markdown(m, model_name="track_a_v1"))
```


## Full task-file compliance (every line of task_8_shared_evaluator_hard.md)

After review, the remaining literal items were implemented so the module matches
the task file 1:1:

- **Deliverable "Runtime and false-positive-per-hour metrics"** — `aggregate_metrics`
  now accepts `runtime_s` and reports `runtime_per_video_minute` (alongside `fp_per_hour`).
  Test: `test_runtime_per_video_minute`.
- **Deliverable "Markdown/HTML report"** — `report.render_html` added next to
  `render_markdown`. Test: `test_render_html_and_markdown`.
- **Implementation step 7 "slices … multiple-person"** — `slice_metrics` now emits a
  `multiple_person` slice wherever events carry an `n_person` attribute > 1
  ("where metadata is available", per §17.5). Test: `test_multiple_person_slice`.

Every Deliverable, Implementation Step (1–9), and Acceptance Criterion in
`task_8_shared_evaluator_hard.md` is now satisfied and covered by tests
(**38 passed, 1 skipped**). The only item still produced outside this module is Stage B
proposal recall, which by spec belongs to Layer 0B / task_7.

## Repository integration notes (at merge time)

- This package's `pyproject.toml` is for standalone testing. Do NOT replace the
  repository root `pyproject.toml`; instead add `scipy>=1.10` (and `numpy`) to the
  repository's existing dependencies/requirements and keep its `requires-python >= 3.12`.
- Run the repository's own `make` lint/type/test targets, not only `pytest -q`.
- `configs/evaluation_acceptance.yaml` sets `ignore_rule: any_positive_overlap` — confirm
  against `manifest/labeling-guidelines.md`; once confirmed, drop the inline "confirm" note.
- `tests/test_integration_real_schemas.py` runs against the real `pickup_putdown.common.schemas`
  in-repo (skips standalone).

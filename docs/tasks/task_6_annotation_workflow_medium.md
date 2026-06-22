# task_6_medium: Annotation Workflow and Canonical Import/Export

> This task belongs to the pickup/putdown temporal action detection implementation.
> Read `docs/concepts.md`, the copied `manifest/labeling-guidelines.md`, and
> `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md` before starting.
> The reference case repository is read-only; implementation artifacts belong in the solution repository.

**Task ID:** `task_6`  
**Difficulty:** `medium`  
**Dependencies:** Task 1. Task 5 improves candidate-assisted annotation but is not required to start.  
**Parallel work:** Tasks 2–5 and 8.

## Objective

Configure one interval-annotation workflow that enforces the operational definitions and exports exact case-compatible event tables.

## Inputs

- Copied labeling guidelines
- Person-containing clips and active spans
- Optional Stage B candidate suggestions

## Deliverables

- Configured annotation tool or purpose-built minimal UI
- Canonical `events.csv` export
- Internal `ignore_intervals.parquet` export
- Support for confidence, hard case, annotator, item count, and review status
- Candidate import as suggestions, not ground truth
- Annotation operating procedure

## Expected Files or Modules

- `src/pickup_putdown/annotation/import_export.py`
- `src/pickup_putdown/annotation/schemas.py`
- `manifest/labeling-guidelines.md`
- `docs/ANNOTATION_WORKFLOW.md`
- `tests/test_annotation_export.py`

## Implementation Steps

1. Choose one tool for all annotators. Prefer an existing temporal annotation tool unless candidate import makes a minimal custom interface materially simpler.
2. Embed the exact pickup, putdown, negative, multi-item, immediate-return, occlusion, confidence, and hard-case rules.
3. Require annotators to review the complete active span, even when candidates are supplied.
4. Allow zero, one, or multiple events in a candidate or clip.
5. Export two simultaneous items as two event rows sharing an internal `event_group_id`.
6. Use `confidence=low` for visible ambiguity; use ignore intervals only when transfer evidence is unavailable.
7. Exclude fully occluded/out-of-frame actions from `events.csv` while recording internal ignore intervals to prevent negative sampling.
8. Round-trip a sample export back into the tool or converter to prove timestamp fidelity.

## Acceptance Criteria

- [ ] Exported `events.csv` uses exact canonical columns and allowed values.
- [ ] Immediate pickup then putdown creates two ordered rows.
- [ ] Two-item pickup creates two rows.
- [ ] Ignore intervals never appear as official events.
- [ ] Candidate suggestions can be corrected, deleted, or supplemented.

## Out of Scope

- Dataset split
- Model training
- Evaluation metrics

## Handoff Contract

The task owner must provide:

- a pull request containing the implementation and tests;
- the resolved configuration used for the acceptance run;
- one machine-readable sample output or fixture;
- a short note listing assumptions, known limitations, and any interface changes;
- confirmation that no source videos, credentials, or model weights were committed.

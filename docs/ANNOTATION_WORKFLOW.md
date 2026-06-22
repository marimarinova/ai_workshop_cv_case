# Annotation Workflow

> **Authority:** The canonical labeling rules are defined in
> [`manifest/labeling-guidelines.md`](../manifest/labeling-guidelines.md).
> This document covers tool-specific actions and operating procedure.

## 1. Starting Label Studio

```bash
make annotation-up
```

This starts Label Studio Community Edition (pinned to image version 1.15.0) on
port 8080 (configurable via `ANNOTATION_PORT`).

## 2. Accessing the UI

Open `http://localhost:8080` in your browser. Log in with your credentials or
create an account if this is your first time.

## 3. Creating a Project

1. Click **Create New Project**.
2. In **Labeling Setup**, paste the full contents of
   [`annotation/label_studio_config.xml`](../annotation/label_studio_config.xml).
3. Save the project.

The shared XML configuration enforces:
- Temporal interval annotation for `pickup`, `putdown`, and `ignore`.
- Per-region metadata: confidence, hard_case, item_count, review_status.
- Task-level `complete_active_span_reviewed` confirmation.

## 4. Video Mounting

Videos are mounted read-only from the directory specified by
`ANNOTATION_VIDEO_DIR` (default: `./data/videos`). **Do not commit source
videos to Git.** They must reside in this directory or a symlinked location.

## 5. Importing Tasks

1. Go to **Data** → **Import**.
2. Upload a JSON file in the Label Studio task format.
3. Use `make annotation-test` or the Python CLI to generate task JSON from
   candidate predictions:

```bash
python -m pickup_putdown.annotation import-tasks \
  --clips clips.json --candidates candidates.json \
  --output annotation/tasks.json
```

## 6. Candidate Suggestions

Candidates from Stage B (pose-based proposals) appear as **pre-annotated
predictions**. They are:
- **Editable** — annotators can resize, move, or delete them.
- **Supplementable** — annotators can add events not present in candidates.
- **Traceable** — each prediction carries `candidate_id`, `candidate_score`,
  and `model_source` metadata.

**Candidates are suggestions only.** They must never be imported as completed
ground truth.

## 7. Complete Active Span Review

Annotators **must** review the entire active span of each clip, not just the
candidate intervals. The `complete_active_span_reviewed` checkbox is required
before export. A reviewed clip with zero events is valid and distinguishable
from an unreviewed clip.

## 8. Annotating Events

### Zero events
If no pickup or putdown occurs in the active span, leave the timeline empty
and check the review confirmation box.

### One or multiple events
- Draw temporal regions on the timeline.
- Assign labels: `pickup`, `putdown`, or `ignore`.
- Set per-region metadata: confidence, hard_case, item_count, review_status.

### Immediate pickup then putdown
Create **two separate ordered events**. Do not merge them.

### Multiple simultaneous items
For an event with `item_count=N`, create **N separate events** with identical
intervals. The export will produce N canonical rows sharing one
`event_group_id`.

## 9. Confidence vs. Ignore

| Condition | Action |
|-----------|--------|
| Visible transfer, clear | `confidence=high` |
| Visible transfer, likely | `confidence=med` |
| Visible transfer, uncertain | `confidence=low` — **still an official event** |
| Hand/item fully occluded | **Ignore interval** — not an official event |
| Hand/item out of frame | **Ignore interval** — not an official event |

**Low-confidence visible events remain in `events.csv`.** Ignore intervals
never appear as official events.

## 10. Review Status

Set per-region review status:
- `draft` — initial annotation
- `reviewed` — checked by annotator
- `accepted` — finalized
- `needs_adjudication` — disagreement between annotators

## 11. Exporting Label Studio JSON

After annotation, export from Label Studio:
1. Go to **Exports** → select format.
2. Download the JSON file.

The export must include the `complete_active_span_reviewed` metadata field.

## 12. Generating Canonical Outputs

Convert the Label Studio export to canonical repository formats:

```bash
python -m pickup_putdown.annotation export \
  --input annotation/export.json \
  --events events.csv \
  --ignore ignore_intervals.parquet
```

Or programmatically:

```python
from pickup_putdown.annotation import export_events_csv, export_ignore_intervals_parquet

with open("annotation/export.json") as f:
    data = json.load(f)

export_events_csv(data, "events.csv")
export_ignore_intervals_parquet(data, "ignore_intervals.parquet")
```

### Official events (events.csv)
- Only accepted visible `pickup` and `putdown` annotations.
- Requires `complete_active_span_reviewed=true`.
- Chronological ordering within each clip.
- Multi-item expansion: N rows with shared `event_group_id`.

### Ignore intervals (ignore_intervals.parquet)
- Only `ignore`-label regions.
- Used for excluding occluded/out-of-frame spans from negative sampling.

## 13. Acceptance Round Trip

Verify timestamp fidelity after export:

```python
from pickup_putdown.annotation import round_trip_check

original = [...]  # CanonicalEvent objects
export_data = json.loads(Path("export.json").read_text())

assert round_trip_check(original, export_data, fps=30.0)
```

Tolerance is 1 frame by default.

## 14. Files That Must NOT Be Committed

| Artifact | Reason |
|----------|--------|
| `data/videos/*.mp4` | Source videos — mounted read-only |
| `annotation/tasks.json` | Generated per-session |
| `annotation/export.json` | Generated per-session |
| `annotation/label_studio_data/` | Label Studio database |
| `.env` | Credentials |
| Any file containing real annotator data | Privacy |

These paths are covered by `.gitignore`.

## Troubleshooting

### Docker service not starting
```bash
make annotation-status
make annotation-logs
```
Check if port 8080 is in use. Override with `ANNOTATION_PORT=8081`.

### Media not loading
- Verify `ANNOTATION_VIDEO_DIR` points to a directory containing videos.
- Check Docker volume mount: `docker compose -f docker-compose.annotation.yml ps`.
- Ensure videos use supported codecs (H.264 MP4 recommended).

### Unsupported video codec
Label Studio uses browser video playback. Use H.264 codec in MP4 container.
Convert with:
```bash
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

### Wrong local media path
Set `ANNOTATION_VIDEO_DIR` to the correct path:
```bash
ANNOTATION_VIDEO_DIR=/path/to/videos make annotation-up
```

### Invalid XML configuration
```bash
make annotation-config-validate
```
Checks that `label_studio_config.xml` exists, is well-formed, and contains
required controls and labels.

### Export validation failure

## Task 6 Acceptance Matrix

| Requirement | Automated proof |
|---|---|
| Exact canonical columns and values | `TestTask6Acceptance::test_01_exact_canonical_columns_and_values` |
| Immediate pickup followed by putdown | `TestTask6Acceptance::test_02_immediate_pickup_then_putdown` |
| Two-item pickup | `TestTask6Acceptance::test_03_two_item_pickup` |
| Ignore intervals excluded from events | `TestTask6Acceptance::test_04_ignore_intervals_excluded` |
| Candidate correction (human overrides) | `TestTask6Acceptance::test_05_candidate_correction` |
| Candidate deletion (no event) | `TestTask6Acceptance::test_06_candidate_deletion` |
| Candidate supplementation (manually added) | `TestTask6Acceptance::test_07_candidate_supplementation` |
| Unconfirmed clip emits no events | `TestTask6Acceptance::test_08a_unconfirmed_no_events` |
| Confirmed zero-event clip valid | `TestTask6Acceptance::test_08b_confirmed_zero_events` |
| Timestamp round-trip fidelity | `TestTask6Acceptance::test_09_timestamp_round_trip` |
| Deterministic export | `TestTask6Acceptance::test_10_deterministic_export` |

Run all acceptance tests:

```bash
make annotation-acceptance
```

### Export validation failure
```bash
python -c "
from pickup_putdown.annotation import validate_export
import json
errors = validate_export(json.load(open('annotation/export.json')))
for e in errors.errors:
    print(f'{e.task_id}/{e.region_id}: {e.message}')
"
```

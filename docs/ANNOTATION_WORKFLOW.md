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

### Candidate clip review (Task 6.2)

For candidate-backed tasks (those built from Task 6.1 candidate metadata), the
annotator reviews only the trimmed candidate clip window, not the full active
span. These tasks carry `annotation_unit: candidate_clip` in their data.

During export, candidate-backed tasks accept either:
- `candidate_clip_reviewed = true` (preferred for candidate tasks); or
- `complete_active_span_reviewed = true` (backward compatibility).

A candidate task with neither confirmation fails validation. The exported
events are never interpreted as having the full active span reviewed — they
are traceable only to the candidate clip window via `candidate_id` in the
provenance export.

Legacy (non-candidate) tasks still require `complete_active_span_reviewed =
true` and are unaffected by this change.

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
- For legacy tasks: requires `complete_active_span_reviewed=true`.
- For candidate-backed tasks: requires `candidate_clip_reviewed=true` (or
  `complete_active_span_reviewed=true` for backward compatibility).
- Contains **only the approved canonical columns**: `event_id`, `clip_id`,
  `type`, `t_start`, `t_end`, `hard_case`, `annotator`, `confidence`, `notes`.
- Chronological ordering within each clip.
- Multi-item expansion: N rows with shared `event_group_id`.
- Compatible with Task 8 evaluator without downstream filtering.

### Ignore intervals (ignore_intervals.parquet)
- Only `ignore`-label regions.
- Contains **only the approved canonical columns**: `ignore_id`, `clip_id`,
  `t_start`, `t_end`, `reason`, `annotator`, `notes`.
- Used for excluding occluded/out-of-frame spans from negative sampling.

### Provenance export (event_provenance.parquet)
- Optional artifact produced with `--provenance` flag.
- Contains candidate traceability metadata: `event_id`, `candidate_id`,
  `clip_id`, `actor_id`, `hand_side`, `region_id`, `event_group_id`.
- Not consumed by Task 8 evaluator. Preserves full traceability from exported
  events back to originating candidates and generation configuration.

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

## 15. Candidate-Backed Annotation (Task 6.2)

This section covers the production annotation handoff workflow for candidate
clips generated by Task 6.1. Candidate clips are trimmed video windows that
may contain zero, one, or multiple events. The annotator decides the event
type explicitly — no default label is assigned.

### Required candidate metadata

Each candidate metadata record must include:

| Field | Required | Description |
|---|---|---|
| `candidate_id` | yes | Unique candidate identifier |
| `clip_id` | yes | Original source video identifier |
| `source_start_s` | yes | Candidate window start in source-video seconds |
| `source_end_s` | yes | Candidate window end in source-video seconds |
| `candidate_video` or `candidate_key` | yes | S3 key, URL, or local path accessible to Label Studio |
| `actor_id` | no | Actor track identifier (e.g. `track_3`) |
| `hand_side` | no | Hand side (`left` or `right`) |
| `region_id` | no | Shelf/surface region identifier |
| `proposal_score` | no | Candidate proposal score |
| `config_fingerprint` | no | Generation configuration fingerprint |
| `duration_s` | no | Candidate clip duration in seconds |
| `fps` | no | Candidate clip frame rate |

Missing optional fields do not prevent task creation. Missing required fields
produce a clear validation error naming the candidate and the missing field.

### How candidate videos are referenced

Candidate videos are stored at:

```text
s3://chillnbite-cameras/anon/candidates/videos/<source_video_id>/<candidate_id>.mp4
```

or locally at:

```text
.local/candidate_staging/candidates/<source_video_id>/<candidate_id>.mp4
```

The `candidate_video` or `candidate_key` field in the metadata must point to
a location Label Studio can access.

### Task-generation command

All candidates:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_candidates.json
```

With pilot sampling (30-50 candidates):

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_pilot.json \
  --limit 40 \
  --seed 42
```

With S3 storage integration:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_candidates.json \
  --video-url-mode s3_storage \
  --s3-bucket chillnbite-cameras
```

With local video serving:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_candidates.json \
  --video-url-mode local \
  --local-video-dir /data/candidates
```

The command:
- Processes all `.json` files in the directory (sorted deterministically).
- Produces one Label Studio task per valid candidate.
- With `--limit`, selects a deterministic subset using `--seed` for
  reproducibility.
- Reports the number of generated and rejected tasks.
- Fails with exit code 1 when any candidate has validation errors.
- Assigns no default event label — the candidate is only a possible
  interaction interval.
- Sets `annotation_unit: candidate_clip` in task data.

### Import procedure

1. Generate tasks:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_candidates.json
```

2. Start Label Studio:

```bash
make annotation-up
```

3. Import tasks in Label Studio UI:
- Go to **Data** → **Import**.
- Upload `annotation/tasks_candidates.json`.

4. Annotate each candidate video:
- Watch the complete candidate clip.
- Draw temporal regions for events.
- Assign labels: `pickup`, `putdown`, or `ignore`.
- Set per-region metadata: confidence, hard_case, item_count, review_status.
- Mark `candidate_clip_reviewed` as confirmed (or `complete_active_span_reviewed`
  for backward compatibility).

### Candidate-relative to source-video timestamp conversion

Label Studio records timestamps relative to the candidate video. During export,
these are converted to source-video timestamps using:

```text
source_event_start_s = source_start_s + candidate_relative_start_s
source_event_end_s   = source_start_s + candidate_relative_end_s
```

Example:
```text
Candidate source interval: 102.0–110.0 s
Annotation in candidate:   2.3–3.7 s
Canonical source event:    104.3–105.7 s
```

For legacy tasks without `source_start_s`, timestamps pass through unchanged
(zero-offset behavior).

### Export and validation commands

Export with source offset conversion (official canonical outputs only):

```bash
pickup-putdown annotation-export \
  --input annotation/label_studio_export.json \
  --events annotation/exports/events.csv \
  --ignore annotation/exports/ignore_intervals.parquet \
  --candidate-mode
```

Export with provenance traceability:

```bash
pickup-putdown annotation-export \
  --input annotation/label_studio_export.json \
  --events annotation/exports/events.csv \
  --ignore annotation/exports/ignore_intervals.parquet \
  --provenance annotation/exports/event_provenance.parquet \
  --candidate-mode
```

Export without offset conversion (legacy mode):

```bash
pickup-putdown annotation-export \
  --input annotation/label_studio_export.json \
  --events annotation/exports/events.csv \
  --ignore annotation/exports/ignore_intervals.parquet
```

Validate export:

```bash
pickup-putdown annotation-validate \
  --input annotation/label_studio_export.json
```

### Official canonical schemas

**events.csv** (Task 8 compatible, no provenance fields):

```
event_id,clip_id,type,t_start,t_end,hard_case,annotator,confidence,notes
```

**ignore_intervals.parquet** (Task 8 compatible, no provenance fields):

```
ignore_id,clip_id,t_start,t_end,reason,annotator,notes
```

**event_provenance.parquet** (optional, candidate traceability only):

```
event_id,candidate_id,clip_id,actor_id,hand_side,region_id,event_group_id
```

### Timestamp validation rules

The exporter validates:
- `source_start_s < source_end_s`
- Candidate-relative start is non-negative
- Candidate-relative end does not exceed candidate duration (tolerance: 0.05 s)
- Event start is before event end
- Required source mapping fields exist
- Computed source timestamps stay within the declared source interval

Violations produce validation errors identifying the candidate and annotation.

### How to prepare a 30–50 candidate pilot

1. Generate candidates from source videos (if not already done):

```bash
# Using remote S3 pipeline
make candidates-remote \
  CANDIDATE_TARGET_COUNT=5 \
  CANDIDATE_WORKERS=2 \
  CANDIDATE_TRANSFER_WORKERS=4 \
  CANDIDATE_GPU_WORKERS=1 \
  CANDIDATE_ENCODE_WORKERS=4
```

2. Select a deterministic pilot subset from existing candidate metadata:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_pilot.json \
  --limit 40 \
  --seed 42
```

The `--limit` flag selects at most that many valid candidates. The `--seed`
ensures reproducible selection. Selection is deterministic for the same seed
and inputs. If fewer candidates exist than requested, all valid candidates are
exported.

3. Verify media references before import:

```bash
pickup-putdown annotation-check-media \
  --tasks annotation/tasks_pilot.json \
  --video-url-mode s3_key
```

4. Import into Label Studio and annotate.

5. Export and validate:

```bash
pickup-putdown annotation-export \
  --input annotation/label_studio_export_pilot.json \
  --events annotation/exports/pilot_events.csv \
  --ignore annotation/exports/pilot_ignore.parquet \
  --provenance annotation/exports/pilot_provenance.parquet \
  --candidate-mode
```

6. Manually verify several exported timestamps against the original source
   videos (see below).

### Manual source-video timestamp verification procedure

1. Open the original source video for a given `clip_id`.
2. For each exported event, navigate to the exported `t_start` timestamp.
3. Confirm the frame matches the expected action onset.
4. Navigate to `t_end` and confirm the action has stabilized.
5. Cross-check: `t_start - source_start_s` should match the candidate-relative
   timestamp recorded in Label Studio.

Example verification:
```text
Exported event: t_start=104.3, t_end=105.7, clip_id=source_clip_001
Candidate:      source_start_s=102.0, source_end_s=110.0
Expected in candidate video: 2.3–3.7 s
Verify: 104.3 - 102.0 = 2.3 ✓, 105.7 - 102.0 = 3.7 ✓
```

### Known limitations

- Candidate tasks do not include pre-annotated predictions. The annotator
  must draw all temporal regions manually.
- The boundary tolerance of 0.05 s is fixed and not currently configurable.

### Supported video playback modes

Candidate videos can be referenced in Label Studio using different playback
modes, configured via `--video-url-mode` during task building:

| Mode | Description | Configuration |
|------|-------------|---------------|
| `s3_key` (default) | Raw S3 object key passed through | None |
| `s3_storage` | `s3://bucket/key` format for Label Studio cloud-storage integration | `--s3-bucket` required |
| `local` | Local file path under `--local-video-dir` | `--local-video-dir` required |
| `presigned` | Presigned S3 URL (expires) | URL must be http(s) |

**Important:** A raw S3 key like `anon/candidates/videos/cand_001.mp4` is NOT
automatically playable by Label Studio. Choose the mode matching your Label
Studio deployment:

- **Local serving:** Download candidates locally, then use `--video-url-mode
  local --local-video-dir /path/to/candidates`. Label Studio serves files from
  its mounted document root.
- **S3 cloud-storage integration:** Use `--video-url-mode s3_storage --s3-bucket
  chillnbite-cameras`. Requires Label Studio to be configured with S3
  cloud-storage credentials.
- **Presigned URLs:** Generate presigned URLs externally, then use
  `--video-url-mode presigned`. URLs expire — regenerate at task-build time.

### Media verification command

Before importing tasks, verify video references:

```bash
pickup-putdown annotation-check-media \
  --tasks annotation/tasks_pilot.json \
  --video-url-mode local \
  --local-video-dir /data/candidates
```

This checks:
- Each task has a video reference
- URL/path format matches the selected mode
- Local files exist (for local mode)
- S3 objects exist (for s3_storage mode, with credentials)
- Reports unsupported or malformed references

### Troubleshooting videos that do not load

1. Run `annotation-check-media` to identify broken references.
2. For local mode, verify `--local-video-dir` contains the candidate videos.
3. For s3_storage mode, verify Label Studio cloud-storage integration is
   configured with correct S3 credentials.
4. Ensure videos use H.264 codec in MP4 container (see codec conversion
   above).

## Task 6.2 Acceptance Matrix

| Requirement | Automated proof |
|---|---|
| Required metadata in task JSON | `TestRequiredMetadataInTask::test_all_required_fields_present` |
| No default event label | `TestNoDefaultEventLabel::test_no_predictions_in_task` |
| Zero offset preserves timestamps | `TestZeroSourceOffset::test_zero_offset_preserves_timestamps` |
| Non-zero offset added correctly | `TestNonZeroSourceOffset::test_offset_added_correctly` |
| Pickup export conversion | `TestPickupExportConversion::test_pickup_with_offset` |
| Putdown export conversion | `TestPutdownExportConversion::test_putdown_with_offset` |
| Ignore interval export conversion | `TestIgnoreIntervalExportConversion::test_ignore_with_offset` |
| Candidate-boundary annotations | `TestCandidateBoundaryAnnotations::test_at_start_boundary` |
| Negative relative timestamps rejected | `TestNegativeRelativeTimestamps::test_negative_start_rejected` |
| Beyond-duration timestamps rejected | `TestBeyondDurationTimestamps::test_beyond_duration_rejected` |
| Missing required metadata error | `TestMissingRequiredMetadata::test_missing_candidate_id` |
| Invalid source intervals rejected | `TestInvalidSourceIntervals::test_source_start_equals_end` |
| Optional metadata round trip | `TestOptionalMetadataRoundTrip::test_actor_hand_region_survive` |
| Multi-item determinism | `TestMultiItemDeterminism::test_multi_item_deterministic` |
| Legacy export compatibility | `TestLegacyExportCompatibility::test_legacy_export_still_works` |
| Deterministic repeated export | `TestDeterministicExport::test_repeated_candidate_export_identical` |
| End-to-end round trip | `TestEndToEndFixture::test_full_round_trip` |

Run Task 6.2 tests:

```bash
python -m pytest tests/test_candidate_annotation.py -v
```

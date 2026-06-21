# Implementation Manifest Schema

This file defines the canonical dataset and prediction exports used by the solution repository.

The original case manifest remains unchanged in the case repository. This version keeps the same three canonical CSV tables while making several ambiguous behaviors explicit and adding internal-only tables needed by the implementation.

---

## 1. Canonical table: `clips.csv`

One row per source video file.

| column | type | meaning |
|---|---|---|
| `clip_id` | string | Stable unique ID assigned to one source file. Never reused. |
| `s3_key` | string | Stable source object key or repository-relative source path. Never a machine-specific cache path. |
| `duration_s` | float | Source clip duration in seconds. |
| `fps` | float | Source frames per second. |
| `width` | int | Source width in pixels. |
| `height` | int | Source height in pixels. |
| `n_person_tracks` | int | Number of stable person tracklets; `0` means no person was found. |
| `usable` | bool | Whether the clip is eligible for pickup/putdown annotation or model use. |
| `active_start_s` | float or null | Earliest person-active timestamp in the clip. |
| `active_end_s` | float or null | Latest person-active timestamp in the clip. |
| `split` | enum | `train`, `val`, or `test`. |
| `session_id` | string or null | Recording/session grouping used for leakage-safe splits. |
| `notes` | string or null | Quality issues, duplicate notes, additional active spans, or other remarks. |

### Active-span rule

When one or more internal active spans exist:

```text
active_start_s = minimum active-span start
active_end_s   = maximum active-span end
```

When no active span exists, both fields must be null.

Exact disjoint spans are stored internally in `active_spans.parquet`.

---

## 2. Canonical table: `events.csv`

One row per human-verified ground-truth event.

| column | type | meaning |
|---|---|---|
| `event_id` | string | Unique event ID. |
| `clip_id` | string | Foreign key to `clips.clip_id`. |
| `type` | enum | `pickup` or `putdown`. |
| `t_start` | float | Event start in seconds from source-clip start. |
| `t_end` | float | Event end in seconds from source-clip start. |
| `hard_case` | bool | `true` for difficult but labelable events. |
| `annotator` | string | Annotator identifier. |
| `confidence` | enum | `high`, `med`, or `low`. |
| `notes` | string or null | Free text about ambiguity, item count, or event context. |

Rules:

- Store every event as `[t_start, t_end]`.
- Two simultaneous items produce two rows with unique `event_id` values.
- Immediate pickup followed by putdown produces two ordered rows.
- Fully occluded, out-of-frame, or otherwise unlabelable actions do not appear here.
- Visible but uncertain events remain official rows with `confidence=low`.

---

## 3. Canonical table: `predictions.csv`

One row per predicted event.

| column | type | meaning |
|---|---|---|
| `pred_id` | string | Unique prediction ID. |
| `clip_id` | string | Foreign key to `clips.clip_id`. |
| `type` | enum | `pickup` or `putdown`. |
| `t_start` | float | Predicted start in seconds from source-clip start. |
| `t_end` | float | Predicted end in seconds from source-clip start. |
| `score` | float | Finite normalized confidence in `[0, 1]`. |
| `model` | string | Immutable model/run identifier that produced the row. |

Rules:

- Candidates and annotation suggestions must never be exported as predictions.
- Two predicted items produce two rows with unique `pred_id` values.
- Preserve raw model outputs separately from fused or post-processed predictions.

---

## 4. Internal-only tables

These tables support the implementation but are not part of the canonical case CSV schema.

### `active_spans.parquet`

```text
clip_id
active_span_id
t_start
t_end
n_person_tracks
```

### `ignore_intervals.parquet`

```text
ignore_id
clip_id
t_start
t_end
reason
annotator
notes
```

Allowed reasons should be versioned, for example:

```text
ACTION_OCCLUDED
ACTION_OUT_OF_FRAME
CLIP_BOUNDARY
UNLABELABLE
CORRUPT_SECTION
```

Ignore intervals:

- never appear in `events.csv`;
- receive zero training and evaluation weight;
- must not be sampled as background.

Additional internal columns such as `actor_id`, `event_group_id`, `item_index`, `candidate_id`, `item_count`, and `boundary_method` are allowed only outside the canonical CSV exports.

---

## 5. Canonical validation rules

### `clips.csv`

```text
clip_id is unique and non-empty
s3_key is non-empty
duration_s > 0
fps > 0
width > 0
height > 0
n_person_tracks >= 0
split in {train, val, test}
active_start_s and active_end_s are both null or both populated
0 <= active_start_s < active_end_s <= duration_s
```

### `events.csv`

```text
event_id is unique
clip_id exists in clips.csv
type in {pickup, putdown}
0 <= t_start < t_end <= clip duration
confidence in {high, med, low}
annotator is non-empty
```

### `predictions.csv`

```text
pred_id is unique
clip_id exists in clips.csv
type in {pickup, putdown}
0 <= t_start < t_end <= clip duration
score is finite and 0 <= score <= 1
model is non-empty
```

Canonical CSV exports must contain exactly the documented columns in the documented order.

---

## 6. Versioning and reproducibility

- Never silently overwrite a labeled dataset version.
- Store human ground truth separately from VLM or model suggestions.
- Record dataset version, split version, Git commit, resolved configuration, model/run ID, and checkpoint hash.
- Duplicate files still receive distinct `clip_id` values; duplicate relationships are recorded internally.

---

## 7. Evaluation contract

For each clip:

1. Perform class-aware one-to-one temporal matching for precision, recall, F1, tIoU, and timing error.
2. Perform a separate class-agnostic temporal match to measure pickup/putdown type confusion.
3. Do not collapse duplicated multi-item rows before official matching.
4. Use deterministic tie-breaking.
5. Support score-threshold sweeps and precision-recall reporting.

---

# Differences from the original case manifest

1. **Multiple active spans are explicit.**  
   The canonical columns store the enclosing interval, while exact disjoint spans are kept in `active_spans.parquet`. This avoids losing valid events while preserving precise internal timing.

2. **No-person active timestamps are null.**  
   Both `active_start_s` and `active_end_s` are null when no active span exists. Using `0.0` would create a misleading zero-length interval.

3. **`usable` has one fixed meaning.**  
   It means eligible for pickup/putdown annotation or model use. Decode validity and person presence are tracked separately in internal metadata.

4. **Confidence is categorical only.**  
   Canonical ground truth uses `high`, `med`, or `low`, rather than allowing both strings and numeric values. This prevents mixed-type CSV columns and inconsistent filtering.

5. **Ignore intervals are formalized internally.**  
   They distinguish unlabelable footage from visible low-confidence events and prevent hidden actions from becoming false background.

6. **Prediction scores must be normalized.**  
   `score` must be finite and within `[0, 1]`, so thresholding and precision-recall curves are comparable across runs.

7. **Canonical CSVs are strict.**  
   Additional implementation metadata is stored only in internal tables, keeping exports exactly compatible with the case schema.

8. **Evaluation behavior is explicit.**  
   Matching is deterministic, class-aware metrics and class-confusion analysis are separate, and score-threshold sweeps are required.

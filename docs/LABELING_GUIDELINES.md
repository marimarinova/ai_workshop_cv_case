# Pickup and Putdown Labeling Guidelines

This document is the annotation rulebook for the pickup/putdown temporal action detection project. It must remain synchronized with `docs/concepts.md` and the operational definitions in `PICKUP_PUTDOWN_IMPLEMENTATION_PLAN_CONCEPTS_ALIGNED.md`.

Use these rules consistently across annotators, annotation-tool configuration, canonical exports, model training targets, prompts, evaluation, and error analysis.

---

## 1. Annotation scope

Annotate human-visible `pickup` and `putdown` events in untrimmed video clips.

For every selected person-containing clip:

1. Review the **complete active span**, not only generated interaction candidates.
2. Treat pose/shelf candidates as suggestions that may be corrected, split, extended, or deleted.
3. Record each accepted event as an interval `[t_start, t_end]` in seconds from the start of the original source clip.
4. Preserve chronological order when reviewing and labeling.

An interaction candidate is not a ground-truth event. One candidate may contain zero, one, or multiple ordered events.

---

## 2. Event definitions

### 2.1 `pickup`

A person removes an item from a shelf or surface and takes it into their hand or hands so that the item leaves its resting place and becomes held or carried.

The defining state transition is:

```text
shelf/surface → hand
```

### 2.2 `putdown`

A person places an item that they were already holding onto a shelf or surface and releases it so that the item remains resting there.

The defining state transition is:

```text
hand → shelf/surface
```

A generic placement is not automatically a putdown. The visible evidence must establish that the item was already being held and is being returned or placed as part of the relevant interaction context.

---

## 3. Event boundaries

Candidate boundaries and event boundaries are different.

### 3.1 Start time

Set `t_start` to the onset of the **final purposeful action that causes the object transfer**.

Do not automatically use:

- the start of the interaction candidate;
- the moment the person first becomes visible;
- the moment the wrist first enters an expanded shelf region;
- an unrelated earlier reach or browsing motion.

### 3.2 Pickup end time

Set `t_end` when the item:

- has left its resting place; and
- is stably controlled, held, or carried by the person.

### 3.3 Putdown end time

Set `t_end` when the item:

- has been released by the person; and
- remains stably resting on the shelf or surface.

A hand entering or leaving a shelf region is proposal evidence only. It is not automatically an event boundary.

### 3.4 Timestamp rules

- Store timestamps in seconds from the beginning of the original clip.
- Do not store times relative to a candidate preview or active-span excerpt.
- Require finite timestamps satisfying `0 <= t_start <= t_end <= clip duration`.
- Use the finest reliable temporal precision supported by the annotation tool and video.

---

## 4. Non-events and hard negatives

Do **not** create an event row for:

- touching or inspecting an item without removing it;
- looking or reaching past an item;
- browsing, standing, or walking near shelves;
- hand motion near a shelf without persistent object transfer;
- carrying an item past a shelf without placing it;
- visible restocking or placement of newly introduced goods;
- empty or no-person clips.

Visible restocking is normally retained as background or a hard-negative example. Do not create an ignore interval merely because an action is restocking.

A different staff/restocking scope policy may be used only when it is decided before annotation, documented, and applied consistently to the whole dataset.

---

## 5. Edge-case rules

Apply these rules exactly.

### 5.1 Multiple items

Taking or placing two items simultaneously produces **two official event rows**, one per item.

The rows must have:

- unique `event_id` values;
- the same `clip_id`, type, and interval when the transfers are simultaneous;
- an optional shared internal `event_group_id`;
- an optional internal `item_index`.

Do not collapse a two-item action into one canonical event row.

### 5.2 Immediate pickup followed by return

A pickup followed immediately by a putdown produces **two ordered events**:

1. `pickup`;
2. `putdown`.

Do not merge them merely because the gap is short or both occur inside one interaction candidate.

### 5.3 Multiple people

Label every visible event performed by every actor, including simultaneous events.

When the annotation workflow supports it, retain a clip-local internal `actor_id` such as `track_3`. Actor IDs are tracking references, not personal identities, and are not part of the canonical `events.csv` export.

### 5.4 Very brief or ambiguous actions

Keep a visible action as an official event with `confidence=low` when it is more likely than not to satisfy the event definition, but its type, count, or exact boundaries remain uncertain.

Do not discard an event only because it is brief.

### 5.5 Partial visibility

Annotate a partially obscured action when the object transfer and event type remain sufficiently observable.

Use:

- `hard_case=true` when the event is difficult but still labelable;
- `confidence=low` when type, count, or boundaries are uncertain;
- notes describing the visibility limitation when useful.

### 5.6 Fully occluded or out-of-frame actions

Do not add an official event row when the evidence required to determine whether transfer occurred is unavailable because the hand, item, or decisive transition is fully occluded, outside the frame, or technically unusable.

Create an internal ignore interval instead, when the interval can be identified approximately.

---

## 6. Confidence

Use exactly one of:

```text
high
med
low
```

### `high`

The event type, item count, and boundaries are clearly visible.

### `med`

The event is clear, but one aspect such as the exact start/end boundary or item count requires judgment.

### `low`

The action is visible and more likely than not to be a pickup or putdown, but its type, count, or boundaries are materially uncertain.

Do not mix numeric confidence values with these labels in canonical annotation exports.

---

## 7. `hard_case`

Set `hard_case=true` for events that are difficult but still labelable, including:

- multiple people in the interaction area;
- partial obscuring;
- unusual or atypical motion;
- very short transitions;
- difficult item-count judgments;
- close consecutive events.

`hard_case` and `confidence` describe different properties:

- `hard_case` identifies a difficult example for later analysis;
- `confidence` records annotation certainty.

A hard case may still have `confidence=high` or `confidence=med`.

Fully unobservable actions are excluded and represented by internal ignore intervals rather than `hard_case` event rows.

---

## 8. Low confidence versus ignore

Use `confidence=low` when transfer evidence is visible but uncertain.

Use an internal ignore interval when the evidence needed to decide whether transfer occurred is unavailable.

```text
visible but uncertain
    → official event row with confidence=low

evidence unavailable
    → no official event row; internal ignore interval
```

Ignore intervals have zero training and evaluation weight and must never be sampled as background.

Suggested ignore reasons:

```text
ACTION_OCCLUDED
ACTION_OUT_OF_FRAME
CLIP_BOUNDARY
UNLABELABLE
CORRUPT_SECTION
```

---

## 9. Annotation procedure

For each person-containing clip:

1. Watch the complete active span once at normal speed.
2. Inspect generated proposals only as suggestions.
3. Rewatch each possible interaction frame by frame or at reduced speed.
4. Decide whether a persistent shelf/surface-to-hand or hand-to-shelf transfer occurred.
5. Set `t_start` at the onset of the final purposeful action causing transfer.
6. Set `t_end` when the resulting object state becomes stable.
7. Assign `pickup` or `putdown`.
8. Create separate rows for multiple items.
9. Create separate ordered rows for immediate pickup/putdown sequences.
10. Label all visible actors.
11. Assign `confidence` as `high`, `med`, or `low`.
12. Set `hard_case=true` when appropriate.
13. Add concise notes only when they help explain uncertainty or an unusual case.
14. Add an internal ignore interval when decisive transfer evidence is unavailable.
15. Mark the complete active span as reviewed in the annotation workflow.

Do not infer an event solely from a candidate, wrist trajectory, shelf proximity, or model suggestion.

---

## 10. Canonical event export

Export official labels to `manifest/events.csv` using exactly:

```text
event_id
clip_id
type
t_start
t_end
hard_case
annotator
confidence
notes
```

Allowed canonical values:

```text
type: pickup | putdown
confidence: high | med | low
hard_case: true | false
```

Requirements:

- `event_id` is stable and unique.
- `clip_id` references a valid clip manifest row.
- `t_start` and `t_end` use source-clip seconds.
- Every item transfer has its own event row.
- Fully occluded, out-of-frame, or unlabelable actions do not appear in `events.csv`.
- Interaction candidates never appear in `events.csv` unless a human verifies an actual event.

Richer internal annotation records may additionally contain:

```text
event_group_id
actor_id
item_index
review_status
```

These fields must not change the required canonical CSV schema.

---

## 11. Internal ignore export

Store ignored intervals separately, for example in `manifest/ignore_intervals.parquet`:

```text
ignore_id
clip_id
t_start
t_end
reason
annotator
notes
```

Rules:

- ignored intervals are not ground-truth events;
- ignored intervals are excluded from official event matching;
- ignored intervals receive zero training weight;
- ignored intervals must not be sampled as negatives;
- visible restocking is normally not an ignore interval.

---

## 12. Agreement and quality control

Before annotation scales, all annotators must label the same pilot clips and resolve disagreements.

Double-label at least the configured shared subset, with a target of 15% where feasible.

Compare:

- event existence;
- event type;
- `t_start`;
- `t_end`;
- item count or number of event rows;
- confidence;
- `hard_case` status;
- ignore decisions.

When annotators disagree systematically, clarify this document before continuing. Do not silently resolve inconsistent rules per annotator.

Check event previews against exported timestamps before freezing a dataset version.

---

## 13. Quick decision guide

| Observation | Annotation decision |
|---|---|
| Item leaves shelf and becomes held/carried | `pickup` |
| Previously held item is released and remains on surface | `putdown` |
| Touch, inspection, browsing, or reach with no transfer | No event; background/hard negative |
| Visible newly introduced restocking | No event; background/hard negative |
| Pickup immediately followed by return | Two rows: `pickup`, then `putdown` |
| Two items transferred together | Two official event rows |
| Multiple actors act simultaneously | Label every visible event |
| Visible but uncertain transfer | Event row with `confidence=low` |
| Partial obscuring but still labelable | Event row; usually `hard_case=true` |
| Decisive transfer fully occluded or out of frame | No event row; internal ignore interval |
| Pose/shelf candidate without verified transfer | No event |

---

## 14. Terminology

Use these terms consistently:

| Term | Meaning |
|---|---|
| Active span | Interval in which at least one person is visible |
| Interaction candidate | Broad actor/hand/region interval that may contain an interaction |
| Ground-truth event | Human-verified `pickup` or `putdown` interval |
| Event prediction | Model-generated claim that a pickup or putdown occurred |
| Ignore interval | Internal interval with unavailable decisive evidence and zero training/evaluation weight |

A candidate is never an event prediction or ground-truth event by itself.

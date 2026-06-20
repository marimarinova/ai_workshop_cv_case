# Pickup and Putdown Event Detection

## Concepts-Aligned Architecture and Small-Team Implementation Plan

**Status:** refined against the case operational definitions, edge-case rules, manifest schema, Layer 0–3 guidance, deliverables, privacy rules, and best practices.

**Scope:** batch processing of video files to detect `pickup` and `putdown` intervals.

---

# 1. Objective

Build a reproducible computer-vision system that accepts store video files and returns zero or more event predictions:

```text
clip_id
event_type: pickup | putdown
t_start
t_end
score
```

The system must detect:

- **what happened** — `pickup` or `putdown`;
- **when it happened** — an interval `[t_start, t_end]` measured in seconds from the beginning of the source clip.

The initial implementation deliberately excludes:

- product identification;
- customer identification;
- face recognition;
- inventory counting;
- theft detection;
- SAM-based segmentation;
- live streaming or RTSP input;
- distributed or agentic orchestration.

The supported runtime is batch inference on video files:

```bash
pickup-putdown infer --input videos/example.mp4
```

or:

```bash
pickup-putdown infer --input videos/
```

The final prediction export must follow the case `predictions.csv` schema.

---

# 2. Operational Event Definitions

These definitions are the system contract. They must remain identical across:

- labeling guidelines;
- annotation exports;
- Track A state-transition rules;
- Track B training targets and decoders;
- Qwen prompts;
- evaluation and error analysis.

## 2.1 Problem formulation

The task is **temporal action detection on untrimmed video**.

The system must find zero or more events in a source clip and output both:

```text
event type: pickup | putdown
event interval: [t_start, t_end]
```

A trimmed-window classifier is only an internal component. The complete system must still accept an untrimmed clip and return event intervals.

## 2.2 Pickup

A person removes an item from a shelf or surface and takes it into their hand or hands, so that the item leaves its resting place and becomes held or carried.

The defining transition is:

```text
shelf/surface → hand
```

## 2.3 Putdown

A person places an item that they were already holding onto a shelf or surface and releases it so that it remains resting there.

The defining transition is:

```text
hand → shelf/surface
```

A generic placement or restocking action is not a putdown unless the visible evidence establishes that the item is being returned after being held/taken in the relevant interaction context.

## 2.4 Non-events and hard negatives

Do not label the following as events:

- touching or inspecting an item without removing it;
- looking or reaching past an item;
- browsing, standing, or walking near shelves;
- hand movement near a shelf without a persistent object transfer;
- visible restocking or placement of newly introduced goods;
- empty or no-person clips.

Visible restocking is normally retained as a **background/hard-negative example**, not placed in an ignore interval.

## 2.5 Edge-case rules

- Taking two items simultaneously produces **two event rows**, one per item.
- Immediate pickup followed by return produces **two ordered events**: pickup, then putdown.
- Fully occluded or out-of-frame actions are excluded from `events.csv`.
- Multiple simultaneous actors are all labeled and processed separately.
- Visible but ambiguous or very brief actions are retained with `confidence=low`.
- Difficult but labelable cases use `hard_case=true`.
- Every event is stored as an interval `[t_start, t_end]`.

## 2.6 Event interval semantics

Candidate intervals and event intervals are different artifacts.

```text
candidate_start / candidate_end:
    broad interval in which an interaction may occur

event t_start:
    onset of the final purposeful action that results in transfer

event t_end for pickup:
    the item has left its resting place and is stably held/carried

event t_end for putdown:
    the item has been released and remains stably resting on the surface
```

A hand entering or leaving an expanded shelf region is a useful proposal signal, but it is not automatically the event boundary.

## 2.7 Low confidence versus ignore

Use `confidence=low` when the action is visible and is more likely than not to satisfy the definition, but its exact type, count, or boundaries are uncertain.

Use an internal ignore interval when the evidence required to decide whether transfer occurred is unavailable, for example because the hand/item is fully occluded or outside the frame.

```text
low confidence → official event row remains in events.csv
ignore interval → no official event row; zero training/evaluation weight
```

## 2.8 Temporal direction

Pickup and putdown are approximate time reversals. All feature extraction, frame sampling, training, decoding, and prompting must preserve chronological order.

Never shuffle frames or aggregate them as an unordered bag.

## 2.9 Terminology invariant

Use these terms consistently:

| Artifact | Meaning |
|---|---|
| Active span | At least one person is visible |
| Interaction candidate | A person/hand may interact with a shelf or surface |
| Event prediction | A model claims a pickup or putdown occurred |
| Ground-truth event | A human-verified pickup or putdown |

Stage A and Stage B outputs are not event predictions. A candidate must never be exported to `predictions.csv` until Track A, Track B, or Layer 2 classifies it as an event.

# 3. Final Architecture

```text
Read-only cloud bucket / local video files
                    │
                    ▼
Layer 0A — Inventory, person triage, and active spans
YOLO person detection + ByteTrack at low sampling rate
                    │
        ┌───────────┴────────────┐
        │                        │
   no person                person present
        │                        │
retain manifest row         derive active span(s)
exclude from event          and person tracklets
annotation/modeling              │
                                 ▼
Layer 0B — Actor-specific pose and interaction proposals
YOLO pose + fixed shelf/surface regions at higher sampling rate
                                 │
                                 ▼
Broad candidates + wrist/actor trajectories
Each candidate may contain zero, one, or multiple events
                                 │
                                 ▼
Human annotation of complete active spans
                                 │
                                 ▼
clips.csv + events.csv + internal ignore intervals
                                 │
             ┌───────────────────┴────────────────────┐
             │                                        │
             ▼                                        ▼
Layer 1 Track A                               Layer 1 Track B
Pose/hand-region baseline                     Learned actor-conditioned detector
repeated state transitions                    │
+ hand/shelf appearance                       ├─ Track B1: VideoMAE window classifier
             │                                 └─ Track B2: cached VideoMAE features + TCN
             ▼                                        │
zero/one/multiple events                              ▼
                                             zero/one/multiple events
             │                                        │
             └───────────────────┬────────────────────┘
                                 │
                      Shared non-VLM item count
                     1 item | 2+ items | uncertain
                                 │
                                 ▼
                 Expand multi-item actions into rows
                                 │
                                 ▼
                         Shared evaluation
                                 │
                                 ▼
Layer 2 — Standalone Qwen3.6-27B detector
Qwen scans active-span windows independently of Layer 1
                                 │
                                 ▼
Independent VLM event predictions
                                 │
                                 ▼
Layer 3 — Optional verification and fusion
Layer 1 proposes type/interval; Qwen verifies event, type,
item count, and visibility
                                 │
                                 ▼
Deterministic final predictions.csv
```

## 3.1 Architectural principles

1. Layer 0 creates trustworthy ground truth and person-active spans.
2. Layer 0B proposals accelerate work but do not define ground truth or predictions.
3. A candidate is a container and may contain zero, one, or multiple ordered events.
4. Track A is the first official non-VLM detector.
5. Track B1 is a learned fixed-window baseline.
6. Track B2 is the stronger feature-based temporal model.
7. Track B inputs are actor-conditioned so simultaneous people can produce independent events.
8. Layer 1 and Layer 2 are independently evaluable on the same held-out clips.
9. Qwen verification of Layer 1 predictions belongs to Layer 3.
10. Multi-item actions are exported as separate canonical prediction rows.
11. All systems use the same prediction schema and two-pass evaluator.
12. Streaming is deferred; inputs are encoded video files.

# 4. Repository, Storage, and Privacy

## 4.1 Repository separation

The case repository is read-only. All generated work belongs in the team's own repository and private storage.

| Artifact | Location |
|---|---|
| Raw footage | Provided read-only bucket |
| Cached working subset | Private local/cloud storage |
| Source code | Team solution repository |
| Small manifests and predictions | Solution repository or versioned private storage |
| Candidate clips and previews | Private storage, not Git |
| Features and model checkpoints | Private artifact storage, not Git |
| Credentials | Environment variables or ignored secret files |

## 4.2 Required `.gitignore`

```gitignore
.env
.env.*
data/
cache/
artifacts/features/
artifacts/checkpoints/
*.mp4
*.avi
*.mov
*.pt
*.pth
*.onnx
*.gguf
__pycache__/
.pytest_cache/
```

## 4.3 Privacy rules

- Do not identify individuals.
- Do not redistribute source clips.
- Keep footage in controlled working storage.
- Blur faces in reports and presentations.
- Use clip-local track identifiers such as `track_3`; these are not identities.

---

# 5. Canonical Schemas

The implementation may maintain richer internal Parquet tables, but it must export the exact case-compatible CSV schemas.

## 5.1 `clips.csv`

```text
clip_id
s3_key
duration_s
fps
width
height
n_person_tracks
usable
active_start_s
active_end_s
split
session_id
notes
```

Internal `clips.parquet` may additionally contain:

```text
source_uri
size_bytes
etag
capture_time
decode_status
has_person
triage_status
dataset_version
```

## 5.2 `active_spans.parquet`

The official schema supports one main active interval. Preserve multiple active spans internally:

```text
clip_id
active_span_id
t_start
t_end
n_person_tracks
```

Export the main or enclosing interval to `active_start_s` and `active_end_s`.

## 5.3 `events.csv`

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

Allowed values:

```text
type: pickup | putdown
confidence: high | med | low
```

Internal additions may include:

```text
event_group_id
actor_id
item_index
review_status
```

For two simultaneous items, export two official event rows with unique IDs and the same interval.

## 5.4 `ignore_intervals.parquet`

Internal only:

```text
ignore_id
clip_id
t_start
t_end
reason
annotator
notes
```

Suggested reasons:

```text
ACTION_OCCLUDED
ACTION_OUT_OF_FRAME
CLIP_BOUNDARY
UNLABELABLE
CORRUPT_SECTION
```

Rules:

- Excluded actions must not appear in `events.csv`.
- Ignore intervals must never be sampled as background.
- Visible but ambiguous actions are not ignore intervals; they remain event rows with `confidence=low`.
- Visible restocking is normally a hard negative, not an ignore interval.

## 5.5 `predictions.csv`

```text
pred_id
clip_id
type
t_start
t_end
score
model
```

Recommended model identifiers:

```text
layer1_track_a_pose_state_v1
layer1_track_b1_videomae_window_v1
layer1_track_b2_videomae_tcn_v1
layer2_qwen36_27b_standalone_v1
layer3_track_b_qwen_verifier_v1
```

Internal prediction records may additionally contain:

```text
event_group_id
candidate_id
actor_id
hand_side
region_id
item_count
item_count_score
boundary_method
source_prediction_id
```

For a two-item action, export two official prediction rows with unique `pred_id` values and the same type and interval. Retain `event_group_id` internally to link them.

# 6. Layer 0A — Inventory, Person Triage, and Active Spans

## 6.1 Purpose

Layer 0A answers:

```text
Is the file technically usable?
Does it contain a person?
When is any person visible?
How many stable person tracklets exist?
```

It does not detect pickup or putdown.

## 6.2 Inventory implementation

The inventory command must:

1. List video objects in the bucket.
2. Generate stable `clip_id` values.
3. Read duration, FPS, dimensions, codec, and size using `ffprobe`.
4. Record the exact object key as `s3_key`.
5. Detect likely duplicates using key, size, ETag, or checksum.
6. Record decoding failures.
7. Avoid downloading the entire source bucket.

```bash
pickup-putdown index \
  --source s3://bucket/prefix \
  --output manifest/clips.parquet
```

Support S3-compatible endpoints:

```yaml
storage:
  bucket_uri: s3://bucket/prefix
  region: null
  endpoint_url: null
  anonymous: false
```

## 6.3 Bounded cache

Download files only when selected for:

- triage;
- annotation;
- training;
- validation;
- testing.

Cache entries must be addressable by `clip_id`, source key, and ETag/version.

## 6.4 Direct video processing

Use a small pretrained YOLO person detector with ByteTrack.

The encoded video file can be passed directly to Ultralytics:

```python
from ultralytics import YOLO

model = YOLO(settings.person_model)

results = model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    stream=True,
    classes=[0],
    vid_stride=calculated_stride,
    verbose=False,
)
```

Ultralytics decodes the file internally. The application still iterates over per-frame results and stores:

```text
frame_index
timestamp
track_id
person_bbox
detection_confidence
```

Do not save all frames to disk.

## 6.5 Sampling rate

Stage A only needs coarse person presence, but it must not systematically miss short appearances.

Recommended starting point:

```yaml
triage:
  target_fps: 1.0
  minimum_visible_duration_s: 0.75
  minimum_observations: 2
  minimum_person_confidence: 0.35
```

Use 2 FPS for very short motion-triggered clips when one FPS misses brief person appearances.

Calculate:

```text
vid_stride = max(1, round(source_fps / target_fps))
```

Track duration must be calculated from source timestamps, not inferred only from the number of sampled frames.

## 6.6 Stable track rule

Mark `PERSON_PRESENT` when at least one track:

- spans at least the configured visible duration using source timestamps;
- has at least the configured number of confident observations;
- is not an isolated accidental detection.

When no stable track exists:

```text
n_person_tracks = 0
usable = false for event annotation/modeling
```

Keep the row in `clips.csv` for Stage A evaluation. Do not use no-person clips as Layer 1 action-classification negatives.

## 6.7 Active spans

Derive intervals where at least one person is visible.

Active spans are used to:

- focus annotation;
- avoid dead footage in Layer 1;
- create standalone Layer 2 windows;
- reduce repeated decoding.

## 6.8 Quality control

Review:

- 5–10% of `NO_PERSON` clips;
- every decode failure;
- low-confidence partial detections;
- a random sample from every session or day.

The main triage metric is **person-containing clip recall**.

## 6.9 Outputs

```text
manifest/clips.parquet
manifest/active_spans.parquet
tracks/person/<clip_id>.parquet
artifacts/triage_previews/<clip_id>.mp4  # sampled only
```

## 6.10 Exit criteria

- Inventory is reproducible.
- Decode status is recorded.
- No-person clips remain in the manifest.
- Active spans are generated.
- A sample of rejected clips is reviewed.
- No manual frame extraction is required.

---

# 7. Layer 0B — Pose and Interaction Proposal Generation

## 7.1 Purpose

Layer 0B answers:

```text
When is a tracked person likely interacting with a shelf or surface?
```

It produces:

- actor and wrist trajectories;
- broad candidate interaction intervals;
- actor/shelf context for Track A and Track B;
- high-recall suggestions for annotation.

It does **not** decide pickup or putdown.

A candidate may contain:

```text
zero events
one event
multiple ordered events
```

Examples include a touch-only negative, one pickup, or a pickup immediately followed by a putdown.

## 7.2 Fixed shelf and surface regions

Because the camera is fixed, define shelf and placement polygons once and version them.

```yaml
camera_id: store_camera_01
regions:
  - region_id: shelf_left
    type: shelf
    polygon: [[115, 90], [520, 85], [530, 620], [110, 625]]
  - region_id: center_table
    type: surface
    polygon: [[600, 420], [1120, 410], [1190, 770], [580, 780]]
interaction_margin_px: 60
```

Maintain exact regions and expanded interaction regions.

## 7.3 Direct pose-video processing

Pass the video path directly to YOLO pose tracking. The library decodes frames internally; the application stores timestamped structured results.

Store at least:

```text
frame_index
timestamp
actor_id
person_bbox
left_wrist_xy
right_wrist_xy
wrist_confidences
```

Save:

```text
tracks/pose/<clip_id>.parquet
```

## 7.4 Sampling rate

Start at approximately 8 FPS:

```yaml
proposals:
  target_fps: 8
```

Benchmark 2, 4, and 8 FPS on validation data. Measure proposal recall and temporal error, not only runtime.

## 7.5 Candidate signals

Required initial signals:

1. Wrist enters an expanded shelf region.
2. Wrist approaches within a configured shelf distance.
3. Wrist remains near the shelf for a minimum duration.
4. Person box overlaps the shelf interaction region.

Optional later signals:

- wrist-direction reversal;
- local shelf motion;
- object-like motion near the hand.

## 7.6 Candidate rule

Create a raw interaction when a confident wrist remains inside an expanded shelf region for at least the configured duration.

Then:

1. Merge intervals only for the same actor, hand, and shelf region.
2. Add context before and after.
3. Clamp to source duration.
4. Preserve raw and padded boundaries separately.
5. Cap excessively long windows.

```yaml
proposals:
  target_fps: 8
  minimum_wrist_confidence: 0.30
  minimum_interaction_duration_s: 0.25
  merge_gap_s: 1.0
  context_before_s: 2.0
  context_after_s: 2.0
  maximum_candidate_duration_s: 10.0
```

Candidate merging must not imply event merging. Later decoders may emit multiple events from one candidate.

## 7.7 Candidate schema

```text
candidate_id
clip_id
actor_id
hand_side
region_id
raw_start_s
raw_end_s
window_start_s
window_end_s
proposal_reason
proposal_score
review_status
```

## 7.8 Proposal recall

Annotators must review complete active spans, not only proposals.

```text
proposal_recall =
number of ground-truth events overlapped by at least one candidate
/
total number of ground-truth events
```

A reasonable initial target is at least 90% on the reviewed validation subset. Candidate precision may be low; recall is the priority.

## 7.9 Outputs

```text
manifest/candidates.parquet
tracks/pose/<clip_id>.parquet
artifacts/candidate_previews/<candidate_id>.mp4
```

No Stage B row is written to `predictions.csv`.

## 7.10 Exit criteria

- Shelf regions are version-controlled.
- Actor-specific wrist proposals are generated automatically.
- Candidate previews are inspectable.
- Full active spans remain human-reviewed.
- Proposal recall is measurable.
- Candidates may contain zero, one, or multiple events.
- Pose trajectories are available for Track A and actor-conditioned Track B.

# 8. Annotation Protocol

## 8.1 Annotation tool

Use one tool consistently:

- CVAT;
- Label Studio;
- VIA;
- ELAN.

A custom Streamlit tool is acceptable only when importing Stage B proposals into an existing tool is harder than implementing the required timeline workflow.

The selected tool must support:

- interval annotation;
- complete active-span review;
- proposal correction and deletion;
- multiple events inside one candidate;
- metadata fields;
- reproducible export.

## 8.2 Annotation procedure

For every selected person-containing clip:

1. Watch the complete active span.
2. Inspect Stage B proposals as suggestions only.
3. Rewatch possible events frame by frame.
4. Mark `t_start` at the onset of the final purposeful action that causes transfer.
5. Mark `t_end` when the resulting object state is stable:
   - pickup: item is stably held/carried;
   - putdown: item is released and stably resting.
6. Assign `pickup` or `putdown`.
7. Create separate rows for multiple items.
8. Create separate ordered rows for pickup followed by immediate putdown.
9. Label every visible actor event.
10. Set `confidence` to `high`, `med`, or `low`.
11. Set `hard_case=true` when appropriate.
12. Add an internal ignore interval only when transfer evidence is unavailable.
13. Mark the complete active span reviewed.

## 8.3 Restocking and negatives

Record before annotation scales:

```text
Restocking observed: yes/no
Staff/restocking scope decision: included as negatives / clips excluded by documented policy
```

Rules:

- Visible restocking is not a putdown.
- When retained in the dataset, visible restocking is background/hard negative.
- Do not create an ignore interval merely because an action is restocking.
- Use ignore only when visibility is insufficient to determine transfer.

## 8.4 Confidence and ignore decision

```text
confidence=low:
    transfer is visible and likely, but type/count/boundaries are uncertain

ignore interval:
    necessary transfer evidence is hidden, outside frame, or technically unusable
```

Recommended training weights:

```yaml
label_weights:
  high: 1.0
  med: 1.0
  low: 0.5
  ignore: 0.0
```

The primary official evaluation includes all event rows. Also report a high/med-only slice and a low-confidence slice.

## 8.5 Annotation budget

Agree on a target before labeling expands:

```yaml
annotation_budget:
  target_pickup_events: 100
  target_putdown_events: 100
  target_hard_negative_intervals: 200
  double_annotation_fraction: 0.15
```

Adjust after measuring actual event frequency.

## 8.6 Agreement check

Double-label at least 15% of selected clips.

Compare:

```text
event existence
event type
t_start
t_end
item count
confidence
hard-case status
```

Resolve disagreements before freezing the test set.

## 8.7 Event previews

Generate:

```text
2 seconds before t_start
+ event interval
+ 2 seconds after t_end
```

Save to:

```text
artifacts/event_previews/<event_id>.mp4
```

## 8.8 Split policy

Split by the strongest available grouping:

1. recording session;
2. contiguous customer sequence where safely inferable without identifying a person;
3. recording day;
4. whole clip as fallback.

Never split derived windows, candidates, frames, or feature sequences independently.

Freeze the test split before model or threshold tuning.

## 8.9 Outputs

```text
manifest/clips.csv
manifest/events.csv
manifest/ignore_intervals.parquet
manifest/labeling-guidelines.md
manifest/splits.json
artifacts/event_previews/
```

The copied labeling guidelines must remain synchronized with Section 2.

# 9. Layer 1 Track A — Pose and Hand/Shelf State Baseline

## 9.1 Purpose

Track A is the first official non-VLM detector.

It converts actor-specific interaction candidates into zero, one, or multiple ordered event predictions by combining:

- wrist trajectory;
- shelf interaction geometry;
- hand appearance before/after contact;
- shelf appearance before/after contact;
- deterministic state-transition logic.

## 9.2 Inputs

```text
candidate_id
clip_id
actor_id
hand_side
region_id
broad candidate interval
wrist trajectory
person trajectory
```

Track A processes each actor, hand, and shelf region independently.

## 9.3 Repeating temporal state machine

The state machine must be able to emit more than one event inside a candidate.

```text
OUTSIDE
  → APPROACHING
  → CONTACT
  → TRANSFER/STABILIZATION
  → WITHDRAWING
  → OUTSIDE
```

After an event is emitted, the state machine remains active for the rest of the candidate and can enter another contact/transfer cycle.

Example:

```text
CONTACT → shelf-to-hand transition → PICKUP
        → later CONTACT → hand-to-shelf transition → PUTDOWN
```

## 9.4 Candidate versus event boundaries

Retain both:

```text
candidate_start_s / candidate_end_s
event_start_s / event_end_s
```

Use wrist-region entry/exit only to delimit the broad interaction context.

Estimate event boundaries from the final transition:

```text
event t_start:
    onset of the final purposeful approach/contact sequence producing transfer

event t_end for pickup:
    item is stably separated from shelf and held/carried

event t_end for putdown:
    item is released and remains stably on the surface
```

When transition boundaries cannot be estimated reliably, use wrist entry/exit as a versioned fallback and record:

```text
boundary_method = WRIST_REGION_FALLBACK
```

## 9.5 Pre/contact/post sampling points

For each potential transition, sample:

```text
pre_contact
contact/transfer
post_transfer_stable
```

Sampling points must be based on timestamps and chronological evidence, not arbitrary frame order.

## 9.6 Hand and shelf crops

Extract actor/hand-specific hand crops and a local shelf patch around the estimated contact point:

```text
hand_before_crop
hand_after_crop
shelf_before_crop
shelf_after_crop
```

The shelf patch must be large enough to observe an item disappearing, appearing, or remaining unchanged.

## 9.7 Lightweight appearance features

Start with a frozen image encoder:

- DINOv2-small;
- SigLIP;
- CLIP;
- MobileNet feature extractor.

Cache:

```text
hand_before_embedding
hand_after_embedding
shelf_before_embedding
shelf_after_embedding
```

## 9.8 Lightweight state classifiers

### Hand-state classifier

```text
empty
carrying
uncertain
```

### Shelf-transition classifier

```text
object_removed
object_placed
no_meaningful_change
uncertain
```

Acceptable first models:

- logistic regression;
- small MLP;
- gradient-boosted trees.

## 9.9 Deterministic decision rules

### Pickup

```text
hand approaches shelf
AND transfer is visible
AND hand becomes carrying and/or shelf indicates object removal
AND the object state remains changed after contact
```

### Putdown

```text
hand approaches while carrying
AND transfer is visible
AND hand becomes empty and/or shelf indicates object placement
AND the item remains resting after release
```

### Background interaction

```text
hand enters/exits interaction region
BUT no persistent shelf-to-hand or hand-to-shelf transfer occurs
```

This includes touching, inspecting, reaching, browsing, and visible restocking.

## 9.10 Multi-event decoding

Track A returns a list, not a single label:

```text
[]
[pickup]
[putdown]
[pickup, putdown]
[other ordered combinations supported by evidence]
```

Merge events only when they have:

- the same actor;
- the same type;
- compatible overlapping/adjacent intervals;
- evidence that they are duplicate detections of one transfer.

Never merge adjacent pickup and putdown predictions merely because the gap is short.

## 9.11 Shared non-VLM item-count estimator

After an event is detected, estimate:

```text
1 item
2+ items
uncertain
```

Use an event-aligned hand/object crop and a lightweight classifier or deterministic two-hand logic.

- One object in each hand can naturally yield two linked events.
- Two objects in one hand requires the count estimator.
- `uncertain` remains one prediction internally flagged for review unless Layer 3 resolves it.

For `item_count=2`, export two canonical prediction rows with the same event interval and an internal shared `event_group_id`.

## 9.12 Confidence score

Combine normalized evidence:

```text
wrist trajectory confidence
hand-state transition confidence
shelf-transition confidence
boundary confidence
visibility quality
item-count confidence
```

Keep the formula deterministic and version-controlled.

## 9.13 Outputs

Canonical:

```text
results/layer1_track_a/predictions.csv
```

Internal diagnostics:

```text
candidate_id
actor_id
hand_side
region_id
event_group_id
item_count
item_count_score
boundary_method
hand_before_score
hand_after_score
shelf_transition_score
visibility_score
decision_rule
```

## 9.14 Track A exit criteria

- Track A emits canonical pickup/putdown predictions, not candidates.
- One candidate can yield zero, one, or multiple events.
- Immediate pickup/putdown can produce two ordered rows.
- Simultaneous actors are processed separately.
- Multi-item actions can be expanded into separate rows.
- Visible restocking and touch-only interactions are background.
- Event boundaries represent transfer/stabilization rather than the full candidate whenever possible.
- Thresholds are selected on validation data.
- Event-level metrics and failure previews are generated.

# 10. Layer 1 Track B1 — VideoMAE Fixed-Window Classifier

## 10.1 Purpose

Track B1 is the simplest learned video baseline. It classifies chronological actor-conditioned windows as:

```text
pickup
putdown
background
```

Sliding-window scores are decoded into zero or more event intervals for the original untrimmed clip.

## 10.2 Actor-conditioned inputs

Build one input stream per:

```text
clip_id + actor_id + region_id + candidate_id
```

Recommended spatial crop:

```text
union of actor bounding boxes across the window
+ active shelf/surface region
+ 15–20% margin
```

Keep an optional full-scene view for debugging, but use the actor-conditioned crop as the primary training input. This allows simultaneous people to produce independent events.

## 10.3 Training windows

Start with shorter, overlapping windows to separate close pickup/putdown transitions:

```yaml
track_b1:
  window_duration_s: 2.5
  window_stride_s: 0.5
  sampled_frames: 16
  labels:
    - background
    - pickup
    - putdown
```

### Positive windows

- pickup events;
- putdown events;
- immediate pickup/return sequences represented by separate windows where possible;
- hard but labelable events;
- low-confidence events with reduced sample/loss weight.

### Hard negatives

- touching without removal;
- reaching or browsing;
- standing near shelves;
- carrying an item near another shelf;
- visible restocking;
- Stage B candidates without events.

### Exclusions

- no-person clips;
- ignore intervals;
- corrupt sections;
- fully occluded or out-of-frame actions.

## 10.4 Window labeling when events are close

A single-label window cannot represent two types simultaneously. Use this deterministic rule:

1. Prefer windows whose temporal center lies inside exactly one event interval.
2. When multiple event intervals overlap the window, assign the event whose midpoint is closest to the window center.
3. Skip only cases whose event midpoints are effectively indistinguishable at the configured temporal resolution.
4. Preserve both original event rows for inference/evaluation.

Do not remove all immediate pickup/putdown examples from training.

## 10.5 Window manifest

```text
sample_id
clip_id
candidate_id
actor_id
region_id
window_start_s
window_end_s
label
event_id
label_confidence
sample_weight
split
```

## 10.6 Video loading

The loader accepts:

```text
video path
actor-conditioned crop track
window_start_s
window_end_s
```

It must:

1. seek to the requested interval;
2. decode only that interval;
3. reconstruct the actor/shelf crop;
4. sample frames uniformly in chronological order;
5. resize and normalize using the VideoMAE processor;
6. return tensor, label, and metadata.

Do not pre-extract the full dataset into JPEG images.

## 10.7 Training gates

### Gate A — visual inspection

Render at least 20 samples with chronological frames, crop box, label, timestamps, actor ID, and region ID.

### Gate B — tiny overfit

Overfit approximately 8–16 samples before full training.

### Gate C — baseline training

Start with:

- pretrained VideoMAE-Small;
- frozen or mostly frozen encoder;
- trained classification head;
- weighted loss/sampling using label confidence and class frequency;
- checkpoint selection by validation event F1, not frame/window accuracy.

## 10.8 Inference and multi-event decoding

For each actor-conditioned active span or candidate:

1. slide the configured window;
2. predict class probabilities;
3. smooth scores chronologically;
4. apply class-specific validation thresholds;
5. detect distinct class peaks;
6. merge only duplicate intervals of the same type and actor;
7. preserve adjacent different-type peaks as separate events;
8. derive coarse event intervals from accepted score regions;
9. run the shared non-VLM item-count estimator;
10. expand two-item actions into separate canonical rows.

## 10.9 Outputs

```text
results/layer1_track_b1/predictions.csv
```

## 10.10 Track B1 exit criteria

- Actor-conditioned crops and frame order are visually validated.
- Tiny overfit succeeds.
- One candidate can produce multiple ordered event predictions.
- Immediate pickup/putdown is not forcibly merged.
- Simultaneous actors can produce independent rows.
- Canonical predictions are produced and compared with Track A using the same evaluator.

# 11. Layer 1 Track B2 — Cached VideoMAE Features and TCN

## 11.1 Purpose

Track B2 separates expensive actor-conditioned feature extraction from cheap temporal detector training.

```text
actor-conditioned video sequence
→ cached VideoMAE features
→ temporal detector
→ zero/one/multiple event intervals
```

## 11.2 Feature extraction

For every actor/region active span or candidate:

1. create the actor+shelf union crop sequence;
2. divide it into overlapping chronological micro-clips;
3. pass each micro-clip through the frozen VideoMAE encoder;
4. save one embedding and representative timestamp per micro-clip;
5. reuse the features for all temporal-head experiments.

```yaml
track_b2:
  backbone: videomae-small
  input_fps: 8
  micro_clip_duration_s: 2.0
  micro_clip_stride_s: 0.5
  sampled_frames: 16
  freeze_backbone: true
```

Save:

```text
features/<clip_id>/<actor_id>/<candidate_or_span_id>.npz
```

Contents:

```text
clip_id
candidate_or_span_id
actor_id
region_id
timestamps: [T]
embeddings: [T, D]
```

## 11.3 Temporal labels

Per timestep:

```text
background
pickup
putdown
ignore
```

- Center inside pickup interval → `pickup`.
- Center inside putdown interval → `putdown`.
- Center inside ignore interval → `ignore`.
- Otherwise → `background`.

When event intervals are very close, retain the ordered sequence at the available temporal resolution rather than collapsing it into one label.

Use confidence weights:

```text
high/med = 1.0
low = 0.5
ignore = 0.0
```

## 11.4 Temporal head

Use a small TCN:

```text
VideoMAE sequence [T, D]
        │
Linear projection
        │
Conv1D residual block
        │
Dilated Conv1D residual block
        │
Dilated Conv1D residual block
        │
Classification head
        │
background / pickup / putdown per timestep
```

```yaml
temporal_head:
  hidden_size: 256
  convolution_blocks: 3
  kernel_size: 3
  dropout: 0.2
```

Use focal loss or weighted cross-entropy with ignore masking.

## 11.5 Interval decoding

1. Smooth temporal probabilities.
2. Apply class-specific validation thresholds.
3. Detect contiguous/peaked score regions per class.
4. Fill only short internal gaps for the same class.
5. Never merge adjacent intervals of different types.
6. Remove intervals below minimum duration.
7. Apply class-specific temporal NMS only to duplicates.
8. Estimate event boundaries around transfer/stabilization; retain the first/last accepted timestep as a coarse fallback.
9. Run the shared non-VLM item-count estimator.
10. Expand multi-item actions into separate canonical rows.

## 11.6 Outputs

```text
results/layer1_track_b2/predictions.csv
```

## 11.7 Track B2 exit criteria

- Actor-conditioned features are cached reproducibly.
- Sequence labels and chronological order are visually inspected.
- Tiny sequence overfit succeeds.
- Close pickup/putdown transitions can remain separate.
- Multiple actors are independently represented.
- Canonical intervals and multi-item rows are produced.
- Track B2 is retained only if it improves useful metrics or localization over simpler baselines.

# 12. Optional Stronger Track B — ActionFormer

ActionFormer is optional and must not block the main delivery.

Use it only after:

- Track A works;
- Track B1 works;
- cached VideoMAE features are validated;
- Track B2 TCN works;
- the evaluator and splits are frozen.

Input:

```text
timestamped VideoMAE feature sequence
```

Output:

```text
candidate temporal segments
class labels
confidence scores
```

Do not replace a working TCN with a partially integrated ActionFormer implementation merely to claim a more advanced model.

---

# 13. Hardware and Feature-Caching Strategy

## 13.1 Laptop/CPU work

Use laptops or CPU for:

- indexing and metadata;
- annotation conversion;
- shelf configuration;
- Track A state machine;
- lightweight state classifiers;
- TCN training on cached features;
- evaluation and reporting.

## 13.2 GPU/Colab work

Use GPU bursts for:

- YOLO pose extraction;
- VideoMAE feature extraction;
- Track B1 training;
- optional partial VideoMAE fine-tuning.

## 13.3 Cache once

Persist:

```text
tracks/person/
tracks/pose/
crops/hand/
crops/shelf/
features/image_encoder/
features/videomae/
```

Do not repeatedly decode and encode the complete dataset for every experiment.

## 13.4 FPS benchmark

Measure 2, 4, and 8 FPS on validation data.

For each rate report:

```text
Stage B proposal recall
Track A F1
Track B F1
temporal error
feature-extraction runtime
```

Start at 8 FPS for hand interaction and reduce only if recall and event quality remain acceptable.

---

# 14. Layer 2 — Standalone Qwen3.6-27B Detector

## 14.1 Purpose

Layer 2 independently detects events. It must not receive Layer 1 predictions.

```text
active span → overlapping chronological Qwen windows → independent event predictions
```

Each window may contain zero, one, or multiple events.

## 14.2 Model-size note

Qwen3.6-27B is larger than the case's intended small-VLM profile.

Record:

- exact model and quantization;
- runtime backend;
- GPU;
- peak VRAM;
- inference speed;
- reason for choosing the larger model.

## 14.3 Input windows

```yaml
layer2:
  window_duration_s: 8.0
  window_stride_s: 4.0
  target_fps: 4
```

Each window includes:

- chronological video frames or a short MP4;
- frame numbers or relative timestamps;
- clip ID;
- source-window start time.

Use active spans rather than Layer 1 proposals so the Layer 2 comparison remains independent.

## 14.4 Response schema

```json
{
  "events": [
    {
      "type": "pickup",
      "t_start": 2.4,
      "t_end": 3.7,
      "item_count": 1,
      "visible": true,
      "confidence": 0.82
    }
  ]
}
```

An empty list and multiple ordered events are both valid.

## 14.5 Prompt requirements

Include:

- exact pickup and putdown definitions;
- event-boundary semantics;
- touching/browsing negatives;
- visible restocking is not putdown;
- occluded/out-of-frame actions must be excluded;
- visible ambiguity may be returned with low confidence;
- two-item action means two event rows or `item_count=2` for expansion;
- immediate pickup/putdown means two ordered events;
- multiple actors must all be reported;
- temporal-order requirement;
- strict JSON schema;
- instruction that times are relative to the supplied window.

Use deterministic or low-temperature decoding. Do not request verbose reasoning.

## 14.6 Parsing and merging

- Validate with Pydantic.
- Retry invalid JSON once.
- Preserve raw responses.
- Count parse failures.
- Convert relative times to source times deterministically.
- Merge only duplicate same-type overlapping predictions from adjacent windows.
- Never merge a pickup and putdown solely because they are close.
- Expand `item_count=2` into two canonical rows.

## 14.7 Outputs

```text
results/layer2/predictions.csv
results/layer2/raw_responses.jsonl
results/layer2/run_metadata.json
```

## 14.8 Exit criteria

- Qwen runs independently of Layer 1.
- It scans active spans, not dead footage.
- Zero, one, or multiple events per window are supported.
- Canonical multi-item rows are produced.
- Duplicate merging is deterministic and type-aware.
- Parse failures and runtime are reported.

# 15. Layer 3 — Qwen Verification and Deterministic Fusion

## 15.1 Purpose

Layer 3 uses Qwen to verify individual Layer 1 event predictions.

Qwen verifies:

```text
event / no event
pickup / putdown
item count
visibility
```

Layer 1 remains the timing source in the first version. Multiple Layer 1 events inside one candidate are verified independently.

## 15.2 Verification input

For each Layer 1 event prediction:

```text
4 seconds before Layer 1 t_start
through
2 seconds after Layer 1 t_end
```

Clamp to source boundaries. Overlay relative timestamps or frame numbers. Include enough pre-context to distinguish a return from generic restocking.

## 15.3 Response schema

```json
{
  "event_visible": true,
  "event_present": true,
  "event_type": "pickup",
  "item_count": 1,
  "confidence": 0.91,
  "reason_code": "ITEM_LEAVES_SURFACE_WITH_HAND"
}
```

Allowed types:

```text
pickup
putdown
none
uncertain
```

Suggested reason codes:

```text
ITEM_LEAVES_SURFACE_WITH_HAND
ITEM_RELEASED_ON_SURFACE
TOUCH_ONLY
NO_OBJECT_TRANSFER
RESTOCKING_NOT_PUTDOWN
ACTION_OCCLUDED
AMBIGUOUS
```

## 15.4 Fusion rules

### Invisible

```text
event_visible = false → REJECTED_NOT_VISIBLE
```

### No event or visible restocking

```text
event_present = false → REJECTED_NO_EVENT
reason_code = RESTOCKING_NOT_PUTDOWN → REJECTED_NO_EVENT
```

### Type confirmed

```text
Layer 1 type = Qwen type → ACCEPTED
```

Use the Layer 1 interval and verified item count.

### Type changed

```text
Layer 1 type != Qwen type and event_present = true
→ ACCEPTED_TYPE_CHANGED
```

Retain the Layer 1 interval and record both types.

### Uncertain

```text
Qwen type = uncertain or confidence below threshold
→ NEEDS_REVIEW
```

### Multiple items

If `item_count=2`, create two final event rows sharing the same interval and internal event group.

## 15.5 Audit output

```text
results/layer3/qwen_verifications.jsonl
```

Preserve:

```text
prediction_id
clip_id
candidate_id
actor_id
layer1_model
layer1_type
layer1_start_s
layer1_end_s
qwen_event_visible
qwen_event_present
qwen_event_type
qwen_item_count
qwen_confidence
qwen_reason_code
raw_response
prompt_version
model_version
```

Never overwrite the original Layer 1 prediction.

# 16. Batch Inference

## 16.1 Inputs

```bash
pickup-putdown infer --input clip.mp4
```

```bash
pickup-putdown infer --input directory/
```

No RTSP, camera stream, Kafka, or live buffer is required.

## 16.2 Batch flow

For each video:

1. Read metadata.
2. Run Layer 0A person tracking.
3. Finish early if no person is detected.
4. Generate active spans.
5. Run Layer 0B pose and proposals.
6. Run Track A and/or selected Track B model.
7. Optionally run standalone Layer 2.
8. Optionally render Layer 1 verification clips.
9. Run Qwen verification.
10. Apply deterministic fusion.
11. Write canonical predictions and audit artifacts.

## 16.3 Required CLI commands

```bash
pickup-putdown index
pickup-putdown triage
pickup-putdown propose
pickup-putdown annotate
pickup-putdown validate-manifest
pickup-putdown build-track-a-dataset
pickup-putdown train-track-a-state
pickup-putdown infer-track-a
pickup-putdown build-track-b1-dataset
pickup-putdown train-track-b1
pickup-putdown extract-videomae-features
pickup-putdown train-track-b2
pickup-putdown infer-layer1
pickup-putdown infer-layer2
pickup-putdown verify-qwen
pickup-putdown fuse
pickup-putdown evaluate
pickup-putdown infer
```

Each command must:

- resolve a configuration file;
- log parameters;
- fail with non-zero exit status on errors;
- avoid silent overwrites;
- emit a machine-readable summary;
- record Git commit and dataset version.

## 16.4 Single-video output

```text
outputs/example/
├── metadata.json
├── tracks_person.parquet
├── tracks_pose.parquet
├── active_spans.parquet
├── candidates.parquet
├── predictions_track_a.csv
├── predictions_track_b1.csv
├── predictions_track_b2.csv
├── predictions_layer2.csv
├── qwen_verifications.jsonl
├── predictions_final.csv
└── previews/
```

---

# 17. Shared Evaluation

Use one evaluator for Track A, Track B1, Track B2, standalone Layer 2, and Layer 3.

## 17.1 Pass 1 — class-aware event matching

For every clip and event type:

1. Construct prediction/ground-truth pairs of the same type.
2. Calculate temporal IoU and midpoint distance.
3. Perform one-to-one maximum-weight matching.
4. Match only pairs meeting the selected criterion.
5. Count unmatched predictions as false positives.
6. Count unmatched ground truth as false negatives.

Use this pass for precision, recall, F1, tIoU metrics, and timing errors.

## 17.2 Pass 2 — class-agnostic temporal matching

To measure pickup/putdown reversal:

1. Match predictions and ground truth using temporal alignment without requiring the same type.
2. Compare the types after matching.
3. Count:

```text
GT pickup  + predicted pickup   → correct pickup
GT pickup  + predicted putdown  → pickup→putdown confusion
GT putdown + predicted pickup   → putdown→pickup confusion
GT putdown + predicted putdown  → correct putdown
```

This pass is reported separately and does not replace class-aware precision/recall matching.

## 17.3 Required metrics

```text
Precision / Recall / F1 at tIoU 0.3
Precision / Recall / F1 at tIoU 0.5
Precision / Recall / F1 at midpoint tolerance ±1 second
pickup → putdown confusion
putdown → pickup confusion
start-time MAE
end-time MAE
false positives per video hour
Stage B proposal recall
runtime per video minute
```

Optional:

```text
mAP at selected temporal IoU thresholds
```

## 17.4 Multi-item and event-count evaluation

Two ground-truth rows for a two-item pickup require two matched prediction rows to achieve full recall.

Also report:

```text
absolute event-count error per clip
multi-item event recall
```

Do not collapse duplicated item rows into one event before official matching.

## 17.5 Confidence and hard-case slices

Report where sample size permits:

```text
all official event rows
high/med-confidence events only
low-confidence events only
normal vs hard_case
single-person vs multiple-person
short vs long events
```

Ignore intervals never enter ground-truth matching.

## 17.6 Threshold discipline

Choose on validation data:

- Track A state thresholds;
- class thresholds;
- smoothing widths;
- same-type merge gaps;
- minimum event durations;
- class-specific temporal NMS settings;
- item-count threshold;
- Qwen confidence threshold.

Example:

```yaml
inference:
  pickup_threshold: 0.61
  putdown_threshold: 0.58
  same_type_merge_gap_s: 0.75
  minimum_event_duration_s: 0.30
```

Never tune on the test set.

# 18. Recommended Repository Layout

```text
pickup-putdown-solution/
├── README.md
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── configs/
│   ├── storage.yaml
│   ├── camera.yaml
│   ├── shelves.yaml
│   ├── triage.yaml
│   ├── proposals.yaml
│   ├── track_a.yaml
│   ├── track_b1.yaml
│   ├── track_b2.yaml
│   ├── layer2_qwen.yaml
│   ├── layer3_fusion.yaml
│   └── evaluation.yaml
├── src/pickup_putdown/
│   ├── ingestion/
│   │   ├── index_bucket.py
│   │   ├── video_probe.py
│   │   └── cache.py
│   ├── annotation/
│   │   ├── import_export.py
│   │   ├── schemas.py
│   │   ├── validation.py
│   │   └── agreement.py
│   ├── perception/
│   │   ├── person_tracker.py
│   │   ├── pose_tracker.py
│   │   ├── shelf_regions.py
│   │   └── proposals.py
│   ├── layer1/
│   │   ├── track_a/
│   │   │   ├── crop_extractor.py
│   │   │   ├── image_features.py
│   │   │   ├── hand_state.py
│   │   │   ├── shelf_state.py
│   │   │   ├── state_machine.py
│   │   │   └── inference.py
│   │   ├── track_b1/
│   │   │   ├── dataset.py
│   │   │   ├── videomae_classifier.py
│   │   │   ├── train.py
│   │   │   └── inference.py
│   │   ├── track_b2/
│   │   │   ├── feature_extractor.py
│   │   │   ├── temporal_head.py
│   │   │   ├── train.py
│   │   │   └── inference.py
│   │   └── common/
│   │       ├── actor_crops.py
│   │       ├── item_count.py
│   │       ├── decoding.py
│   │       └── schemas.py
│   ├── layer2/
│   │   ├── window_generator.py
│   │   ├── renderer.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── qwen_client.py
│   │   └── merge_predictions.py
│   ├── layer3/
│   │   ├── verifier.py
│   │   └── fusion.py
│   ├── evaluation/
│   │   ├── class_aware_matching.py
│   │   ├── confusion_matching.py
│   │   ├── metrics.py
│   │   ├── failure_gallery.py
│   │   └── report.py
│   └── cli.py
├── tests/
├── manifest/
│   ├── clips.csv
│   ├── events.csv
│   ├── labeling-guidelines.md
│   └── versions/
├── results/
├── notebooks/
└── docker/
```

---

# 19. Minimal Technology Stack

```text
Python 3.11
pyenv
uv or pip with locked dependencies
PyTorch
Hugging Face Transformers
Ultralytics YOLO person/pose
ByteTrack or BoT-SORT
FFmpeg / ffprobe
PyAV or decord
OpenCV
DINOv2, SigLIP, CLIP, or MobileNet for Track A features
scikit-learn or XGBoost for Track A state classifiers
Pandas or Polars
PyArrow / Parquet
Pydantic
Typer
CVAT, Label Studio, VIA, or ELAN
MLflow or structured local run directories
Docker Compose
```

Do not introduce initially:

- Kubernetes;
- Kafka;
- MCP;
- autonomous agents;
- feature stores;
- SAM;
- live-camera infrastructure.

---

# 20. Reproducibility Controls

Every run records:

```text
run_id
git_commit
dataset_version
split_version
config path
resolved config
random seed
model identifier
checkpoint hash
```

Example:

```json
{
  "run_id": "20260620_track_b2_v003",
  "git_commit": "abc1234",
  "dataset_version": "manifest_v3",
  "split_version": "split_v1",
  "config": "configs/track_b2.yaml",
  "seed": 42,
  "model": "videomae-small-tcn"
}
```

Rules:

- fix random seeds;
- version datasets immutably;
- never silently overwrite manifests;
- keep configuration outside source code;
- record exact code version;
- use the same evaluator for all systems;
- preserve raw predictions before fusion.

---

# 21. Small-Team Implementation Sequence

## 21.1 Team structure

### Person A — Data and annotation

Responsible for:

- bucket inventory;
- metadata and caching;
- annotation tooling;
- concept-aligned labeling protocol;
- agreement checks;
- splits and dataset versions.

### Person B — Standard CV and Layer 1

Responsible for:

- person tracking;
- pose inference;
- shelf regions;
- Stage B candidates;
- Track A;
- actor-conditioned Track B1 and Track B2;
- shared non-VLM item counting.

### Person C — VLM, evaluation, and integration

Responsible for:

- standalone Qwen Layer 2;
- Qwen verification;
- deterministic fusion;
- two-pass evaluator;
- CLI integration;
- reporting.

All members annotate the same pilot before independent annotation.

## Day 1 — Repository, inventory, and Stage A

### Person A

1. Create the repository and dependency lock.
2. Implement canonical/internal clip schemas.
3. Implement bucket listing and endpoint configuration.
4. Implement `ffprobe` metadata extraction.
5. Implement bounded cache.
6. Index an initial subset.
7. Record duplicates and decode failures.

### Person B

1. Pin Ultralytics and select a small person model.
2. Implement direct video-file person tracking.
3. Save timestamped person tracklets.
4. Derive active spans using timestamp-based duration.
5. Generate one preview with boxes and IDs.
6. Test 1 FPS and 2 FPS triage settings.

### Person C

1. Create Pydantic schemas.
2. Create run metadata and structured logging.
3. Implement canonical prediction export, including duplicate rows for multi-item actions.
4. Create class-aware and class-agnostic evaluator skeletons.
5. Prepare Qwen response schemas.

### Day 1 acceptance criteria

- Source videos can be indexed reproducibly.
- A video file can be triaged directly.
- Active spans and stable tracks are stored.
- Short person appearances are not rejected due to an inconsistent sampling rule.
- Empty clips remain in the manifest.
- Decode errors fail cleanly.

## Day 2 — Stage B and annotation workflow

### Person A

1. Configure one annotation tool.
2. Copy and adapt the labeling guidelines from Section 2.
3. Document visible restocking as a hard negative unless a pre-declared scope policy excludes staff clips.
4. Implement export to `events.csv`.
5. Implement ignore intervals and the low-confidence/ignore distinction.
6. Define annotation budget.
7. Implement manifest validation.

### Person B

1. Pin a YOLO pose model.
2. Define shelf/surface polygons.
3. Implement direct video pose tracking.
4. Implement actor-specific wrist-to-region interactions.
5. Merge only broad candidates for the same actor, hand, and region.
6. Add temporal context.
7. Save `candidates.parquet`.
8. Generate candidate previews showing that a candidate may contain multiple events.

### Person C

1. Validate timestamps and IDs.
2. Generate event previews.
3. Implement proposal-recall measurement.
4. Implement annotation agreement summaries.
5. Prepare multi-item and multiple-event evaluation data structures.

### Whole team

1. Annotate the same pilot clips.
2. Compare event existence, type, interval, count, confidence, and hard-case decisions.
3. Verify immediate pickup/putdown and two-item examples explicitly.
4. Refine the guideline.
5. Start complete active-span annotation.

### Day 2 acceptance criteria

- Shelf regions are version-controlled.
- Candidate intervals are generated but never exported as predictions.
- Annotators review complete active spans.
- Canonical event export works for multiple events and multiple items.
- Low-confidence and ignore handling are distinct.
- Proposal recall can be measured.

## Day 3 — Dataset freeze and Track A

### Person A

1. Continue annotation to the agreed budget.
2. Double-label at least 15%.
3. Resolve disagreements.
4. Assign session grouping.
5. Create train/validation/test splits.
6. Freeze the test split.
7. Export canonical manifests.

### Person B

1. Extract pre/contact/post frames for each candidate transition.
2. Extract actor-specific hand and shelf crops.
3. Visually validate crops on at least 30 examples.
4. Extract frozen image embeddings.
5. Build hand-state and shelf-transition training data.
6. Train lightweight state classifiers.
7. Implement the repeating state machine.
8. Implement transfer/stabilization boundary estimation and fallback metadata.
9. Implement the shared item-count estimator.
10. Export Track A predictions.

### Person C

1. Complete class-aware temporal matching.
2. Complete class-agnostic confusion matching.
3. Implement tIoU, midpoint tolerance, precision, recall, F1, and type confusion.
4. Implement multi-item count evaluation.
5. Generate Track A failure previews.
6. Prepare Qwen standalone window generation.

### Day 3 acceptance criteria

- Test split is frozen.
- Track A produces zero, one, or multiple canonical events per candidate.
- Immediate pickup/putdown remains two ordered events.
- Multi-item actions can produce two rows.
- Visible restocking and touch-only interactions are background.
- Event boundaries represent transfer/stabilization or are explicitly marked as fallback.
- Both evaluator passes work.

## Day 4 — Actor-conditioned Track B1 and standalone Layer 2

### Person A

1. Review Track A false positives and false negatives.
2. Correct labels only through documented dataset versions.
3. Add missing hard-negative annotations.
4. Do not inspect the test split for tuning.

### Person B

1. Generate actor-conditioned Track B1 windows.
2. Use center-based labels for close events rather than dropping all pickup/putdown sequences.
3. Exclude ignore intervals and no-person clips.
4. Render sampled-frame and crop debug views.
5. Run tiny overfit.
6. Train VideoMAE-Small classifier.
7. Implement type-aware multi-peak decoding.
8. Apply the shared item-count estimator.
9. Compare Track B1 with Track A on validation data.

### Person C

1. Implement active-span Qwen windows.
2. Overlay frame numbers/timestamps.
3. Implement Qwen3.6-27B client.
4. Implement concepts-aligned standalone prompt and Pydantic validation.
5. Retry invalid JSON once.
6. Merge only duplicate same-type window predictions.
7. Expand multi-item outputs.
8. Record runtime, quantization, hardware, and parse errors.

### Day 4 acceptance criteria

- Track B1 passes tiny overfit.
- Actor-conditioned Track B1 can separate simultaneous actors.
- Close pickup/putdown events are not forcibly merged.
- Qwen runs independently of Layer 1.
- Qwen supports zero, one, or multiple events per window.
- Qwen parsing failures are measured.

## Day 5 — Track B2, Layer 3, and final integration

### Person A

1. Review Layer 1/Qwen disagreements.
2. Categorize failure modes using the concepts taxonomy.
3. Confirm final manifest versions.
4. Prepare privacy-safe examples.

### Person B

1. Extract and cache actor-conditioned VideoMAE features.
2. Generate confidence-weighted timestep labels.
3. Implement the TCN.
4. Run tiny sequence overfit.
5. Train Track B2.
6. Decode type-aware temporal intervals.
7. Apply the shared item-count estimator.
8. Compare Track A, B1, and B2.
9. Package selected checkpoints and configs.

### Person C

1. Implement the Qwen verification prompt.
2. Implement deterministic fusion per Layer 1 prediction.
3. Implement single-file and directory inference.
4. Run final validation comparison.
5. Run the untouched test set once.
6. Produce both matching-pass metrics and failure gallery.
7. Document exact reproduction commands.

### Day 5 acceptance criteria

```bash
pickup-putdown infer \
  --input example.mp4 \
  --config configs/inference.yaml \
  --output outputs/example/
```

The final report includes:

```text
Track A metrics
Track B1 metrics
Track B2 metrics
standalone Layer 2 metrics
Layer 3 fusion metrics
pickup precision / recall / F1
putdown precision / recall / F1
pickup → putdown confusion
putdown → pickup confusion
tIoU and midpoint metrics
multi-item recall and event-count error
high/med and low-confidence slices
Qwen hardware and quantization
```

# 22. Mandatory Engineering Gates

## Gate 1 — Concept and dataset validity

Do not train until:

- pickup and putdown definitions are copied into the labeling guideline;
- event intervals represent transfer/stabilization, not full candidate duration;
- low confidence and ignore are distinct;
- visible restocking is handled as a negative unless excluded by a documented scope policy;
- videos decode correctly;
- timestamps and previews match labels;
- split leakage checks pass.

## Gate 2 — Proposal recall

Do not rely on Stage B filtering until recall is measured. Candidate precision may be low; recall must be high.

A candidate is never an event prediction.

## Gate 3 — Track A completeness

Track A is complete only when it independently outputs zero, one, or multiple pickup/putdown intervals per candidate.

It must support:

- immediate pickup followed by putdown;
- simultaneous actors;
- multi-item row expansion;
- touch/restocking negatives.

## Gate 4 — Tiny overfit and actor conditioning

Track B1 and Track B2 must pass tiny overfit tests, and debug views must prove that:

- frames are chronological;
- actor/shelf crops correspond to the intended actor;
- simultaneous actors are represented separately;
- ignore positions have zero loss.

## Gate 5 — Type-aware decoding

No decoder may merge pickup and putdown solely because they are temporally adjacent.

Only same-type duplicate detections may be merged or suppressed.

## Gate 6 — Independent Layer 2

Qwen verification is not Layer 2. Layer 2 requires independent scanning of active-span windows.

## Gate 7 — Two-pass evaluation

The evaluator must provide:

- class-aware matching for precision/recall/F1;
- class-agnostic temporal matching for pickup/putdown confusion;
- multi-item row-level evaluation.

## Gate 8 — Test isolation

All thresholds, merge rules, boundary settings, count thresholds, and prompt revisions are selected on validation data.

## Gate 9 — Auditability

Preserve:

- ground truth and ignore intervals;
- broad candidates;
- Track A predictions;
- Track B1 predictions;
- Track B2 predictions;
- standalone Layer 2 predictions;
- Qwen verification records;
- fusion decisions;
- prompts, configs, dataset versions, and model versions.

# 23. Priority Order if Time Is Limited

## Required

1. Trustworthy Layer 0 dataset.
2. Shared evaluator.
3. Complete Track A baseline.
4. At least one learned Track B baseline, preferably B1.
5. Standalone Qwen Layer 2.

## Next

6. Track B2 temporal head.
7. Layer 3 verification and fusion.
8. Optional ActionFormer.
9. Extended ablations.

If annotation falls behind, defer Track B2 before reducing label quality.

---

# 24. Deferred Work

Do not implement initially:

- SAM or product segmentation;
- product identity;
- live streaming;
- RTSP ingestion;
- causal online detection;
- multi-camera fusion;
- inventory state;
- agent orchestration;
- VLM fine-tuning;
- Kubernetes deployment.

---

# 25. Final Implementation Decision

## Layer 0A

```text
Video file
→ YOLO person detection + ByteTrack at 1–2 FPS
→ timestamped person tracklets and active spans
```

## Layer 0B

```text
Active-span video
→ actor-specific YOLO pose + fixed shelf polygons at ~8 FPS
→ broad wrist/shelf interaction candidates
```

A candidate is only a container and may contain zero, one, or multiple events.

## Layer 1 Track A

```text
Actor-specific pose candidate
+ hand/shelf pre/contact/post crops
+ lightweight frozen image features
+ repeating deterministic state machine
→ zero/one/multiple pickup or putdown intervals
```

Boundaries represent the transfer and stabilization where possible, with an explicit fallback method when only coarse wrist-region boundaries are available.

## Shared Layer 1 item count

```text
Detected event
→ event-aligned hand/object crop + lightweight count logic
→ 1 item | 2+ items | uncertain
→ separate canonical rows for multiple items
```

## Layer 1 Track B1

```text
Actor-conditioned chronological windows
→ VideoMAE classifier
→ type-aware temporal peaks
→ zero/one/multiple coarse event intervals
```

## Layer 1 Track B2

```text
Cached actor-conditioned VideoMAE features
→ small TCN
→ temporal class scores
→ type-aware refined intervals
```

## Layer 2

```text
Active-span overlapping windows
→ Qwen3.6-27B standalone detection
→ independent zero/one/multiple event intervals
```

## Layer 3

```text
Each selected Layer 1 event with context
→ Qwen verification
→ deterministic accept / reject / relabel / item-count resolution
→ final canonical predictions
```

## Evaluation

```text
Pass 1: class-aware matching → precision / recall / F1 / timing
Pass 2: class-agnostic temporal matching → pickup↔putdown confusion
Row-level multi-item matching → count correctness
```

## Runtime

```text
MP4/video files only
batch inference only
no streaming
no live-camera integration
```


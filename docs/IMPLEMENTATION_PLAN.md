# Pickup and Putdown Event Detection

## Clean Architecture and Small-Team Implementation Plan

**Status:** cross-referenced against the complete case repository

**Scope:** batch processing of video files for detection of `pickup` and `putdown` event intervals

---

## 1. Objective

Build a reproducible computer-vision system that accepts store video files and returns zero or more event predictions:

```text
clip_id
event_type: pickup | putdown
t_start
t_end
score
```

The system must detect both:

- **what happened** — `pickup` or `putdown`;
- **when it happened** — an interval `[t_start, t_end]` measured in seconds from the start of the source clip.

The project deliberately excludes:

- product identification;
- customer identification;
- inventory counting;
- theft detection;
- face recognition;
- live streaming in the first implementation;
- SAM-based segmentation in the first implementation.

The first implementation operates only on video files:

```bash
pickup-putdown infer --input videos/example.mp4
```

or:

```bash
pickup-putdown infer --input videos/
```

The required model output is exported as `predictions.csv` using the case schema.

---

## 2. Operational Event Definitions

These definitions must be used consistently in annotation, prompting, training, and evaluation.

### Pickup

A person removes an item from a shelf or surface and takes it into their hand or hands, so that the item leaves its resting place and becomes held or carried.

### Putdown

A person places an item that they were already holding onto a shelf or surface and releases it so that it remains resting there.

### Non-events

Do not label the following as events:

- touching or inspecting an item without removing it;
- looking or reaching past an item;
- walking, browsing, or standing near shelves;
- generic restocking of items that were not previously taken;
- empty clips or clips with no person.

### Fixed edge-case rules

- Two items taken together produce **two pickup rows**.
- Immediate pickup followed by return produces **one pickup and one putdown**.
- Fully occluded or out-of-frame actions are excluded from `events.csv`.
- Multiple simultaneous actors are all labeled.
- Brief or ambiguous but visible actions are retained with `confidence=low`.
- Difficult but labelable cases use `hard_case=true`.
- Every event is stored as an interval `[t_start, t_end]`.

---

## 3. Final Simplified Architecture

```text
Read-only cloud bucket / local input video files
                    │
                    ▼
Layer 0A — Inventory and person triage
YOLO person detection + ByteTrack at low sampling rate
                    │
        ┌───────────┴────────────┐
        │                        │
   no person                person present
        │                        │
retain manifest row         derive active span(s)
mark unusable for           and person tracklets
annotation/modeling              │
                                 ▼
Layer 0B — Interaction proposal generation
YOLO pose + fixed shelf/surface regions at higher sampling rate
                                 │
                                 ▼
High-recall candidate intervals for annotation and modeling
                                 │
                                 ▼
Human annotation of complete active spans
                                 │
                                 ▼
clips.csv + events.csv + internal ignore intervals
                                 │
                ┌────────────────┴────────────────┐
                │                                 │
                ▼                                 ▼
Layer 1A — Standard baseline              Layer 2 — Standalone VLM
VideoMAE fixed-window classifier          Qwen3.6-27B scans active-span
pickup / putdown / background             windows independently
                │                                 │
                ▼                                 ▼
Coarse event proposals                    Independent VLM events
                │                                 │
                ▼                                 │
Layer 1B — Temporal refinement                     │
VideoMAE embeddings + small Conv1D head            │
                │                                 │
                ▼                                 │
Refined event type and interval                     │
                └────────────────┬─────────────────┘
                                 │
                    Shared independent evaluation
                                 │
                                 ▼
Layer 3 — Optional fusion
Layer 1 proposes; Qwen verifies uncertain/proposed events
                                 │
                                 ▼
Deterministic fusion and final predictions.csv
```

### Architectural principles

1. **Layer 0 produces trustworthy labels and active spans.**
2. **Stage B is a high-recall proposal mechanism, not the ground truth.**
3. **Layer 1 and Layer 2 must be independently evaluable on the same test clips.**
4. **Qwen verification of Layer 1 predictions belongs to Layer 3 fusion.**
5. **All systems use the same prediction schema and evaluator.**
6. **The first implementation uses batch video files only; streaming is deferred.**

---

## 4. Repository, Storage, and Privacy Rules

### 4.1 Repository separation

The case repository is read-only.

All implementation artifacts must live in the team's own solution repository and private storage.

| Artifact | Location |
|---|---|
| Raw source footage | Provided read-only bucket |
| Cached working subset | Private local/cloud storage |
| Source code | Team solution repository |
| `clips.csv`, `events.csv` | Solution repository or private versioned storage |
| Extracted candidate clips | Private storage, not Git |
| Embeddings and checkpoints | Private artifact storage, not Git |
| Predictions, metrics, reports | Solution repository |
| Credentials | Environment variables or ignored secret files |

### 4.2 Required `.gitignore`

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

### 4.3 Privacy

- Do not identify or attempt to recognize individuals.
- Keep footage in controlled working storage.
- Do not redistribute source clips.
- Blur faces in published examples, reports, and presentations.
- Use clip-local actor IDs such as `track_3`; these are not identities.

---

## 5. Canonical Case Schemas

The implementation may maintain richer internal Parquet files, but it must export the exact case-compatible CSV schemas.

## 5.1 `clips.csv`

One row per source video file:

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

### Internal additions

The internal `clips.parquet` may also include:

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

### Active spans

The official schema supports one main active span. Internally, preserve multiple spans in:

```text
active_spans.parquet
```

with:

```text
clip_id
active_span_id
t_start
t_end
n_person_tracks
```

Export the main or enclosing span to `active_start_s` and `active_end_s`, and describe unusual additional spans in `notes`.

## 5.2 `events.csv`

One row per human-verified ground-truth event:

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

Allowed `type`:

```text
pickup
putdown
```

Allowed `confidence`:

```text
high
med
low
```

### Internal additions

The internal `events.parquet` may include:

```text
event_group_id
actor_id
item_index
review_status
```

For two objects picked up simultaneously, create two official event rows with the same interval and two unique `event_id` values.

## 5.3 Internal `ignore_intervals.parquet`

This is not part of the official case schema, but it prevents excluded actions from being sampled as background.

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

Occluded events are not included in `events.csv`; their time ranges are kept only in the internal ignore table.

## 5.4 `predictions.csv`

One row per predicted event:

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
layer1_videomae_window_v1
layer1_videomae_tcn_v1
layer2_qwen36_27b_standalone_v1
layer3_videomae_qwen_verifier_v1
```

Richer audit information must be preserved separately and must never replace the canonical prediction export.

---

# 6. Layer 0A — Inventory, Person Triage, and Active Spans

## 6.1 Purpose

Layer 0A answers:

```text
Is the video technically usable?
Does it contain a person?
When is a person present?
How many person tracklets are detected?
```

It does not attempt to detect pickup or putdown.

Its main outputs are:

- populated clip metadata;
- `n_person_tracks`;
- `usable`;
- active person span or spans;
- cached person tracklets.

## 6.2 Inventory

The inventory process must:

1. List all video objects in the bucket.
2. Generate a stable `clip_id` for each object.
3. Read duration, FPS, resolution, codec, and size.
4. Record the exact object key as `s3_key`.
5. detect duplicate objects using key, size, ETag, or checksum;
6. record decode failures;
7. avoid downloading the complete 80 GB dataset.

Suggested command:

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

## 6.3 Local cache policy

Download videos only when they are selected for:

- triage;
- annotation;
- training;
- validation;
- testing.

The cache must be bounded and reproducible. Cache entries should be addressable by `clip_id`, source key, and object version or ETag.

## 6.4 Person detection and tracking

Use a pretrained YOLO person detector with ByteTrack or BoT-SORT.

For the simplest implementation, pass the encoded video file directly to Ultralytics:

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

The library decodes the video internally. The application still iterates over one result object per processed frame and extracts:

```text
frame_index
timestamp
track_id
person_bbox
detection_confidence
```

Do not save every decoded frame.

Save structured tracks to:

```text
tracks/person/<clip_id>.parquet
```

## 6.5 Sampling rate

Layer 0A only needs coarse person presence.

Recommended initial rate:

```text
one processed frame every 1–2 seconds
```

Equivalent target FPS:

```yaml
triage:
  target_fps: 0.5
  minimum_track_duration_s: 0.75
  minimum_person_confidence: 0.35
```

Calculate video stride from source FPS:

```text
vid_stride = max(1, round(source_fps / target_fps))
```

Do not hard-code one stride for all videos.

## 6.6 Stable person-track rule

Mark a clip as containing a person when at least one tracklet:

- lasts for at least the configured minimum duration;
- contains multiple confident detections;
- is not a single-frame detection.

Suggested initial rule:

```text
track duration >= 0.75 s
and
at least 3 confident observations
```

## 6.7 Empty clips

When no stable person track is found:

```text
n_person_tracks = 0
usable = false
```

Keep the row in `clips.csv`. Do not delete it.

No-person clips are used to evaluate the triage stage and end-to-end compute savings. They are not used as Layer 1 event-classification negatives.

## 6.8 Active spans

For each kept clip, calculate when one or more people are visible.

Store:

```text
active_start_s
active_end_s
```

If there are separate bursts of person presence, store all of them internally in `active_spans.parquet`.

Active spans are used to:

- focus annotation;
- avoid processing dead footage in Layer 1;
- generate standalone Layer 2 windows;
- reduce repeated video decoding.

## 6.9 Triage quality control

Review:

- 5–10% of automatically rejected no-person clips;
- all clips with decode failures;
- clips with partial or low-confidence person detections;
- a random sample from each recording session or day.

The key triage metric is **person-containing clip recall**. False-positive retention is acceptable; false-negative removal may discard real events.

## 6.10 Layer 0A outputs

```text
manifest/clips.parquet
manifest/active_spans.parquet
tracks/person/<clip_id>.parquet
artifacts/triage_previews/<clip_id>.mp4  # optional sample only
```

## 6.11 Layer 0A exit criteria

Layer 0A is complete when:

- bucket inventory is reproducible;
- metadata and decode status are recorded;
- no-person clips remain in the manifest with `usable=false`;
- active person spans are generated;
- person-track recall is sampled and reviewed;
- no manual frame extraction is required.

---

# 7. Layer 0B — Interaction Proposal Generation

## 7.1 Purpose

Layer 0B answers:

```text
When is a tracked person likely interacting with a shelf or surface?
```

It does not decide whether the interaction is pickup, putdown, or a negative action.

It produces high-recall candidate intervals used to:

- prioritize human annotation;
- create training windows;
- reduce Layer 1 computation;
- optionally reduce the number of standalone VLM windows.

## 7.2 Fixed shelf and surface regions

Because the camera is fixed, define shelf and placement regions once.

Example `configs/shelves.yaml`:

```yaml
camera_id: store_camera_01

regions:
  - region_id: shelf_left
    type: shelf
    polygon:
      - [115, 90]
      - [520, 85]
      - [530, 620]
      - [110, 625]

  - region_id: center_table
    type: surface
    polygon:
      - [600, 420]
      - [1120, 410]
      - [1190, 770]
      - [580, 780]

interaction_margin_px: 60
```

Store shelf-region configuration in version control.

## 7.3 Pose inference

Use a pretrained YOLO pose model with ByteTrack.

The encoded video path can again be passed directly to Ultralytics:

```python
pose_model = YOLO(settings.pose_model)

results = pose_model.track(
    source=str(video_path),
    tracker="bytetrack.yaml",
    stream=True,
    classes=[0],
    vid_stride=calculated_stride,
    verbose=False,
)
```

The application extracts:

```text
frame_index
timestamp
track_id
person_bbox
left_wrist_xy
right_wrist_xy
keypoint_confidences
```

Save:

```text
tracks/pose/<clip_id>.parquet
```

## 7.4 Sampling rate

Fine hand interaction requires a higher sampling rate than Layer 0A.

Recommended initial setting:

```yaml
proposals:
  target_fps: 8
```

Three or four FPS may miss very brief interactions. Start around 8 FPS and tune using measured proposal recall.

## 7.5 Candidate signals

Initial high-recall signals:

1. Left or right wrist enters an expanded shelf region.
2. Wrist approaches within a configured distance of a shelf polygon.
3. Wrist remains near a shelf for a minimum duration.
4. Person box overlaps the shelf interaction region.

Optional later signals:

- wrist-direction reversal;
- local shelf-region motion;
- object-like motion near the hand.

## 7.6 Initial candidate rule

Create a raw interaction when:

```text
a confident wrist lies inside an expanded shelf region
for at least 0.25 seconds
```

Then:

1. Merge intervals from the same actor and shelf when the gap is below the merge threshold.
2. Add temporal context before and after.
3. Clamp intervals to clip duration.
4. Cap abnormally long candidate windows.
5. Preserve both raw and padded times.

Suggested configuration:

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

## 7.7 Candidate schema

Create `candidates.parquet`:

```text
candidate_id
clip_id
actor_id
region_id
raw_start_s
raw_end_s
window_start_s
window_end_s
proposal_reason
proposal_score
review_status
```

## 7.8 Critical annotation rule

Stage B proposals must never define the ground truth.

For every person-containing clip selected for the reference dataset:

1. Show proposed intervals as suggestions.
2. Require the annotator to review the full active span.
3. Allow manual events outside proposals.
4. Record which ground-truth events were covered by proposals.

Measure:

```text
proposal_recall =
number of ground-truth events overlapped by a candidate
/
total number of ground-truth events
```

Prioritize recall over precision.

A practical initial target is at least 90% recall on the reviewed validation subset. If recall is low, increase shelf margins, lower confidence thresholds, increase temporal padding, or process at a higher FPS.

## 7.9 Layer 0B outputs

```text
manifest/candidates.parquet
tracks/pose/<clip_id>.parquet
artifacts/candidate_previews/<candidate_id>.mp4
```

## 7.10 Layer 0B exit criteria

Layer 0B is complete when:

- shelf regions are version-controlled;
- wrist-based proposals are generated automatically;
- proposal previews can be inspected;
- complete active spans are still reviewed by humans;
- proposal recall is measurable;
- Stage B is not used as the sole source of ground truth.

---

# 8. Annotation Protocol

## 8.1 Annotation tool

Choose one real video annotation tool and use it consistently across the team:

- CVAT;
- Label Studio;
- VIA;
- ELAN.

A custom Streamlit tool is acceptable only when importing Stage B proposals into the selected annotation tool is more difficult than implementing the required timeline annotation workflow.

The selected tool must support:

- precise interval annotation;
- complete active-span review;
- editing and deleting proposals;
- pickup and putdown labels;
- confidence and hard-case metadata;
- reproducible export.

## 8.2 Annotation procedure

For each selected person-containing clip:

1. Watch the complete active span once.
2. Inspect Stage B candidate intervals.
3. Rewatch possible actions frame by frame.
4. Mark `t_start` when the physical transfer action begins.
5. Mark `t_end` when the object is carried away or settled.
6. Assign `pickup` or `putdown`.
7. Create separate rows for multiple items.
8. Set `confidence` to `high`, `med`, or `low`.
9. Set `hard_case=true` when appropriate.
10. Add an internal ignore interval for fully occluded or unlabelable actions.
11. Mark the clip as fully reviewed.

## 8.3 Restocking decision

Before large-scale annotation, inspect whether restocking occurs and record the project decision in the copied labeling guidelines:

```text
Restocking observed: yes/no
Restocking handling: hard negative / excluded interval
```

Generic restocking must not be labeled as putdown.

## 8.4 Annotation budget

Cap labeling effort before annotation scales.

Example initial target:

```yaml
annotation_budget:
  target_pickup_events: 100
  target_putdown_events: 100
  target_hard_negative_intervals: 200
  double_annotation_fraction: 0.15
```

Adjust the target after inspecting actual class frequency. The team should agree on the budget and a stopping rule.

## 8.5 Agreement check

At least 15% of selected clips should be independently annotated by two people.

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

Resolve disagreements and update the labeling guideline before freezing the test set.

## 8.6 Event previews

Generate a short preview for every event:

```text
2 seconds before t_start
event interval
2 seconds after t_end
```

Save to:

```text
artifacts/event_previews/<event_id>.mp4
```

This is the main label-quality audit artifact.

## 8.7 Dataset split

Split by the strongest available grouping:

1. recording session;
2. contiguous customer sequence where safely inferable;
3. recording day;
4. whole clip as the minimum fallback.

Never split derived windows independently.

Freeze the test split before threshold tuning or model comparison.

## 8.8 Layer 0 final outputs

```text
manifest/clips.csv
manifest/events.csv
manifest/labeling-guidelines.md
manifest/ignore_intervals.parquet
manifest/active_spans.parquet
manifest/candidates.parquet
manifest/split_version.json
```

---

# 9. Layer 1A — VideoMAE Fixed-Window Baseline

## 9.1 Purpose

Layer 1A is the simplest complete non-VLM detector.

It answers:

```text
Does this fixed window contain:
- pickup;
- putdown;
- background?
```

It uses the standard VideoMAE classification head and produces coarse event intervals from overlapping windows.

## 9.2 Training examples

Generate fixed windows only from person-containing active spans.

Suggested initial configuration:

```yaml
layer1a:
  window_duration_s: 4.0
  window_stride_s: 1.0
  sampled_frames: 16
  labels:
    - background
    - pickup
    - putdown
```

Positive samples:

- windows overlapping pickup events;
- windows overlapping putdown events.

Negative samples:

- touch without removal;
- reaching or browsing;
- standing near shelves;
- walking past;
- carrying an item without transfer;
- hand motion near shelves;
- Stage B candidate intervals with no event.

Do not use no-person clips as Layer 1 training negatives.

Do not use windows that overlap internal ignore intervals.

## 9.3 Window manifest

Create:

```text
sample_id
clip_id
window_start_s
window_end_s
label
event_id
actor_id
split
```

For the initial baseline, skip windows containing simultaneous incompatible event labels if the single-label classifier cannot represent them.

## 9.4 Video decoding

The dataset loader accepts:

```text
video path
window_start_s
window_end_s
```

It must:

1. seek to the required interval;
2. decode only that interval;
3. sample frames uniformly in chronological order;
4. resize and normalize using the VideoMAE processor;
5. return the frame tensor and label.

Do not pre-extract the complete dataset into JPEG files.

## 9.5 Training gates

### Gate A — visual data inspection

Render at least 20 examples showing sampled frames, label, source timestamps, and clip ID.

### Gate B — tiny overfit

Train on approximately 8–16 samples until the model nearly memorizes them.

Do not run a full experiment until this succeeds.

### Gate C — baseline training

Initially:

- freeze most or all of the VideoMAE encoder;
- train the classification head;
- use weighted sampling, focal loss, or weighted cross-entropy;
- select checkpoints using validation F1, not accuracy.

If stable, unfreeze the final encoder blocks.

## 9.6 Inference

For each Stage B candidate:

1. Slide a 4-second window over the candidate.
2. Use a 1-second stride.
3. Predict pickup, putdown, and background probabilities.
4. Smooth adjacent scores.
5. Merge adjacent windows with the same class.
6. Produce a coarse event interval.

Export canonical predictions as:

```text
results/layer1a/predictions.csv
```

## 9.7 Layer 1A exit criteria

- Frame order and sampling are visually validated.
- Tiny overfit succeeds.
- Inference produces canonical predictions.
- Precision, recall, F1, and pickup/putdown confusion are calculated.
- False-positive and false-negative previews can be generated.

---

# 10. Layer 1B — VideoMAE Embeddings and Temporal Head

## 10.1 Purpose

Layer 1B improves interval localization while retaining a simple implementation.

It converts overlapping VideoMAE micro-clips into a temporal embedding sequence and predicts pickup, putdown, or background at each timestep.

## 10.2 Feature extraction

For each active span or candidate interval:

1. Divide the interval into overlapping micro-clips.
2. Run each micro-clip through the VideoMAE encoder.
3. Save one embedding and representative timestamp per micro-clip.
4. Cache the embeddings so temporal-head training is fast.

Suggested settings:

```yaml
layer1b:
  micro_clip_duration_s: 2.0
  micro_clip_stride_s: 0.5
  sampled_frames: 16
```

Save:

```text
features/<clip_id>/<candidate_id>.npz
```

containing:

```text
timestamps: [T]
embeddings: [T, D]
actor_id
candidate_id
```

## 10.3 Temporal labels

Convert event intervals into per-timestep labels:

```text
background
pickup
putdown
ignore
```

- A timestep centered inside a pickup event receives `pickup`.
- A timestep centered inside a putdown event receives `putdown`.
- A timestep inside an internal ignore interval receives `ignore`.
- All other valid positions receive `background`.

Ignore positions do not contribute to the loss.

## 10.4 Temporal head

Use a small temporal convolutional network:

```text
VideoMAE embedding sequence
        │
Linear projection
        │
Conv1D + activation
        │
Dilated Conv1D
        │
Dilated Conv1D
        │
Classification head
        │
background / pickup / putdown per timestep
```

Suggested starting configuration:

```yaml
temporal_head:
  hidden_size: 256
  convolution_blocks: 3
  kernel_size: 3
  dropout: 0.2
```

Use focal loss or weighted cross-entropy.

Do not introduce transformers, multi-scale feature pyramids, or boundary-regression heads until the simple head has been evaluated.

## 10.5 Interval decoding

1. Smooth class probabilities with a short moving average.
2. Select timesteps above a validation-tuned threshold.
3. Combine adjacent timesteps of the same class.
4. Fill very short internal gaps.
5. Remove intervals shorter than the minimum duration.
6. Apply temporal non-maximum suppression where necessary.
7. Set `t_start` and `t_end` from the first and last accepted timesteps.

## 10.6 Layer 1B exit criteria

- Embeddings are reproducibly generated and cached.
- Sequence labels are visually inspected.
- Tiny sequence overfit succeeds.
- Canonical intervals are produced.
- Layer 1B is compared directly against Layer 1A.
- The more complex model is retained only if it improves useful metrics.

---

# 11. Layer 2 — Standalone Qwen3.6-27B Detector

## 11.1 Purpose

Layer 2 must be an independent VLM detector, not only a verifier.

It receives active person spans and produces event predictions without seeing Layer 1 predictions.

This creates a fair comparison:

```text
Layer 1 standard detector
versus
Layer 2 standalone VLM detector
```

## 11.2 Model-size note

The case describes a small VLM as the intended baseline. Qwen3.6-27B is larger than that profile.

The final report must explicitly record:

- model name;
- quantization;
- runtime backend;
- GPU hardware;
- peak VRAM;
- inference speed;
- reason for selecting the larger model.

Using Qwen3.6-27B is acceptable as a deliberate resource-enabled choice, but it should not be presented as a small-model baseline.

## 11.3 Input windows

Qwen scans active spans, not complete raw clips containing long dead periods.

Suggested initial standalone windowing:

```yaml
layer2:
  window_duration_s: 8.0
  window_stride_s: 4.0
  target_fps: 4
```

Each window should include:

- chronological frames or a short MP4;
- visible frame numbers or relative timestamps;
- source clip ID;
- source window start time.

The VLM must return zero or more events relative to the window. The application converts them into source-clip timestamps.

## 11.4 Standalone response schema

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

Allowed types:

```text
pickup
putdown
```

An empty event list is valid.

## 11.5 Prompt requirements

The standalone prompt must include:

- exact pickup definition;
- exact putdown definition;
- negatives;
- occlusion rule;
- two-item rule;
- immediate pickup/putdown rule;
- instruction to preserve temporal order;
- instruction not to infer hidden actions;
- strict JSON schema;
- instruction that times are relative to the supplied window.

Use low-temperature or deterministic decoding.

Do not request verbose chain-of-thought reasoning.

## 11.6 Parsing and validation

- Validate every response with Pydantic.
- Strip common JSON wrappers or code fences.
- Retry once on invalid JSON.
- Preserve the raw response.
- Count invalid, retried, and permanently unparseable outputs.

## 11.7 Duplicate merging

Overlapping VLM windows may predict the same event multiple times.

Merge predictions when:

- types match;
- temporal IoU or midpoint tolerance exceeds a validation-tuned threshold;
- predictions originate from overlapping source windows.

Use the highest confidence or a confidence-weighted merged interval.

## 11.8 Layer 2 output

Export:

```text
results/layer2/predictions.csv
results/layer2/raw_responses.jsonl
results/layer2/run_metadata.json
```

## 11.9 Layer 2 reporting

Record:

```text
model
quantization
backend
frame sampling rate
frames per window
window duration
window stride
prompt version
temperature
invalid JSON count
retry count
unparseable count
GPU
peak VRAM
seconds per video minute
```

## 11.10 Layer 2 exit criteria

- Qwen scans active spans independently of Layer 1.
- It produces canonical predictions.
- Duplicate merging is deterministic.
- Parsing failures are measured.
- Layer 2 is evaluated on the same held-out clips using the shared evaluator.

---

# 12. Layer 3 — Qwen Verification and Deterministic Fusion

## 12.1 Purpose

Layer 3 uses Qwen as a verifier of Layer 1 event proposals.

Qwen verifies:

```text
event / no event
pickup / putdown
item count
visibility
```

Qwen does not replace the Layer 1 interval in the first implementation.

## 12.2 Verification input

For every Layer 1 prediction:

1. Add context before and after the predicted interval.
2. Extract a short MP4.
3. Overlay relative timestamps or frame numbers.
4. Preserve chronological order.
5. Pass the full scene initially.

Suggested context:

```text
4 seconds before Layer 1 t_start
2 seconds after Layer 1 t_end
```

The larger pre-event context helps distinguish putdown from generic placement or restocking.

## 12.3 Verification response schema

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

Allowed `event_type`:

```text
pickup
putdown
none
uncertain
```

Suggested `reason_code` values:

```text
ITEM_LEAVES_SURFACE_WITH_HAND
ITEM_RELEASED_ON_SURFACE
TOUCH_ONLY
NO_OBJECT_TRANSFER
ACTION_OCCLUDED
MULTIPLE_ACTIONS
AMBIGUOUS
```

## 12.4 Audit output

Create:

```text
results/layer3/qwen_verifications.jsonl
```

Preserve:

```text
prediction_id
clip_id
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

## 12.5 Initial fusion rules

### Invisible action

```text
qwen_event_visible = false
```

Result:

```text
REJECTED_NOT_VISIBLE
```

### No event

```text
qwen_event_present = false
```

Result:

```text
REJECTED_NO_EVENT
```

### Type confirmed

```text
layer1_type = qwen_event_type
```

Result:

```text
ACCEPTED
```

Use the Layer 1 interval and Qwen item count.

### Type changed

```text
layer1_type != qwen_event_type
and qwen_event_present = true
```

Result:

```text
ACCEPTED_TYPE_CHANGED
```

Retain the Layer 1 interval, use the Qwen type, and preserve both values in the audit record.

### Uncertain

```text
qwen_event_type = uncertain
or qwen_confidence < threshold
```

Result:

```text
NEEDS_REVIEW
```

Exclude `NEEDS_REVIEW` from the fully automatic accepted export.

### Multiple items

When `item_count=2`, create two final rows sharing the same interval and an internal `event_group_id`.

## 12.6 Layer 3 evaluation

Evaluate Layer 3 separately from standalone Layer 2.

The final comparison contains:

| System | Role |
|---|---|
| VideoMAE Layer 1A | Standard baseline |
| VideoMAE Layer 1B | Standard temporal model |
| Qwen standalone | Layer 2 VLM detector |
| VideoMAE + Qwen verifier | Layer 3 fusion |

---

# 13. Batch Inference

## 13.1 Supported inputs

```bash
pickup-putdown infer --input clip.mp4
```

and:

```bash
pickup-putdown infer --input directory/
```

No camera stream, RTSP input, live buffer, Kafka, or streaming service is required.

## 13.2 Batch flow

For each input video:

1. Read metadata.
2. Run Layer 0A person detection and tracking.
3. Finish early if no person is detected.
4. Generate active spans.
5. Run Layer 0B pose and interaction proposals.
6. Run the selected Layer 1 model on proposals.
7. Optionally run standalone Layer 2 over active spans.
8. Optionally render Layer 1 verification clips.
9. Run Qwen verification.
10. Apply deterministic fusion.
11. Write canonical predictions and audit artifacts.

## 13.3 Required CLI commands

```bash
pickup-putdown index
pickup-putdown triage
pickup-putdown propose
pickup-putdown annotate
pickup-putdown validate-manifest
pickup-putdown build-dataset
pickup-putdown train-layer1a
pickup-putdown extract-features
pickup-putdown train-layer1b
pickup-putdown infer-layer1
pickup-putdown infer-layer2
pickup-putdown verify-qwen
pickup-putdown fuse
pickup-putdown evaluate
pickup-putdown infer
```

Each command must:

- use a configuration file;
- log resolved parameters;
- return a non-zero exit code on failure;
- avoid silent overwrites;
- produce a machine-readable summary;
- record the Git commit and dataset version.

## 13.4 Single-video output

```text
outputs/example/
├── metadata.json
├── tracks_person.parquet
├── tracks_pose.parquet
├── active_spans.parquet
├── candidates.parquet
├── predictions_layer1.csv
├── predictions_layer2.csv
├── qwen_verifications.jsonl
├── predictions_final.csv
└── previews/
```

---

# 14. Shared Evaluation

## 14.1 Matching

For each clip and class:

1. Construct valid prediction/ground-truth pairs.
2. Calculate temporal IoU.
3. Perform one-to-one matching.
4. Match only pairs meeting the selected criterion.
5. Count unmatched predictions as false positives.
6. Count unmatched ground truth as false negatives.

Use the same evaluation code for Layer 1, standalone Layer 2, and Layer 3.

## 14.2 Required metrics

```text
Precision / Recall / F1 at tIoU 0.3
Precision / Recall / F1 at tIoU 0.5
Precision / Recall / F1 at midpoint tolerance ±1 s
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

## 14.3 Stratified metrics

Report separately where sample size permits:

```text
pickup vs putdown
normal vs hard_case
high/med vs low confidence
single-person vs multiple-person
short vs long events
```

## 14.4 Test discipline

Use validation data for:

- probability thresholds;
- smoothing width;
- merge gaps;
- minimum duration;
- temporal NMS;
- VLM window settings;
- Qwen confidence threshold.

Do not tune on the test set.

---

# 15. Recommended Repository Layout

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
│   ├── layer1a.yaml
│   ├── layer1b.yaml
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
│   │   ├── video_dataset.py
│   │   ├── videomae_classifier.py
│   │   ├── feature_extractor.py
│   │   ├── temporal_head.py
│   │   ├── train_layer1a.py
│   │   ├── train_layer1b.py
│   │   └── inference.py
│   ├── layer2/
│   │   ├── window_generator.py
│   │   ├── candidate_renderer.py
│   │   ├── prompts.py
│   │   ├── schemas.py
│   │   ├── qwen_client.py
│   │   └── merge_predictions.py
│   ├── layer3/
│   │   ├── verifier.py
│   │   └── fusion.py
│   ├── evaluation/
│   │   ├── temporal_matching.py
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

# 16. Minimal Technology Stack

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
Pandas or Polars
PyArrow / Parquet
Pydantic
Typer
CVAT, Label Studio, VIA, or ELAN
MLflow or structured run directories
Docker Compose
```

Do not introduce in the initial implementation:

- Kubernetes;
- Kafka;
- MCP;
- autonomous agent frameworks;
- distributed orchestration;
- feature stores;
- SAM;
- live-camera infrastructure.

---

# 17. Reproducibility Controls

Every run must record:

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
  "run_id": "20260620_layer1b_v003",
  "git_commit": "abc1234",
  "dataset_version": "manifest_v3",
  "split_version": "split_v1",
  "config": "configs/layer1b.yaml",
  "seed": 42,
  "model": "videomae-small"
}
```

Rules:

- fix random seeds;
- version labeled datasets immutably;
- never overwrite manifests silently;
- keep configuration outside source code;
- record the exact code version for every result;
- use the same evaluator for all systems;
- preserve raw Layer 1 and VLM outputs before fusion.

---

# 18. Small-Team Implementation Sequence

## 18.1 Team structure

Recommended three-person team:

### Person A — Data and annotation

Responsible for:

- bucket inventory;
- metadata and caching;
- annotation-tool setup;
- labeling protocol;
- agreement checks;
- dataset splits and versions.

### Person B — Standard CV and VideoMAE

Responsible for:

- person tracking;
- pose inference;
- shelf regions;
- Stage B proposals;
- Layer 1A and Layer 1B.

### Person C — VLM, evaluation, and integration

Responsible for:

- standalone Qwen Layer 2;
- Qwen verification;
- fusion;
- shared evaluator;
- CLI integration;
- final reporting.

All team members should annotate a controlled shared subset before independent annotation begins.

---

## Day 1 — Repository, inventory, and person triage

### Person A

1. Create the solution repository.
2. Configure environment and dependency locking.
3. Create canonical and internal clip schemas.
4. Implement bucket listing and S3-compatible endpoint support.
5. Implement `ffprobe` metadata extraction.
6. Implement bounded local caching.
7. Index an initial source subset.
8. Record decode failures and duplicate candidates.

### Person B

1. Install and pin Ultralytics.
2. Select a small pretrained person model.
3. Implement direct video-file tracking with ByteTrack.
4. Calculate timestamped person tracklets.
5. Derive active span or spans.
6. Save track records to Parquet.
7. Generate at least one preview video with person boxes and track IDs.

### Person C

1. Create shared Pydantic schemas.
2. Create the run-metadata format.
3. Implement structured logging.
4. Create the shared evaluator skeleton.
5. Create the canonical prediction exporter.
6. Prepare Qwen response schemas.

### Day 1 acceptance criteria

- One command inventories source videos.
- One command triages a video file directly.
- `clips.parquet` contains technical metadata.
- Stable person tracklets are stored.
- Active spans are generated.
- Empty clips remain in the manifest with `usable=false`.
- Corrupt files fail cleanly.

---

## Day 2 — Stage B and annotation workflow

### Person A

1. Select and configure one annotation tool.
2. Copy and adapt the labeling guidelines.
3. Define restocking handling.
4. Implement annotation export to `events.csv`.
5. Implement internal ignore intervals.
6. Implement manifest validation.
7. Define the annotation budget.

### Person B

1. Select and pin a YOLO pose model.
2. Define fixed shelf and surface polygons.
3. Implement direct video-file pose tracking.
4. Implement wrist-to-region interaction detection.
5. Merge nearby intervals.
6. Add temporal context.
7. Save `candidates.parquet`.
8. Generate candidate preview clips.

### Person C

1. Validate event timestamps against clip duration.
2. Detect duplicate IDs.
3. Detect illegal events inside ignore intervals.
4. Generate event previews.
5. Implement proposal-recall measurement.
6. Implement agreement-summary utilities.

### Whole team

1. Read the event definitions together.
2. Independently annotate the same small pilot set.
3. Compare disagreements.
4. Refine the guideline.
5. Begin annotation of complete active spans.

### Day 2 acceptance criteria

- Shelf configuration is version-controlled.
- Candidate intervals are generated automatically.
- Annotators review complete active spans, not only proposals.
- Canonical event exports work.
- Ignore intervals work internally.
- Event previews and proposal recall are available.

---

## Day 3 — Dataset freeze and Layer 1A

### Person A

1. Continue annotation up to the agreed budget.
2. Double-label at least 15% of selected clips.
3. Resolve disagreements.
4. Assign `session_id` or the strongest available grouping.
5. Create train, validation, and test splits.
6. Freeze test split version 1.
7. Export canonical `clips.csv` and `events.csv`.

### Person B

1. Implement fixed-window generation.
2. Exclude no-person clips and ignore intervals.
3. Generate pickup, putdown, and hard-negative windows.
4. Implement the VideoMAE loader.
5. Render sampled-frame debug views.
6. Run a tiny overfit test.
7. Train the first Layer 1A model.
8. Implement coarse event decoding.

### Person C

1. Complete one-to-one temporal matching.
2. Implement tIoU.
3. Implement midpoint tolerance.
4. Implement precision, recall, and F1.
5. Implement class-confusion counts.
6. Generate the first evaluation report and failure previews.

### Day 3 acceptance criteria

- Test split is frozen.
- Split leakage checks pass.
- Ignore intervals are excluded from negative sampling.
- VideoMAE frames are in correct order.
- Tiny overfit succeeds.
- Layer 1A produces canonical event predictions.
- Event-level metrics are calculated.

If the tiny overfit test fails, stop model scaling and debug the data path.

---

## Day 4 — Layer 1B and standalone Layer 2

### Person A

1. Review Layer 1A false positives.
2. Add or correct hard-negative annotations where justified.
3. Review false negatives for missed labels.
4. Document any changes as a new dataset version.
5. Do not tune against the test split.

### Person B

1. Extract overlapping VideoMAE embeddings.
2. Cache timestamps and embedding arrays.
3. Generate per-timestep labels.
4. Implement the Conv1D temporal head.
5. Run a tiny sequence overfit test.
6. Train Layer 1B.
7. Decode temporal scores into intervals.
8. Compare Layer 1A and Layer 1B on validation data.

### Person C

1. Implement active-span VLM windows.
2. Render timestamps or frame numbers.
3. Implement the Qwen3.6-27B client.
4. Implement the standalone Layer 2 prompt.
5. Validate responses with Pydantic.
6. Retry invalid JSON once.
7. Merge duplicate events across overlapping windows.
8. Record runtime, quantization, hardware, and parse failures.

### Day 4 acceptance criteria

- Layer 1B produces temporal intervals.
- Layer 1A and Layer 1B are directly comparable.
- Qwen runs independently of Layer 1.
- Qwen scans only active spans, not full dead footage.
- Qwen outputs canonical event predictions.
- Invalid-response rate is measured.

---

## Day 5 — Layer 3, batch inference, and final evaluation

### Person A

1. Review Layer 1/Qwen disagreements.
2. Categorize common failure modes.
3. Confirm final manifest and labeling guideline versions.
4. Prepare privacy-safe example material.

### Person B

1. Finalize the selected Layer 1 model.
2. Confirm Stage B proposal recall.
3. Optimize repeated video decoding where necessary.
4. Package model checkpoints and configs.
5. Measure runtime per video minute.

### Person C

1. Implement the Qwen verifier prompt.
2. Implement deterministic fusion rules.
3. Generate final canonical predictions.
4. Implement single-file inference.
5. Implement directory batch inference.
6. Run the untouched test set.
7. Produce the final metrics table.
8. Produce the final failure gallery.
9. Document exact reproduction commands.

### Day 5 acceptance criteria

The following command works:

```bash
pickup-putdown infer \
  --input example.mp4 \
  --config configs/inference.yaml \
  --output outputs/example/
```

The final report includes:

```text
Layer 1A metrics
Layer 1B metrics
standalone Layer 2 metrics
Layer 3 fusion metrics
pickup precision / recall / F1
putdown precision / recall / F1
pickup-to-putdown confusion
putdown-to-pickup confusion
tIoU metrics
midpoint-tolerance metrics
false positives per video hour
Stage B proposal recall
runtime per video minute
Qwen invalid-response rate
Qwen hardware and quantization
```

---

# 19. Mandatory Engineering Gates

## Gate 1 — Dataset validity

Do not train until:

- videos decode correctly;
- timestamps are validated;
- event previews match labels;
- ignore intervals work;
- split leakage checks pass.

## Gate 2 — Proposal recall

Do not rely on Stage B filtering until proposal recall has been measured.

Candidate precision may be low. Candidate recall must be high.

## Gate 3 — Tiny overfit

Layer 1A and Layer 1B must pass tiny overfit tests.

Failure commonly indicates:

- wrong labels;
- wrong frame order;
- broken sampling;
- loss-mask errors;
- wrong tensor shapes;
- incorrectly frozen parameters;
- model/data mismatch.

## Gate 4 — Independent Layer 2

Do not call Qwen verification “Layer 2.”

Layer 2 is complete only when Qwen independently scans active-span windows and produces its own event predictions.

## Gate 5 — Test isolation

Use validation data for all thresholds and decoding settings.

Do not tune on the test set.

## Gate 6 — Auditability

Preserve:

- human ground truth;
- Layer 1 predictions;
- standalone Layer 2 predictions;
- Qwen verification outputs;
- fusion decisions;
- prompt versions;
- model versions;
- configs;
- source timestamps.

Never retain only the final accepted event list.

---

# 20. Priority Order if Time Is Limited

The case explicitly values depth over reaching every optional layer.

## Required priority

1. Trustworthy Layer 0 dataset.
2. Shared evaluator.
3. Layer 1A standard baseline.
4. Standalone Layer 2 Qwen detector.

## Next priority

5. Layer 1B temporal head.
6. Layer 3 Qwen verification and fusion.
7. Error taxonomy and ablations.

If annotation quality falls behind schedule, defer Layer 1B before compromising the dataset.

---

# 21. Deferred Work

Do not implement in the first delivery:

- SAM or object segmentation;
- product identity;
- live streaming;
- RTSP ingestion;
- causal online detection;
- multi-camera fusion;
- inventory state;
- agent orchestration;
- VLM fine-tuning;
- Kubernetes deployment.

These may be considered only after the batch pipeline is reproducible and evaluated.

---

# 22. Final Implementation Decision

## Layer 0A

```text
Video file
→ YOLO person detection + ByteTrack at low rate
→ person tracklets and active spans
→ canonical clips manifest
```

## Layer 0B

```text
Active-span video
→ YOLO pose + fixed shelf polygons at higher rate
→ high-recall interaction candidates
```

## Layer 1A

```text
Candidate windows
→ VideoMAE classifier
→ pickup / putdown / background
→ coarse event intervals
```

## Layer 1B

```text
Overlapping VideoMAE embeddings
→ small Conv1D temporal head
→ pickup / putdown scores over time
→ refined event intervals
```

## Layer 2

```text
Active-span overlapping windows
→ Qwen3.6-27B standalone detection
→ independent event intervals
```

## Layer 3

```text
Layer 1 proposal with context
→ Qwen3.6-27B verification
→ deterministic accept / reject / relabel
→ final canonical predictions
```

## Runtime

```text
MP4/video files only
batch inference only
no streaming
no live-camera integration
```

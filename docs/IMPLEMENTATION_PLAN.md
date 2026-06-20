# Pickup and Putdown Event Detection

## Simplified Architecture and Small-Team Implementation Plan

## 1. Objective

Build a batch-processing system that accepts store video files and returns:

```text
clip_id
event_type: pickup | putdown
t_start
t_end
item_count
confidence
```

The system does not perform:

* product identification;
* customer identification;
* inventory counting;
* theft detection;
* live streaming;
* real-time inference;
* SAM-based segmentation.

The first implementation operates entirely on video files:

```bash
pickup-putdown infer --input videos/example.mp4
```

or:

```bash
pickup-putdown infer --input videos/
```

The output is:

```text
predictions.csv
```

---

# 2. Final Simplified Architecture

```text
Input video files
        │
        ▼
Layer 0A: Clip triage
Person detection + pose + tracking
        │
        ├── no person → exclude from event annotation/inference
        │
        └── person present
                    │
                    ▼
Layer 0B: Interaction proposal generation
Wrist tracks + fixed shelf regions
                    │
                    ▼
Candidate video intervals
                    │
                    ▼
Layer 1A: VideoMAE baseline
Fixed-window pickup / putdown / background classification
                    │
                    ▼
Layer 1B: VideoMAE temporal model
VideoMAE embeddings + lightweight temporal head
                    │
                    ▼
Proposed event type and interval
                    │
                    ▼
Layer 2: Qwen3.6-27B verifier
Event / no event
Pickup / putdown
Item count
Visibility
                    │
                    ▼
Layer 3: Deterministic fusion
Accept, reject, relabel or flag
                    │
                    ▼
Final predictions.csv
```

The architecture contains four main components:

1. Dataset creation and interaction proposals.
2. VideoMAE event detection.
3. Qwen verification.
4. Deterministic result fusion.

There is no agentic orchestration requirement.

---

# 3. Layer 0A: Dataset Architecture and Clip Triage

## 3.1 Purpose

Layer 0A answers:

```text
Does this video contain a visible person?
Is the video technically usable?
```

It does not detect pickup or putdown.

Its purpose is to:

* avoid manually watching empty clips;
* remove corrupt and unusable videos;
* index the source material;
* create a review queue;
* preserve metadata needed for reproducibility.

---

## 3.2 Canonical dataset files

Create the following tables.

### `clips.parquet`

One row per source video:

```text
clip_id
source_uri
source_filename
source_etag
duration_s
fps
width
height
capture_time
recording_group
decode_status
has_person
usable
triage_status
split
notes
```

Recommended values for `triage_status`:

```text
PENDING
NO_PERSON
PERSON_PRESENT
UNUSABLE
REVIEWED
```

### `events.parquet`

One row per labeled pickup or putdown:

```text
event_id
event_group_id
clip_id
actor_id
event_type
t_start
t_end
item_index
label_confidence
hard_case
annotator
review_status
notes
```

Allowed `label_confidence` values:

```text
high
medium
low
```

For two objects picked up simultaneously, create two rows:

```text
event_101 | group_12 | clip_7 | pickup | 4.20 | 5.10 | item_index=1
event_102 | group_12 | clip_7 | pickup | 4.20 | 5.10 | item_index=2
```

### `ignore_intervals.parquet`

Store intervals that must not be treated as either positive events or negative background:

```text
ignore_id
clip_id
t_start
t_end
reason
annotator
notes
```

Reasons include:

```text
ACTION_OCCLUDED
ACTION_OUT_OF_FRAME
CLIP_BOUNDARY
UNLABELABLE
CORRUPT_SECTION
```

This table is important because an excluded, occluded action must not accidentally become a background training example.

---

## 3.3 Automated metadata indexing

The indexing process must:

1. List available video objects in the cloud bucket.
2. Generate a stable `clip_id`.
3. Read metadata with `ffprobe`.
4. Record duration, FPS, resolution and codec.
5. Detect duplicate files using URI, ETag or checksum.
6. Mark decode failures.
7. Avoid downloading all source videos.

Command:

```bash
pickup-putdown index \
  --source s3://bucket/prefix \
  --output manifests/clips.parquet
```

The local cache should download videos only when they are:

* selected for triage;
* selected for annotation;
* selected for training;
* selected for evaluation.

---

## 3.4 YOLO pose and ByteTrack triage

Use one small pretrained YOLO pose model with ByteTrack.

At the application level, pass the video filename directly:

```python
from ultralytics import YOLO

model = YOLO(settings.pose_model)

results = model.track(
    source=video_path,
    tracker="bytetrack.yaml",
    stream=True,
    classes=[0],
    vid_stride=settings.triage_stride,
    verbose=False,
)
```

The library decodes the video internally.

Your code must iterate over the results and extract:

```text
frame_index
timestamp
track_id
person_bbox
detection_confidence
keypoints
keypoint_confidences
```

Do not save every decoded frame.

Save structured tracking results to:

```text
tracks/<clip_id>.parquet
```

Optionally save one preview video with:

* person boxes;
* track IDs;
* wrist keypoints;
* timestamps.

---

## 3.5 Triage frame rate

For an initial 25–30 FPS video, process approximately 3–5 FPS.

Configure this through `vid_stride`.

Example:

```yaml
triage:
  target_fps: 4
  minimum_track_duration_s: 0.75
  minimum_person_confidence: 0.35
```

The stride should be calculated from the actual source FPS:

```text
vid_stride = round(source_fps / target_fps)
```

Do not hard-code the same stride for every video.

---

## 3.6 Person-presence rule

Mark a clip `PERSON_PRESENT` when at least one person track:

* lasts longer than the configured minimum duration;
* contains enough confident detections;
* is not restricted to one accidental frame.

Suggested initial rule:

```text
stable track duration ≥ 0.75 seconds
and
at least 3 confident observations
```

Mark the clip `NO_PERSON` when no stable track exists.

Do not delete `NO_PERSON` records. Keep them in `clips.parquet` so triage performance can be evaluated later.

---

## 3.7 Human quality control for triage

Automatically rejected videos must still be sampled.

Review:

* 5–10% of `NO_PERSON` clips;
* all clips with decode errors;
* clips containing very low-confidence partial detections;
* a random sample from each recording day or session.

The main triage metric is:

```text
person-containing clip recall
```

False-positive triage is acceptable because a human can reject an unnecessary clip. False-negative triage is dangerous because it can remove real events.

---

# 4. Layer 0B: Interaction Proposal Generation

## 4.1 Purpose

Layer 0B answers:

```text
When is a tracked person likely interacting with a shelf or surface?
```

It does not decide whether the action is pickup or putdown.

Its output is a list of high-recall candidate intervals.

Layer 0B is used for:

* prioritizing annotation;
* generating training windows;
* reducing Layer 1 computation;
* reducing the number of clips sent to Qwen.

---

## 4.2 Configure shelf regions once

Because the camera is fixed, manually define shelf and surface polygons once.

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
```

For each exact region, generate an expanded interaction region:

```yaml
interaction_margin_px: 60
```

The exact polygon represents the shelf.

The expanded polygon represents the area in which hand interaction is plausible.

---

## 4.3 Candidate-generation signals

Use the following signals:

1. Left or right wrist enters an expanded shelf region.
2. Wrist approaches within a configured distance of a shelf polygon.
3. Wrist remains near the shelf for a minimum duration.
4. The person bounding box overlaps the shelf interaction region.
5. Wrist direction changes near the shelf.
6. Motion increases inside the shelf region.

For the first implementation, wrist proximity is sufficient. Local shelf motion and direction changes can be added after the basic version works.

---

## 4.4 Initial candidate rule

Create a raw interaction when:

```text
a confident wrist lies inside an expanded shelf region
for at least 0.25 seconds
```

Then:

1. Merge interactions from the same actor and shelf when the gap is less than one second.
2. Add two seconds before the interaction.
3. Add two seconds after the interaction.
4. Clamp the interval to the video duration.
5. Cap very long candidate windows at approximately 8–10 seconds.
6. Preserve the original unpadded interaction timestamps.

Suggested configuration:

```yaml
proposals:
  minimum_wrist_confidence: 0.30
  minimum_interaction_duration_s: 0.25
  merge_gap_s: 1.0
  context_before_s: 2.0
  context_after_s: 2.0
  maximum_candidate_duration_s: 10.0
```

These are starting values, not fixed truths.

---

## 4.5 Candidate table

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

Example:

```text
cand_001
clip_007
actor_3
shelf_left
12.8
14.1
10.8
16.1
WRIST_IN_REGION
0.82
PENDING
```

---

## 4.6 Critical annotation rule

Do not annotate only the proposed candidate windows.

Layer 0B may miss events.

For every person-containing clip selected for the ground-truth dataset:

1. Show candidate intervals as suggestions.
2. Require the annotator to review the complete video.
3. Allow the annotator to create events outside proposed intervals.
4. Record which true events were covered by proposals.

Measure:

```text
Stage B proposal recall =
ground-truth events covered by at least one candidate
/
all ground-truth events
```

Target high recall rather than high precision.

A practical initial target is at least 90% proposal recall on the reviewed subset. If recall is lower, widen the shelf margin, reduce wrist-confidence thresholds or increase temporal padding.

---

# 5. Layer 0 Annotation Protocol

## 5.1 Annotation interface

Use a small Streamlit application.

The interface should contain:

* video player;
* current timestamp;
* frame-step buttons;
* playback-speed controls;
* candidate interval markers;
* event list;
* add pickup button;
* add putdown button;
* add ignore interval button;
* confidence selector;
* hard-case checkbox;
* item-count selector;
* save and next-clip buttons.

Keyboard shortcuts are strongly recommended:

```text
P = add pickup
D = add putdown
I = add ignore interval
H = hard case
L = low confidence
Space = play/pause
Left/Right = frame step
```

---

## 5.2 Annotation procedure

For every `PERSON_PRESENT` clip in the selected dataset:

1. Watch the entire clip once at normal or increased speed.
2. Review every proposed interaction window.
3. Rewatch possible actions frame by frame.
4. Mark `t_start` when the physical action begins.
5. Mark `t_end` when the object is carried away or settled.
6. Assign `pickup` or `putdown`.
7. Record item count.
8. Set confidence.
9. Set `hard_case` where applicable.
10. Add ignore intervals for fully occluded or unlabelable actions.
11. Mark the whole clip `REVIEWED`.

---

## 5.3 Annotation rules

### Pickup

The item leaves its resting position and becomes held or carried.

### Putdown

An item already held by the person is released and remains resting on a shelf or surface.

### Negative examples

Do not label:

* touching without removal;
* looking;
* reaching past;
* browsing;
* standing;
* walking by;
* generic restocking;
* hand movement without a visible object transfer.

### Edge cases

* Two items at once: create two event rows.
* Immediate pickup and return: create pickup and putdown events.
* Fully occluded or out-of-frame action: add an ignore interval.
* Multiple actors: label every visible event.
* Ambiguous but visible: label with `confidence=low`.
* Difficult but labelable: set `hard_case=true`.

---

## 5.4 Annotation quality assurance

At least 15% of selected clips should be independently annotated by two people.

For disagreements, review:

```text
event existence
event type
start time
end time
item count
visibility
hard-case status
```

Resolve disagreements before freezing the test set.

Generate a preview clip for every annotated event:

```text
artifacts/event_previews/<event_id>.mp4
```

The preview should include approximately:

```text
2 seconds before t_start
event interval
2 seconds after t_end
```

---

## 5.5 Dataset split

Split by:

1. recording session, when available;
2. customer sequence, when reliably inferable without identification;
3. recording day;
4. complete source clip as the minimum fallback.

Never split extracted candidate windows independently.

Freeze the test split before model tuning.

---

# 6. Layer 1A: VideoMAE Baseline Without a Custom Temporal Head

## 6.1 Purpose

Layer 1A is the simplest complete standard-model baseline.

It answers:

```text
Does this fixed video window contain:
- pickup;
- putdown;
- background?
```

This stage uses the standard VideoMAE video-classification head.

It does not use a custom temporal sequence head.

---

## 6.2 Training examples

Generate fixed-length windows from reviewed videos.

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

Create windows from:

* all annotated events;
* Stage B candidate windows without events;
* touching and browsing negatives;
* random person-present background;
* a limited number of no-person negatives.

Do not generate training windows that overlap `ignore_intervals`.

---

## 6.3 Window labeling

For each fixed window:

* label `pickup` when it overlaps a pickup event sufficiently;
* label `putdown` when it overlaps a putdown event sufficiently;
* label `background` when it overlaps no event and no ignore interval;
* skip windows containing incompatible overlapping events during the initial baseline.

Store:

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

---

## 6.4 Video decoding

The dataset loader accepts:

```text
video path
window_start_s
window_end_s
```

It must:

1. seek to the required interval;
2. decode only that interval;
3. sample frames uniformly;
4. resize and normalize using the VideoMAE processor;
5. return a tensor and label.

Do not pre-extract all frames to disk.

Optional optimization:

* cache decoded fixed windows as compressed MP4 files;
* do not cache millions of JPEG frames.

---

## 6.5 Training progression

Run three checks in order.

### Check 1: Data-loader inspection

Render at least 20 training samples with:

* sampled frames;
* label;
* source timestamps.

Confirm temporal order is correct.

### Check 2: Tiny overfit test

Train on approximately 8–16 samples until the model nearly memorizes them.

Do not begin a full run until the tiny overfit test succeeds.

### Check 3: Baseline training

Initially:

* freeze the VideoMAE encoder;
* train only the classification head;
* use weighted sampling or weighted loss;
* select checkpoints using validation F1, not accuracy.

If the baseline is stable, unfreeze the final encoder blocks.

---

## 6.6 Baseline inference

For each Stage B candidate:

1. Slide a four-second window over the candidate.
2. Use a one-second stride.
3. Obtain pickup, putdown and background probabilities.
4. Smooth adjacent scores.
5. Merge adjacent windows with the same predicted class.
6. Produce an approximate interval.

Output:

```text
predictions_layer1a.parquet
```

Fields:

```text
prediction_id
clip_id
actor_id
event_type
t_start
t_end
confidence
source_candidate_id
```

The interval from Layer 1A will be coarse. That is acceptable for the baseline.

---

## 6.7 Layer 1A exit criteria

Layer 1A is complete when:

* data loading has been visually validated;
* the model passes the tiny overfit test;
* inference produces the required prediction schema;
* event-level precision, recall and F1 can be calculated;
* pickup/putdown confusion is reported;
* failure previews can be generated automatically.

---

# 7. Layer 1B: VideoMAE Encoder With a Temporal Head

## 7.1 Purpose

Layer 1B improves interval localization.

Instead of classifying one complete fixed window, it creates a temporal sequence of VideoMAE embeddings and predicts a class for each temporal step.

Use a simple temporal segmentation head rather than a complex research-grade temporal action detector.

---

## 7.2 Feature extraction

For every reviewed candidate or clip:

1. Divide the interval into overlapping micro-clips.
2. Run each micro-clip through the VideoMAE encoder.
3. Save one embedding per temporal position.
4. Preserve the timestamp represented by each embedding.

Suggested initial configuration:

```yaml
layer1b:
  micro_clip_duration_s: 2.0
  micro_clip_stride_s: 0.5
  sampled_frames: 16
```

Output:

```text
features/<clip_id>/<candidate_id>.npz
```

Containing:

```text
timestamps: [T]
embeddings: [T, D]
actor_id
candidate_id
```

Cache these embeddings so the temporal head can be trained quickly.

---

## 7.3 Temporal labels

Convert each event interval into per-timestep labels:

```text
background
pickup
putdown
ignore
```

A timestep receives:

* `pickup` when its center falls inside a pickup interval;
* `putdown` when its center falls inside a putdown interval;
* `ignore` when it falls inside an ignore interval;
* `background` otherwise.

Ignore positions must not contribute to the loss.

---

## 7.4 Temporal head

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

A suitable first version:

```yaml
temporal_head:
  hidden_size: 256
  convolution_blocks: 3
  kernel_size: 3
  dropout: 0.2
```

Use focal loss or weighted cross-entropy.

Do not add boundary regression, transformers or multi-scale feature pyramids until the simpler temporal head has been evaluated.

---

## 7.5 Interval decoding

Convert timestep probabilities into events:

1. Smooth probabilities with a short moving average.
2. Select timesteps above the class threshold.
3. Combine adjacent timesteps of the same class.
4. Fill very short gaps.
5. Remove intervals shorter than the minimum event duration.
6. Apply temporal non-maximum suppression.
7. Use the first and last active timestep as `t_start` and `t_end`.

Output:

```text
predictions_layer1b.parquet
```

This becomes the input to the Qwen verifier.

---

## 7.6 Layer 1B exit criteria

Layer 1B is complete when:

* embeddings are reproducibly generated;
* the temporal head passes a tiny overfit test;
* predicted intervals are produced;
* results are compared against Layer 1A;
* tIoU and midpoint metrics are reported;
* Layer 1B either improves interval detection or is explicitly rejected in favour of Layer 1A.

Do not assume the more complex model is automatically better.

---

# 8. Layer 2: Qwen3.6-27B Verifier

## 8.1 Purpose

Qwen does not scan full raw videos.

Qwen receives only event proposals produced by Layer 1.

For each proposed interval, Qwen verifies:

```text
Is an event visibly present?
Is it pickup or putdown?
How many items are involved?
Is the action sufficiently visible?
```

Qwen does not train or fine-tune.

Qwen does not generate the primary interval.

The Layer 1 interval remains the final timing source.

---

## 8.2 Verifier input

For every Layer 1 prediction:

1. Add approximately two seconds of context before.
2. Add approximately two seconds of context after.
3. Extract a short MP4 section.
4. Overlay relative timestamps or frame numbers.
5. Preserve chronological order.
6. Keep the full scene for the initial implementation.

Example:

```text
Layer 1 interval: [14.2, 15.4]
Qwen clip: [12.2, 17.4]
```

Optional later improvement:

* send an actor-and-shelf crop when multiple-person scenes cause confusion.

Do not implement actor crops initially unless necessary.

---

## 8.3 Qwen response schema

Require strict JSON:

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

Allowed `event_type` values:

```text
pickup
putdown
none
uncertain
```

Allowed `reason_code` examples:

```text
ITEM_LEAVES_SURFACE_WITH_HAND
ITEM_RELEASED_ON_SURFACE
TOUCH_ONLY
NO_OBJECT_TRANSFER
ACTION_OCCLUDED
MULTIPLE_ACTIONS
AMBIGUOUS
```

Validate every response with Pydantic.

Retry once when:

* JSON is invalid;
* a required field is missing;
* a value is outside the allowed enumeration.

After a failed retry, mark the verification as:

```text
VERIFICATION_ERROR
```

---

## 8.4 Qwen prompt requirements

The prompt must include:

* exact pickup definition;
* exact putdown definition;
* negatives;
* occlusion rule;
* two-item rule;
* immediate pickup/putdown rule;
* instruction to use temporal direction;
* instruction not to infer hidden actions;
* strict JSON schema.

Use low-temperature or deterministic decoding.

Do not request lengthy chain-of-thought reasoning.

---

## 8.5 Verifier output

Create:

```text
qwen_verifications.jsonl
```

Each record should preserve:

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

---

# 9. Layer 3: Simplified Fusion

## 9.1 Purpose

Layer 3 combines Layer 1 and Qwen results using deterministic rules.

There is:

* no SAM;
* no additional learned fusion model;
* no streaming;
* no agent;
* no further training.

---

## 9.2 Fusion rules

Use the following initial rules.

### Rule 1: Invisible action

```text
qwen_event_visible = false
```

Result:

```text
reject prediction
```

Preserve it in the audit table with:

```text
final_status = REJECTED_NOT_VISIBLE
```

### Rule 2: No event

```text
qwen_event_present = false
```

Result:

```text
reject prediction
```

### Rule 3: Qwen confirms Layer 1 type

```text
layer1_type = qwen_event_type
```

Result:

```text
accept event
```

Use:

* Layer 1 interval;
* confirmed event type;
* Qwen item count;
* combined audit information.

### Rule 4: Qwen changes the event type

```text
layer1_type != qwen_event_type
and
qwen_event_present = true
```

Result:

```text
accept Qwen type
retain Layer 1 interval
mark type_changed = true
```

Preserve both types for evaluation.

### Rule 5: Qwen uncertain

```text
qwen_event_type = uncertain
or
qwen_confidence below threshold
```

Result:

```text
final_status = NEEDS_REVIEW
```

For a fully automatic demo, exclude `NEEDS_REVIEW` predictions from final accepted events.

### Rule 6: Multiple items

When:

```text
qwen_item_count = 2
```

Create two final event rows sharing the same:

```text
event_group_id
t_start
t_end
```

---

## 9.3 Final prediction table

Create `predictions_final.parquet` and `predictions_final.csv`:

```text
final_event_id
event_group_id
clip_id
actor_id
event_type
t_start
t_end
item_index
layer1_confidence
qwen_confidence
type_changed
final_status
source_prediction_id
```

Only rows with:

```text
final_status = ACCEPTED
```

are exported as official predictions.

---

# 10. Batch Inference

## 10.1 Input

Support:

```bash
pickup-putdown infer --input clip.mp4
```

and:

```bash
pickup-putdown infer --input directory/
```

No camera stream, RTSP input or live buffer is required.

---

## 10.2 Batch inference flow

For each input video:

1. Read video metadata.
2. Run YOLO pose and ByteTrack.
3. Reject or finish early if no person is present.
4. Generate Stage B interaction candidates.
5. Run Layer 1 on candidate intervals.
6. Convert scores into proposed events.
7. Render Qwen verification clips.
8. Call Qwen3.6-27B.
9. Validate Qwen JSON.
10. Apply deterministic fusion.
11. Write final results.

---

## 10.3 Required CLI commands

Implement the following commands:

```bash
pickup-putdown index
pickup-putdown triage
pickup-putdown propose
pickup-putdown annotate
pickup-putdown build-dataset
pickup-putdown train-layer1a
pickup-putdown extract-features
pickup-putdown train-layer1b
pickup-putdown infer-layer1
pickup-putdown verify-qwen
pickup-putdown fuse
pickup-putdown evaluate
pickup-putdown infer
```

Each command must:

* use a configuration file;
* log its parameters;
* fail with a non-zero exit code on errors;
* avoid silently overwriting outputs;
* produce a machine-readable summary.

---

# 11. Implementation Sequence for a Small Team

## 11.1 Team structure

Recommended three-person team:

### Person A — Data and annotation

Responsible for:

* bucket indexing;
* metadata;
* local cache;
* triage validation;
* annotation application;
* labeling quality;
* dataset splits.

### Person B — Standard CV and VideoMAE

Responsible for:

* YOLO pose and tracking;
* shelf regions;
* Stage B proposals;
* VideoMAE datasets;
* Layer 1A;
* Layer 1B.

### Person C — VLM, evaluation and integration

Responsible for:

* Qwen service;
* prompt and schema;
* Qwen clip rendering;
* fusion;
* metrics;
* command-line integration;
* final report.

All team members should annotate data for at least one fixed session each day.

---

## Day 1: Repository, ingestion and triage

### Person A

1. Create the repository.
2. Create `clips.parquet` schema.
3. Implement bucket listing.
4. Implement `ffprobe` metadata extraction.
5. Implement bounded local caching.
6. Index an initial subset of source videos.
7. Record decode failures.

### Person B

1. Install and pin the YOLO package.
2. Select one small pose checkpoint.
3. Implement direct video-file tracking.
4. Extract person boxes, track IDs and wrists.
5. Save tracking records to Parquet.
6. Generate one annotated preview video.

### Person C

1. Create the shared configuration structure.
2. Create Pydantic schemas for clips, events and predictions.
3. Implement logging and run metadata.
4. Create an empty evaluation command.
5. Prepare the Qwen response schema.

### Day 1 acceptance criteria

* One command indexes videos.
* One command triages a video.
* No manual frame extraction is required.
* Tracking data is saved as structured records.
* At least one preview video displays person IDs and wrists.
* Corrupt files fail cleanly.
* Configuration and outputs are reproducible.

---

## Day 2: Stage B and annotation system

### Person A

1. Build the Streamlit annotation interface.
2. Implement event creation and editing.
3. Implement ignore intervals.
4. Add confidence and hard-case fields.
5. Add item-count support.
6. Add complete-clip review status.
7. Test saving and reloading annotations.

### Person B

1. Create a shelf-polygon configuration tool or simple editor.
2. Define shelf regions for the camera.
3. Implement wrist-to-region interaction detection.
4. Merge nearby interaction intervals.
5. Add temporal context.
6. Save `candidates.parquet`.
7. Generate candidate preview clips.

### Person C

1. Implement dataset validators.
2. Check timestamps against video duration.
3. Detect duplicate IDs.
4. Detect events overlapping ignore intervals.
5. Generate event preview clips.
6. Implement proposal-recall measurement.

### Whole team

1. Read the labeling definitions together.
2. Label the same small set independently.
3. Compare disagreements.
4. Resolve ambiguous interpretation before scaling annotation.
5. Begin reviewing complete person-containing clips.

### Day 2 acceptance criteria

* Shelf regions are version-controlled.
* Candidate intervals can be generated automatically.
* Annotators see candidate suggestions.
* Annotators must still review complete clips.
* Events and ignore intervals are saved.
* Event previews can be generated.
* Proposal recall can be measured.

Do not proceed to serious training until complete-clip review works reliably.

---

## Day 3: Dataset freeze and Layer 1A

### Person A

1. Continue full-clip annotation.
2. Double-label at least 15% of selected clips.
3. Resolve annotation disagreements.
4. Assign recording groups.
5. Create train, validation and test splits.
6. Freeze the initial test split.

### Person B

1. Implement fixed-window generation.
2. Exclude ignore intervals.
3. Add pickup, putdown and background windows.
4. Implement the VideoMAE dataset loader.
5. Render sampled-frame debug views.
6. Run the tiny overfit test.
7. Train the first Layer 1A model.

### Person C

1. Implement temporal matching.
2. Implement midpoint tolerance.
3. Implement tIoU.
4. Implement precision, recall and F1.
5. Implement pickup/putdown confusion counts.
6. Generate a basic HTML or Markdown evaluation report.

### Day 3 acceptance criteria

* The test split is frozen.
* Training windows contain no test clips.
* Ignore intervals are excluded.
* VideoMAE sees frames in correct temporal order.
* The tiny dataset can be overfit.
* Layer 1A produces event predictions.
* Event-level metrics are calculated.

If the tiny overfit test fails, stop and debug before continuing.

---

## Day 4: Layer 1B and Qwen verifier

### Person A

1. Review false positives from Layer 1A.
2. Add missing hard negatives.
3. Review false negatives.
4. Correct annotation errors where justified.
5. Do not modify the frozen test labels based on model predictions without documented review.

### Person B

1. Extract overlapping VideoMAE embeddings.
2. Cache timestamps and embeddings.
3. Generate per-timestep labels.
4. Implement the Conv1D temporal head.
5. Run a tiny sequence overfit test.
6. Train Layer 1B.
7. Decode timestep predictions into intervals.
8. Compare Layer 1A and Layer 1B.

### Person C

1. Render short verification MP4s.
2. Add timestamp or frame overlays.
3. Implement the Qwen client.
4. Implement the strict prompt.
5. Validate output with Pydantic.
6. Retry invalid JSON once.
7. Save raw and parsed verification records.
8. Test Qwen on known positive and negative examples.

### Day 4 acceptance criteria

* Layer 1B produces temporal intervals.
* Layer 1A and Layer 1B are compared on identical data.
* Qwen receives only short proposed sections.
* Qwen does not scan entire raw videos.
* Qwen output is machine-validated.
* Raw model responses are preserved.
* No fine-tuning of Qwen is required.

---

## Day 5: Fusion, batch inference and final evaluation

### Person A

1. Review Layer 1/Qwen disagreements.
2. Categorize failure modes.
3. Confirm the final annotation manifest.
4. Prepare privacy-safe example clips.

### Person B

1. Optimize repeated video decoding where necessary.
2. Confirm Stage B proposal recall.
3. Finalize Layer 1 model selection.
4. Package checkpoints and configuration.
5. Measure inference runtime.

### Person C

1. Implement fusion rules.
2. Generate final predictions.
3. Implement the single-file inference command.
4. Implement directory batch inference.
5. Run the untouched test set.
6. Produce the final metrics report.
7. Produce the final failure gallery.
8. Document exact reproduction commands.

### Day 5 acceptance criteria

The following command must work:

```bash
pickup-putdown infer \
  --input example.mp4 \
  --config configs/inference.yaml \
  --output outputs/example/
```

It must produce:

```text
outputs/example/
├── metadata.json
├── tracks.parquet
├── candidates.parquet
├── predictions_layer1.parquet
├── qwen_verifications.jsonl
├── predictions_final.csv
└── previews/
```

The final report must include:

```text
Layer 1A metrics
Layer 1B metrics
Layer 1 + Qwen metrics
pickup precision / recall / F1
putdown precision / recall / F1
pickup-to-putdown confusion
putdown-to-pickup confusion
tIoU metrics
midpoint-tolerance metrics
false positives per video hour
proposal recall
runtime per video minute
```

---

# 12. Mandatory Engineering Gates

## Gate 1: Dataset validity

Do not train until:

* videos decode correctly;
* timestamps are validated;
* event previews match labels;
* ignore intervals work;
* split leakage checks pass.

## Gate 2: Proposal recall

Do not rely on Stage B filtering until proposal recall has been measured.

Candidate precision may be low.

Candidate recall must be high.

## Gate 3: Tiny overfit

Both Layer 1A and Layer 1B must pass a tiny overfit test.

Failure usually indicates:

* incorrect labels;
* incorrect frame order;
* incorrect sampling;
* broken loss masking;
* wrong tensor shape;
* frozen parameters;
* data/model mismatch.

## Gate 4: Independent evaluation

Do not tune thresholds on the test set.

Use validation data for:

* probability thresholds;
* smoothing;
* merge gaps;
* minimum duration;
* Qwen confidence threshold.

## Gate 5: Auditability

Preserve:

* Layer 1 prediction;
* Qwen verification;
* final fusion decision;
* prompt version;
* model version;
* configuration;
* source timestamps.

Never keep only the final accepted event.

---

# 13. Minimal Technology Stack

```text
Python 3.11
pyenv
PyTorch
Transformers
Ultralytics YOLO pose
ByteTrack through Ultralytics
FFmpeg / ffprobe
PyAV
OpenCV
Pandas or Polars
Parquet
Pydantic
Typer
Streamlit
MLflow or structured local run directories
Docker Compose
```

Use one Python repository.

Do not introduce:

* Kubernetes;
* Kafka;
* MCP;
* agent frameworks;
* distributed orchestration;
* feature stores;
* SAM;
* live-stream infrastructure.

---

# 14. Final Implementation Decision

## Layer 0A

```text
Video file
→ YOLO pose + ByteTrack
→ person-presence triage
→ tracks.parquet
```

## Layer 0B

```text
Person/wrist tracks
+ fixed shelf polygons
→ interaction candidate intervals
→ candidates.parquet
```

## Layer 1A

```text
Fixed candidate windows
→ VideoMAE video classifier
→ pickup / putdown / background
→ coarse intervals
```

## Layer 1B

```text
Overlapping VideoMAE embeddings
→ small Conv1D temporal head
→ pickup / putdown probabilities over time
→ refined intervals
```

## Layer 2

```text
Layer 1 proposed interval with context
→ Qwen3.6-27B
→ event, type, count and visibility verification
```

## Layer 3

```text
Layer 1 interval
+ Qwen verification
→ deterministic fusion
→ final event rows
```

## Runtime

```text
MP4 files only
batch inference only
no streaming
no live camera integration
```

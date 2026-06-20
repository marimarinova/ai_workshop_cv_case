# Pickup and Putdown Event Detection

## Updated Architecture and Small-Team Implementation Plan

**Status:** aligned with the case README, concepts, manifest schema, Layer 0, Layer 1, Layer 2, Layer 3, deliverables, and best-practices guidance.

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

These definitions must remain identical across:

- labeling guidelines;
- annotation;
- Track A rules;
- Track B training labels;
- Qwen prompts;
- evaluation.

## 2.1 Pickup

A person removes an item from a shelf or surface and takes it into their hand or hands, so that the item leaves its resting place and becomes held or carried.

## 2.2 Putdown

A person places an item that they were already holding onto a shelf or surface and releases it so that it remains resting there.

## 2.3 Non-events

Do not label:

- touching or inspecting without removal;
- looking or reaching past an item;
- browsing, standing, or walking near shelves;
- generic restocking of goods that were not previously taken;
- empty or no-person clips.

## 2.4 Edge-case rules

- Taking two items simultaneously produces **two pickup rows**.
- Immediate pickup followed by return produces **one pickup and one putdown**.
- Fully occluded or out-of-frame actions are excluded from `events.csv`.
- Multiple simultaneous actors are all labeled.
- Visible but ambiguous actions are retained with `confidence=low`.
- Difficult but labelable cases use `hard_case=true`.
- Every event is stored as an interval `[t_start, t_end]`.

## 2.5 Temporal direction

Pickup and putdown are approximate time reversals:

```text
pickup:  shelf → hand
putdown: hand → shelf
```

All feature extraction, classification, and prompting must preserve chronological frame order.

---

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
Layer 0B — Pose and shelf-interaction proposals
YOLO pose + fixed shelf/surface regions at higher sampling rate
                                 │
                                 ▼
High-recall interaction candidates + wrist/actor trajectories
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
Pose/hand-region baseline                     Learned temporal detector
wrist trajectory + hand/shelf state           │
transition + deterministic logic               ├─ Track B1: VideoMAE window classifier
             │                                 └─ Track B2: cached VideoMAE features + TCN
             ▼                                        │
Independent pickup/putdown events                     ▼
                                             Independent pickup/putdown events
             │                                        │
             └───────────────────┬────────────────────┘
                                 │
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

1. Layer 0 creates trustworthy labels and active spans.
2. Layer 0B proposals accelerate annotation but do not define ground truth.
3. Track A is the first official non-VLM baseline.
4. Track B1 is a learned fixed-window baseline.
5. Track B2 is the stronger feature-based temporal model.
6. Layer 1 and Layer 2 must be independently evaluable.
7. Qwen verification of Layer 1 predictions belongs to Layer 3.
8. All systems use the same prediction schema and evaluator.
9. Streaming is deferred; all inputs are encoded video files.

---

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

Excluded actions must not appear in `events.csv`, but their intervals must not be sampled as background.

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

---

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

Stage A only needs coarse person presence.

Recommended starting point:

```yaml
triage:
  target_fps: 0.5
  minimum_track_duration_s: 0.75
  minimum_person_confidence: 0.35
```

Calculate:

```text
vid_stride = max(1, round(source_fps / target_fps))
```

## 6.6 Stable track rule

Mark `PERSON_PRESENT` when at least one track:

- lasts at least 0.75 seconds;
- has at least three confident observations;
- is not a one-frame false detection.

When no stable track exists:

```text
n_person_tracks = 0
usable = false
```

Keep the row in `clips.csv` for triage evaluation.

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
- candidate interaction intervals;
- the signals consumed by Layer 1 Track A;
- high-recall windows for annotation and Track B.

It does not define ground truth.

## 7.2 Fixed shelf and surface regions

Define camera-specific polygons once:

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

Commit this configuration to Git.

## 7.3 Direct pose-video processing

Use YOLO pose with ByteTrack:

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

Extract:

```text
frame_index
timestamp
track_id
person_bbox
left_wrist_xy
right_wrist_xy
wrist_confidence
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

Benchmark 2, 4, and 8 FPS later. Short hand interactions may be missed at very low rates.

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

Create a raw interaction when:

```text
a confident wrist remains inside an expanded shelf region
for at least 0.25 seconds
```

Then:

1. Merge intervals from the same actor, hand, and shelf when the gap is below the threshold.
2. Add context before and after.
3. Clamp to video duration.
4. Preserve raw and padded intervals.
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

Measure:

```text
proposal_recall =
number of ground-truth events overlapped by a candidate
/
total number of ground-truth events
```

Target high recall. A reasonable initial target is at least 90% on the reviewed validation subset.

## 7.9 Outputs

```text
manifest/candidates.parquet
tracks/pose/<clip_id>.parquet
artifacts/candidate_previews/<candidate_id>.mp4
```

## 7.10 Exit criteria

- Shelf regions are version-controlled.
- Wrist proposals are generated automatically.
- Candidate previews are inspectable.
- Full active spans remain human-reviewed.
- Proposal recall is measurable.
- Pose trajectories are available for Track A.

---

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
- metadata fields;
- reproducible export.

## 8.2 Annotation procedure

For every selected person-containing clip:

1. Watch the complete active span.
2. Inspect Stage B proposals.
3. Rewatch possible events frame by frame.
4. Mark `t_start` when the physical transfer action begins.
5. Mark `t_end` when the object is carried away or settled.
6. Assign `pickup` or `putdown`.
7. Create separate rows for multiple items.
8. Set `confidence` to `high`, `med`, or `low`.
9. Set `hard_case=true` when appropriate.
10. Add internal ignore intervals for excluded actions.
11. Mark the clip fully reviewed.

## 8.3 Restocking decision

Before annotation scales, record:

```text
Restocking observed: yes/no
Restocking handling: hard negative / excluded interval
```

Generic restocking must not be labeled as putdown.

## 8.4 Annotation budget

Agree on a target before labeling expands:

```yaml
annotation_budget:
  target_pickup_events: 100
  target_putdown_events: 100
  target_hard_negative_intervals: 200
  double_annotation_fraction: 0.15
```

Adjust after measuring actual event frequency.

## 8.5 Agreement check

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

## 8.6 Event previews

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

## 8.7 Split policy

Split by the strongest available grouping:

1. recording session;
2. contiguous customer sequence where safely inferable;
3. recording day;
4. whole clip as fallback.

Never split derived windows independently.

Freeze the test split before tuning models or thresholds.

## 8.8 Outputs

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

# 9. Layer 1 Track A — Pose and Hand/Shelf State Baseline

## 9.1 Purpose

Track A is the first official non-VLM event detector.

It must independently output:

```text
pickup or putdown
start time
end time
confidence
```

Track A extends Layer 0B from a proposal generator into an interpretable event detector.

## 9.2 Inputs

For every interaction candidate:

```text
actor track
hand side
wrist trajectory
shelf region
raw and padded interval
source video
```

## 9.3 Temporal state machine

Operate per:

```text
actor_id + hand_side + region_id
```

State sequence:

```text
OUTSIDE
   │ wrist enters interaction region
   ▼
APPROACHING
   │ wrist reaches shelf/contact area
   ▼
CONTACT
   │ wrist begins moving away
   ▼
WITHDRAWING
   │ wrist exits interaction region
   ▼
STATE COMPARISON
   ├── shelf → hand = pickup
   ├── hand → shelf = putdown
   └── no persistent transition = background
```

## 9.4 Pre/post sampling points

For each interaction, select stable observations:

```text
before: shortly before final approach
contact: closest wrist/shelf interaction
post: after the hand exits and state stabilizes
```

Avoid frames where the hand is fully occluded.

## 9.5 Hand crops

Extract a crop centered on the wrist before and after interaction.

Store:

```text
hand_before_crop
hand_after_crop
```

The crop must include enough surrounding context to show an object being carried, not only a few hand pixels.

## 9.6 Shelf crops

Extract a local shelf patch around the contact point:

```text
shelf_before_crop
shelf_after_crop
```

The patch should be large enough to capture an item disappearing, appearing, or changing position.

## 9.7 Lightweight appearance features

Start with a frozen image encoder:

- DINOv2-small;
- SigLIP;
- CLIP;
- MobileNet feature extractor.

Extract:

```text
hand_before_embedding
hand_after_embedding
shelf_before_embedding
shelf_after_embedding
```

Cache these features.

## 9.8 Lightweight state classifiers

Train small classifiers, not a new large detector.

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

Use training labels derived from human-verified event intervals and hard negatives.

## 9.9 Deterministic decision rules

### Pickup

```text
hand approaches shelf
AND interaction occurs
AND hand becomes carrying and/or shelf indicates removal
AND hand exits with the new state
```

### Putdown

```text
hand approaches while carrying
AND interaction occurs
AND hand becomes empty and/or shelf indicates placement
AND the item remains after the hand exits
```

### Background interaction

```text
hand enters and exits shelf region
BUT no persistent hand/shelf state transition occurs
```

This must classify touching, reaching, and browsing as background.

## 9.10 Event interval

Initial approximation:

```text
t_start = first stable wrist entry into expanded shelf region
t_end   = stable wrist exit after the interaction
```

A later refinement may use the contact point and stabilization point.

## 9.11 Confidence score

Combine normalized evidence:

```text
wrist trajectory confidence
hand-state transition confidence
shelf-transition confidence
visibility quality
```

Keep the formula deterministic and version-controlled.

## 9.12 Outputs

Canonical:

```text
results/layer1_track_a/predictions.csv
```

Internal diagnostics:

```text
actor_id
hand_side
region_id
hand_before_score
hand_after_score
shelf_transition_score
visibility_score
decision_rule
```

## 9.13 Track A exit criteria

- Track A emits canonical pickup/putdown predictions.
- Simultaneous actors are processed separately.
- Hard negative interactions are classified.
- Pre/post crop extraction is visually validated.
- Decision rules are deterministic and auditable.
- Thresholds are selected on validation data.
- Event-level metrics and failure previews are generated.

---

# 10. Layer 1 Track B1 — VideoMAE Fixed-Window Classifier

## 10.1 Purpose

Track B1 is the simplest learned video baseline and corresponds to the guideline's lighter middle option.

It classifies a fixed chronological window as:

```text
pickup
putdown
background
```

It produces coarse event intervals by sliding, smoothing, and merging window predictions.

## 10.2 Training windows

Use only person-containing active spans.

```yaml
track_b1:
  window_duration_s: 4.0
  window_stride_s: 1.0
  sampled_frames: 16
  labels:
    - background
    - pickup
    - putdown
```

### Positive windows

- pickup events;
- putdown events;
- immediate pickup/return sequences where separable;
- hard but labelable events.

### Hard negatives

- touching without removal;
- reaching or browsing;
- standing near shelves;
- carrying an item near another shelf;
- hand movement near products;
- Stage B candidates without events.

### Exclusions

- no-person clips;
- ignore intervals;
- corrupt sections;
- fully occluded actions.

## 10.3 Window manifest

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

Skip windows with incompatible simultaneous labels if the initial single-label classifier cannot represent them.

## 10.4 Video loading

The loader accepts:

```text
video path
window_start_s
window_end_s
```

It must:

1. Seek to the requested interval.
2. Decode only that interval.
3. Sample frames uniformly in chronological order.
4. Resize and normalize using the VideoMAE processor.
5. Return the tensor and label.

Do not pre-extract the full dataset into JPEG images.

## 10.5 Training gates

### Gate A — visual inspection

Render at least 20 samples with frame sequence, label, timestamps, and clip ID.

### Gate B — tiny overfit

Overfit approximately 8–16 samples before full training.

### Gate C — baseline training

Start with:

- pretrained VideoMAE-Small;
- frozen or mostly frozen encoder;
- trained classification head;
- weighted loss or weighted sampling;
- checkpoint selection by validation F1, not accuracy.

Then optionally unfreeze final encoder blocks.

## 10.6 Inference

For each active span or Stage B candidate:

1. Slide a four-second window.
2. Use a one-second stride.
3. Predict class probabilities.
4. Smooth adjacent scores.
5. Apply validation-selected thresholds.
6. Merge adjacent windows of the same type.
7. Produce coarse intervals.

## 10.7 Outputs

```text
results/layer1_track_b1/predictions.csv
```

## 10.8 Track B1 exit criteria

- Frame order is visually validated.
- Tiny overfit succeeds.
- Canonical predictions are produced.
- Pickup/putdown confusion is reported.
- Track B1 is compared with Track A using the same evaluator.

---

# 11. Layer 1 Track B2 — Cached VideoMAE Features and TCN

## 11.1 Purpose

Track B2 is the stronger learned model.

It separates expensive feature extraction from cheap temporal detector training.

```text
video → cached VideoMAE features → temporal detector → event intervals
```

## 11.2 Feature extraction

For every active span or candidate:

1. Divide it into overlapping micro-clips.
2. Pass each micro-clip through the frozen VideoMAE encoder.
3. Save one embedding and representative timestamp per micro-clip.
4. Reuse the feature files for all temporal-head experiments.

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
features/<clip_id>/<span_or_candidate_id>.npz
```

Contents:

```text
timestamps: [T]
embeddings: [T, D]
actor_id
span_or_candidate_id
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

Ignore positions do not contribute to loss.

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

Use focal loss or weighted cross-entropy.

## 11.5 Interval decoding

1. Smooth temporal probabilities.
2. Apply class-specific validation thresholds.
3. Combine adjacent timesteps.
4. Fill short internal gaps.
5. Remove intervals below minimum duration.
6. Apply temporal non-maximum suppression where required.
7. Use first/last accepted positions as `t_start` and `t_end`.

## 11.6 Outputs

```text
results/layer1_track_b2/predictions.csv
```

## 11.7 Track B2 exit criteria

- Features are cached reproducibly.
- Sequence labels are visually inspected.
- Tiny sequence overfit succeeds.
- Canonical intervals are produced.
- Track B2 is compared with Track A and Track B1.
- Track B2 is retained only if it improves useful metrics or localization.

---

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

Layer 2 must independently detect events. It must not receive Layer 1 predictions.

```text
active span → overlapping Qwen windows → independent event predictions
```

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

An empty event list is valid.

## 14.5 Prompt requirements

Include:

- exact pickup and putdown definitions;
- non-events;
- occlusion rule;
- two-item rule;
- immediate pickup/putdown rule;
- temporal-order requirement;
- strict JSON schema;
- instruction that times are relative to the supplied window.

Use deterministic or low-temperature decoding. Do not request verbose reasoning.

## 14.6 Parsing and merging

- Validate with Pydantic.
- Retry invalid JSON once.
- Preserve raw responses.
- Count parse failures.
- Merge duplicate predictions from overlapping windows deterministically.

## 14.7 Outputs

```text
results/layer2/predictions.csv
results/layer2/raw_responses.jsonl
results/layer2/run_metadata.json
```

## 14.8 Exit criteria

- Qwen runs independently of Layer 1.
- It scans active spans, not dead footage.
- Canonical predictions are produced.
- Duplicate merging is deterministic.
- Parse failures and runtime are reported.

---

# 15. Layer 3 — Qwen Verification and Deterministic Fusion

## 15.1 Purpose

Layer 3 uses Qwen to verify Layer 1 proposals.

Qwen verifies:

```text
event / no event
pickup / putdown
item count
visibility
```

Layer 1 remains the timing source in the first version.

## 15.2 Verification input

For each Layer 1 prediction:

```text
4 seconds before Layer 1 t_start
through
2 seconds after Layer 1 t_end
```

Clamp to source boundaries. Overlay relative timestamps or frame numbers.

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
ACTION_OCCLUDED
MULTIPLE_ACTIONS
AMBIGUOUS
```

## 15.4 Fusion rules

### Invisible

```text
event_visible = false → REJECTED_NOT_VISIBLE
```

### No event

```text
event_present = false → REJECTED_NO_EVENT
```

### Type confirmed

```text
Layer 1 type = Qwen type → ACCEPTED
```

Use Layer 1 interval and Qwen item count.

### Type changed

```text
Layer 1 type != Qwen type and event_present = true
→ ACCEPTED_TYPE_CHANGED
```

Retain Layer 1 interval and record both types.

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

---

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

## 17.1 Matching

For every clip and class:

1. Construct prediction/ground-truth pairs.
2. Calculate temporal IoU.
3. Perform one-to-one matching.
4. Match only pairs meeting the criterion.
5. Count unmatched predictions as false positives.
6. Count unmatched truth as false negatives.

Use one evaluator for Track A, Track B1, Track B2, Layer 2, and Layer 3.

## 17.2 Required metrics

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

## 17.3 Stratified metrics

Where sample size permits:

```text
pickup vs putdown
normal vs hard_case
high/med vs low confidence
single-person vs multiple-person
short vs long events
```

## 17.4 Threshold discipline

Choose on validation data:

- Track A state thresholds;
- class thresholds;
- smoothing widths;
- merge gaps;
- minimum event durations;
- temporal NMS settings;
- Qwen confidence threshold.

Example:

```yaml
inference:
  pickup_threshold: 0.61
  putdown_threshold: 0.58
  merge_gap_s: 0.75
  minimum_event_duration_s: 0.30
```

Never tune on the test set.

---

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
- labeling protocol;
- agreement checks;
- splits and dataset versions.

### Person B — Standard CV and Layer 1

Responsible for:

- person tracking;
- pose inference;
- shelf regions;
- Stage B proposals;
- Track A;
- Track B1 and Track B2.

### Person C — VLM, evaluation, and integration

Responsible for:

- standalone Qwen Layer 2;
- Qwen verification;
- fusion;
- shared evaluator;
- CLI integration;
- reporting.

All members annotate a shared pilot before independent annotation.

---

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
4. Derive active spans.
5. Generate one preview with boxes and IDs.
6. Test 0.5–1 FPS triage settings.

### Person C

1. Create Pydantic schemas.
2. Create run metadata and structured logging.
3. Implement canonical prediction export.
4. Create evaluator skeleton.
5. Prepare Qwen response schemas.

### Day 1 acceptance criteria

- Source videos can be indexed reproducibly.
- A video file can be triaged directly.
- Active spans and stable tracks are stored.
- Empty clips remain in the manifest.
- Decode errors fail cleanly.

---

## Day 2 — Stage B and annotation workflow

### Person A

1. Configure one annotation tool.
2. Copy and adapt the labeling guidelines.
3. Document restocking handling.
4. Implement export to `events.csv`.
5. Implement ignore intervals.
6. Define annotation budget.
7. Implement manifest validation.

### Person B

1. Pin a YOLO pose model.
2. Define shelf/surface polygons.
3. Implement direct video pose tracking.
4. Implement wrist-to-region interactions.
5. Merge candidate intervals.
6. Add temporal context.
7. Save `candidates.parquet`.
8. Generate candidate previews.

### Person C

1. Validate timestamps and IDs.
2. Generate event previews.
3. Implement proposal-recall measurement.
4. Implement annotation agreement summaries.
5. Prepare the shared evaluation data structures.

### Whole team

1. Annotate the same pilot clips.
2. Compare disagreements.
3. Refine the guideline.
4. Start complete active-span annotation.

### Day 2 acceptance criteria

- Shelf regions are version-controlled.
- Candidate intervals are generated.
- Annotators review complete active spans.
- Canonical event export works.
- Proposal recall can be measured.

---

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

1. Extract pre/contact/post frames for candidates.
2. Extract hand and shelf crops.
3. Visually validate crops on at least 30 examples.
4. Extract frozen image embeddings.
5. Build hand-state and shelf-transition training data.
6. Train lightweight state classifiers.
7. Implement the deterministic state machine.
8. Export Track A predictions.

### Person C

1. Complete one-to-one temporal matching.
2. Implement tIoU and midpoint tolerance.
3. Implement precision, recall, F1, and confusion.
4. Generate Track A failure previews.
5. Prepare Qwen standalone window generation.

### Day 3 acceptance criteria

- Test split is frozen.
- Track A produces canonical pickup/putdown intervals.
- Hard negatives are processed.
- Crop extraction and state transitions are auditable.
- Event-level metrics are calculated.

---

## Day 4 — Track B1 and standalone Layer 2

### Person A

1. Review Track A false positives and false negatives.
2. Correct labels only through documented dataset versions.
3. Add missing hard-negative annotations.
4. Do not inspect the test split for tuning.

### Person B

1. Generate Track B1 fixed-window data.
2. Exclude ignore intervals and no-person clips.
3. Render sampled-frame debug views.
4. Run tiny overfit.
5. Train VideoMAE-Small classifier.
6. Implement sliding-window decoding.
7. Compare Track B1 with Track A on validation data.

### Person C

1. Implement active-span Qwen windows.
2. Overlay frame numbers/timestamps.
3. Implement Qwen3.6-27B client.
4. Implement standalone prompt and Pydantic validation.
5. Retry invalid JSON once.
6. Merge duplicate window predictions.
7. Record runtime, quantization, hardware, and parse errors.

### Day 4 acceptance criteria

- Track B1 passes tiny overfit.
- Track B1 produces canonical events.
- Qwen runs independently of Layer 1.
- Qwen scans active spans only.
- Qwen parsing failures are measured.

---

## Day 5 — Track B2, Layer 3, and final integration

### Person A

1. Review Layer 1/Qwen disagreements.
2. Categorize failure modes.
3. Confirm final manifest versions.
4. Prepare privacy-safe examples.

### Person B

1. Extract and cache VideoMAE features.
2. Generate timestep labels.
3. Implement the TCN.
4. Run tiny sequence overfit.
5. Train Track B2.
6. Decode temporal intervals.
7. Compare Track A, B1, and B2.
8. Package selected checkpoints and configs.

### Person C

1. Implement Qwen verification prompt.
2. Implement deterministic fusion.
3. Implement single-file and directory inference.
4. Run final validation comparison.
5. Run the untouched test set once.
6. Produce final metrics and failure gallery.
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
pickup ↔ putdown confusion
tIoU metrics
midpoint-tolerance metrics
false positives per video hour
Stage B proposal recall
runtime per video minute
Qwen invalid-response rate
Qwen hardware and quantization
```

---

# 22. Mandatory Engineering Gates

## Gate 1 — Dataset validity

Do not train until:

- videos decode correctly;
- timestamps are validated;
- event previews match labels;
- ignore intervals work;
- split leakage checks pass.

## Gate 2 — Proposal recall

Do not rely on Stage B filtering until recall is measured. Precision may be low; recall must be high.

## Gate 3 — Track A completeness

Track A is complete only when it independently outputs pickup/putdown intervals. Candidate generation alone is not a Layer 1 baseline.

## Gate 4 — Tiny overfit

Track B1 and Track B2 must pass tiny overfit tests.

Common failure causes:

- wrong labels;
- shuffled frame order;
- broken sampling;
- incorrect ignore masking;
- wrong tensor shapes;
- frozen trainable parameters.

## Gate 5 — Independent Layer 2

Qwen verification is not Layer 2. Layer 2 requires independent scanning of active-span windows.

## Gate 6 — Test isolation

All thresholds and decoding settings are selected on validation data.

## Gate 7 — Auditability

Preserve:

- ground truth;
- Track A predictions;
- Track B1 predictions;
- Track B2 predictions;
- standalone Layer 2 predictions;
- Qwen verification records;
- fusion decisions;
- prompts, configs, and model versions.

---

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
→ YOLO person detection + ByteTrack at low rate
→ person tracklets and active spans
```

## Layer 0B

```text
Active-span video
→ YOLO pose + fixed shelf polygons at higher rate
→ wrist trajectories and high-recall candidates
```

## Layer 1 Track A

```text
Pose candidate
+ hand/shelf pre/post crops
+ lightweight frozen image features
+ deterministic state machine
→ pickup / putdown / background intervals
```

## Layer 1 Track B1

```text
Chronological fixed windows
→ VideoMAE classifier
→ pickup / putdown / background
→ coarse intervals
```

## Layer 1 Track B2

```text
Cached overlapping VideoMAE features
→ small TCN
→ temporal class scores
→ refined intervals
```

## Layer 2

```text
Active-span overlapping windows
→ Qwen3.6-27B standalone detection
→ independent event intervals
```

## Layer 3

```text
Selected Layer 1 proposal with context
→ Qwen verification
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

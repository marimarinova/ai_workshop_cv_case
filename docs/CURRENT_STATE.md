Here's the implementation plan to reach Layer 1 Track A.

# Current State Assessment
Completed (Tasks 1–9): Ingestion, triage, shelf regions, proposals, annotation workflow, dataset quality/splits, evaluator, Track A feature extraction pipeline.

Data available:
- ~48 events in .local/vlm_annotations/events.csv (~20 pickup, ~28 putdown)
- ~200 source videos in .local/source_videos/
- Feature extraction infra: crop_extractor.py, sampling.py, cache.py, image_features.py, dataset_builder.py, contracts.py, manifest.py
- Evaluator: class_aware_matching.py, metrics.py, report.py (Task 8)

# Missing for Layer 1 Track A (Task 10): State classifiers, state machine, inference pipeline, CLI commands, configs/track_a.yaml

## Implementation Plan
### Phase 1: Build Feature Dataset from Existing Data

Goal: Run build_feature_dataset on the VLM-annotated candidates to produce cached embeddings keyed by label/split.
1. Map clip_ids in events.csv to source videos in .local/source_videos/
2. Generate candidates for those clips (or reuse existing from .local/candidate_staging/)
3. Assign train/val/test splits at the clip level (rule: by recording day, per IMPLEMENTATION_PLAN.md §8.8)
4. Run build_feature_dataset() → FeatureDataset with embeddings cached to .local/track_a_features/
5. Validate split isolation with validate_split_isolation()

New files: scripts/build_track_a_dataset.py (or CLI command)
Risk: Small dataset (~48 events). Need enough negatives. Use candidates without matching events as negatives.

### Phase 2: Hand-State + Shelf-Transition Classifiers

New modules:

```bash
File	Purpose
src/pickup_putdown/layer1/track_a/hand_state.py	Hand-state classifier (empty/carrying/uncertain) trained on hand crop embeddings
src/pickup_putdown/layer1/track_a/shelf_state.py	Shelf-transition classifier (object_removed/object_placed/no_change/uncertain) trained on shelf patch embeddings
```

Approach: Logistic regression (via sklearn) on frozen MobileNetV3 embeddings. Labels derived from sample_position + event label:
- Hand empty: pre sample of putdown, post sample of pickup
- Hand carrying: post sample of pickup, pre sample of putdown
- Shelf no_change: negative candidates
- Shelf object_removed: post sample of pickup
- Shelf object_placed: post sample of putdown

```bash
Tests: tests/test_hand_state.py, tests/test_shelf_state.py
Dependency: Add scikit-learn to pyproject.toml optional deps
```

### Phase 3: Repeating State Machine

New module: src/pickup_putdown/layer1/track_a/state_machine.py

Implements the repeating state cycle from IMPLEMENTATION_PLAN.md §9.3:
```
OUTSIDE → APPROACHING → CONTACT → TRANSFER → WITHDRAWING → OUTSIDE
```

Key logic:
- Processes per (actor_id, hand_side, region_id) preserving time order
- Uses hand-state + shelf-state classifier outputs as evidence
- Emits 0, 1, or multiple events per candidate
- Pickup: shelf→hand transition (hand empty→carrying AND shelf object_removed)
- Putdown: hand→shelf transition (hand carrying→empty AND shelf object_placed)
- Background: hand enters/exits but no persistent transfer
- Never merges adjacent pickup+putdown

```bash
Tests: tests/test_state_machine.py
```

### Phase 4: Inference Pipeline
```bash
New module: src/pickup_putdown/layer1/track_a/inference.py
```

End-to-end inference on a video or directory:
1. Load candidates + pose observations
2. Run feature extraction (with cache)
3. Run state machine per actor/hand/region
4. Estimate event boundaries from transfer/stabilization frames
5. Compute confidence score from classifier evidence + trajectory confidence
6. Export canonical predictions.csv

```bash
Config: configs/track_a.yaml with threshold settings, classifier paths, etc.
Tests: tests/test_inference.py
```

### Phase 5: CLI Integration

```bash
New CLI commands in src/pickup_putdown/cli.py:
Command	Purpose
build-track-a-dataset	Build feature dataset from candidates + ground truth
train-track-a	Train hand-state + shelf-state classifiers, save to artifacts
infer-track-a	Run Track A inference on video(s), export predictions
```

Also add Makefile targets: make track-a-dataset, make train-track-a, make infer-track-a

### Phase 6: Evaluation + Validation Report

Use existing Task 8 evaluator on Track A predictions:
- Run evaluate against events.csv ground truth
- Report P/R/F1 at tIoU 0.3 and 0.5
- Generate failure previews for missed/false events

Affected Modules Summary

```bash
src/pickup_putdown/layer1/track_a/hand_state.py
src/pickup_putdown/layer1/track_a/shelf_state.py
src/pickup_putdown/layer1/track_a/state_machine.py
src/pickup_putdown/layer1/track_a/inference.py
configs/track_a.yaml
src/pickup_putdown/cli.py
pyproject.toml
Makefile
tests/test_hand_state.py
tests/test_shelf_state.py
tests/test_state_machine.py
tests/test_inference.py
```

### Assumptions & Risks
1. Small dataset (~48 events): Logistic regression on 576-d embeddings is appropriate; no deep learning needed. Validation set will be small; thresholds tuned conservatively.
2. Label derivation for classifiers: Hand/shelf labels are inferred from event type + sample position, not explicitly annotated. This is per Task 9/10 spec.
3. Clip-to-video mapping: Events.csv clip_ids (e.g., D2_S20260520141725_E20260520142151_anon) must map to .local/source_videos/ files.
4. Negatives: Candidates without matching ground truth events serve as negatives. With 1527 candidates and 48 events, negatives are plentiful.
5. Restocking: Per spec, visible restocking → background/negative, not putdown. VLM annotations should already reflect this.

Validation Commands

### After implementation

```bash
ruff check src/pickup_putdown/layer1/track_a/
ruff format --check .
python -m pytest tests/test_hand_state.py tests/test_shelf_state.py tests/test_state_machine.py tests/test_inference.py -v
python -m compileall src
```

### End-to-end smoke test on one video

```bash
pickup-putdown infer-track-a .local/source_videos/D2_S20260520141725_E20260520142151_anon.mp4 --config configs/track_a.yaml -o .local/track_a_output/
```

# Evaluate predictions
# (using Task 8 evaluator on generated predictions.csv vs events.csv)

Recommended Order
1. Phase 1 (dataset build) — prerequisite for everything
2. Phase 2 (classifiers) — depends on dataset
3. Phase 3 (state machine) — depends on classifiers
4. Phase 4 (inference) — integrates all above
5. Phase 5 (CLI) — exposes to user
6. Phase 6 (evaluation) — measures quality


# S3 Bucket Analysis — s3://chillnbite-cameras/anon/
What's in S3

```bash
Prefix	Count	Size	Content
anon/*.mp4 (root)	~395 files	~5.8 GB	Source videos
anon/candidates/videos/	1527 files	~25 GB	Candidate MP4 clips
anon/candidates/metadata/	75 files	~1 MB	Candidate metadata JSONs (per source video)
anon/vlm/2026-06-26/vlm_annotations/	3059 files	~30 KB	VLM annotations: events.csv (48 events), normalized/ (1527 JSONs), raw/ (1527 JSONs), processing.csv, summary.json
anon/vlm/2026-06-26/task_7_vlm/	6 files	~365 KB	Task 7: clips.csv, events.csv (41 events), processing.csv, dedup_audit.json, provenance.json, summary.json
anon/vlm/2026-06-26/task_7_review/	1 file	~14 KB	review_manifest.csv
```

Answer: Yes, everything is in S3.

All the data needed for the Track A plan is already uploaded to s3://chillnbite-cameras/anon/vlm/2026-06-26/. No local transfer required.

On another device:

Download ground truth + annotations:
```bash
aws s3 sync s3://chillnbite-cameras/anon/vlm/2026-06-26/vlm_annotations/ .local/vlm_annotations/
aws s3 cp s3://chillnbite-cameras/anon/vlm/2026-06-26/task_7_review/review_manifest.csv .local/task_7_review/
```

Download source videos on-demand (only clips with events needed)

Or download all candidate videos for the event-bearing clips

The two events.csv files differ slightly: vlm_annotations/events.csv has 48 events, task_7_vlm/events.csv has 41 (deduplicated post-review). 

Use task_7_vlm/events.csv as the canonical ground truth since it's post-manual-review.
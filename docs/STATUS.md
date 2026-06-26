# Project Status

Pickup and putdown event detection in store video.

## Milestones

| # | Area | Status |
|---|------|--------|
| 1 | Bucket inventory, clip registry, bounded cache | ✅ Done |
| 2 | Person triage, active spans, ByteTrack | ✅ Done |
| 3 | Pose inference, shelf regions, actor association | ✅ Done |
| 4 | Candidate proposal generation, previews | ✅ Done |
| 5 | Label Studio annotation, export, manifest validation | ✅ Done |
| 6 | Remote candidate pipeline (S3 download/generate/upload) | ✅ Done |
| 7 | VLM annotation (Qwen3.6), reviewed JSONs | ✅ Done |
| 8 | Shared evaluation framework, canonical export | ✅ Done |
| 9 | Track A Phase 1 — reviewed feature dataset (embeddings, splits, manifest) | ✅ Done |
| 10 | Track A Phase 2 — hand-state + shelf-transition classifiers | ✅ Done |
| 11 | Track A Phase 3 — repeating temporal state machine | ✅ Done |
| 12 | Track A Phase 4 — inference pipeline, boundary refinement, dedup | ✅ Done |
| 13 | Track A Phase 5 — CLI integration, Makefile, real-data smoke test | ✅ Done |

## Track A — Phase 1 (Feature Dataset)

Built reviewed feature dataset from manually annotated candidates.

- Source: 50 reviewed candidates (40 positive, 10 negative)
- Embeddings: MobileNetV3-small, 576-dim, cached per crop
- Splits: 11 train clips, 4 val clips (by recording day)
- Output: 222 records (111 hand, 111 shelf) across pre/contact/post positions
- Artifacts: `.local/track_a_features/`

## Track A — Phase 2 (Classifiers)

Two logistic-regression classifiers on frozen embeddings.

### Hand-State Classifier

- Labels: `empty`, `carrying` (derived from event type + pre/post position)
- Uncertain: returned when confidence < 0.60 or margin < 0.15
- Training records: 41 (21 carrying, 20 empty)
- Validation records: 10 (7 carrying, 3 empty)
- Artifacts: `.local/track_a_artifacts/hand_state.{joblib,metadata.json,metrics.json}`

### Shelf-Transition Classifier

- Labels: `object_removed`, `object_placed`, `no_change` (derived from event type + post position)
- Uncertain: returned when confidence < 0.60 or margin < 0.15
- Training records: 30 (15 object_removed, 8 object_placed, 7 no_change)
- Validation records: 10 (5 object_removed, 2 object_placed, 3 no_change)
- Artifacts: `.local/track_a_artifacts/shelf_state.{joblib,metadata.json,metrics.json}`

### Files

- `src/pickup_putdown/layer1/track_a/classifier.py` — shared sklearn pipeline base
- `src/pickup_putdown/layer1/track_a/hand_state.py` — hand-state label derivation + training
- `src/pickup_putdown/layer1/track_a/shelf_state.py` — shelf-state label derivation + training
- `configs/track_a.yaml` — classifier configuration
- `tests/test_hand_state.py`, `test_shelf_state.py`, `test_track_a_training.py` — 49 tests

### CLI / Makefile

```bash
pickup-putdown train-track-a --config configs/track_a.yaml --output-dir .local/track_a_artifacts
make train-track-a
```

## Track A — Phase 3 (State Machine)

Deterministic repeating state machine that converts per-frame classifier
probabilities into pickup/putdown events.

### States

`OUTSIDE → APPROACHING → CONTACT → TRANSFER → WITHDRAWING → OUTSIDE`

Processes observations per `(clip_id, actor_id, hand_side, region_id)` stream.
Supports multiple interaction cycles per stream.

### Evidence rules

- **Pickup**: pre-transfer hand empty + post-transfer hand carrying + shelf object\_removed
- **Putdown**: pre-transfer hand carrying + post-transfer hand empty + shelf object\_placed
- Both require configurable probability thresholds and minimum transfer duration
- Uncertain/contradictory interactions remain background (no event emitted)

### Configuration

All thresholds in `configs/track_a.yaml` under `state_machine:`:
temporal (approach, contact, transfer, withdrawal, gap, timeout),
probability (hand, shelf, uncertainty ratio),
confidence (weights for hand/shelf/trajectory, emission threshold).

### API

```python
machine = RepeatingInteractionStateMachine(config)
events = machine.process(observations)  # batch
event = machine.update(observation)     # incremental
```

### Files

- `src/pickup_putdown/layer1/track_a/state_machine.py` — enums, config, observation/event contracts, state machine
- `configs/track_a.yaml` — extended with `state_machine` section
- `tests/test_state_machine.py` — 36 tests (fully synthetic, no real models)

### Test results

36/36 pass. Total Track A test suite: 140/140 pass.

## Track A — Phase 4 (Inference Pipeline)

End-to-end callable pipeline that integrates feature extraction, trained classifiers, and the repeating state machine into canonical predictions.

### Data flow

```
candidates + poses
  → validation (identity, video, pose, shelf region)
  → sliding-window sampling (configurable FPS, default 4 Hz)
  → hand/shelf feature extraction with cache reuse
  → classifier probabilities (empty/carrying/uncertain, removed/placed/no_change/uncertain)
  → TrackAObservation construction (pose trajectory + classifier evidence)
  → RepeatingInteractionStateMachine per (clip, actor, hand, region) stream
  → transition-frame grace window (0.25 s default)
  → boundary refinement (clip/candidate bounds, min duration)
  → cross-candidate deduplication (temporal IoU + transfer proximity)
  → CanonicalPrediction + predictions.csv + diagnostics
```

### Features

- **Artifact validation**: embedding dimension, encoder name/version, required classes
- **Sliding-window sampling**: uniform samples at configurable FPS across candidate window
- **Grace window**: recovers events lost when wrist exits on the transition frame
- **Boundary refinement**: enforces clip/candidate bounds, start < end, minimum duration
- **Deduplication**: temporal IoU + transfer-time tolerance, keeps highest-confidence, preserves audit
- **Diagnostics**: per-candidate trace, confidence distribution, skip reasons, raw events, dedup audit
- **Canonical output**: `predictions.csv` compatible with evaluator schema

### State machine fix

`_last_event_s` no longer updated for rejected events. A failed low-confidence emission attempt cannot suppress a later valid event through `minimum_event_separation_s`.

### API

```python
pipeline = TrackAInferencePipeline(config)
result = pipeline.run(
    candidates=candidates,
    pose_observations=poses,
    source_videos=video_paths,
    hand_classifier_path=hand_path,
    shelf_classifier_path=shelf_path,
    output_dir=output_dir,
)
```

### Configuration

`configs/track_a.yaml` under `inference:`: sampling FPS, boundary refinement, deduplication thresholds, grace window, debug traces.

### Files

- `src/pickup_putdown/layer1/track_a/inference.py` — pipeline, config, evidence, sampling, extraction, grace window, boundary refinement, dedup, canonical output
- `src/pickup_putdown/layer1/track_a/state_machine.py` — `_last_event_s` fix
- `configs/track_a.yaml` — extended with `inference` section
- `tests/test_inference.py` — 55 tests (fully synthetic, no GPU/videos)

### Test results

55/55 pass. Total Track A test suite: 195/195 pass.

## Track A — Phase 5 (CLI Integration & Smoke Test)

CLI command and Makefile target that expose the Phase 4 inference pipeline for
real-data execution.

### CLI Command

```bash
pickup-putdown infer-track-a \
  --config configs/track_a.yaml \
  --candidate-metadata .local/candidate_staging/metadata \
  --source-video-dir .local/source_videos \
  --shelves-config configs/shelves.yaml \
  --camera-id store_camera_01 \
  --artifact-dir .local/track_a_artifacts \
  --cache-dir .local/track_a_features \
  --output-dir .local/track_a_output \
  --clip-id D2_S..._anon \
  --debug-traces --force -v
```

Arguments: `--config`, `--candidate-metadata`, `--candidates`, `--pose-observations`,
`--source-video-dir`, `--shelves-config`, `--camera-id`, `--artifact-dir`,
`--cache-dir`, `--output-dir`, `--clip-id`, `--candidate-id`, `--debug-traces`,
`--force`, `-v`.

Scope: single candidate (`--candidate-id`), single clip (`--clip-id`), or all
resolvable candidates.

### Makefile

```bash
make infer-track-a
make infer-track-a TRACK_A_CLIP_ID=D2_S20260520141725_E20260520142151_anon
make infer-track-a TRACK_A_DEBUG_TRACES=1 TRACK_A_FORCE=1
```

### Input Resolution

- **Candidates**: Loaded from `<metadata>/<clip_id>/<clip_id>.json` or
  `<metadata>/candidates/<clip_id>/<clip_id>.json`.
- **Identity enrichment**: `actor_id`/`hand_side`/`region_id` filled from
  `feature_dataset.parquet` when metadata lacks them.
- **Pose data**: Auto-detected from `.local/remote_candidates/*/tracks_pose.parquet`.
  `clip_` prefix stripped to match candidate clip_id format.
- **Videos**: `<source_video_dir>/<clip_id>.mp4`.
- **Shelves**: From `configs/shelves.yaml` + `--camera-id`.
- **Artifacts**: `<artifact_dir>/hand_state.joblib` + `shelf_state.joblib`.

### Smoke Test Results

```
Clip: D2_S20260520141725_E20260520142151_anon
Candidates processed:  5
Candidates skipped:    69 (missing identity fields)
Total samples:         94
Cache hits:            2
Cache misses:          184
Raw events:            5
Final predictions:     5
  Pickups:             3
  Putdowns:            2
  Mean confidence:     0.5410
```

### Cache Reuse

- **Hand embeddings**: Deterministic keys, full reuse on re-run (94/94 hits).
- **Shelf embeddings**: Cache key depends on runtime-computed crop geometry.
  Added fallback lookup for contact-point geometry to reuse Phase 1 build cache.
  Subsequent runs may miss when `extract_shelf_patch` produces different geometry
  than the contact point used during Phase 1 build.

### Phase 4 Integration Checks

- **Transition grace**: Requires BOTH hand AND shelf directional evidence
  (`hand_carrying + shelf_removed` for pickup, `hand_empty + shelf_placed` for
  putdown). Does NOT emit based on post-transfer hand state alone. ✅
- **Deduplication**: Uses temporal IoU threshold AND transfer-time tolerance,
  keeps highest-confidence prediction. ✅

### Files

- `src/pickup_putdown/cli.py` — `infer_track_a` command + helper functions
- `src/pickup_putdown/layer1/track_a/inference.py` — shelf cache fallback lookup
- `Makefile` — `infer-track-a` target
- `tests/test_track_a_cli.py` — 26 tests (mocked pipeline, no GPU/videos)

### Test Results

26/26 CLI tests pass. Total Track A suite: 221/221 pass.

## Known Limitations

- Small dataset: 50 reviewed candidates, imbalanced classes, limited val coverage
- Shelf `object_placed` has only 2 val samples
- Contact/mid positions excluded from supervised training (no reliable label)
- Negative candidates excluded from hand-state training (no explicit hand-state annotation)
- Thresholds are baseline defaults, not tuned on held-out data
- State machine transfer detection requires transition observation inside region; Phase 4 adds grace window to recover transition-frame withdrawals
- Confidence formula is a simple weighted mean; no learned calibration
- Cross-candidate deduplication is within-clip only; no global cross-video dedup
- Event boundaries are refined but not calibrated against ground truth
- Grace window uses fixed probability thresholds (0.55/0.50) matching state machine config
- Shelf comparison during inference uses per-timestamp patches; not explicit pre/post reference comparison
- **Shelf cache reuse**: Runtime geometry from `extract_shelf_patch` differs from
  contact-point geometry used in Phase 1 build, so shelf cache keys don't match
  across builds. Hand cache keys are deterministic and fully reusable.
- **Candidate identity**: Only candidates present in Phase 1 feature dataset get
  `actor_id`/`hand_side`/`region_id` enrichment. Candidates outside the build
  are skipped with `missing_identity_fields`.

## Next

- Track A Phase 6: shelf cache key stabilization, full-dataset inference, threshold tuning
- Task 8: evaluation (precision/recall/F1 against reviewed ground truth)
- Track B1/B2: VideoMAE window classifiers
- Layer 2/3: Qwen VLM verification and fusion

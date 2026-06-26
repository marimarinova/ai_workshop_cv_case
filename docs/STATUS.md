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

## Known Limitations

- Small dataset: 50 reviewed candidates, imbalanced classes, limited val coverage
- Shelf `object_placed` has only 2 val samples
- Contact/mid positions excluded from supervised training (no reliable label)
- Negative candidates excluded from hand-state training (no explicit hand-state annotation)
- Thresholds are baseline defaults, not tuned on held-out data

## Next

- Track A Phase 3: repeating state machine, event emission, boundary estimation
- Track A end-to-end inference and `predictions.csv`
- Track B1/B2: VideoMAE window classifiers
- Layer 2/3: Qwen VLM verification and fusion

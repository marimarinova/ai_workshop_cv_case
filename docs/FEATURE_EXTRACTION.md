# Feature Extraction

Pipeline for building the reviewed Track A feature dataset from manually reviewed Task 7 data.

## Overview

```
review_manifest.csv + reviewed candidate JSONs
         │
         ▼
  Regenerate reviewed_events.csv from JSON events
  (candidate-relative → source-video timestamps)
         │
         ▼
  Resolve reviewed examples
  (JSON events → positives, empty JSON → negatives)
         │
         ▼
  Assign train/val/test splits by recording day
         │
         ▼
  Run YOLO pose inference on source video windows
         │
         ▼
  Extract hand/shelf crops → compute embeddings → cache
         │
         ▼
  reviewed_events.csv + feature_dataset.parquet + splits.json + build_summary.json
```

## Ground Truth

The reviewed candidate JSON files (`.local/vlm_annotations/normalized/cand_*.json`) are the ground truth. Each reviewed JSON contains:

- `events[]` — reviewed event list with `label`, `start_s`, `end_s` (candidate-relative)
- `source_start_s` — offset to convert to source-video timestamps
- `clip_id` — source video identifier

Events are regenerated from these JSONs into a compliant `reviewed_events.csv`. The original VLM `events.csv` is not used for labeling.

## Prerequisites

- Source videos at `.local/source_videos/<clip_id>.mp4`
- Pose model at `models/pose_detector.pt` (YOLO11n-pose)
- Shelf config at `configs/shelves.yaml`
- Review manifest with local paths (see Path Rewrite below)
- Reviewed candidate JSONs at `.local/vlm_annotations/normalized/cand_*.json`

## Path Rewrite

Review manifest and VLM annotation JSONs use S3 paths after upload. Revert to local paths before building:

```bash
# Preview
python -m scripts.vlm_annotations.rewrite_s3_paths_to_local --dry-run

# Execute (creates .bak backups)
python -m scripts.vlm_annotations.rewrite_s3_paths_to_local
```

Forward rewrite (local → S3) remains available:

```bash
python -m scripts.vlm_annotations.rewrite_local_paths_to_s3
```

## CLI Usage

```bash
# With repository defaults
pickup-putdown build-track-a-dataset

# With explicit paths
pickup-putdown build-track-a-dataset \
  --events-csv .local/task_7_vlm/events.csv \
  --clips-csv .local/task_7_vlm/clips.csv \
  --review-manifest .local/task_7_review/review_manifest.csv \
  --candidate-metadata-dir .local/candidate_staging \
  --source-video-dir .local/source_videos \
  --output-dir .local/track_a_features \
  --split-seed 42 \
  --config configs/proposals.yaml \
  --shelves-config configs/shelves.yaml \
  --camera-id store_camera_01 \
  -v
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--events-csv -e` | `.local/task_7_vlm/events.csv` | Canonical events CSV |
| `--clips-csv` | `.local/task_7_vlm/clips.csv` | Clips CSV |
| `--review-manifest -r` | `.local/task_7_review/review_manifest.csv` | Review manifest CSV |
| `--candidate-metadata-dir` | `.local/candidate_staging` | Candidate staging directory |
| `--source-video-dir` | `.local/source_videos` | Source video directory |
| `--output-dir -o` | `.local/track_a_features` | Output directory |
| `--split-seed` | `42` | Random seed for split assignment |
| `--config -c` | `configs/proposals.yaml` | Configuration YAML |
| `--shelves-config` | `configs/shelves.yaml` | Shelf region configuration |
| `--camera-id` | `store_camera_01` | Camera ID |
| `--verbose -v` | | Enable debug logging |

## Makefile Usage

```bash
# With defaults
make track-a-dataset

# With overrides
make track-a-dataset \
  TRACK_A_OUTPUT_DIR=.local/my_features \
  TRACK_A_SPLIT_SEED=123
```

### Makefile Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TRACK_A_EVENTS_CSV` | `.local/task_7_vlm/events.csv` | Events CSV path |
| `TRACK_A_CLIPS_CSV` | `.local/task_7_vlm/clips.csv` | Clips CSV path |
| `TRACK_A_REVIEW_MANIFEST` | `.local/task_7_review/review_manifest.csv` | Review manifest path |
| `TRACK_A_CANDIDATE_DIR` | `.local/candidate_staging` | Candidate staging directory |
| `TRACK_A_SOURCE_VIDEO_DIR` | `.local/source_videos` | Source video directory |
| `TRACK_A_OUTPUT_DIR` | `.local/track_a_features` | Output directory |
| `TRACK_A_SPLIT_SEED` | `42` | Split seed |
| `TRACK_A_CONFIG` | `configs/proposals.yaml` | Config YAML |
| `TRACK_A_SHELVES_CONFIG` | `configs/shelves.yaml` | Shelf config |
| `TRACK_A_CAMERA_ID` | `store_camera_01` | Camera ID |

## Outputs

| File | Description |
|------|-------------|
| `reviewed_events.csv` | Regenerated events from reviewed JSONs (ground truth) |
| `feature_dataset.parquet` | Feature dataset manifest with all records |
| `splits.json` | Clip-to-split assignments and counts |
| `build_summary.json` | Build statistics (positives, negatives, records by split/label/position) |
| `embeddings/` | Cached embedding vectors (.npy) |
| `crops/` | Cached crop images (if `save_crops=true`) |

## Labeling Rules

Only reviewed data is used for supervised training:

- **Reviewed positive**: candidate with `reviewed=true` and non-empty `events[]` in JSON → labeled from JSON `label` field
- **Reviewed negative**: candidate with `reviewed=true` and empty `events[]` in JSON → labeled `negative`
- **Excluded**: unreviewed candidates, candidates without metadata, candidates without reviewed JSON

Unreviewed candidates are never treated as negatives. Labels come from the reviewed JSON events, not from the VLM `events.csv`.

## Split Assignment

- Clips grouped by recording day (extracted from clip ID timestamp)
- Days shuffled with deterministic seed, then assigned 70/15/15 (train/val/test)
- All clips from the same day stay in one split
- Split isolation validated: no clip appears in multiple splits

## Configuration

Feature extraction uses `track_a_features` section from the config YAML:

```yaml
track_a_features:
  encoder_name: mobilenet_v3_small
  hand_crop_size: 224
  shelf_patch_size: 224
  cache_dir: .local/track_a_features
  save_crops: true
  min_samples: 3
  max_interval_s: 99999.0
```

Pose inference uses `pose` section:

```yaml
pose:
  model_path: models/pose_detector.pt
  target_fps: 8.0
  image_size: 640
  device: auto
```

## Programmatic Usage

```python
from pickup_putdown.layer1.track_a.reviewed_dataset import (
    build_reviewed_feature_dataset,
    load_review_manifest,
    load_events_csv,
    resolve_reviewed_examples,
    assign_splits_by_recording_day,
)
from pickup_putdown.perception.shelf_regions import (
    load_shelf_config,
    get_regions_for_camera,
)
from pickup_putdown.config import load_config

cfg = load_config("configs/proposals.yaml")
shelf_cfg = load_shelf_config("configs/shelves.yaml")
camera_cfg = get_regions_for_camera(shelf_cfg, "store_camera_01")
shelf_regions = {r.region_id: r.points for r in camera_cfg.regions}

dataset, summary = build_reviewed_feature_dataset(
    review_manifest_path=".local/task_7_review/review_manifest.csv",
    events_path=".local/task_7_vlm/events.csv",
    clips_path=".local/task_7_vlm/clips.csv",
    candidate_staging_dir=".local/candidate_staging",
    source_video_dir=".local/source_videos",
    output_dir=".local/track_a_features",
    pose_cfg=cfg.pose,
    track_a_cfg=cfg.track_a_features,
    shelf_regions=shelf_regions,
    split_seed=42,
)
```

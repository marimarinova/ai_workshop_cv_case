# VLM Annotations Upload & Path Rewrite

Tools for rewriting local candidate video paths to S3 paths and uploading VLM annotation artifacts.

## Scripts

| Script | Purpose |
|--------|---------|
| `rewrite_local_paths_to_s3.py` | In-place path rewrite in `.local/` metadata files |
| `upload_s3.py` | Upload annotation artifacts to S3 with dated prefix |

## Usage

### 1. Rewrite local paths to S3

Replaces `.local/candidate_staging/candidates/` with `s3://chillnbite-cameras/anon/candidates/videos/` across all annotation metadata.

```bash
# Preview
python -m scripts.vlm_annotations.rewrite_local_paths_to_s3 --dry-run

# Execute (creates .bak backups)
python -m scripts.vlm_annotations.rewrite_local_paths_to_s3

# Skip backups
python -m scripts.vlm_annotations.rewrite_local_paths_to_s3 --no-backup
```

**Targets:**
- `.local/vlm_annotations/normalized/*.json` — `video_path` field
- `.local/vlm_annotations/raw/*.json` — `video_path` field
- `.local/task_7_review/review_manifest.csv` — `video_path` column
- `.local/task_7_vlm/processing.csv` — video path column
- `.local/candidate_staging/candidates/*/` metadata JSONs — `candidate_key` field

**Expected result:** ~3000 files scanned, ~3000 modified, ~4600 replacements, 0 errors.

### 2. Upload artifacts to S3

Uploads VLM annotation artifacts to `s3://chillnbite-cameras/anon/vlm/YYYY-MM-DD/`.

```bash
# Preview
python -m scripts.vlm_annotations.upload_s3 --dry-run

# Upload (uses today's date)
python -m scripts.vlm_annotations.upload_s3

# Specific date
python -m scripts.vlm_annotations.upload_s3 --date 2026-06-26
```

**Uploads to:**
```
anon/vlm/{date}/
  vlm_annotations/
    normalized/*.json
    raw/*.json
    events.csv
    processing.csv
    summary.json
  task_7_review/
    review_manifest.csv
  task_7_vlm/
    clips.csv
    events.csv
    processing.csv
    provenance.json
    summary.json
    dedup_audit.json
```

**Skips:** `review_frames/`, `logs/`, `.bak` files, symlinks.

**Expected result:** ~3064 files (~4-5 MB), 0 failures.

## Execution Order

Always rewrite paths first, then upload:

```bash
python -m scripts.vlm_annotations.rewrite_local_paths_to_s3
python -m scripts.vlm_annotations.upload_s3
```

## Dependencies

Uses `boto3` and `pickup_putdown.config` from existing project dependencies. Reads `configs/storage.s3.yaml` for S3 credentials.

# Label Studio Pilot Setup

This workflow creates the annotation pilot directly from candidate videos stored in AWS S3. Candidate videos do not need to be downloaded locally.

## 1. Download candidate metadata

```bash
mkdir -p .local/candidate_staging/metadata

aws s3 sync \
  s3://chillnbite-cameras/anon/candidates/metadata/ \
  .local/candidate_staging/metadata/ \
  --only-show-errors
```

Task 6.1 metadata contains one JSON file per source video, with candidates under `.candidates[]`.

Future uploads should set `ContentType="video/mp4"` directly.

## 3. Configure Label Studio S3 storage

In **Project Settings → Cloud Storage**, add Amazon S3 source storage:

```text
Bucket: chillnbite-cameras
Prefix: anon/candidates/videos
Region: eu-central-1
```

Credentials:

* Access Key ID: required
* Secret Access Key: required
* Session Token: leave blank unless using temporary STS credentials
* S3 Endpoint: leave blank for normal AWS S3

Enable **Proxy through the platform** where available.

Test and save the connection.

Do not sync the storage. Tasks will be imported from generated JSON.

## 4. Generate the pilot

Create a deterministic pilot of 40 candidates:

```bash
pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir .local/candidate_staging/metadata \
  --output annotation/tasks_pilot.json \
  --limit 40 \
  --seed 42 \
  --video-url-mode s3_storage \
  --s3-bucket chillnbite-cameras
```

The generated video references should look like:

```text
s3://chillnbite-cameras/anon/candidates/videos/<source_id>/<candidate_id>.mp4
```

## 5. Validate S3 references

```bash
pickup-putdown annotation-check-media \
  annotation/tasks_pilot.json \
  --video-url-mode s3_storage \
  --s3-bucket chillnbite-cameras \
  --s3-region eu-central-1
```

All tasks should pass.

## 6. Import and annotate

In Label Studio:

1. Open **Data Import**.
2. Upload `annotation/tasks_pilot.json`.
3. Confirm that videos load, play, and seek.
4. Label:

   * `pickup`
   * `putdown`
   * `ignore`
   * or leave the task without an event

Candidate clips are interaction proposals, not pickup/putdown predictions. Annotators must choose the event type and boundaries.

## 7. Export annotations

Export completed annotations from Label Studio as JSON, then run:

```bash
mkdir -p annotation/exports

pickup-putdown annotation-export \
  --input annotation/label_studio_export_pilot.json \
  --events annotation/exports/pilot_events.csv \
  --ignore annotation/exports/pilot_ignore.parquet \
  --candidate-mode \
  --provenance annotation/exports/pilot_provenance.parquet
```

The exporter converts candidate-relative timestamps to original source-video timestamps:

```text
source timestamp = source_start_s + candidate-relative timestamp
```

## 8. Validate

```bash
pickup-putdown annotation-validate \
  --input annotation/label_studio_export_pilot.json
```

Manually verify several exported timestamps against the original source videos before starting bulk annotation.

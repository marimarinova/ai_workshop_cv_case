# task_6_1_hard: Remote S3 Candidate Generation for Annotation

> This task is a follow-up to Task 6. It operationalizes the pipeline implemented through Tasks 3–5 against source videos stored in Amazon S3.
>
> The main goal is to generate annotation-ready candidate clips in S3 as soon as possible, beginning with a 5-video validation run and followed by a 100-video batch.

**Task ID:** `task_6_1`
**Difficulty:** `hard`
**Dependencies:** Tasks 1–6
**Parallel work:** Annotation preparation and evaluator development

## Objective

Implement a remote batch-processing workflow that:

1. Connects from the remote compute server to Amazon S3.

2. Discovers source videos under:

   ```text
   s3://chillnbite-cameras/anon/
   ```

3. Creates and maintains:

   ```text
   s3://chillnbite-cameras/anon/process_for_candidates.csv
   ```

4. Automatically selects a requested number of unprocessed source videos.

5. Runs the existing Tasks 3–5 pipeline on those videos.

6. Supports configurable parallel processing.

7. Encodes all generated candidate clips as browser-compatible H.264 MP4.

8. Uploads the candidates back to S3 for direct use by local Label Studio instances.

9. Marks each source video as processed only after its candidate-processing outputs have been successfully uploaded.

The existing Tasks 3–5 implementations must be reused rather than reimplemented.

---

## S3 layout

The existing source videos remain under the current prefix.

```text
s3://chillnbite-cameras/anon/
├── <existing source videos and source subdirectories>
│
├── process_for_candidates.csv
│
└── candidates/
    ├── videos/
    │   └── <source_video_id>/
    │       ├── candidate_0001.mp4
    │       ├── candidate_0002.mp4
    │       └── ...
    │
    ├── metadata/
    │   └── <source_video_id>.json
    │
    └── runs/
        └── <run_id>.json
```

Label Studio source storage should point only to:

```text
s3://chillnbite-cameras/anon/candidates/videos/
```

The source discovery process must exclude:

```text
anon/candidates/
anon/process_for_candidates.csv
```

S3 folders are object-key prefixes rather than physical directories.

---

## Processing ledger

Create the following file in the same S3 prefix as the source videos:

```text
s3://chillnbite-cameras/anon/process_for_candidates.csv
```

Required schema:

```csv
file_name,processed
camera_01/video_001.mp4,false
camera_01/video_002.mp4,true
camera_02/video_003.mp4,false
```

### `file_name`

`file_name` must contain the complete object key relative to:

```text
s3://chillnbite-cameras/anon/
```

Do not store only the basename because different source directories may contain files with identical names.

### `processed`

Allowed values:

```text
true
false
```

A video is marked `true` only after:

1. The Tasks 3–5 pipeline completes successfully.
2. All generated candidates are encoded and validated.
3. All candidate files are uploaded to S3.
4. The source metadata file is uploaded successfully.

If a video produces zero candidates but the pipeline completes successfully, it is still marked as processed.

If processing or upload fails, the value remains `false`.

### Ledger initialization and synchronization

When the command starts, it must:

1. List supported source-video objects under `anon/`.
2. Exclude generated candidates and non-video artifacts.
3. Create the CSV if it does not exist.
4. Add newly discovered source videos with `processed=false`.
5. Preserve existing `processed=true` values.
6. Never reset an existing processed video to `false` automatically.

The main coordinator process must update the CSV. Parallel workers must not write to it directly.

The updated CSV must be uploaded after each successfully completed source video so progress survives interruption.

Only one remote candidate-generation batch may update the CSV at a time. Distributed locking and multiple simultaneous batch runners are outside this task.

---

## Automatic video selection

The user must not need to preselect source videos manually.

The command must select up to the requested number of entries where:

```text
processed == false
```

The default selection order should be deterministic:

```text
sort by file_name
```

Example:

```bash
pickup-putdown candidates-remote --target-count 20
```

This processes the first 20 unprocessed source videos.

If fewer unprocessed videos remain, process all remaining videos and report the actual count.

---

## CLI command

Add a command such as:

```bash
pickup-putdown candidates-remote \
  --storage-config configs/storage.s3.yaml \
  --pipeline-config configs/candidates.yaml \
  --target-count 20 \
  --workers 8 \
  --transfer-workers 8 \
  --gpu-workers 1 \
  --encode-workers 8
```

Required options:

```text
--target-count
--workers
--transfer-workers
--gpu-workers
--encode-workers
--storage-config
--pipeline-config
```

Recommended optional arguments:

```text
--work-dir
--keep-local-files
--fail-fast
--overwrite
--dry-run
```

### Argument meanings

* `--target-count`: maximum number of unprocessed source videos selected for this run.
* `--workers`: maximum number of active source-video jobs.
* `--transfer-workers`: maximum concurrent S3 downloads and uploads.
* `--gpu-workers`: maximum concurrent GPU inference jobs.
* `--encode-workers`: maximum concurrent H.264 encoding jobs.

Command-line values must override configuration-file defaults.

---

## Make target

Add a Make target that exposes the target count and parallelism settings.

Example:

```makefile
CANDIDATE_TARGET_COUNT ?= 5
CANDIDATE_WORKERS ?= 4
CANDIDATE_TRANSFER_WORKERS ?= 4
CANDIDATE_GPU_WORKERS ?= 1
CANDIDATE_ENCODE_WORKERS ?= 4

candidates-remote:
	pickup-putdown candidates-remote \
		--storage-config $(STORAGE_CONFIG) \
		--pipeline-config $(CANDIDATE_CONFIG) \
		--target-count $(CANDIDATE_TARGET_COUNT) \
		--workers $(CANDIDATE_WORKERS) \
		--transfer-workers $(CANDIDATE_TRANSFER_WORKERS) \
		--gpu-workers $(CANDIDATE_GPU_WORKERS) \
		--encode-workers $(CANDIDATE_ENCODE_WORKERS)
```

Example usage:

```bash
make candidates-remote CANDIDATE_TARGET_COUNT=5
```

```bash
make candidates-remote \
  CANDIDATE_TARGET_COUNT=100 \
  CANDIDATE_WORKERS=8 \
  CANDIDATE_TRANSFER_WORKERS=8 \
  CANDIDATE_GPU_WORKERS=1 \
  CANDIDATE_ENCODE_WORKERS=8
```

---

## Parallel execution model

Parallelism must be applied across source videos while preserving independent limits for network transfer, GPU work, and encoding.

Each source-video job follows:

```text
download source
    → run Tasks 3–5
    → generate candidates
    → encode candidates as H.264
    → validate candidates
    → upload candidates and metadata
    → mark source as processed
    → clean temporary files
```

Use a bounded producer-consumer workflow so downloads, inference, encoding, and uploads can overlap.

The remote server has substantial CPU, RAM, and GPU capacity. The implementation must therefore avoid a single sequential worker. However, it must not assume unlimited GPU concurrency.

The likely bottleneck is network transfer, so concurrent downloads and uploads must be configurable independently from inference.

### GPU handling

GPU work must be protected by a semaphore or dedicated GPU worker pool.

The implementation must avoid loading a separate model instance for every general worker when the model can be safely reused by a smaller number of GPU workers.

### Worker isolation

Each source video must use an isolated local directory:

```text
<work-dir>/<run_id>/<source_video_id>/
├── source/
├── intermediate/
├── candidates/
└── metadata/
```

Workers must not share mutable intermediate files.

Temporary files should be removed after successful upload unless `--keep-local-files` is enabled.

---

## Candidate encoding

All candidate clips uploaded to S3 must use:

```text
Container: MP4
Video codec: H.264/AVC
Pixel format: yuv420p
Fast-start metadata: enabled
```

Equivalent FFmpeg settings:

```bash
ffmpeg \
  -i input.mp4 \
  -c:v libx264 \
  -pix_fmt yuv420p \
  -movflags +faststart \
  -an \
  output.mp4
```

Audio should be removed by default unless the existing annotation requirements explicitly require it.

Recommended configurable defaults:

```yaml
candidate_encoding:
  codec: libx264
  pixel_format: yuv420p
  preset: fast
  crf: 23
  faststart: true
  retain_audio: false
  keyframe_interval_s: 2
```

Using `libx264` is preferred initially because the remote server has substantial CPU capacity and GPU resources may be needed by the perception pipeline.

Hardware H.264 encoding may be added as a configurable alternative, but it must generate equivalent Label Studio-compatible output.

---

## Candidate validation

Every candidate must be inspected with `ffprobe` before upload.

Required validation:

```text
codec_name == h264
pix_fmt == yuv420p
container includes mp4
duration > 0
video stream exists
```

A candidate that fails validation must not be uploaded as a successful result.

The source video must remain `processed=false` if any required candidate fails encoding, validation, or upload.

---

## Candidate naming

Candidate identifiers must be stable and deterministic.

Recommended structure:

```text
candidates/videos/<source_video_id>/<candidate_id>.mp4
```

Example:

```text
anon/candidates/videos/camera_01_video_001/
    camera_01_video_001_candidate_0001.mp4
    camera_01_video_001_candidate_0002.mp4
```

The source video ID must avoid collisions between files with identical basenames in different directories.

It may be derived from:

* the normalized relative source key; or
* a normalized filename plus a short hash of the complete S3 object key.

---

## Candidate metadata

Upload one metadata document per processed source video:

```text
anon/candidates/metadata/<source_video_id>.json
```

Minimum content:

```json
{
  "source_bucket": "chillnbite-cameras",
  "source_key": "anon/camera_01/video_001.mp4",
  "source_video_id": "camera_01_video_001",
  "candidate_count": 2,
  "candidates": [
    {
      "candidate_id": "camera_01_video_001_candidate_0001",
      "candidate_key": "anon/candidates/videos/camera_01_video_001/camera_01_video_001_candidate_0001.mp4",
      "source_start_s": 34.2,
      "source_end_s": 49.8,
      "duration_s": 15.6,
      "codec": "h264",
      "pixel_format": "yuv420p"
    }
  ]
}
```

The metadata must allow annotations made against candidate-local timestamps to be mapped back to the original source video timeline.

---

## Run report

Upload one run report to:

```text
anon/candidates/runs/<run_id>.json
```

It should include:

```text
requested video count
selected video count
completed video count
failed video count
skipped video count
candidate count
start and completion timestamps
worker configuration
failed source keys and errors
```

This report is diagnostic only. The authoritative processed state remains `process_for_candidates.csv`.

---

## S3 credentials and permissions

Credentials must use the normal AWS credential chain, such as:

* an IAM role attached to the remote server;
* environment variables;
* an AWS shared credentials file.

Credentials must not be committed to the repository.

Required permissions should be restricted to the relevant bucket and prefixes:

```text
s3:ListBucket
s3:GetObject
s3:HeadObject
s3:PutObject
```

The remote process requires:

* read access to `anon/`;
* write access to `anon/process_for_candidates.csv`;
* write access to `anon/candidates/`.

Local Label Studio participants require read access only to:

```text
anon/candidates/videos/
```

---

## Failure and resume behavior

A failure for one source video must not stop unrelated jobs unless `--fail-fast` is enabled.

On failure:

1. Record the error in the run report and logs.
2. Do not mark the source as processed.
3. Do not upload incomplete metadata as a successful result.
4. Remove partial local outputs unless debugging retention is enabled.
5. Allow the video to be selected again in a later run.

On restart, the command must select only entries where:

```text
processed == false
```

Previously completed source videos must not be regenerated unless `--overwrite` is explicitly used.

---

## Logging

Log at least:

```text
source videos discovered
CSV entries created or preserved
unprocessed videos available
videos selected
active jobs
download progress
pipeline progress
candidates generated
encoding results
validation results
uploaded S3 keys
processed ledger updates
failed videos
final batch summary
```

---

## Tests

Automated tests must cover:

* source discovery under `anon/`;
* exclusion of `anon/candidates/`;
* creation of `process_for_candidates.csv`;
* preservation of existing processed flags;
* addition of newly discovered source videos;
* deterministic selection of the requested number of unprocessed videos;
* successful processing-state update;
* failure leaving `processed=false`;
* zero-candidate source marked as processed after successful completion;
* bounded source-job concurrency;
* bounded transfer concurrency;
* bounded GPU concurrency;
* bounded encoding concurrency;
* H.264 MP4 encoding;
* `ffprobe` validation;
* candidate and metadata upload;
* resume after interruption;
* source filenames with duplicate basenames in different directories.

Use mocked S3 storage or an S3-compatible local test service for automated tests.

---

## Acceptance criteria

The task is complete when:

1. The remote server can connect to `s3://chillnbite-cameras/anon/`.
2. Source videos are discovered without including generated outputs.
3. `process_for_candidates.csv` is created and synchronized with the source objects.
4. A target count can be supplied from the CLI and Makefile.
5. The requested number of unprocessed videos is selected automatically.
6. The existing Tasks 3–5 pipeline runs against downloaded source videos.
7. Multiple source videos can be processed concurrently.
8. Network, GPU, and encoding concurrency are independently configurable.
9. Candidate clips are uploaded under `anon/candidates/videos/`.
10. Every uploaded candidate is H.264 MP4 with `yuv420p` and fast-start metadata.
11. Candidate metadata maps each candidate back to the original source video and timeline.
12. A source video is marked processed only after successful candidate and metadata publication.
13. Failed and interrupted videos remain eligible for a later retry.
14. Local Label Studio instances can load the candidate clips directly from S3.
15. No AWS credentials are stored in source control.

---

## Initial execution sequence

After implementation, run the following batches.

### Validation batch

```bash
make candidates-remote \
  CANDIDATE_TARGET_COUNT=5 \
  CANDIDATE_WORKERS=2 \
  CANDIDATE_TRANSFER_WORKERS=4 \
  CANDIDATE_GPU_WORKERS=1 \
  CANDIDATE_ENCODE_WORKERS=4
```

Validate:

* all five ledger entries become `true`;
* candidates exist in S3;
* candidate metadata is correct;
* every candidate passes `ffprobe`;
* local Label Studio can load and seek through the videos.

### First production batch

```bash
make candidates-remote \
  CANDIDATE_TARGET_COUNT=100 \
  CANDIDATE_WORKERS=8 \
  CANDIDATE_TRANSFER_WORKERS=8 \
  CANDIDATE_GPU_WORKERS=1 \
  CANDIDATE_ENCODE_WORKERS=8
```

Tune the worker counts after observing:

* S3 transfer throughput;
* GPU utilization;
* CPU utilization;
* local temporary disk consumption;
* candidate-processing throughput;
* failure rate.

---

## Non-goals

* Hosting Label Studio on the remote server.
* Copying candidate videos into each participant’s local filesystem.
* Merging local Label Studio annotation exports.
* Modifying the original source videos.
* Distributed processing across multiple remote servers.
* Multiple simultaneous processes updating `process_for_candidates.csv`.
* Reimplementing Tasks 3–5.
* Automatically creating Label Studio annotations.

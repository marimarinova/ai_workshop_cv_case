# ai_workshop_cv_case
A repo to hold the work on the Summer School 2026 CV case.

# Case details are separate at:
https://github.com/Marchev-Science/case-pickup-putdown-event-detection


# Local ingestion and Task 3 smoke test

This workflow configures access to the source bucket, downloads and indexes videos through ingestion, and runs Layer 0A person triage against a cached MP4.

Local credentials, cached videos, model weights, previews, and generated outputs must not be committed.

## Prerequisites

Activate the project environment and install the package:

```bash
make install-dev
```

Confirm that the CLI and Makefile targets are available:

```bash
pickup-putdown --help
make help
```

Ensure `.local/` and downloaded model weights are ignored by Git.

## 1. Configure the storage environment

Run:

```bash
make env-setup
```

Enter:

* AWS access key ID;
* AWS secret access key;
* S3 bucket URI;
* S3 region;
* optional custom endpoint;
* whether anonymous access should be used.

The values are saved to:

```text
.local/env/storage.env
```

The file is created with permissions restricted to the current user. It is sourced automatically by the `ingest` target, so the credentials do not need to be exported manually in every terminal.

Verify the file permissions without printing its contents:

```bash
ls -l .local/env/storage.env
```

Do not display, commit, or share this file.

## 2. Run ingestion

Run:

```bash
make ingest
```

This executes:

```bash
pickup-putdown ingest --config configs/storage.yaml
```

using the environment saved by `make env-setup`.

The ingestion step indexes the configured bucket, probes video metadata, and downloads or populates the configured local video cache.

Find the cached MP4 files:

```bash
find .local -type f -iname '*.mp4' | sort
```

Inspect one cached video:

```bash
ffprobe -v error \
  -select_streams v:0 \
  -show_entries stream=codec_name,width,height,avg_frame_rate,duration \
  -of default=noprint_wrappers=1 \
  /path/to/cached/video.mp4
```

Choose a clip with at least one clearly visible person for the initial positive smoke test.

## 3. Run Task 3 person triage

Run:

```bash
make task-3
```

By default, the target selects the first cached MP4 found under `.local`, excluding generated triage previews and output directories.

Before starting, it prints the selected input:

```text
Triage input: ...
Triage output: .local/triage_acceptance/output_single
```

If `models/person_detector.pt` does not exist, the target downloads the small pretrained YOLO11n detector automatically.

To select a specific cached clip, override `TRIAGE_INPUT`:

```bash
make task-3 \
  TRIAGE_INPUT=.local/path/to/cache/example.mp4
```

To use a different output directory:

```bash
make task-3 \
  TRIAGE_INPUT=.local/path/to/cache/example.mp4 \
  TRIAGE_OUTPUT=.local/triage_acceptance/example_run
```

The command executed by the target is equivalent to:

```bash
pickup-putdown triage \
  /path/to/cached/video.mp4 \
  --config configs/triage.yaml \
  --output-dir .local/triage_acceptance/output_single \
  --verbose
```

## Expected runtime output

A successful run should report information similar to:

```text
Triage: 1 video(s) to process.
Processing example.mp4...
Loading YOLO model from models/person_detector.pt on cuda
Triage complete: ... observations, ... tracks (... stable)
Active spans for ...: ... spans
```

The `on cuda` message confirms that YOLO inference is running on the GPU.

## Preview rendering

Detection and tracking may complete before the CLI exits because the preview video is rendered afterward.

For long 4K inputs, preview rendering can take substantially longer than detection because decoding, overlay drawing, and encoding are largely CPU-bound.

Find the active process:

```bash
ps aux | grep '[p]ickup-putdown triage'
```

Check which output file it is writing:

```bash
lsof -p <PID> | grep 'triage_acceptance'
```

Monitor the preview size:

```bash
watch -n 3 ls -lh \
  .local/triage_acceptance/output_single/triage_previews/*.mp4
```

If the file size continues to increase and the process is consuming CPU, the renderer is still working and is not hung.

## Inspect generated artifacts

After the command exits, list the outputs:

```bash
find .local/triage_acceptance/output_single \
  -type f \
  -printf '%p\t%k KB\n' |
sort
```

Expected outputs include Task 3 artifacts such as:

```text
tracks_person.parquet
active_spans.parquet
clips.parquet
triage_sampling_report.parquet
triage_previews/
```

The exact paths depend on the implemented output layout.

## One-step interactive alternative

To configure the environment and immediately run ingestion:

```bash
make env-ingest
```

The recommended repeatable workflow remains:

```bash
make env-setup
make ingest
make task-3
```

## Cleanup and security

Do not commit:

* `.local/env/storage.env`;
* cached source videos;
* generated previews;
* Task 3 output artifacts;
* `models/person_detector.pt`;
* AWS credentials or endpoint secrets.

Before committing, check:

```bash
git status --short
```


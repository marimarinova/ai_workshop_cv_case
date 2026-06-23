# Parallel Download and Processing

Run S3 downloads and local candidate processing as background jobs that survive SSH disconnect.

## Prerequisites

Activate the project environment in every terminal before running commands:

```bash
source /path/to/venv/bin/activate
cd /home/naim/repos/ai_workshop_cv_case
```

## Clean up interrupted downloads

If a previous download was interrupted, remove any leftover `.part` files:

```bash
find .local/source_videos -name '*.part.*' -delete
```

## Step 1: Start downloads (Terminal 1)

```bash
ssh user@host
source /path/to/venv/bin/activate
cd /home/naim/repos/ai_workshop_cv_case

nohup bash scripts/download_all_s3.sh > /dev/null 2>&1 &
```

Monitor while connected:

```bash
tail -f scripts/download_all.log
```

When you see `All clips downloaded. Done.` — downloads finished.

## Step 2: Start processing (Terminal 2)

**Sequential (after downloads finish):**

```bash
ssh user@host
source /path/to/venv/bin/activate
cd /home/naim/repos/ai_workshop_cv_case

nohup bash scripts/process_all_local.sh > /dev/null 2>&1 &
```

**Continuous (overlap with downloads — recommended for large sets):**

```bash
ssh user@host
source /path/to/venv/bin/activate
cd /home/naim/repos/ai_workshop_cv_case

nohup bash scripts/process_all_local.sh --continuous > /dev/null 2>&1 &
```

In continuous mode, the script processes available videos, then waits for new downloads instead of exiting. It auto-exits after 10 minutes of idle time (no new videos arriving).

Monitor while connected:

```bash
tail -f scripts/process_all.log
cat .local/process_status.txt
```

## After SSH disconnect

Reconnect and check status:

```bash
source /path/to/venv/bin/activate
cd /home/naim/repos/ai_workshop_cv_case

# Quick status
cat .local/process_status.txt
cat .local/candidate_staging/local_processing.csv

# Recent log tail
tail -50 scripts/download_all.log
tail -50 scripts/process_all.log

# Check if jobs are still running
ps aux | grep -E 'download_all_s3|process_all_local'

# Stop a running job if needed
pkill -f download_all_s3
pkill -f process_all_local
```

## Estimated runtime

| Stage | Videos | Est. time |
|---|---|---|
| Download | ~30 clips | 10–20 min |
| Processing | ~30 clips (8 GPU workers) | 10–15 min |

## Failure handling

The process script tracks failures in `.local/processing_failed.json`. Videos exceeding 3 retries are added to `.local/processing_skip.txt` and excluded from future runs.

To retry a skipped video:

```bash
# Remove from skip list
grep -v 'video_name.mp4' .local/processing_skip.txt > .local/processing_skip.tmp
mv .local/processing_skip.tmp .local/processing_skip.txt

# Rerun
nohup bash scripts/process_all_local.sh > /dev/null 2>&1 &
```

## Step 3: Upload (when S3 write access is available)

```bash
make candidates-upload CANDIDATE_TARGET_COUNT=0
```

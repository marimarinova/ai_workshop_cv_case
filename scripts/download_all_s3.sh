#!/usr/bin/env bash
set -Eeuo pipefail

COUNT=10
SLEEP_BETWEEN=5
MAX_RETRIES=3
LOG_FILE="scripts/download_all.log"
OFFSET=0

log() {
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"
}

log "Starting download loop (count=$COUNT, max_retries=$MAX_RETRIES)" | tee -a "$LOG_FILE"

while true; do
    retries=0
    batch_ok=false

    while true; do
        log "=== Batch: offset=$OFFSET count=$COUNT (attempt $((retries + 1))/$MAX_RETRIES) ===" | tee -a "$LOG_FILE"

        output=""
        exit_code=0
        output=$(python scripts/download_s3_sample.py --count "$COUNT" --offset "$OFFSET" 2>&1) || exit_code=$?

        echo "$output" >> "$LOG_FILE"

        if echo "$output" | grep -q "All matching clips already downloaded"; then
            log "All clips downloaded. Done." | tee -a "$LOG_FILE"
            exit 0
        fi

        if echo "$output" | grep -q "No clips at offset"; then
            log "No more clips at offset $OFFSET. Done." | tee -a "$LOG_FILE"
            exit 0
        fi

        if [ "$exit_code" -eq 0 ]; then
            batch_ok=true
            break
        fi

        retries=$((retries + 1))
        if [ "$retries" -ge "$MAX_RETRIES" ]; then
            log "FAILED: batch at offset=$OFFSET exceeded $MAX_RETRIES retries. Aborting." | tee -a "$LOG_FILE"
            exit 1
        fi

        wait_time=$((SLEEP_BETWEEN * retries))
        log "Batch failed (exit $exit_code). Retry $retries/$MAX_RETRIES in ${wait_time}s..." | tee -a "$LOG_FILE"
        sleep "$wait_time"
    done

    OFFSET=$((OFFSET + COUNT))
    log "Batch succeeded. Sleeping ${SLEEP_BETWEEN}s before next batch..." | tee -a "$LOG_FILE"
    sleep "$SLEEP_BETWEEN"
done

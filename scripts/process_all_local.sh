#!/usr/bin/env bash
set -Eeuo pipefail

CONTINUOUS=false
for arg in "$@"; do
    case "$arg" in
        -c|--continuous) CONTINUOUS=true ;;
    esac
done

MAX_RUNS=10
MAX_RETRIES=3
SLEEP_BETWEEN=10
IDLE_TIMEOUT=600
LOG_FILE="scripts/process_all.log"
LEDGER=".local/candidate_staging/local_processing.csv"
FAILED_JSON=".local/processing_failed.json"
SKIP_FILE=".local/processing_skip.txt"
STATUS_FILE=".local/process_status.txt"

count_remaining() {
    grep -c ',true,false,false,' "$LEDGER" 2>/dev/null || echo 0
}

count_skipped() {
    if [ -f "$SKIP_FILE" ]; then
        grep -c '.' "$SKIP_FILE" 2>/dev/null || echo 0
    else
        echo 0
    fi
}

log() {
    local ts
    ts="[$(date -u '+%Y-%m-%dT%H:%M:%SZ')]"
    echo "$ts $*" | tee -a "$LOG_FILE"
}

update_status() {
    local run_num="$1" completed="$2" failed="$3" remaining="$4"
    local mode="once"
    [ "$CONTINUOUS" = true ] && mode="continuous"
    echo "[${mode}] Run ${run_num}/${MAX_RUNS} | Completed: ${completed} | Failed: ${failed} | Remaining: ${remaining} | Last: $(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS_FILE"
}

load_failed_json() {
    if [ -f "$FAILED_JSON" ]; then
        cat "$FAILED_JSON"
    else
        echo '{}'
    fi
}

save_failed_json() {
    local data="$1"
    echo "$data" | python -c "
import sys, json
d = json.load(sys.stdin)
with open('$FAILED_JSON', 'w') as f:
    json.dump(d, f, indent=2)
"
}

mark_failed_video() {
    local video="$1" error="$2"
    local ts
    ts="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    local data
    data=$(load_failed_json)

    data=$(echo "$data" | python -c "
import sys, json
d = json.load(sys.stdin)
v = '$video'
e = '''$error'''
if v in d:
    d[v]['count'] += 1
    d[v]['last_seen'] = '$ts'
else:
    d[v] = {'error': e, 'count': 1, 'first_seen': '$ts', 'last_seen': '$ts'}
print(json.dumps(d))
")

    save_failed_json "$data"

    local cnt
    cnt=$(echo "$data" | python -c "import sys,json; d=json.load(sys.stdin); print(d.get('$video',{}).get('count',0))")

    if [ "$cnt" -ge "$MAX_RETRIES" ]; then
        if ! grep -qF "$video" "$SKIP_FILE" 2>/dev/null; then
            echo "$video $(date -u '+%Y-%m-%dT%H:%M:%SZ')" >> "$SKIP_FILE"
            log "PERMANENT: $video exceeded $MAX_RETRIES retries → added to $SKIP_FILE"
        fi
    fi
}

log "=== PROCESS ALL START ==="
log "Config: gpu_workers=8 encode_workers=12 max_retries=$MAX_RETRIES continuous=$CONTINUOUS"

remaining=$(count_remaining)
skipped=$(count_skipped)
log "Videos ready: $remaining"
log "Skipped: $skipped (from $SKIP_FILE)"

if [ "$remaining" -eq 0 ]; then
    if [ "$CONTINUOUS" = true ]; then
        log "No videos ready yet. Waiting for downloads (idle timeout: ${IDLE_TIMEOUT}s)..."
    else
        log "No videos ready for processing. Done."
        exit 0
    fi
fi

total_completed=0
total_failed=0
idle_start=0

run=0
while [ "$run" -lt "$MAX_RUNS" ]; do
    run=$((run + 1))
    remaining=$(count_remaining)

    if [ "$remaining" -eq 0 ]; then
        if [ "$CONTINUOUS" = true ]; then
            if [ "$idle_start" -eq 0 ]; then
                idle_start=$(date +%s)
            else
                idle_elapsed=$(( $(date +%s) - idle_start ))
                if [ "$idle_elapsed" -ge "$IDLE_TIMEOUT" ]; then
                    log "Idle for ${idle_elapsed}s (timeout ${IDLE_TIMEOUT}s). Downloads likely done."
                    break
                fi
                log "Idle ${idle_elapsed}s/${IDLE_TIMEOUT}s. Waiting for new downloads..."
            fi
            sleep "$SLEEP_BETWEEN"
            continue
        else
            log "All videos processed."
            break
        fi
    fi

    if [ "$CONTINUOUS" = true ]; then
        idle_start=0
    fi

    log ""
    log "─── Run $run/$MAX_RUNS ───"
    log "Selected: $remaining"

    output=$(pickup-putdown candidates-process-local \
        --target-count 9999 \
        --gpu-workers 5 \
        --encode-workers 12 \
        --skip-file "$SKIP_FILE" \
        -v 2>&1) || true

    echo "$output" >> "$LOG_FILE"

    completed=$(echo "$output" | grep -oP '(?<=Completed:\s)\d+' | head -1 || echo 0)
    failed=$(echo "$output" | grep -oP '(?<=Failed:\s)\d+' | head -1 || echo 0)
    candidates=$(echo "$output" | grep -oP '(?<=Candidates:\s)\d+' | head -1 || echo 0)

    total_completed=$((total_completed + completed))
    total_failed=$((total_failed + failed))

    log "Completed: $completed | Failed: $failed | Candidates: $candidates"

    if [ "$failed" -gt 0 ]; then
        while IFS= read -r line; do
            video=$(echo "$line" | sed -n 's/^[[:space:]]*\([^:]*\):.*/\1/p')
            error=$(echo "$line" | sed -n 's/^[[:space:]]*[^:]*: \(.*\)/\1/p')
            if [ -n "$video" ] && [ -n "$error" ]; then
                log "  Failed: $video ($error)"
                mark_failed_video "$video" "$error"
            fi
        done <<< "$(echo "$output" | grep -A1 "Failed sources:" | grep '^[[:space:]]')"
    fi

    update_status "$run" "$total_completed" "$total_failed" "$remaining"

    new_remaining=$(count_remaining)

    if [ "$new_remaining" -eq "$remaining" ] && [ "$failed" -gt 0 ]; then
        log ""
        log "STUCK: no progress after run $run ($remaining videos remain, all failed)"
        log ""
        log "Permanently failed videos:"
        if [ -f "$FAILED_JSON" ]; then
            python -c "
import json
with open('$FAILED_JSON') as f:
    d = json.load(f)
for v, info in sorted(d.items(), key=lambda x: -x[1]['count']):
    print(f'  {v} ({info[\"count\"]} retries)')
" 2>/dev/null | tee -a "$LOG_FILE"
        fi
        break
    fi

    remaining=$new_remaining

    if [ "$remaining" -eq 0 ]; then
        if [ "$CONTINUOUS" = true ]; then
            idle_start=$(date +%s)
            log "All current videos processed. Waiting for new downloads..."
            sleep "$SLEEP_BETWEEN"
            continue
        else
            break
        fi
    fi

    log "Remaining: $remaining"
    log "Sleeping ${SLEEP_BETWEEN}s..."
    sleep "$SLEEP_BETWEEN"
done

remaining=$(count_remaining)
log ""
log "=== PROCESS ALL DONE ==="
log "Total completed: $total_completed | Total failed: $total_failed | Remaining: $remaining"

if [ -f "$FAILED_JSON" ]; then
    skipped_videos=$(python -c "
import json
with open('$FAILED_JSON') as f:
    d = json.load(f)
perma = [v for v, info in d.items() if info['count'] >= $MAX_RETRIES]
if perma:
    print('Permanently failed (max retries reached):')
    for v in perma:
        print(f'  {v} ({d[v][\"count\"]} retries)')
    print(f'Skipped list: $SKIP_FILE')
    print(f'Failure details: $FAILED_JSON')
    print('To retry: remove video from skip list, rerun script')
" 2>/dev/null)
    if [ -n "$skipped_videos" ]; then
        echo "$skipped_videos" | tee -a "$LOG_FILE"
    fi
fi

update_status "$run" "$total_completed" "$total_failed" "$remaining"

if [ "$remaining" -eq 0 ]; then
    log "All videos processed successfully."
    exit 0
else
    log "Stopped with $remaining video(s) remaining."
    exit 1
fi

#!/bin/bash
export VIDEO=".local/triage_acceptance/videos/person_clear.mp4"
export TRIAGE_DIR=".local/triage_acceptance/output_single"

export PERSON_TRACKS="$TRIAGE_DIR/tracks_person.parquet"
export ACTIVE_SPANS="$TRIAGE_DIR/active_spans.parquet"
export CLIPS="$TRIAGE_DIR/clips.parquet"

export CONFIG="configs/proposals.yaml"
export SHELF_CONFIG="configs/shelves.yaml"

export RUN_ID="$(date +%Y%m%d_%H%M%S)"
export OUTPUT_DIR=".local/task5_acceptance/$RUN_ID"

export mkdir -p "$OUTPUT_DIR"
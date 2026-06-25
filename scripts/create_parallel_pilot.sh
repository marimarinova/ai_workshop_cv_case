#!/usr/bin/env bash
set -Eeuo pipefail

METADATA_DIR=".local/candidate_staging/metadata"
OUTPUT_DIR="annotation/pilot_assignments"
TASKS_PER_ANNOTATOR=40
OVERLAP_COUNT=5
SEED=42
S3_BUCKET="chillnbite-cameras"
VIDEO_URL_MODE="s3_storage"

ANNOTATORS=("galya" "marieta" "todor" "lyubomir")

usage() {
  cat <<'EOF'
Create deterministic Label Studio pilot assignments for multiple annotators.

Each annotator receives:
  - a unique subset;
  - the same shared overlap subset for agreement measurement.

Defaults:
  Annotators:             galya, marieta, todor, lyubomir
  Tasks per annotator:    40
  Shared overlap tasks:   5
  Seed:                   42
  Metadata directory:     .local/candidate_staging/metadata
  Output directory:       annotation/pilot_assignments
  S3 bucket:              chillnbite-cameras
  Video URL mode:         s3_storage

Usage:
  create_parallel_pilot.sh [options]

Options:
  --metadata-dir PATH          Candidate metadata directory.
  --output-dir PATH            Output directory.
  --tasks-per-annotator N      Total tasks in each annotator JSON.
  --overlap-count N            Shared tasks included in every JSON.
  --seed N                     Deterministic sampling seed.
  --s3-bucket NAME             Candidate-video S3 bucket.
  --video-url-mode MODE        Task-builder video mode.
  -h, --help                   Show this help.

Example:
  ./scripts/create_parallel_pilot.sh

Outputs:
  annotation/pilot_assignments/tasks_pilot_master.json
  annotation/pilot_assignments/tasks_pilot_galya.json
  annotation/pilot_assignments/tasks_pilot_marieta.json
  annotation/pilot_assignments/tasks_pilot_todor.json
  annotation/pilot_assignments/tasks_pilot_lyubomir.json
  annotation/pilot_assignments/assignment_manifest.tsv
EOF
}

while (($#)); do
  case "$1" in
    --metadata-dir)
      METADATA_DIR="${2:?Missing value for --metadata-dir}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:?Missing value for --output-dir}"
      shift 2
      ;;
    --tasks-per-annotator)
      TASKS_PER_ANNOTATOR="${2:?Missing value for --tasks-per-annotator}"
      shift 2
      ;;
    --overlap-count)
      OVERLAP_COUNT="${2:?Missing value for --overlap-count}"
      shift 2
      ;;
    --seed)
      SEED="${2:?Missing value for --seed}"
      shift 2
      ;;
    --s3-bucket)
      S3_BUCKET="${2:?Missing value for --s3-bucket}"
      shift 2
      ;;
    --video-url-mode)
      VIDEO_URL_MODE="${2:?Missing value for --video-url-mode}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

command -v jq >/dev/null 2>&1 || {
  echo "Error: jq is required." >&2
  exit 1
}

command -v pickup-putdown >/dev/null 2>&1 || {
  echo "Error: pickup-putdown is not available in PATH." >&2
  exit 1
}

[[ -d "$METADATA_DIR" ]] || {
  echo "Error: metadata directory does not exist: $METADATA_DIR" >&2
  exit 1
}

[[ "$TASKS_PER_ANNOTATOR" =~ ^[1-9][0-9]*$ ]] || {
  echo "Error: --tasks-per-annotator must be a positive integer." >&2
  exit 2
}

[[ "$OVERLAP_COUNT" =~ ^[0-9]+$ ]] || {
  echo "Error: --overlap-count must be a non-negative integer." >&2
  exit 2
}

if (( OVERLAP_COUNT >= TASKS_PER_ANNOTATOR )); then
  echo "Error: overlap count must be smaller than tasks per annotator." >&2
  exit 2
fi

annotator_count="${#ANNOTATORS[@]}"
unique_per_annotator=$((TASKS_PER_ANNOTATOR - OVERLAP_COUNT))
master_count=$((OVERLAP_COUNT + annotator_count * unique_per_annotator))

mkdir -p "$OUTPUT_DIR"

master_file="$OUTPUT_DIR/tasks_pilot_master.json"
manifest_file="$OUTPUT_DIR/assignment_manifest.tsv"

echo "Creating deterministic master pilot..."
echo "  Annotators: ${ANNOTATORS[*]}"
echo "  Tasks per annotator: $TASKS_PER_ANNOTATOR"
echo "  Shared overlap: $OVERLAP_COUNT"
echo "  Unique per annotator: $unique_per_annotator"
echo "  Required unique candidates: $master_count"

pickup-putdown annotation-build-tasks \
  --candidate-metadata-dir "$METADATA_DIR" \
  --output "$master_file" \
  --limit "$master_count" \
  --seed "$SEED" \
  --video-url-mode "$VIDEO_URL_MODE" \
  --s3-bucket "$S3_BUCKET"

actual_count="$(jq 'length' "$master_file")"
if (( actual_count != master_count )); then
  echo "Error: expected $master_count master tasks, but generated $actual_count." >&2
  exit 1
fi

printf 'annotator\tcandidate_id\tclip_id\tshared_overlap\n' > "$manifest_file"

for index in "${!ANNOTATORS[@]}"; do
  annotator="${ANNOTATORS[$index]}"
  output_file="$OUTPUT_DIR/tasks_pilot_${annotator}.json"

  unique_start=$((OVERLAP_COUNT + index * unique_per_annotator))
  unique_end=$((unique_start + unique_per_annotator))

  jq \
    --arg annotator "$annotator" \
    --argjson overlap "$OVERLAP_COUNT" \
    --argjson unique_start "$unique_start" \
    --argjson unique_end "$unique_end" \
    '
      (.[0:$overlap]
        | map(
            .data.assigned_annotator = $annotator
            | .data.pilot_overlap = true
          )
      )
      +
      (.[$unique_start:$unique_end]
        | map(
            .data.assigned_annotator = $annotator
            | .data.pilot_overlap = false
          )
      )
    ' "$master_file" > "$output_file"

  assignment_count="$(jq 'length' "$output_file")"
  if (( assignment_count != TASKS_PER_ANNOTATOR )); then
    echo "Error: ${annotator} received ${assignment_count} tasks; expected ${TASKS_PER_ANNOTATOR}." >&2
    exit 1
  fi

  jq -r \
    --arg annotator "$annotator" \
    '.[] |
      [
        $annotator,
        .data.candidate_id,
        .data.clip_id,
        (.data.pilot_overlap | tostring)
      ] | @tsv
    ' "$output_file" >> "$manifest_file"

  echo "Wrote ${assignment_count} tasks: ${output_file}"
done

echo
echo "Checking assignment integrity..."

all_unique_count="$(
  jq -r '.[].data.candidate_id' "$master_file" |
  sort -u |
  wc -l |
  tr -d ' '
)"

if (( all_unique_count != master_count )); then
  echo "Error: duplicate candidate IDs found in the master file." >&2
  exit 1
fi

for annotator in "${ANNOTATORS[@]}"; do
  file="$OUTPUT_DIR/tasks_pilot_${annotator}.json"
  duplicate_count="$(
    jq -r '.[].data.candidate_id' "$file" |
    sort |
    uniq -d |
    wc -l |
    tr -d ' '
  )"

  if (( duplicate_count != 0 )); then
    echo "Error: duplicate candidates found in ${file}." >&2
    exit 1
  fi
done

echo "Assignment integrity check passed."
echo
echo "Files ready to send:"
for annotator in "${ANNOTATORS[@]}"; do
  echo "  $OUTPUT_DIR/tasks_pilot_${annotator}.json"
done
echo
echo "Shared overlap candidates: $OVERLAP_COUNT"
echo "Manifest: $manifest_file"
echo
echo "Each teammate should import only their named JSON through Label Studio Data Import."

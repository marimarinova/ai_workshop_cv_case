#!/usr/bin/env bash
set -Eeuo pipefail

BUCKET="chillnbite-cameras"
PREFIX="anon/candidates/videos/"
REGION="eu-central-1"
DRY_RUN=false

usage() {
  cat <<'HELP'
Fix the Content-Type of candidate MP4 objects in S3.

Usage:
  fix_candidate_s3_content_type.sh [options]

Options:
  --bucket NAME     S3 bucket name.
                    Default: chillnbite-cameras
  --prefix PREFIX   Candidate object prefix.
                    Default: anon/candidates/videos/
  --region REGION   AWS region.
                    Default: eu-central-1
  --dry-run         Show objects that would be updated without changing them.
  -h, --help        Show this help.

Examples:
  ./scripts/fix_candidate_s3_content_type.sh --dry-run
  ./scripts/fix_candidate_s3_content_type.sh
HELP
}

while (($#)); do
  case "$1" in
    --bucket)
      BUCKET="${2:?Missing value for --bucket}"
      shift 2
      ;;
    --prefix)
      PREFIX="${2:?Missing value for --prefix}"
      shift 2
      ;;
    --region)
      REGION="${2:?Missing value for --region}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=true
      shift
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

command -v aws >/dev/null 2>&1 || {
  echo "Error: aws CLI is required." >&2
  exit 1
}

command -v jq >/dev/null 2>&1 || {
  echo "Error: jq is required." >&2
  exit 1
}

tmp_keys="$(mktemp)"
trap 'rm -f "$tmp_keys"' EXIT

echo "Listing MP4 objects under s3://${BUCKET}/${PREFIX} ..."

aws s3api list-objects-v2 \
  --bucket "$BUCKET" \
  --prefix "$PREFIX" \
  --region "$REGION" \
  --output json |
jq -r '
  .Contents[]?.Key
  | select(ascii_downcase | endswith(".mp4"))
' > "$tmp_keys"

total="$(wc -l < "$tmp_keys" | tr -d ' ')"

if [[ "$total" -eq 0 ]]; then
  echo "No MP4 objects found."
  exit 0
fi

echo "Found ${total} MP4 object(s)."

checked=0
updated=0
skipped=0
failed=0

while IFS= read -r key; do
  ((checked += 1))

  if ! head_json="$(
    aws s3api head-object \
      --bucket "$BUCKET" \
      --key "$key" \
      --region "$REGION" \
      --output json
  )"; then
    echo "[${checked}/${total}] ERROR: could not inspect ${key}" >&2
    ((failed += 1))
    continue
  fi

  current_type="$(jq -r '.ContentType // ""' <<<"$head_json")"

  if [[ "$current_type" == "video/mp4" ]]; then
    echo "[${checked}/${total}] OK: ${key}"
    ((skipped += 1))
    continue
  fi

  if [[ "$DRY_RUN" == true ]]; then
    echo "[${checked}/${total}] WOULD UPDATE: ${key} (${current_type:-unset} -> video/mp4)"
    ((updated += 1))
    continue
  fi

  copy_args=(
    s3api copy-object
    --bucket "$BUCKET"
    --copy-source "${BUCKET}/${key}"
    --key "$key"
    --region "$REGION"
    --metadata-directive REPLACE
    --content-type "video/mp4"
  )

  metadata="$(jq -c '.Metadata // {}' <<<"$head_json")"
  copy_args+=(--metadata "$metadata")

  cache_control="$(jq -r '.CacheControl // empty' <<<"$head_json")"
  content_disposition="$(jq -r '.ContentDisposition // empty' <<<"$head_json")"
  content_encoding="$(jq -r '.ContentEncoding // empty' <<<"$head_json")"
  content_language="$(jq -r '.ContentLanguage // empty' <<<"$head_json")"

  [[ -n "$cache_control" ]] && copy_args+=(--cache-control "$cache_control")
  [[ -n "$content_disposition" ]] && copy_args+=(--content-disposition "$content_disposition")
  [[ -n "$content_encoding" ]] && copy_args+=(--content-encoding "$content_encoding")
  [[ -n "$content_language" ]] && copy_args+=(--content-language "$content_language")

  if aws "${copy_args[@]}" >/dev/null; then
    echo "[${checked}/${total}] UPDATED: ${key} (${current_type:-unset} -> video/mp4)"
    ((updated += 1))
  else
    echo "[${checked}/${total}] ERROR: failed to update ${key}" >&2
    ((failed += 1))
  fi
done < "$tmp_keys"

echo
echo "Completed."
echo "  Checked: ${checked}"
if [[ "$DRY_RUN" == true ]]; then
  echo "  Would update: ${updated}"
else
  echo "  Updated: ${updated}"
fi
echo "  Already correct: ${skipped}"
echo "  Failed: ${failed}"

if [[ "$failed" -gt 0 ]]; then
  exit 1
fi
